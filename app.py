import streamlit as st
import os, time, tempfile, re, math, uuid
import arxiv, requests
from datetime import datetime
from streamlit_agraph import agraph, Node, Edge, Config

# ================= 1. 环境检查 =================
try:
    import zhipuai, langchain_community, fitz
except ImportError as e:
    st.error(f"🚑 环境缺失库 -> {e.name}"); st.stop()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= 2. 页面配置 =================
st.set_page_config(page_title="AI 深度研读助手", layout="wide", page_icon="🎓")
st.markdown("""
<style>
    .stButton>button { width:100%; border-radius:8px; }
    .abstract-box {
        background:#f0f2f6; padding:15px; border-radius:8px;
        border-left:5px solid #4CAF50; font-size:.95em;
        line-height:1.6; margin-bottom:6px;
    }
    .contribution-box {
        background:linear-gradient(90deg,#fffbeb,#fef3c7);
        border-left:4px solid #f59e0b; padding:8px 14px;
        border-radius:6px; font-size:.88em; color:#78350f;
        margin-bottom:10px; font-weight:500;
    }
    .cite-badge {
        background:#ff4b4b; color:white; padding:2px 8px;
        border-radius:12px; font-size:.8em; font-weight:bold;
    }
    /* Topic 徽章 */
    .topic-badge {
        display:inline-block; background:#6366f1; color:white;
        padding:2px 10px; border-radius:20px; font-size:.78em;
        font-weight:600; margin-right:4px;
    }
    /* 知识漏洞提示框 */
    .gap-box {
        background:#fef2f2; border:1px solid #fca5a5;
        border-left:4px solid #ef4444; border-radius:8px;
        padding:12px 16px; margin:10px 0;
    }
    /* 笔记卡片 */
    .note-card {
        background:#f8fafc; border:1px solid #e2e8f0;
        border-radius:10px; padding:14px 18px; margin-bottom:10px;
    }
    .note-tag {
        display:inline-block; background:#e0e7ff; color:#3730a3;
        padding:1px 8px; border-radius:12px; font-size:.75em;
        margin-right:4px; font-weight:500;
    }
</style>
""", unsafe_allow_html=True)
st.title("📖 AI 深度研读助手 v3")

# ================= API Key =================
USER_API_KEY = "3bc598c9bf544f4fb3ecb23d771994df.l7gZBe4mawinxS31"
SS_API_KEY   = "8SwYzCFlra3KhzLD4A0KM2ejrtpz4FsGiGVx7xCb"

# ================= 3. 状态初始化 =================
defaults = {
    # ── 检索 ──
    "search_results": [],
    "suggested_query": "",
    "focus_paper_id": None,
    "contributions_cache": {},
    # ── 对话 ──
    "chat_history": [],
    # ── ★ Topic 系统 ──
    # topics = { topic_name: {"files": [], "chunks": [], "db": FAISS|None} }
    "topics": {"默认主题": {"files": [], "chunks": [], "db": None}},
    "active_topic": "默认主题",
    "selected_scope": "🌐 对比所有论文",
    # ── ★ 笔记系统 ──
    # notes = [{"id", "content", "question", "tags", "topic", "ts"}]
    "notes": [],
    # 暂存待保存的回答（由问答区写入）
    "pending_note": None,
    # ── ★ 知识漏洞：缓存最近图谱的 references ──
    "graph_references_cache": [],   # [{"title", "arxiv_id", "abstract"}]
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── 兼容旧 session（loaded_files / db 迁移到 topics）──
if "loaded_files" in st.session_state and st.session_state.loaded_files:
    t = st.session_state.topics["默认主题"]
    for f in st.session_state.loaded_files:
        if f not in t["files"]: t["files"].append(f)
    if "db" in st.session_state and st.session_state.db:
        t["db"] = st.session_state.db
    if "all_chunks" in st.session_state:
        t["chunks"] = st.session_state.all_chunks

# ================= 4. 工具函数 =================

def active_topic_data():
    return st.session_state.topics[st.session_state.active_topic]

def get_pure_arxiv_id(url_or_id):
    m = re.search(r'(\d{4}\.\d{4,5})', url_or_id)
    return m.group(1) if m else url_or_id.split('/')[-1].split('v')[0]

def fetch_citations(arxiv_id, ss_key=None):
    try:
        clean_id = get_pure_arxiv_id(arxiv_id)
        url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields=citationCount"
        headers = {"x-api-key": ss_key} if ss_key else {}
        time.sleep(0.02 if ss_key else 1.0)
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            return r.json().get('citationCount', 0)
    except: pass
    return 0

@st.cache_data(ttl=3600)
def fetch_graph_data(arxiv_id, ss_key=None):
    clean_id = get_pure_arxiv_id(arxiv_id)
    fields = (
        "paperId,title,year,citationCount,abstract,"
        "references.paperId,references.title,references.citationCount,"
        "references.year,references.abstract,references.externalIds,"
        "citations.paperId,citations.title,citations.citationCount,"
        "citations.year,citations.abstract,citations.externalIds"
    )
    url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields={fields}"
    headers = {"x-api-key": ss_key} if ss_key else {}
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=12)
            if r.status_code == 200: return r.json()
            elif r.status_code == 429: time.sleep((attempt+1)*2)
        except:
            if attempt == 2: return None
    return None

def get_one_line_contribution(abstract, title, api_key):
    key = title[:60]
    if key in st.session_state.contributions_cache:
        return st.session_state.contributions_cache[key]
    try:
        llm = ChatZhipuAI(model="glm-4-flash", api_key=api_key, temperature=0.0)
        res = llm.invoke(
            f"请用一句话（不超过40个汉字或20个英文单词）总结这篇论文的核心创新贡献。"
            f"只输出这一句话，不要前缀或解释。\n\n标题：{title}\n摘要：{abstract[:600]}"
        )
        result = res.content.strip()
    except: result = "（生成失败）"
    st.session_state.contributions_cache[key] = result
    return result

def fix_latex(text):
    if not text: return text
    return text.replace(r"\(","$").replace(r"\)","$").replace(r"\[","$$").replace(r"\]","$$")

# ── ★ 核心：支持 Topic 的入库函数 ──
def process_and_add_to_topic(file_path, file_name, api_key, topic_name=None):
    topic_name = topic_name or st.session_state.active_topic
    t = st.session_state.topics[topic_name]
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source_paper'] = file_name
            doc.metadata['topic'] = topic_name
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=200,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        chunks = [c for c in splitter.split_documents(docs) if len(c.page_content.strip()) > 20]
        t["chunks"].extend(chunks)
        embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
        batch = 10
        if t["db"] is None:
            t["db"] = FAISS.from_documents(chunks[:batch], embeddings)
            for i in range(batch, len(chunks), batch):
                t["db"].add_documents(chunks[i:i+batch]); time.sleep(0.1)
        else:
            for i in range(0, len(chunks), batch):
                t["db"].add_documents(chunks[i:i+batch]); time.sleep(0.1)
        if file_name not in t["files"]:
            t["files"].append(file_name)
        st.session_state.chat_history.append({
            "role": "system_notice",
            "content": f"📚 《{file_name}》已加入主题「{topic_name}」。"
        })
        return True
    except Exception as e:
        st.error(f"处理失败: {e}"); return False

def rebuild_topic_index(topic_name, api_key):
    t = st.session_state.topics[topic_name]
    if not t["chunks"]: t["db"] = None; return
    embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
    t["db"] = FAISS.from_documents(t["chunks"], embeddings)

# ── ★ 知识漏洞检测 ──
def detect_knowledge_gap(answer_text, docs):
    """判断回答是否存在知识漏洞"""
    gap_signals = ["资料不足", "没有找到", "无法回答", "未提及", "不清楚", "没有相关", "cannot find", "not mentioned"]
    if len(docs) < 3:
        return True
    for sig in gap_signals:
        if sig.lower() in answer_text.lower():
            return True
    return False

def get_gap_recommendations():
    """从图谱 references 缓存中找出尚未入库的推荐论文"""
    all_loaded = set()
    for t in st.session_state.topics.values():
        all_loaded.update(t["files"])
    recs = []
    for ref in st.session_state.graph_references_cache:
        title = ref.get("title", "")
        # 粗略匹配：标题没出现在已加载文件名中
        if not any(title[:20].lower() in f.lower() for f in all_loaded):
            recs.append(ref)
    return recs[:4]   # 最多推荐4篇

# ================= 5. 图谱渲染 =================
def render_connected_graph(data):
    if not data: return None, {}
    nodes, edges, paper_details = [], [], {}
    current_year = 2026

    def get_color(year, rel_type):
        if not year or year == 'Unknown': return "#94a3b8"
        age = max(0, current_year - int(year))
        if rel_type == 'seed': return "#FF4B4B"
        if rel_type == 'cite':
            return "#059669" if age < 2 else "#10b981" if age < 5 else "#6ee7b7"
        return "#2563eb" if age < 2 else "#3b82f6" if age < 5 else "#93c5fd"

    seed_id = data.get('paperId', 'root')
    paper_details[seed_id] = {
        "title": data.get('title', 'Seed Paper'),
        "abstract": data.get('abstract') or "无摘要",
        "year": data.get('year', 'Unknown'),
        "cites": data.get('citationCount', 0),
        "url": f"https://www.semanticscholar.org/paper/{seed_id}",
        "arxiv_id": None,
    }
    nodes.append(Node(id=seed_id, label="THIS PAPER", size=35, color=get_color(data.get('year'), 'seed')))
    seen = {seed_id}

    # ★ 缓存 references 用于知识漏洞推荐
    refs_for_gap = []

    combined = []
    for p in data.get('references', [])[:15]: p['rel_type']='ref'; combined.append(p)
    for p in data.get('citations', [])[:15]:  p['rel_type']='cite'; combined.append(p)

    for item in combined:
        p_id = item.get('paperId')
        if not p_id or p_id in seen: continue
        seen.add(p_id)
        title = item.get('title','Unknown')
        year  = item.get('year')
        cites = item.get('citationCount', 0)
        ext   = item.get('externalIds') or {}
        arxiv_id = ext.get('ArXiv')
        paper_details[p_id] = {
            "title": title, "abstract": item.get('abstract') or "暂无摘要。",
            "year": year, "cites": cites,
            "url": f"https://www.semanticscholar.org/paper/{p_id}",
            "arxiv_id": arxiv_id,
        }
        if item['rel_type'] == 'ref' and arxiv_id:
            refs_for_gap.append({"title": title, "arxiv_id": arxiv_id, "abstract": item.get('abstract','')})
        node_size = 15 + math.log(cites+1)*3.5
        nodes.append(Node(id=p_id, label=f"{title[:20]}...", size=node_size, color=get_color(year, item['rel_type'])))
        if item['rel_type']=='cite':
            edges.append(Edge(source=p_id, target=seed_id, color="#d1d5db", width=1, dashed=True))
        else:
            edges.append(Edge(source=seed_id, target=p_id, color="#94a3b8", width=1.5))

    st.session_state.graph_references_cache = refs_for_gap

    config = Config(width="100%", height=620, directed=True, physics=True,
                    nodeHighlightBehavior=True, highlightColor="#F7D154",
                    d3={'alphaTarget':0.05,'gravity':-250,'linkLength':150,'linkStrength':0.1})
    clicked_id = agraph(nodes=nodes, edges=edges, config=config)
    return clicked_id, paper_details

# ================= 6. 侧边栏 =================
with st.sidebar:
    st.header("🎛️ 控制台")
    user_api_key = USER_API_KEY
    ss_api_key   = SS_API_KEY
    st.success("🚀 高速调研模式已激活")

    st.markdown("---")

    # ── ★ Topic 管理 ──
    st.subheader("🗂️ 研究主题")
    topic_names = list(st.session_state.topics.keys())
    active_idx = topic_names.index(st.session_state.active_topic) if st.session_state.active_topic in topic_names else 0
    chosen = st.selectbox("当前主题", topic_names, index=active_idx)
    if chosen != st.session_state.active_topic:
        st.session_state.active_topic = chosen
        st.session_state.selected_scope = "🌐 对比所有论文"
        st.rerun()

    col_new, col_del = st.columns([3,1])
    with col_new:
        new_topic_name = st.text_input("新建主题", placeholder="输入名称后回车", label_visibility="collapsed")
    with col_del:
        if st.button("➕") and new_topic_name.strip():
            name = new_topic_name.strip()
            if name not in st.session_state.topics:
                st.session_state.topics[name] = {"files": [], "chunks": [], "db": None}
                st.session_state.active_topic = name
                st.rerun()

    # 删除主题（不能删最后一个）
    if len(st.session_state.topics) > 1:
        if st.button(f"🗑️ 删除「{st.session_state.active_topic}」", type="secondary"):
            del st.session_state.topics[st.session_state.active_topic]
            st.session_state.active_topic = list(st.session_state.topics.keys())[0]
            st.rerun()

    # 当前主题的论文列表
    t = active_topic_data()
    if t["files"]:
        st.markdown(f"**主题内论文（{len(t['files'])}篇）**")
        for file in list(t["files"]):
            c1, c2 = st.columns([4,1])
            with c1: st.text(f"📄 {file[:18]}..." if len(file)>20 else f"📄 {file}")
            with c2:
                if st.button("🗑️", key=f"del_{file}"):
                    t["files"].remove(file)
                    t["chunks"] = [c for c in t["chunks"] if c.metadata.get('source_paper') != file]
                    rebuild_topic_index(st.session_state.active_topic, user_api_key)
                    st.rerun()
        if st.button("🗑️ 清空主题", type="primary"):
            t["files"], t["chunks"], t["db"] = [], [], None
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.subheader("⚙️ 研读模式")
    reading_mode = st.radio("选择模式:", ["🟢 快速问答", "📖 逐段精读"], index=1)
    if t["files"]:
        scope_opts = ["🌐 对比所有论文"] + t["files"]
        st.session_state.selected_scope = st.selectbox("👁️ 专注范围", scope_opts)

    st.markdown("---")
    st.subheader("📥 手动上传 PDF")
    uploaded_file = st.file_uploader("拖入 PDF", type="pdf")
    if uploaded_file and st.button("确认加载"):
        with st.spinner("解析中..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue()); path = tmp.name
            process_and_add_to_topic(path, uploaded_file.name, user_api_key)
            os.remove(path)
            st.rerun()

# ================= 7. 主 Tab 布局（新增笔记Tab）=================
tab_search, tab_chat, tab_notes = st.tabs(["🔍 文献调研", "💬 研读空间", "📌 我的笔记"])

# ─────────────────────────────────────────────
# TAB 1: 文献调研
# ─────────────────────────────────────────────
with tab_search:
    st.subheader("🌍 学术大数据检索")
    col_q, col_sort, col_n = st.columns([3,1.5,1])
    with col_q:
        search_query = st.text_input("关键词", value=st.session_state.suggested_query, placeholder="例如: education robot")
    with col_sort:
        sort_mode = st.selectbox("排序规则", ["🔥 相关性优先","📅 时间由新到旧","📈 引用量由高到低"])
    with col_n:
        max_results = st.number_input("获取数量", min_value=5, max_value=50, value=15)

    if st.button("🚀 开始检索") and search_query:
        with st.spinner("正在检索并同步引用数据..."):
            try:
                arxiv_sort = arxiv.SortCriterion.Relevance
                if "时间" in sort_mode: arxiv_sort = arxiv.SortCriterion.SubmittedDate
                refined = search_query
                if " " in search_query and "AND" not in search_query and '"' not in search_query:
                    refined = " AND ".join([f'(ti:{w} OR abs:{w})' for w in search_query.split()])
                raw = list(arxiv.Search(query=refined, max_results=max_results, sort_by=arxiv_sort).results())
                results_with_cite = []
                pb = st.progress(0)
                for idx, res in enumerate(raw):
                    cites = fetch_citations(res.entry_id, ss_key=ss_api_key)
                    results_with_cite.append({'obj': res, 'citations': cites})
                    pb.progress((idx+1)/len(raw))
                if "引用量" in sort_mode:
                    results_with_cite.sort(key=lambda x: x['citations'], reverse=True)
                st.session_state.search_results = results_with_cite
                st.session_state.contributions_cache = {}
                st.success(f"✅ 完成！已获取 {len(results_with_cite)} 篇。")
            except Exception as e:
                st.error(f"检索失败: {e}")

    # ── 图谱 ──
    if st.session_state.focus_paper_id:
        st.markdown("---")
        st.subheader("📊 文献关联图谱")
        with st.spinner("请求图谱数据..."):
            g_data = fetch_graph_data(st.session_state.focus_paper_id, ss_key=ss_api_key)
        if not g_data:
            st.warning("⚠️ 暂时无法获取图谱，请稍后再试。")
        else:
            col_graph, col_info = st.columns([2.5,1])
            with col_graph:
                clicked_id, all_details = render_connected_graph(g_data)
                seed_ss_id = g_data.get('paperId','root')
                if seed_ss_id in all_details:
                    all_details[seed_ss_id]['arxiv_id'] = get_pure_arxiv_id(st.session_state.focus_paper_id)
            with col_info:
                if clicked_id and clicked_id in all_details:
                    info = all_details[clicked_id]
                    st.markdown(f"### 📑 文献详情")
                    st.markdown(f"**{info['title']}**")
                    c1, c2 = st.columns(2)
                    c1.metric("📅 年份", info['year'])
                    c2.metric("🔥 引用", info['cites'])
                    st.markdown("---")
                    st.markdown(
                        f"**摘要**\n\n<div style='font-size:.85em;color:#444;height:200px;overflow-y:auto;'>{info['abstract']}</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")
                    # ★ 选择加入哪个 Topic
                    target_topic = st.selectbox(
                        "加入主题", list(st.session_state.topics.keys()),
                        index=list(st.session_state.topics.keys()).index(st.session_state.active_topic),
                        key="graph_topic_select"
                    )
                    arxiv_id = info.get('arxiv_id')
                    if arxiv_id:
                        if st.button("⬇️ 下载并加入研读队列", use_container_width=True, type="primary"):
                            with st.spinner(f"下载中..."):
                                try:
                                    paper = next(arxiv.Search(id_list=[arxiv_id]).results())
                                    pdf_path = paper.download_pdf(dirpath=tempfile.gettempdir())
                                    ok = process_and_add_to_topic(pdf_path, info['title'], user_api_key, topic_name=target_topic)
                                    if ok: st.success("✅ 入库成功！"); st.balloons()
                                except Exception as e: st.error(f"下载失败: {e}")
                    else:
                        st.info("暂无 ArXiv 全文链接，请手动上传 PDF。")
                    st.link_button("🌐 Semantic Scholar", info['url'], use_container_width=True)
                else:
                    st.markdown("""
**图谱交互指南**

- 🖱️ **滚动**：缩放  
- ✋ **拖拽**：固定节点  
- 👆 **点击节点**：查看详情 + 选主题入库  

---
<div style='font-size:.8em;color:#666;'>
🔴 当前论文 &nbsp; 🟢 引用本文 &nbsp; 🔵 本文引用<br>
节点越大 = 引用量越高
</div>
""", unsafe_allow_html=True)

    # ── 检索结果列表 ──
    if st.session_state.search_results:
        st.markdown("---")
        st.subheader(f"📋 检索结果（{len(st.session_state.search_results)} 篇）")
        for i, item in enumerate(st.session_state.search_results):
            res   = item['obj']
            cites = item['citations']
            with st.expander(f"#{i+1} 📄 {res.title} ({res.published.year})"):
                st.markdown(
                    f"**👨‍🏫 作者**: {', '.join([a.name for a in res.authors])} | "
                    f"**📅**: {res.published.strftime('%Y-%m-%d')} | "
                    f"**🔥**: <span class='cite-badge'>{cites}</span>",
                    unsafe_allow_html=True
                )
                # 一句话贡献
                cc, cg = st.columns([5,1])
                ck = res.title[:60]
                with cc:
                    if ck in st.session_state.contributions_cache:
                        st.markdown(f'<div class="contribution-box">💡 {st.session_state.contributions_cache[ck]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="contribution-box" style="color:#aaa;">💡 点击右侧按钮生成一句话贡献摘要</div>', unsafe_allow_html=True)
                with cg:
                    if st.button("✨", key=f"contrib_{i}"):
                        with st.spinner("分析..."): get_one_line_contribution(res.summary, res.title, user_api_key)
                        st.rerun()
                st.markdown(f'<div class="abstract-box"><b>📝 摘要：</b><br>{res.summary.replace(chr(10)," ")}</div>', unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1: st.markdown(f"[🔗 ArXiv]({res.entry_id})")
                with col2:
                    if st.button(f"⬇️ 下载分析", key=f"dl_{i}"):
                        with st.spinner("下载解析中..."):
                            try:
                                pdf_path = res.download_pdf(dirpath=tempfile.gettempdir())
                                process_and_add_to_topic(pdf_path, res.title, user_api_key)
                                st.success("入库成功！")
                            except Exception as e: st.error(f"失败: {e}")
                with col3:
                    if st.button(f"🕸️ 关联图谱", key=f"graph_{i}"):
                        st.session_state.focus_paper_id = res.entry_id
                        st.rerun()

# ─────────────────────────────────────────────
# TAB 2: 研读空间（问答 + 知识漏洞 + 保存笔记）
# ─────────────────────────────────────────────
with tab_chat:
    t = active_topic_data()

    # 顶部状态栏
    if t["files"]:
        st.markdown(
            f'当前主题：<span class="topic-badge">🗂️ {st.session_state.active_topic}</span> '
            f'| 模式：{reading_mode} | 范围：{st.session_state.selected_scope} '
            f'| 已入库：**{len(t["files"])}** 篇',
            unsafe_allow_html=True
        )
    else:
        st.info("👈 请先在「文献调研」下载论文入库，或在侧边栏上传 PDF。")

    # 历史消息
    for msg in st.session_state.chat_history:
        if msg["role"] == "system_notice":
            st.info(msg["content"])
        else:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ── ★ 知识漏洞推荐区（在输入框之前渲染）──
    if st.session_state.pending_note and st.session_state.pending_note.get("has_gap"):
        recs = get_gap_recommendations()
        if recs:
            with st.container():
                st.markdown('<div class="gap-box">', unsafe_allow_html=True)
                st.markdown("#### 🔍 知识漏洞检测：这些论文可能包含你需要的答案")
                st.caption("以下是当前图谱中尚未入库的相关参考文献，点击可下载入库：")
                for r in recs:
                    rc1, rc2 = st.columns([5,1])
                    with rc1:
                        st.markdown(f"**{r['title']}**")
                        if r.get('abstract'):
                            st.caption(r['abstract'][:120]+"...")
                    with rc2:
                        if r.get('arxiv_id') and st.button("⬇️ 入库", key=f"gap_{r['arxiv_id']}"):
                            with st.spinner("下载中..."):
                                try:
                                    paper = next(arxiv.Search(id_list=[r['arxiv_id']]).results())
                                    pdf_path = paper.download_pdf(dirpath=tempfile.gettempdir())
                                    process_and_add_to_topic(pdf_path, r['title'], user_api_key)
                                    st.success("已入库！")
                                    st.rerun()
                                except Exception as e: st.error(str(e))
                st.markdown('</div>', unsafe_allow_html=True)

    # ── ★ 保存笔记区（在输入框之前渲染）──
    if st.session_state.pending_note and st.session_state.pending_note.get("content"):
        with st.expander("📌 保存这条回答为笔记", expanded=False):
            note_tags_raw = st.text_input(
                "标签（逗号分隔）", placeholder="例如：方法论, Transformer, 2024",
                key="note_tags_input"
            )
            if st.button("💾 保存笔记", type="primary"):
                tags = [t.strip() for t in note_tags_raw.split(",") if t.strip()]
                st.session_state.notes.append({
                    "id": str(uuid.uuid4())[:8],
                    "content": st.session_state.pending_note["content"],
                    "question": st.session_state.pending_note.get("question",""),
                    "tags": tags,
                    "topic": st.session_state.active_topic,
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
                })
                st.session_state.pending_note = None
                st.success("✅ 笔记已保存！在「我的笔记」Tab 查看。")
                st.rerun()

    # ── 问答输入 ──
    if prompt := st.chat_input("对已入库的论文提问..."):
        if not t["db"]:
            st.warning("🧠 请先添加论文")
        else:
            st.session_state.chat_history.append({"role":"user","content":prompt})
            with st.chat_message("user"): st.write(prompt)
            with st.chat_message("assistant"):
                try:
                    search_k = 15 if "精读" in reading_mode else 8
                    scope = st.session_state.selected_scope
                    filter_dict = {"source_paper": scope} if scope != "🌐 对比所有论文" else None
                    docs = t["db"].max_marginal_relevance_search(
                        prompt, k=search_k, fetch_k=20, lambda_mult=0.6, filter=filter_dict
                    )
                    if not docs:
                        answer = "未找到相关内容，请尝试换个问法或扩大检索范围。"
                        st.warning(answer)
                    else:
                        context = "\n\n".join([
                            f"📄【{d.metadata.get('source_paper','?')} P{d.metadata.get('page',0)+1}】:\n{d.page_content}"
                            for d in docs
                        ])
                        sys_prompt = (
                            f"你是一位科研助手。请基于以下资料回答用户问题。\n"
                            f"资料：\n{context}\n\n问题：{prompt}\n\n"
                            f"要求：数学公式用 $ 包裹，条理清晰。"
                            f"如果资料中确实找不到答案，请明确说【资料不足】。"
                        )
                        llm = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.1)
                        resp = llm.invoke(sys_prompt)
                        answer = fix_latex(resp.content)
                        st.write(answer)

                    st.session_state.chat_history.append({"role":"assistant","content":answer})

                    # ★ 检测知识漏洞 & 暂存待保存笔记
                    has_gap = detect_knowledge_gap(answer, docs if docs else [])
                    st.session_state.pending_note = {
                        "content": answer,
                        "question": prompt,
                        "has_gap": has_gap,
                    }
                    st.rerun()   # 刷新以显示漏洞推荐 & 保存笔记区

                except Exception as e:
                    st.error(f"生成出错: {e}")

# ─────────────────────────────────────────────
# TAB 3: ★ 我的笔记
# ─────────────────────────────────────────────
with tab_notes:
    st.subheader("📌 我的笔记库")

    if not st.session_state.notes:
        st.info("还没有笔记。在「研读空间」提问后，点击「保存笔记」即可积累。")
    else:
        # ── 搜索 & 过滤 ──
        all_tags = sorted(set(tag for n in st.session_state.notes for tag in n["tags"]))
        all_topics = sorted(set(n["topic"] for n in st.session_state.notes))

        col_st, col_sf, col_ss = st.columns([2,2,1])
        with col_st:
            filter_tag = st.selectbox("按标签筛选", ["全部"] + all_tags)
        with col_sf:
            filter_topic = st.selectbox("按主题筛选", ["全部"] + all_topics)
        with col_ss:
            search_note = st.text_input("关键词搜索", placeholder="搜索笔记内容")

        filtered = st.session_state.notes.copy()
        if filter_tag != "全部":
            filtered = [n for n in filtered if filter_tag in n["tags"]]
        if filter_topic != "全部":
            filtered = [n for n in filtered if n["topic"] == filter_topic]
        if search_note:
            filtered = [n for n in filtered if search_note.lower() in n["content"].lower() or search_note.lower() in n["question"].lower()]

        st.caption(f"共 {len(filtered)} 条笔记")

        for note in reversed(filtered):   # 最新的在前
            with st.container():
                st.markdown('<div class="note-card">', unsafe_allow_html=True)
                # 头部：主题 + 时间 + 删除
                hc1, hc2, hc3 = st.columns([3, 2, 1])
                with hc1:
                    st.markdown(f'<span class="topic-badge">🗂️ {note["topic"]}</span>', unsafe_allow_html=True)
                    for tag in note["tags"]:
                        st.markdown(f'<span class="note-tag">#{tag}</span>', unsafe_allow_html=True)
                with hc2:
                    st.caption(f"🕐 {note['ts']}")
                with hc3:
                    if st.button("🗑️", key=f"delnote_{note['id']}"):
                        st.session_state.notes = [n for n in st.session_state.notes if n["id"] != note["id"]]
                        st.rerun()
                # 问题
                if note.get("question"):
                    st.markdown(f"**❓ {note['question']}**")
                # 内容（折叠长内容）
                content = note["content"]
                if len(content) > 400:
                    with st.expander("展开完整回答"):
                        st.markdown(content)
                    st.markdown(content[:400] + "...")
                else:
                    st.markdown(content)
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑️ 清空所有笔记", type="secondary"):
            st.session_state.notes = []
            st.rerun()
