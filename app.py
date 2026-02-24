import streamlit as st
import os, time, tempfile, re, math, uuid
import arxiv, requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        background:#f0f2f6; padding:12px; border-radius:8px;
        border-left:5px solid #4CAF50; font-size:.9em;
        line-height:1.5; margin-bottom:6px;
    }
    .contribution-box {
        background:linear-gradient(90deg,#fffbeb,#fef3c7);
        border-left:4px solid #f59e0b; padding:7px 12px;
        border-radius:6px; font-size:.85em; color:#78350f;
        margin-bottom:8px; font-weight:500;
    }
    .cite-badge {
        background:#ff4b4b; color:white; padding:2px 7px;
        border-radius:12px; font-size:.78em; font-weight:bold;
    }
    .cite-loading {
        color:#94a3b8; font-size:.78em; font-style:italic;
    }
    .topic-badge {
        display:inline-block; background:#6366f1; color:white;
        padding:2px 10px; border-radius:20px; font-size:.78em;
        font-weight:600; margin-right:4px;
    }
    .gap-box {
        background:#fef2f2; border:1px solid #fca5a5;
        border-left:4px solid #ef4444; border-radius:8px;
        padding:12px 16px; margin:10px 0;
    }
    .note-card {
        background:#f8fafc; border:1px solid #e2e8f0;
        border-radius:10px; padding:14px 18px; margin-bottom:10px;
    }
    .note-tag {
        display:inline-block; background:#e0e7ff; color:#3730a3;
        padding:1px 8px; border-radius:12px; font-size:.75em;
        margin-right:4px; font-weight:500;
    }
    /* ★ 右侧聊天面板固定高度可滚动 */
    .chat-panel {
        height: 520px;
        overflow-y: auto;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
        background: #fafafa;
        margin-bottom: 10px;
    }
    .chat-user { background:#dbeafe; border-radius:8px; padding:8px 12px; margin:6px 0; font-size:.9em; }
    .chat-bot  { background:#f0fdf4; border-radius:8px; padding:8px 12px; margin:6px 0; font-size:.9em; }
    .chat-notice { color:#6366f1; font-size:.82em; font-style:italic; margin:4px 0; }
    /* 分隔线标题 */
    .section-divider {
        font-size:.72em; text-transform:uppercase; letter-spacing:2px;
        color:#94a3b8; margin:18px 0 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📖 AI 深度研读助手 v4")

# ================= API Key =================
USER_API_KEY = st.secrets["ZHIPU_API_KEY"]
SS_API_KEY   = st.secrets["SS_API_KEY"]

# ================= 3. 状态初始化 =================
defaults = {
    "search_results": [],          # [{"obj": arxiv.Result, "citations": int|None}]
    "citations_loaded": False,     # ★ 是否已完成引用数加载
    "suggested_query": "",
    "focus_paper_id": None,
    "contributions_cache": {},
    "chat_history": [],
    "topics": {"默认主题": {"files": [], "chunks": [], "db": None}},
    "active_topic": "默认主题",
    "selected_scope": "🌐 对比所有论文",
    "notes": [],
    "pending_note": None,
    "graph_references_cache": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================= 4. 工具函数 =================

def active_topic_data():
    return st.session_state.topics[st.session_state.active_topic]

def get_pure_arxiv_id(url_or_id):
    m = re.search(r'(\d{4}\.\d{4,5})', url_or_id)
    return m.group(1) if m else url_or_id.split('/')[-1].split('v')[0]

def fetch_one_citation(args):
    """单篇引用数获取，用于并发池"""
    arxiv_id, ss_key = args
    try:
        clean_id = get_pure_arxiv_id(arxiv_id)
        url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields=citationCount"
        headers = {"x-api-key": ss_key} if ss_key else {}
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code == 200:
            return arxiv_id, r.json().get('citationCount', 0)
    except: pass
    return arxiv_id, 0

# ★ 并发获取所有引用数（核心提速）
def fetch_citations_parallel(results, ss_key=None):
    """用线程池并行获取所有论文引用数，比顺序快 10x+"""
    args_list = [(item['obj'].entry_id, ss_key) for item in results]
    id_to_cite = {}
    # 有 key 时并发数可调高；无 key 时适当限速
    max_workers = 8 if ss_key else 3
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_one_citation, args): args[0] for args in args_list}
        for future in as_completed(futures):
            arxiv_id, count = future.result()
            id_to_cite[arxiv_id] = count
    return id_to_cite

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

def detect_knowledge_gap(answer_text, docs):
    gap_signals = ["资料不足", "没有找到", "无法回答", "未提及", "不清楚", "没有相关", "cannot find", "not mentioned"]
    if len(docs) < 3: return True
    for sig in gap_signals:
        if sig.lower() in answer_text.lower(): return True
    return False

def get_gap_recommendations():
    all_loaded = set()
    for t in st.session_state.topics.values():
        all_loaded.update(t["files"])
    recs = []
    for ref in st.session_state.graph_references_cache:
        title = ref.get("title","")
        if not any(title[:20].lower() in f.lower() for f in all_loaded):
            recs.append(ref)
    return recs[:4]

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

    seed_id = data.get('paperId','root')
    paper_details[seed_id] = {
        "title": data.get('title','Seed Paper'),
        "abstract": data.get('abstract') or "无摘要",
        "year": data.get('year','Unknown'),
        "cites": data.get('citationCount',0),
        "url": f"https://www.semanticscholar.org/paper/{seed_id}",
        "arxiv_id": None,
    }
    nodes.append(Node(id=seed_id, label="THIS PAPER", size=35, color=get_color(data.get('year'),'seed')))
    seen = {seed_id}
    refs_for_gap = []

    combined = []
    for p in data.get('references',[])[:15]: p['rel_type']='ref'; combined.append(p)
    for p in data.get('citations',[])[:15]:  p['rel_type']='cite'; combined.append(p)

    for item in combined:
        p_id = item.get('paperId')
        if not p_id or p_id in seen: continue
        seen.add(p_id)
        title = item.get('title','Unknown')
        year  = item.get('year')
        cites = item.get('citationCount',0)
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
    config = Config(width="100%", height=560, directed=True, physics=True,
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

    st.subheader("🗂️ 研究主题")
    topic_names = list(st.session_state.topics.keys())
    active_idx = topic_names.index(st.session_state.active_topic) if st.session_state.active_topic in topic_names else 0
    chosen = st.selectbox("当前主题", topic_names, index=active_idx)
    if chosen != st.session_state.active_topic:
        st.session_state.active_topic = chosen
        st.session_state.selected_scope = "🌐 对比所有论文"
        st.rerun()

    col_new, col_add = st.columns([3,1])
    with col_new:
        new_topic_name = st.text_input("新建主题", placeholder="输入名称", label_visibility="collapsed")
    with col_add:
        if st.button("➕") and new_topic_name.strip():
            name = new_topic_name.strip()
            if name not in st.session_state.topics:
                st.session_state.topics[name] = {"files":[],"chunks":[],"db":None}
                st.session_state.active_topic = name
                st.rerun()

    if len(st.session_state.topics) > 1:
        if st.button(f"🗑️ 删除「{st.session_state.active_topic}」"):
            del st.session_state.topics[st.session_state.active_topic]
            st.session_state.active_topic = list(st.session_state.topics.keys())[0]
            st.rerun()

    t_side = active_topic_data()
    if t_side["files"]:
        st.markdown(f"**已入库（{len(t_side['files'])}篇）**")
        for file in list(t_side["files"]):
            c1, c2 = st.columns([4,1])
            with c1: st.text(f"📄 {file[:16]}..." if len(file)>18 else f"📄 {file}")
            with c2:
                if st.button("🗑️", key=f"del_{file}"):
                    t_side["files"].remove(file)
                    t_side["chunks"] = [c for c in t_side["chunks"] if c.metadata.get('source_paper') != file]
                    rebuild_topic_index(st.session_state.active_topic, user_api_key)
                    st.rerun()
        if st.button("🗑️ 清空主题", type="primary"):
            t_side["files"], t_side["chunks"], t_side["db"] = [], [], None
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.subheader("⚙️ 研读设置")
    reading_mode = st.radio("模式", ["🟢 快速问答","📖 逐段精读"], index=1)
    if t_side["files"]:
        scope_opts = ["🌐 对比所有论文"] + t_side["files"]
        st.session_state.selected_scope = st.selectbox("专注范围", scope_opts)

    st.markdown("---")
    st.subheader("📥 上传 PDF")
    uploaded_file = st.file_uploader("拖入 PDF", type="pdf")
    if uploaded_file and st.button("确认加载"):
        with st.spinner("解析中..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue()); path = tmp.name
            process_and_add_to_topic(path, uploaded_file.name, user_api_key)
            os.remove(path)
            st.rerun()

# ===============================================================
# ★ 7. 主界面：单页纵向流布局（搜索 ← 左列 | 问答 → 右列）
#    笔记保留为顶部 Tab
# ===============================================================
tab_main, tab_notes = st.tabs(["🔍📖 研究工作台", "📌 我的笔记"])

with tab_main:
    # ── 两栏：左=发现区  右=问答区 ──
    col_left, col_right = st.columns([1.45, 1])

    # ══════════════════════════════════════════
    # 左栏：搜索 → 图谱 → 结果列表
    # ══════════════════════════════════════════
    with col_left:
        # ── 检索栏 ──
        st.markdown('<div class="section-divider">🌍 学术检索</div>', unsafe_allow_html=True)
        sq1, sq2, sq3 = st.columns([3,1.5,1])
        with sq1:
            search_query = st.text_input("关键词", value=st.session_state.suggested_query,
                                         placeholder="例如: education robot", label_visibility="collapsed")
        with sq2:
            sort_mode = st.selectbox("排序", ["🔥 相关性","📅 最新","📈 引用量"], label_visibility="collapsed")
        with sq3:
            max_results = st.number_input("数量", 5, 50, 15, label_visibility="collapsed")

        if st.button("🚀 检索", use_container_width=True) and search_query:
            # ★ 第一步：先拿论文列表（快）
            with st.spinner("检索论文中..."):
                try:
                    arxiv_sort = arxiv.SortCriterion.Relevance
                    if "最新" in sort_mode: arxiv_sort = arxiv.SortCriterion.SubmittedDate
                    refined = search_query
                    if " " in search_query and "AND" not in search_query and '"' not in search_query:
                        refined = " AND ".join([f'(ti:{w} OR abs:{w})' for w in search_query.split()])
                    raw = list(arxiv.Search(query=refined, max_results=max_results, sort_by=arxiv_sort).results())
                    # 先存结果，citations 先置 None（显示"加载中"）
                    st.session_state.search_results = [{"obj": r, "citations": None} for r in raw]
                    st.session_state.citations_loaded = False
                    st.session_state.contributions_cache = {}
                    st.session_state.focus_paper_id = None
                except Exception as e:
                    st.error(f"检索失败: {e}")

            # ★ 第二步：并发获取引用数（比顺序快 10x）
            if st.session_state.search_results:
                with st.spinner(f"并发加载 {len(st.session_state.search_results)} 篇引用数..."):
                    id_to_cite = fetch_citations_parallel(st.session_state.search_results, ss_key=ss_api_key)
                    for item in st.session_state.search_results:
                        item["citations"] = id_to_cite.get(item["obj"].entry_id, 0)
                    if "引用量" in sort_mode:
                        st.session_state.search_results.sort(key=lambda x: x["citations"], reverse=True)
                    st.session_state.citations_loaded = True
                st.success(f"✅ 完成！{len(st.session_state.search_results)} 篇，引用数已全部加载。")

        # ── 图谱区（有焦点论文时展示）──
        if st.session_state.focus_paper_id:
            st.markdown('<div class="section-divider">📊 文献关联图谱</div>', unsafe_allow_html=True)
            with st.spinner("加载图谱..."):
                g_data = fetch_graph_data(st.session_state.focus_paper_id, ss_key=ss_api_key)
            if not g_data:
                st.warning("⚠️ 暂时无法获取图谱，请稍后再试。")
            else:
                clicked_id, all_details = render_connected_graph(g_data)
                seed_ss_id = g_data.get('paperId','root')
                if seed_ss_id in all_details:
                    all_details[seed_ss_id]['arxiv_id'] = get_pure_arxiv_id(st.session_state.focus_paper_id)

                if clicked_id and clicked_id in all_details:
                    info = all_details[clicked_id]
                    with st.expander(f"📑 {info['title'][:60]}...", expanded=True):
                        c1, c2 = st.columns(2)
                        c1.metric("📅 年份", info['year'])
                        c2.metric("🔥 引用", info['cites'])
                        st.markdown(
                            f"<div style='font-size:.84em;color:#444;max-height:140px;overflow-y:auto;'>"
                            f"{info['abstract']}</div>", unsafe_allow_html=True
                        )
                        target_topic = st.selectbox(
                            "加入主题", list(st.session_state.topics.keys()),
                            index=list(st.session_state.topics.keys()).index(st.session_state.active_topic),
                            key="graph_topic_sel"
                        )
                        arxiv_id = info.get('arxiv_id')
                        gc1, gc2 = st.columns(2)
                        with gc1:
                            if arxiv_id and st.button("⬇️ 下载入库", type="primary", use_container_width=True):
                                with st.spinner("下载中..."):
                                    try:
                                        paper = next(arxiv.Search(id_list=[arxiv_id]).results())
                                        pdf_path = paper.download_pdf(dirpath=tempfile.gettempdir())
                                        if process_and_add_to_topic(pdf_path, info['title'], user_api_key, topic_name=target_topic):
                                            st.success("✅ 入库成功！"); st.balloons()
                                    except Exception as e: st.error(str(e))
                            elif not arxiv_id:
                                st.info("暂无 ArXiv 全文")
                        with gc2:
                            st.link_button("🌐 Semantic Scholar", info['url'], use_container_width=True)
                else:
                    st.caption("👆 点击图谱节点查看详情并下载入库 | 🔴当前论文 🟢引用本文 🔵本文引用")

        # ── 检索结果列表 ──
        if st.session_state.search_results:
            st.markdown(f'<div class="section-divider">📋 检索结果（{len(st.session_state.search_results)} 篇）</div>', unsafe_allow_html=True)
            for i, item in enumerate(st.session_state.search_results):
                res   = item['obj']
                cites = item['citations']
                # 引用数显示：None=加载中，数字=已加载
                cite_html = (
                    f"<span class='cite-badge'>{cites}</span>" if cites is not None
                    else "<span class='cite-loading'>引用数加载中…</span>"
                )
                with st.expander(f"#{i+1} {res.title[:55]}… ({res.published.year})"):
                    st.markdown(
                        f"**{', '.join([a.name for a in res.authors[:3]])}{'等' if len(res.authors)>3 else ''}** | "
                        f"{res.published.strftime('%Y-%m-%d')} | 引用：{cite_html}",
                        unsafe_allow_html=True
                    )
                    # 一句话贡献
                    ck = res.title[:60]
                    cc, cg = st.columns([5,1])
                    with cc:
                        if ck in st.session_state.contributions_cache:
                            st.markdown(f'<div class="contribution-box">💡 {st.session_state.contributions_cache[ck]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="contribution-box" style="color:#aaa;">💡 点击右侧 ✨ 生成核心贡献摘要</div>', unsafe_allow_html=True)
                    with cg:
                        if st.button("✨", key=f"contrib_{i}"):
                            with st.spinner("分析..."): get_one_line_contribution(res.summary, res.title, user_api_key)
                            st.rerun()

                    st.markdown(f'<div class="abstract-box"><b>摘要：</b>{res.summary.replace(chr(10)," ")[:400]}…</div>', unsafe_allow_html=True)

                    b1, b2, b3 = st.columns(3)
                    with b1: st.markdown(f"[🔗 ArXiv]({res.entry_id})")
                    with b2:
                        if st.button("⬇️ 下载入库", key=f"dl_{i}"):
                            with st.spinner("下载解析..."):
                                try:
                                    pdf_path = res.download_pdf(dirpath=tempfile.gettempdir())
                                    process_and_add_to_topic(pdf_path, res.title, user_api_key)
                                    st.success("入库成功！")
                                except Exception as e: st.error(str(e))
                    with b3:
                        if st.button("🕸️ 图谱", key=f"graph_{i}"):
                            st.session_state.focus_paper_id = res.entry_id
                            st.rerun()

    # ══════════════════════════════════════════
    # 右栏：问答区（和发现区并排，随时可用）
    # ══════════════════════════════════════════
    with col_right:
        t = active_topic_data()
        st.markdown(
            f'<div class="section-divider">💬 研读空间 — '
            f'<span class="topic-badge">🗂️ {st.session_state.active_topic}</span> '
            f'· {len(t["files"])} 篇入库</div>',
            unsafe_allow_html=True
        )

        if not t["files"]:
            st.info("👆 在左侧下载论文后即可在这里提问。")

        # ── 知识漏洞推荐 ──
        if st.session_state.pending_note and st.session_state.pending_note.get("has_gap"):
            recs = get_gap_recommendations()
            if recs:
                st.markdown('<div class="gap-box">', unsafe_allow_html=True)
                st.markdown("**🔍 知识漏洞：这些论文可能有你需要的答案**")
                for r in recs:
                    rc1, rc2 = st.columns([4,1])
                    with rc1: st.caption(r['title'][:60])
                    with rc2:
                        if r.get('arxiv_id') and st.button("⬇️", key=f"gap_{r['arxiv_id']}"):
                            with st.spinner("下载..."):
                                try:
                                    paper = next(arxiv.Search(id_list=[r['arxiv_id']]).results())
                                    pdf_path = paper.download_pdf(dirpath=tempfile.gettempdir())
                                    process_and_add_to_topic(pdf_path, r['title'], user_api_key)
                                    st.success("已入库！"); st.rerun()
                                except Exception as e: st.error(str(e))
                st.markdown('</div>', unsafe_allow_html=True)

        # ── 保存笔记 ──
        if st.session_state.pending_note and st.session_state.pending_note.get("content"):
            with st.expander("📌 保存为笔记", expanded=False):
                note_tags_raw = st.text_input("标签（逗号分隔）", placeholder="方法论, Transformer", key="note_tags_input")
                if st.button("💾 保存", type="primary"):
                    tags = [tg.strip() for tg in note_tags_raw.split(",") if tg.strip()]
                    st.session_state.notes.append({
                        "id": str(uuid.uuid4())[:8],
                        "content": st.session_state.pending_note["content"],
                        "question": st.session_state.pending_note.get("question",""),
                        "tags": tags,
                        "topic": st.session_state.active_topic,
                        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    })
                    st.session_state.pending_note = None
                    st.success("✅ 已保存到「我的笔记」")
                    st.rerun()

        # ── 历史消息（可滚动面板）──
        chat_html = ""
        for msg in st.session_state.chat_history[-20:]:   # 只显示最近20条
            if msg["role"] == "system_notice":
                chat_html += f'<div class="chat-notice">📢 {msg["content"]}</div>'
            elif msg["role"] == "user":
                chat_html += f'<div class="chat-user">🧑 {msg["content"]}</div>'
            else:
                # 简单转义换行，保持可读
                content = msg["content"].replace("\n","<br>")
                chat_html += f'<div class="chat-bot">🤖 {content}</div>'
        st.markdown(f'<div class="chat-panel">{chat_html}</div>', unsafe_allow_html=True)

        # ── 输入框 ──
        chat_col1, chat_col2 = st.columns([5,1])
        with chat_col1:
            user_input = st.text_input("提问", placeholder="输入问题，按发送…", label_visibility="collapsed", key="chat_input_box")
        with chat_col2:
            send_btn = st.button("发送 ➤", use_container_width=True)

        if send_btn and user_input.strip():
            prompt = user_input.strip()
            if not t["db"]:
                st.warning("🧠 请先添加论文")
            else:
                st.session_state.chat_history.append({"role":"user","content":prompt})
                with st.spinner("思考中..."):
                    try:
                        search_k = 15 if "精读" in reading_mode else 8
                        scope = st.session_state.selected_scope
                        filter_dict = {"source_paper": scope} if scope != "🌐 对比所有论文" else None
                        docs = t["db"].max_marginal_relevance_search(
                            prompt, k=search_k, fetch_k=20, lambda_mult=0.6, filter=filter_dict
                        )
                        if not docs:
                            answer = "未找到相关内容，请尝试换个问法。"
                        else:
                            context = "\n\n".join([
                                f"📄【{d.metadata.get('source_paper','?')} P{d.metadata.get('page',0)+1}】:\n{d.page_content}"
                                for d in docs
                            ])
                            sys_prompt = (
                                "你是一位科研助手。请基于以下资料回答用户问题。\n"
                                f"资料：\n{context}\n\n问题：{prompt}\n\n"
                                "要求：数学公式用 $ 包裹，条理清晰。"
                                "如果资料中确实找不到答案，请明确说【资料不足】。"
                            )
                            llm = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.1)
                            resp = llm.invoke(sys_prompt)
                            answer = fix_latex(resp.content)

                        st.session_state.chat_history.append({"role":"assistant","content":answer})
                        has_gap = detect_knowledge_gap(answer, docs if docs else [])
                        st.session_state.pending_note = {"content":answer,"question":prompt,"has_gap":has_gap}
                        st.rerun()
                    except Exception as e:
                        st.error(f"生成出错: {e}")

# ══════════════════════════════════════════
# Tab 2: 我的笔记
# ══════════════════════════════════════════
with tab_notes:
    st.subheader("📌 我的笔记库")
    if not st.session_state.notes:
        st.info("还没有笔记。在右侧研读区提问后点击「保存为笔记」即可积累。")
    else:
        all_tags   = sorted(set(tag for n in st.session_state.notes for tag in n["tags"]))
        all_topics = sorted(set(n["topic"] for n in st.session_state.notes))
        f1, f2, f3 = st.columns([2,2,2])
        with f1: filter_tag   = st.selectbox("按标签", ["全部"]+all_tags)
        with f2: filter_topic = st.selectbox("按主题", ["全部"]+all_topics)
        with f3: search_note  = st.text_input("关键词", placeholder="搜索笔记")

        filtered = st.session_state.notes.copy()
        if filter_tag   != "全部": filtered = [n for n in filtered if filter_tag in n["tags"]]
        if filter_topic != "全部": filtered = [n for n in filtered if n["topic"] == filter_topic]
        if search_note:            filtered = [n for n in filtered if search_note.lower() in (n["content"]+n.get("question","")).lower()]

        st.caption(f"共 {len(filtered)} 条")
        for note in reversed(filtered):
            with st.container():
                st.markdown('<div class="note-card">', unsafe_allow_html=True)
                h1, h2, h3 = st.columns([3,2,1])
                with h1:
                    st.markdown(f'<span class="topic-badge">🗂️ {note["topic"]}</span>', unsafe_allow_html=True)
                    for tag in note["tags"]:
                        st.markdown(f'<span class="note-tag">#{tag}</span>', unsafe_allow_html=True)
                with h2: st.caption(f"🕐 {note['ts']}")
                with h3:
                    if st.button("🗑️", key=f"delnote_{note['id']}"):
                        st.session_state.notes = [n for n in st.session_state.notes if n["id"] != note["id"]]
                        st.rerun()
                if note.get("question"):
                    st.markdown(f"**❓ {note['question']}**")
                content = note["content"]
                if len(content) > 400:
                    with st.expander("展开完整回答"): st.markdown(content)
                    st.markdown(content[:400]+"...")
                else:
                    st.markdown(content)
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑️ 清空所有笔记", type="secondary"):
            st.session_state.notes = []; st.rerun()
