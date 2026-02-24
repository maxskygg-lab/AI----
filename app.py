import streamlit as st
import sys
import os
import time
import tempfile
import arxiv
import requests
import math
import re
from streamlit_agraph import agraph, Node, Edge, Config

# ================= 1. 环境检查 =================
try:
    import zhipuai
    import langchain_community
    import fitz
except ImportError as e:
    st.error(f"🚑 环境缺失库 -> {e.name}")
    st.stop()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= 2. 页面配置 =================
st.set_page_config(page_title="AI 深度研读助手", layout="wide", page_icon="🎓")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px;}
    .abstract-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4CAF50;
        font-size: 0.95em;
        line-height: 1.6;
        margin-bottom: 6px;
    }
    /* ★ 新增：一句话贡献样式 */
    .contribution-box {
        background: linear-gradient(90deg, #fffbeb, #fef3c7);
        border-left: 4px solid #f59e0b;
        padding: 8px 14px;
        border-radius: 6px;
        font-size: 0.88em;
        color: #78350f;
        margin-bottom: 10px;
        font-weight: 500;
    }
    .cite-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)
st.title("📖 AI 深度研读助手")

# ================= API 密钥（请用 st.secrets 或 .env 管理，切勿上传到 GitHub）=================
# 推荐做法：在 .streamlit/secrets.toml 中配置，然后用 st.secrets["ZHIPU_API_KEY"] 读取
USER_API_KEY = st.secrets.get("ZHIPU_API_KEY", "your_zhipu_key_here")
SS_API_KEY   = st.secrets.get("SS_API_KEY", "your_ss_key_here")

# ================= 3. 状态初始化 =================
defaults = {
    "chat_history": [],
    "db": None,
    "loaded_files": [],
    "all_chunks": [],
    "suggested_query": "",
    "search_results": [],
    "selected_scope": "🌐 对比所有论文",
    "focus_paper_id": None,
    # ★ 新增：缓存每篇论文的一句话贡献，避免重复调用
    "contributions_cache": {},
    # ★ 新增：图谱点击后待下载的论文信息
    "graph_download_queue": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================= 4. 工具函数 =================

def get_pure_arxiv_id(url_or_id):
    match = re.search(r'(\d{4}\.\d{4,5})', url_or_id)
    return match.group(1) if match else url_or_id.split('/')[-1].split('v')[0]

def fetch_citations(arxiv_id, ss_key=None):
    try:
        clean_id = get_pure_arxiv_id(arxiv_id)
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields=citationCount"
        headers = {"x-api-key": ss_key} if ss_key else {}
        delay = 0.02 if ss_key else 1.0
        time.sleep(delay)
        response = requests.get(api_url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json().get('citationCount', 0)
    except:
        pass
    return 0

@st.cache_data(ttl=3600)
def fetch_graph_data(arxiv_id, ss_key=None):
    clean_id = get_pure_arxiv_id(arxiv_id)
    fields = ("paperId,title,year,citationCount,abstract,"
              "references.paperId,references.title,references.citationCount,references.year,references.abstract,"
              "citations.paperId,citations.title,citations.citationCount,citations.year,citations.abstract,citations.externalIds")
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields={fields}"
    headers = {"x-api-key": ss_key} if ss_key else {}
    for attempt in range(3):
        try:
            response = requests.get(api_url, headers=headers, timeout=12)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                time.sleep((attempt + 1) * 2)
        except:
            if attempt == 2: return None
    return None

# ★ 新增：用 LLM 生成一句话贡献摘要
def get_one_line_contribution(abstract: str, title: str, api_key: str) -> str:
    """从摘要提炼一句话核心贡献，带缓存"""
    cache_key = title[:60]
    if cache_key in st.session_state.contributions_cache:
        return st.session_state.contributions_cache[cache_key]
    try:
        llm = ChatZhipuAI(model="glm-4-flash", api_key=api_key, temperature=0.0)
        prompt = (
            f"请用一句话（不超过40个汉字或20个英文单词）总结这篇论文的核心创新贡献。"
            f"只输出这一句话，不要任何前缀或解释。\n\n标题：{title}\n摘要：{abstract[:600]}"
        )
        res = llm.invoke(prompt)
        result = res.content.strip()
    except:
        result = "（贡献摘要生成失败）"
    st.session_state.contributions_cache[cache_key] = result
    return result

def fix_latex_errors(text):
    if not text: return text
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    return text

def rebuild_index_from_chunks(api_key):
    if not st.session_state.all_chunks:
        st.session_state.db = None
        return
    embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
    st.session_state.db = FAISS.from_documents(st.session_state.all_chunks, embeddings)

def process_and_add_to_db(file_path, file_name, api_key):
    """解析 PDF 并加入向量库，返回 True/False"""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source_paper'] = file_name
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=200,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        valid_chunks = [c for c in chunks if len(c.page_content.strip()) > 20]
        st.session_state.all_chunks.extend(valid_chunks)
        embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
        batch_size = 10
        total = len(valid_chunks)
        if st.session_state.db is None:
            st.session_state.db = FAISS.from_documents(valid_chunks[:batch_size], embeddings)
            for i in range(batch_size, total, batch_size):
                st.session_state.db.add_documents(valid_chunks[i: i + batch_size])
                time.sleep(0.1)
        else:
            for i in range(0, total, batch_size):
                st.session_state.db.add_documents(valid_chunks[i: i + batch_size])
                time.sleep(0.1)
        if file_name not in st.session_state.loaded_files:
            st.session_state.loaded_files.append(file_name)
        st.session_state.chat_history.append({
            "role": "system_notice",
            "content": f"📚 已加载《{file_name}》，可以开始提问。"
        })
        return True
    except Exception as e:
        st.error(f"处理失败: {e}")
        return False

# ================= 5. 图谱渲染（含节点点击返回 SS paper_id + arxiv_id）=================

def render_connected_graph(data):
    if not data: return None, {}
    nodes, edges = [], []
    paper_details = {}
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
        "arxiv_id": None,  # 种子论文 arxiv_id 由外部传入
    }
    nodes.append(Node(id=seed_id, label="THIS PAPER", size=35, color=get_color(data.get('year'), 'seed')))
    seen_ids = {seed_id}

    combined = []
    for p in data.get('references', [])[:15]:
        p['rel_type'] = 'ref'; combined.append(p)
    for p in data.get('citations', [])[:15]:
        p['rel_type'] = 'cite'; combined.append(p)

    for item in combined:
        p_id = item.get('paperId')
        if not p_id or p_id in seen_ids: continue
        seen_ids.add(p_id)
        title = item.get('title', 'Unknown')
        year = item.get('year')
        cites = item.get('citationCount', 0)
        # ★ 尝试从 externalIds 取 ArXiv ID，用于后续下载
        ext = item.get('externalIds') or {}
        arxiv_id = ext.get('ArXiv')
        paper_details[p_id] = {
            "title": title,
            "abstract": item.get('abstract') or "暂无详细摘要。",
            "year": year,
            "cites": cites,
            "url": f"https://www.semanticscholar.org/paper/{p_id}",
            "arxiv_id": arxiv_id,
        }
        node_size = 15 + (math.log(cites + 1) * 3.5)
        nodes.append(Node(id=p_id, label=f"{title[:20]}...", size=node_size, color=get_color(year, item['rel_type'])))
        if item['rel_type'] == 'cite':
            edges.append(Edge(source=p_id, target=seed_id, color="#d1d5db", width=1, dashed=True))
        else:
            edges.append(Edge(source=seed_id, target=p_id, color="#94a3b8", width=1.5))

    config = Config(
        width="100%", height=650, directed=True, physics=True,
        nodeHighlightBehavior=True, highlightColor="#F7D154", collapsible=False,
        d3={'alphaTarget': 0.05, 'gravity': -250, 'linkLength': 150, 'linkStrength': 0.1}
    )
    clicked_id = agraph(nodes=nodes, edges=edges, config=config)
    return clicked_id, paper_details

# ================= 6. 侧边栏 =================
with st.sidebar:
    st.header("🎛️ 控制台")
    user_api_key = USER_API_KEY
    ss_api_key   = SS_API_KEY

    if ss_api_key:
        st.success("🚀 高速调研模式已激活")
    else:
        st.caption("🐢 处于匿名限速模式")

    st.markdown("---")
    if st.session_state.loaded_files:
        st.subheader("🗂️ 已入库论文")
        for file in list(st.session_state.loaded_files):
            col_f1, col_f2 = st.columns([4, 1])
            with col_f1:
                label = f"📄 {file[:18]}..." if len(file) > 20 else f"📄 {file}"
                st.text(label)
            with col_f2:
                if st.button("🗑️", key=f"del_{file}"):
                    st.session_state.loaded_files.remove(file)
                    st.session_state.all_chunks = [
                        c for c in st.session_state.all_chunks
                        if c.metadata.get('source_paper') != file
                    ]
                    if user_api_key: rebuild_index_from_chunks(user_api_key)
                    st.rerun()

        if st.button("🗑️ 清空全部", type="primary"):
            for k in ["db", "loaded_files", "all_chunks", "chat_history", "contributions_cache"]:
                st.session_state[k] = [] if k != "db" and k != "contributions_cache" else (None if k == "db" else {})
            st.rerun()

    st.subheader("⚙️ 研读模式")
    reading_mode = st.radio("选择模式:", ["🟢 快速问答", "📖 逐段精读 (公式修复版)"], index=1)

    if st.session_state.loaded_files:
        st.markdown("---")
        scope_options = ["🌐 对比所有论文"] + st.session_state.loaded_files
        st.session_state.selected_scope = st.selectbox("👁️ 专注范围", scope_options)

    st.markdown("---")
    st.subheader("📥 手动上传 PDF")
    uploaded_file = st.file_uploader("拖入 PDF", type="pdf")
    if uploaded_file and user_api_key and st.button("确认加载"):
        with st.spinner("解析中..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                path = tmp.name
            process_and_add_to_db(path, uploaded_file.name, user_api_key)
            os.remove(path)
            st.rerun()

# ================= 7. 主界面 =================
tab_search, tab_chat = st.tabs(["🔍 文献调研", "💬 研读空间"])

# ─────────────────────────────────────────────
# TAB 1: 文献调研
# ─────────────────────────────────────────────
with tab_search:
    st.subheader("🌍 学术大数据检索")
    col_q, col_sort, col_n = st.columns([3, 1.5, 1])
    with col_q:
        search_query = st.text_input("关键词", value=st.session_state.suggested_query, placeholder="例如: education robot")
    with col_sort:
        sort_mode = st.selectbox("排序规则", ["🔥 相关性优先", "📅 时间由新到旧", "📈 引用量由高到低"])
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
                search = arxiv.Search(query=refined, max_results=max_results, sort_by=arxiv_sort)
                raw = list(search.results())
                results_with_cite = []
                progress_bar = st.progress(0)
                for idx, res in enumerate(raw):
                    cites = fetch_citations(res.entry_id, ss_key=ss_api_key)
                    results_with_cite.append({'obj': res, 'citations': cites})
                    progress_bar.progress((idx + 1) / len(raw))
                if "引用量" in sort_mode:
                    results_with_cite.sort(key=lambda x: x['citations'], reverse=True)
                st.session_state.search_results = results_with_cite
                # ★ 清除旧贡献缓存（新一轮搜索）
                st.session_state.contributions_cache = {}
                st.success(f"✅ 完成！已获取 {len(results_with_cite)} 篇论文。")
            except Exception as e:
                st.error(f"检索失败: {e}")

    # ── 图谱区域 ──
    if st.session_state.focus_paper_id:
        st.markdown("---")
        st.subheader("📊 文献关联图谱")
        with st.spinner("正在请求图谱数据..."):
            g_data = fetch_graph_data(st.session_state.focus_paper_id, ss_key=ss_api_key)

        if not g_data:
            st.warning("⚠️ 暂时无法获取图谱，请稍后再试。")
        else:
            col_graph, col_info = st.columns([2.5, 1])
            with col_graph:
                # 把种子论文的 arxiv_id 注入 paper_details
                clicked_node_id, all_details = render_connected_graph(g_data)
                # 补充种子节点的 arxiv_id
                seed_ss_id = g_data.get('paperId', 'root')
                if seed_ss_id in all_details:
                    all_details[seed_ss_id]['arxiv_id'] = get_pure_arxiv_id(st.session_state.focus_paper_id)

            with col_info:
                if clicked_node_id and clicked_node_id in all_details:
                    info = all_details[clicked_node_id]
                    st.markdown(f"### 📑 文献详情")
                    st.markdown(f"**{info['title']}**")
                    c1, c2 = st.columns(2)
                    c1.metric("📅 年份", info['year'])
                    c2.metric("🔥 引用", info['cites'])
                    st.markdown("---")
                    st.markdown(
                        f"**摘要**\n\n<div style='font-size:0.85em;color:#444;height:220px;overflow-y:auto;'>"
                        f"{info['abstract']}</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")

                    # ★ 关键新功能：一键下载入库
                    arxiv_id = info.get('arxiv_id')
                    if arxiv_id:
                        if st.button("⬇️ 下载并加入研读队列", use_container_width=True, type="primary"):
                            with st.spinner(f"正在下载《{info['title'][:30]}...》"):
                                try:
                                    paper = next(arxiv.Search(id_list=[arxiv_id]).results())
                                    pdf_path = paper.download_pdf(dirpath=tempfile.gettempdir())
                                    ok = process_and_add_to_db(pdf_path, info['title'], user_api_key)
                                    if ok:
                                        st.success("✅ 入库成功！可在「研读空间」提问。")
                                        st.balloons()
                                except Exception as e:
                                    st.error(f"下载失败: {e}")
                    else:
                        st.info("该论文暂无 ArXiv 全文链接，请手动上传 PDF。")

                    st.link_button("🌐 在 Semantic Scholar 查看", info['url'], use_container_width=True)
                else:
                    st.markdown("""
**图谱交互指南**

- 🖱️ **滚动**：缩放  
- ✋ **拖拽**：固定节点  
- 👆 **点击圆点**：查看详情 + 一键入库  

---
<div style='font-size:0.8em;color:#666;'>
🔴 当前论文 &nbsp;
🟢 引用本文 &nbsp;
🔵 本文引用<br>
节点越大 = 引用量越高
</div>
""", unsafe_allow_html=True)

    # ── 检索结果列表 ──
    if st.session_state.search_results:
        st.markdown("---")
        st.subheader(f"📋 检索结果（{len(st.session_state.search_results)} 篇）")
        for i, item in enumerate(st.session_state.search_results):
            res = item['obj']
            cites = item['citations']
            with st.expander(f"#{i+1} 📄 {res.title} ({res.published.year})"):
                st.markdown(
                    f"**👨‍🏫 作者**: {', '.join([a.name for a in res.authors])} | "
                    f"**📅 发表**: {res.published.strftime('%Y-%m-%d')} | "
                    f"**🔥 引用**: <span class='cite-badge'>{cites}</span>",
                    unsafe_allow_html=True
                )

                # ★ 新功能：一句话贡献
                col_contrib, col_gen = st.columns([5, 1])
                cache_key = res.title[:60]
                with col_contrib:
                    if cache_key in st.session_state.contributions_cache:
                        contrib = st.session_state.contributions_cache[cache_key]
                        st.markdown(f'<div class="contribution-box">💡 {contrib}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="contribution-box" style="color:#aaa;">💡 点击右侧按钮生成一句话贡献摘要</div>', unsafe_allow_html=True)
                with col_gen:
                    if st.button("✨ 生成", key=f"contrib_{i}"):
                        if user_api_key:
                            with st.spinner("分析中..."):
                                get_one_line_contribution(res.summary, res.title, user_api_key)
                            st.rerun()
                        else:
                            st.error("需要 API Key")

                st.markdown(
                    f'<div class="abstract-box"><b>📝 摘要：</b><br>{res.summary.replace(chr(10), " ")}</div>',
                    unsafe_allow_html=True
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"[🔗 ArXiv 原文]({res.entry_id})")
                with col2:
                    if st.button(f"⬇️ 下载分析", key=f"dl_{i}"):
                        if user_api_key:
                            with st.spinner("下载解析中..."):
                                try:
                                    pdf_path = res.download_pdf(dirpath=tempfile.gettempdir())
                                    process_and_add_to_db(pdf_path, res.title, user_api_key)
                                    st.success("入库成功！")
                                except Exception as e:
                                    st.error(f"失败: {e}")
                        else:
                            st.error("请先配置 API Key")
                with col3:
                    if st.button(f"🕸️ 关联图谱", key=f"graph_{i}"):
                        st.session_state.focus_paper_id = res.entry_id
                        st.rerun()

# ─────────────────────────────────────────────
# TAB 2: 研读空间（问答）
# ─────────────────────────────────────────────
with tab_chat:
    if st.session_state.loaded_files:
        st.caption(f"📚 模式：{reading_mode} | 范围：{st.session_state.selected_scope}")
    else:
        st.info("👈 请先在「文献调研」选择论文下载入库，或在侧边栏上传 PDF。")

    for msg in st.session_state.chat_history:
        if msg["role"] == "system_notice":
            st.info(msg["content"])
        else:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("对已入库的论文提问..."):
        if not st.session_state.db:
            st.warning("🧠 请先添加论文")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                try:
                    search_k = 15 if "精读" in reading_mode else 8
                    scope = st.session_state.selected_scope
                    filter_dict = {"source_paper": scope} if scope != "🌐 对比所有论文" else None
                    docs = st.session_state.db.max_marginal_relevance_search(
                        prompt, k=search_k, fetch_k=20, lambda_mult=0.6, filter=filter_dict
                    )
                    if not docs:
                        st.warning("未找到相关内容，请尝试换个问法或扩大检索范围。")
                    else:
                        context = "\n\n".join([
                            f"📄【{d.metadata.get('source_paper','?')} P{d.metadata.get('page',0)+1}】:\n{d.page_content}"
                            for d in docs
                        ])
                        sys_prompt = (
                            f"你是一位科研助手。请基于以下资料回答用户问题。\n"
                            f"资料：\n{context}\n\n"
                            f"问题：{prompt}\n\n"
                            f"要求：数学公式用 $ 包裹，条理清晰，如资料不足请说明。"
                        )
                        llm = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.1)
                        response = llm.invoke(sys_prompt)
                        final = fix_latex_errors(response.content)
                        st.write(final)
                        st.session_state.chat_history.append({"role": "assistant", "content": final})
                except Exception as e:
                    st.error(f"生成出错: {e}")
