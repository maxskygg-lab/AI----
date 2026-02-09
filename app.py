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

# ================= 1. 环境听诊器 =================
try:
    import zhipuai
    import langchain_community
    import fitz  # pymupdf
except ImportError as e:
    st.error(f"🚑 环境缺失库 -> {e.name}")
    st.stop()
# ===============================================

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= 2. 页面配置 =================
st.set_page_config(page_title="AI 深度研读助手 (专业调研版)", layout="wide", page_icon="🎓")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px;}
    .reportview-container { margin-top: -2em; }
    .abstract-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4CAF50;
        font-size: 0.95em;
        line-height: 1.6;
        margin-bottom: 10px;
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
st.title("📖 AI 深度研读助手 (专业调研版)")

# ================= 3. 状态初始化 =================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db" not in st.session_state:
    st.session_state.db = None
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "suggested_query" not in st.session_state:
    st.session_state.suggested_query = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "selected_scope" not in st.session_state:
    st.session_state.selected_scope = "🌐 对比所有论文"
if "focus_paper_id" not in st.session_state: 
    st.session_state.focus_paper_id = None

# ================= 4. 核心逻辑函数 =================

def get_pure_arxiv_id(url):
    """从 URL 中提取 ArXiv ID"""
    match = re.search(r'(\d{4}\.\d{4,5})', url)
    if match: return match.group(1)
    return url.split('/')[-1].split('v')[0]

def fetch_citations(arxiv_id, ss_key=None):
    """从 Semantic Scholar 获取引用数"""
    try:
        clean_id = get_pure_arxiv_id(arxiv_id)
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields=citationCount"
        headers = {"x-api-key": ss_key} if ss_key else {}
        response = requests.get(api_url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json().get('citationCount', 0)
    except: pass
    return 0

@st.cache_data(ttl=3600)
def fetch_graph_data(arxiv_id, ss_key=None):
    """【修复点】在请求字段中显式加入摘要，确保关系图谱能显示摘要"""
    try:
        clean_id = get_pure_arxiv_id(arxiv_id)
        # 精准加入 references.abstract 和 citations.abstract
        fields = "paperId,title,year,citationCount,abstract,references.paperId,references.title,references.citationCount,references.year,references.abstract,citations.paperId,citations.title,citations.citationCount,citations.year,citations.abstract"
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields={fields}"
        headers = {"x-api-key": ss_key} if ss_key else {}
        if not ss_key: time.sleep(1.5)
        response = requests.get(api_url, headers=headers, timeout=12)
        if response.status_code == 200: return response.json()
        return None
    except: return None

def render_connected_graph(data):
    """【修复点】遍历 references 和 citations 两个列表，呈现“关系群”"""
    if not data: return None, {}
    nodes, edges, paper_details = [], [], {}
    
    seed_id = data.get('paperId', 'root')
    paper_details[seed_id] = {
        "title": data.get('title', 'Seed Paper'),
        "abstract": data.get('abstract', '暂无摘要'),
        "year": data.get('year', 'N/A'),
        "cites": data.get('citationCount', 0)
    }
    nodes.append(Node(id=seed_id, label="⭐ SEED", size=30, color="#FF4B4B"))

    seen_ids = {seed_id}
    # 遍历引用和被引用，构建群簇
    for rel_type in ['references', 'citations']:
        for p in data.get(rel_type, [])[:15]:
            p_id = p.get('paperId')
            if not p_id or p_id in seen_ids: continue
            seen_ids.add(p_id)
            
            paper_details[p_id] = {
                "title": p.get('title', 'Unknown'),
                "abstract": p.get('abstract', '该节点暂无详细摘要内容。'),
                "year": p.get('year', 'N/A'),
                "cites": p.get('citationCount', 0)
            }
            node_size = 12 + (math.log2(p.get('citationCount', 0) + 1) * 4)
            color = "#3b82f6" if rel_type == 'references' else "#10b981"
            nodes.append(Node(id=p_id, label=f"{p.get('title')[:15]}...", size=node_size, color=color))
            if rel_type == 'references': edges.append(Edge(source=seed_id, target=p_id, color="#3b82f6"))
            else: edges.append(Edge(source=p_id, target=seed_id, color="#10b981"))

    config = Config(width="100%", height=600, directed=True, physics=True, nodeHighlightBehavior=True, highlightColor="#F7D154")
    clicked_id = agraph(nodes=nodes, edges=edges, config=config)
    return clicked_id, paper_details

def fix_latex_errors(text):
    if not text: return text
    return text.replace(r"\(", "$").replace(r"\)", "$").replace(r"\[", "$$").replace(r"\]", "$$")

def rebuild_index_from_chunks(api_key):
    if not st.session_state.all_chunks:
        st.session_state.db = None
        return
    embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
    st.session_state.db = FAISS.from_documents(st.session_state.all_chunks, embeddings)

def process_and_add_to_db(file_path, file_name, api_key):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs: doc.metadata['source_paper'] = file_name
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        st.session_state.all_chunks.extend(chunks)
        embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
        if st.session_state.db is None:
            st.session_state.db = FAISS.from_documents(chunks, embeddings)
        else:
            st.session_state.db.add_documents(chunks)
        if file_name not in st.session_state.loaded_files:
            st.session_state.loaded_files.append(file_name)
        st.session_state.chat_history.append({"role": "system_notice", "content": f"📚 已加载《{file_name}》"})
    except Exception as e: st.error(f"处理失败: {e}")

# ================= 5. 侧边栏 =================
with st.sidebar:
    st.header("🎛️ 控制台")
    user_api_key = st.text_input("智谱 API Key", type="password")
    ss_api_key = st.text_input("SS API Key", type="password")
    st.markdown("---")
    
    if st.session_state.loaded_files:
        st.subheader("🗂️ 文件管理")
        for file in list(st.session_state.loaded_files):
            col_f1, col_f2 = st.columns([4, 1])
            with col_f1: st.text(f"📄 {file[:18]}...")
            with col_f2:
                if st.button("🗑️", key=f"del_{file}"):
                    st.session_state.loaded_files.remove(file)
                    st.session_state.all_chunks = [c for c in st.session_state.all_chunks if c.metadata.get('source_paper') != file]
                    if user_api_key: rebuild_index_from_chunks(user_api_key)
                    st.rerun()

    st.subheader("⚙️ 研读模式")
    reading_mode = st.radio("选择模式:", ["🟢 快速问答", "📖 逐段精读"], index=1)
    if st.session_state.loaded_files:
        st.session_state.selected_scope = st.selectbox("👁️ 专注范围", ["🌐 对比所有论文"] + st.session_state.loaded_files)

    st.markdown("---")
    uploaded_file = st.file_uploader("📥 上传 PDF", type="pdf")
    if uploaded_file and user_api_key and st.button("确认加载"):
        with st.spinner("解析中..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                path = tmp.name
            process_and_add_to_db(path, uploaded_file.name, user_api_key)
            os.remove(path)
            st.rerun()

# ================= 6. 主界面 =================
tab_search, tab_chat = st.tabs(["🔍 文献调研", "💬 深度对话"])

with tab_search:
    st.subheader("🌍 学术知识检索")
    col_q, col_sort, col_n = st.columns([3, 1, 0.8])
    with col_q: search_query = st.text_input("关键词", placeholder="输入搜索词...")
    with col_sort: sort_mode = st.selectbox("排序", ["🔥 相关性", "📅 时间", "📈 引用量"])
    with col_n: max_results = st.number_input("数量", 5, 50, 10)
        
    if st.button("🚀 开始检索") and search_query:
        with st.spinner("检索中..."):
            try:
                # 严格还原最原始搜索逻辑，不进行任何关键词 AND 拼接
                arxiv_sort = arxiv.SortCriterion.Relevance
                if "时间" in sort_mode: arxiv_sort = arxiv.SortCriterion.SubmittedDate
                search = arxiv.Search(query=search_query, max_results=max_results, sort_by=arxiv_sort)
                raw_results = list(search.results())
                results_with_cite = []
                for res in raw_results:
                    cites = fetch_citations(res.entry_id, ss_key=ss_api_key)
                    results_with_cite.append({'obj': res, 'citations': cites})
                if "引用量" in sort_mode:
                    results_with_cite.sort(key=lambda x: x['citations'], reverse=True)
                st.session_state.search_results = results_with_cite
            except Exception as e: st.error(f"检索失败: {e}")
                
    if st.session_state.search_results:
        if st.session_state.focus_paper_id:
            st.markdown("---")
            st.subheader("📊 文献关联图谱")
            g_data = fetch_graph_data(st.session_state.focus_paper_id, ss_key=ss_api_key)
            if g_data:
                col_graph, col_info = st.columns([2.5, 1])
                with col_graph: clicked_id, all_details = render_connected_graph(g_data)
                with col_info:
                    if clicked_id and clicked_id in all_details:
                        info = all_details[clicked_id]
                        st.markdown(f"### 📄 详情")
                        st.markdown(f"**{info['title']}** ({info['year']})")
                        st.markdown(f'<div class="abstract-box">{info["abstract"]}</div>', unsafe_allow_html=True)
                    else:
                        st.info("💡 点击节点查看摘要")
                        if st.button("❌ 关闭图谱"): st.session_state.focus_paper_id = None; st.rerun()
            st.markdown("---")

        for i, item in enumerate(st.session_state.search_results):
            res, cites = item['obj'], item['citations']
            with st.expander(f"#{i+1} {res.title} ({res.published.year})"):
                st.write(res.summary)
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(f"🕸️ 关系群", key=f"g_{i}"): st.session_state.focus_paper_id = res.entry_id; st.rerun()
                with c2:
                    if st.button(f"⬇️ 加载", key=f"l_{i}"):
                        if user_api_key:
                            with st.spinner("加载中..."):
                                path = res.download_pdf(dirpath=tempfile.gettempdir())
                                process_and_add_to_db(path, res.title, user_api_key)
                                st.success("已加载")

with tab_chat:
    for msg in st.session_state.chat_history:
        if msg["role"] == "system_notice": st.info(msg["content"])
        else:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if prompt := st.chat_input("询问论文内容..."):
        if st.session_state.db:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)
            with st.chat_message("assistant"):
                scope = st.session_state.selected_scope
                f_dict = {"source_paper": scope} if scope != "🌐 对比所有论文" else None
                docs = st.session_state.db.similarity_search(prompt, k=6, filter=f_dict)
                context = "\n\n".join([d.page_content for d in docs])
                llm = ChatZhipuAI(model="glm-4", api_key=user_api_key)
                response = llm.invoke(f"资料：{context}\n问题：{prompt}")
                content = fix_latex_errors(response.content)
                st.write(content)
                st.session_state.chat_history.append({"role": "assistant", "content": content})
