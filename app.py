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
        color: #31333F;
    }
    .cite-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .detail-panel {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        height: 600px;
        overflow-y: auto;
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
    """从 URL 中精准提取 ArXiv ID"""
    match = re.search(r'(\d{4}\.\d{4,5})', url)
    if match:
        return match.group(1)
    return url.split('/')[-1].split('v')[0]

def fetch_citations(arxiv_id, ss_key=None):
    """从 Semantic Scholar API 获取引用数"""
    try:
        clean_id = get_pure_arxiv_id(arxiv_id)
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields=citationCount"
        headers = {"x-api-key": ss_key} if ss_key else {}
        response = requests.get(api_url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json().get('citationCount', 0)
    except:
        pass
    return 0

@st.cache_data(ttl=3600)
def fetch_graph_data(arxiv_id, ss_key=None):
    """获取关联数据（严格注入子级摘要字段）"""
    try:
        clean_id = get_pure_arxiv_id(arxiv_id)
        # 精准修复点：确保 references 和 citations 的内部也有 abstract
        fields = "paperId,title,year,citationCount,abstract,references.paperId,references.title,references.citationCount,references.year,references.abstract,citations.paperId,citations.title,citations.citationCount,citations.year,citations.abstract"
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields={fields}"
        headers = {"x-api-key": ss_key} if ss_key else {}
        
        if not ss_key:
            time.sleep(1.5) # 匿名限流保护
            
        response = requests.get(api_url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"图谱数据抓取失败: {e}")
        return None

def render_connected_graph(data):
    """渲染图谱（恢复双向群簇逻辑）"""
    if not data: 
        return None, {}
    
    nodes, edges = [], []
    paper_details = {} 
    
    # 核心论文
    seed_id = data.get('paperId', 'root')
    seed_title = data.get('title', 'Seed Paper')
    paper_details[seed_id] = {
        "title": seed_title,
        "abstract": data.get('abstract', '无摘要信息'),
        "year": data.get('year', 'Unknown'),
        "cites": data.get('citationCount', 0)
    }
    nodes.append(Node(id=seed_id, label="⭐ SEED", size=30, color="#FF4B4B"))

    seen_ids = {seed_id}
    # 恢复 references 和 citations 两个群簇的提取
    for rel_type in ['references', 'citations']:
        items = data.get(rel_type, [])[:15]
        for p in items:
            p_id = p.get('paperId')
            if not p_id or p_id in seen_ids:
                continue
            seen_ids.add(p_id)
            
            title = p.get('title', 'Unknown')
            paper_details[p_id] = {
                "title": title,
                "abstract": p.get('abstract', '该关联文献暂无详细摘要。'),
                "year": p.get('year', 'Unknown'),
                "cites": p.get('citationCount', 0)
            }
            
            c_count = p.get('citationCount', 0)
            node_size = 12 + (math.log2(c_count + 1) * 4)
            node_color = "#3b82f6" if rel_type == 'references' else "#10b981"
            
            nodes.append(Node(id=p_id, label=f"{title[:15]}...", size=node_size, color=node_color))
            if rel_type == 'references':
                edges.append(Edge(source=seed_id, target=p_id, color="#3b82f6", width=1))
            else:
                edges.append(Edge(source=p_id, target=seed_id, color="#10b981", width=1))

    config = Config(width="100%", height=600, directed=True, physics=True, nodeHighlightBehavior=True, highlightColor="#F7D154")
    clicked_id = agraph(nodes=nodes, edges=edges, config=config)
    return clicked_id, paper_details

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
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source_paper'] = file_name
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200, separators=["\n\n", "\n", "。", ".", " ", ""])
        chunks = splitter.split_documents(docs)
        valid_chunks = [c for c in chunks if len(c.page_content.strip()) > 20]
        
        st.session_state.all_chunks.extend(valid_chunks)
        embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
        
        # 分批处理防止超时
        batch_size = 10
        total = len(valid_chunks)
        if st.session_state.db is None:
            st.session_state.db = FAISS.from_documents(valid_chunks[:batch_size], embeddings)
            if total > batch_size:
                for i in range(batch_size, total, batch_size):
                    st.session_state.db.add_documents(valid_chunks[i: i + batch_size])
                    time.sleep(0.1)
        else:
            for i in range(0, total, batch_size):
                st.session_state.db.add_documents(valid_chunks[i: i + batch_size])
                time.sleep(0.1)
        
        if file_name not in st.session_state.loaded_files:
            st.session_state.loaded_files.append(file_name)
        st.session_state.chat_history.append({"role": "system_notice", "content": f"📚 **系统通知**：已加载《{file_name}》。"})
    except Exception as e:
        st.error(f"处理失败: {e}")

# ================= 5. 侧边栏布局 =================
with st.sidebar:
    st.header("🎛️ 助手控制台")
    user_api_key = st.text_input("智谱 AI API Key", type="password", help="用于大模型对话和向量化")
    ss_api_key = st.text_input("Semantic Scholar Key (可选)", type="password", help="填入可提高接口调用频率限制")
    st.markdown("---")
    
    if st.session_state.loaded_files:
        st.subheader("🗂️ 已加载文献")
        for file in list(st.session_state.loaded_files):
            col_f1, col_f2 = st.columns([4, 1])
            with col_f1:
                st.text(f"📄 {file[:18]}..." if len(file)>20 else f"📄 {file}")
            with col_f2:
                if st.button("🗑️", key=f"del_{file}"):
                    st.session_state.loaded_files.remove(file)
                    st.session_state.all_chunks = [c for c in st.session_state.all_chunks if c.metadata.get('source_paper') != file]
                    if user_api_key:
                        rebuild_index_from_chunks(user_api_key)
                    st.rerun()
        
        if st.button("🗑️ 清空所有文献", type="primary"):
            st.session_state.db = None
            st.session_state.loaded_files = []
            st.session_state.all_chunks = []
            st.session_state.chat_history = []
            st.rerun()

    st.subheader("⚙️ 研读偏好")
    reading_mode = st.radio("对话模式:", ["🟢 快速回答", "📖 逐段精读 (增强公式)"], index=1)

    if st.session_state.loaded_files:
        st.markdown("---")
        if st.button("🪄 自动生成多论文对比表"):
            if user_api_key and st.session_state.db:
                with st.spinner("深度分析中..."):
                    llm = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.1)
                    aggregated_context = ""
                    for filename in st.session_state.loaded_files:
                        sub_docs = st.session_state.db.similarity_search("Abstract and main findings", k=2, filter={"source_paper": filename})
                        if sub_docs:
                            aggregated_context += f"\n=== 文献: {filename} ===\n" + "\n".join([d.page_content for d in sub_docs])
                    res = llm.invoke(f"基于以下内容生成 Markdown 对比表：\n{aggregated_context}")
                    st.session_state.chat_history.append({"role": "assistant", "content": res.content})
                    st.rerun()

        scope_options = ["🌐 对比所有论文"] + st.session_state.loaded_files
        st.session_state.selected_scope = st.selectbox("👁️ 当前对话范围", scope_options)

    st.markdown("---")
    st.subheader("📥 本地论文上传")
    uploaded_file = st.file_uploader("上传 PDF 文献", type="pdf")
    if uploaded_file and user_api_key and st.button("开始解析并学习"):
        with st.spinner("PDF 深度解析中..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                path = tmp.name
            process_and_add_to_db(path, uploaded_file.name, user_api_key)
            os.remove(path)
            st.rerun()

# ================= 6. 主界面布局 =================
tab_search, tab_chat = st.tabs(["🔍 文献调研 (Connected Papers 模式)", "💬 论文深读空间"])

with tab_search:
    st.subheader("🌍 ArXiv 全球文献检索")
    col_q, col_sort, col_n = st.columns([3, 1.5, 1])
    with col_q:
        search_query = st.text_input("检索关键词", placeholder="例如: 'transformer architecture' 或 'LLM reasoning'")
    with col_sort:
        sort_mode = st.selectbox("排序方式", ["🔥 相关性优先", "📅 最新发布", "📈 引用量之最"])
    with col_n:
        max_results = st.number_input("获取数量", min_value=5, max_value=50, value=15)
        
    if st.button("🚀 执行检索") and search_query:
        with st.spinner("正在检索并拉取引用统计信息..."):
            try:
                arxiv_sort = arxiv.SortCriterion.Relevance
                if "最新" in sort_mode: arxiv_sort = arxiv.SortCriterion.SubmittedDate
                search = arxiv.Search(query=search_query, max_results=max_results, sort_by=arxiv_sort)
                raw_results = list(search.results())
                results_with_cite = []
                for res in raw_results:
                    cites = fetch_citations(res.entry_id, ss_key=ss_api_key)
                    results_with_cite.append({'obj': res, 'citations': cites})
                if "引用量" in sort_mode:
                    results_with_cite.sort(key=lambda x: x['citations'], reverse=True)
                st.session_state.search_results = results_with_cite
                st.success(f"✅ 找到 {len(results_with_cite)} 篇相关文献")
            except Exception as e:
                st.error(f"检索失败: {e}")
                
    if st.session_state.search_results:
        # 图谱渲染面板
        if st.session_state.focus_paper_id:
            st.markdown("---")
            st.subheader("📊 文献关联网络 (Connected Graph)")
            g_data = fetch_graph_data(st.session_state.focus_paper_id, ss_key=ss_api_key)
            if not g_data:
                st.warning("⚠️ 无法获取图谱数据。如果是匿名模式，请稍后再试或填入 SS API Key。")
            else:
                col_graph, col_info = st.columns([2.5, 1])
                with col_graph:
                    clicked_node_id, all_details = render_connected_graph(g_data)
                with col_info:
                    if clicked_node_id and clicked_node_id in all_details:
                        info = all_details[clicked_node_id]
                        st.markdown(f"### 📄 选定文献详情")
                        st.markdown(f"**标题**: {info['title']}")
                        st.markdown(f"**年份**: {info['year']} | **引用**: {info['cites']}")
                        st.markdown("---")
                        st.markdown(f'<div class="abstract-box">{info["abstract"]}</div>', unsafe_allow_html=True)
                    else:
                        st.info("💡 **操作提示**\n\n点击左侧圆点即可在此处查看对应论文的摘要。")
                        if st.button("❌ 关闭图谱面板"):
                            st.session_state.focus_paper_id = None
                            st.rerun()
            st.markdown("---")

        for i, item in enumerate(st.session_state.search_results):
            res = item['obj']
            cites = item['citations']
            with st.expander(f"#{i+1} 📄 {res.title} ({res.published.year})"):
                st.markdown(f"**🔥 引用次数**: <span class='cite-badge'>{cites}</span>", unsafe_allow_html=True)
                st.write(res.summary.replace("\n", " "))
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.markdown(f"[🔗 ArXiv 页面]({res.entry_id})")
                with col2:
                    if st.button(f"⬇️ 加入深读库", key=f"dl_search_{i}"):
                        if user_api_key:
                            with st.spinner("下载并解析中..."):
                                pdf_path = res.download_pdf(dirpath=tempfile.gettempdir())
                                process_and_add_to_db(pdf_path, res.title, user_api_key)
                                st.success("已成功入库")
                        else:
                            st.error("请先在侧边栏填入 API Key")
                with col3:
                    if st.button(f"🕸️ 查看关系群", key=f"btn_graph_{i}"):
                        st.session_state.focus_paper_id = res.entry_id
                        st.rerun()

with tab_chat:
    if st.session_state.loaded_files:
        st.caption(f"📚 范围：{st.session_state.selected_scope} | 模式：{reading_mode}")
    for msg in st.session_state.chat_history:
        if msg["role"] == "system_notice":
            st.info(msg["content"])
        else:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
    if prompt := st.chat_input("基于已加载的文献提问..."):
        if not st.session_state.db:
            st.warning("⚠️ 请先在侧边栏上传论文或从检索结果中下载论文。")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                try:
                    scope = st.session_state.selected_scope
                    f_dict = {"source_paper": scope} if scope != "🌐 对比所有论文" else None
                    docs = st.session_state.db.similarity_search(prompt, k=8, filter=f_dict)
                    context = "\n\n".join([f"📄【{d.metadata.get('source_paper','?')}】:\n{d.page_content}" for d in docs])
                    
                    llm = ChatZhipuAI(model="glm-4", api_key=user_api_key)
                    response = llm.invoke(f"背景资料：\n{context}\n\n问题：{prompt}")
                    
                    final_content = fix_latex_errors(response.content)
                    st.write(final_content)
                    st.session_state.chat_history.append({"role": "assistant", "content": final_content})
                except Exception as e:
                    st.error(f"对话引擎故障: {e}")
