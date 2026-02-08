import streamlit as st
import sys
import os
import time
import tempfile
import arxiv
import requests  # 新增：用于调用 Semantic Scholar API
from streamlit_agraph import agraph, Node, Edge, Config # 新增：图谱库

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
if "focus_paper_id" not in st.session_state: # 新增：用于跟踪图谱展示
    st.session_state.focus_paper_id = None

# ================= 4. 核心逻辑函数 =================

def fetch_citations(arxiv_id):
    """从 Semantic Scholar API 获取引用数"""
    try:
        clean_id = arxiv_id.split('/')[-1].split('v')[0]
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields=citationCount,title,year"
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            return response.json().get('citationCount', 0)
    except:
        pass
    return 0

# --- 新增图谱数据获取函数 ---
def fetch_graph_data(arxiv_id):
    try:
        clean_id = arxiv_id.split('/')[-1].split('v')[0]
        fields = "title,year,references,citations"
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields={fields}"
        response = requests.get(api_url, timeout=8)
        if response.status_code == 200: return response.json()
    except: pass
    return None

# --- 新增图谱渲染函数 ---
def render_connected_graph(data):
    if not data: return st.warning("无法获取关联数据")
    nodes, edges = [], []
    # 中心节点
    nodes.append(Node(id="root", label="Seed Paper", size=25, color="#FF4B4B"))
    # 被引 (Citations)
    for i, item in enumerate(data.get('citations', [])[:10]):
        nid = f"c_{i}"
        nodes.append(Node(id=nid, label=item.get('title','')[:20], size=15, color="#2ca02c"))
        edges.append(Edge(source=nid, target="root"))
    # 引用 (References)
    for i, item in enumerate(data.get('references', [])[:10]):
        nid = f"r_{i}"
        nodes.append(Node(id=nid, label=item.get('title','')[:20], size=15, color="#1f77b4"))
        edges.append(Edge(source="root", target=nid))
    
    config = Config(width=1000, height=450, directed=True, physics=True)
    return agraph(nodes=nodes, edges=edges, config=config)

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
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,       
            chunk_overlap=200,    
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
        
        st.session_state.chat_history.append({
            "role": "system_notice",
            "content": f"📚 **系统通知**：已加载《{file_name}》。"
        })
    except Exception as e:
        st.error(f"处理失败: {e}")

def generate_html_report(chat_history):
    html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>AI 研究笔记</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
    .message { margin-bottom: 20px; padding: 15px; border-radius: 8px; }
    .user { background-color: #e3f2fd; border-left: 5px solid #2196F3; }
    .assistant { background-color: #f1f8e9; border-left: 5px solid #4CAF50; }</style></head>
    <body><h1>🎓 AI 深度研读笔记</h1><p>导出时间：""" + time.strftime('%Y-%m-%d %H:%M') + """</p>"""
    for msg in chat_history:
        role_class = msg['role'] if msg['role'] in ['user', 'assistant'] else 'system'
        content_html = msg['content'].replace('\n', '<br>')
        html += f'<div class="message {role_class}"><b>{msg["role"]}</b><br>{content_html}</div>'
    html += "</body></html>"
    return html

# ================= 5. 侧边栏 =================
with st.sidebar:
    st.header("🎛️ 控制台")
    user_api_key = st.text_input("智谱 API Key", type="password")
    st.markdown("---")
    
    if st.session_state.loaded_files:
        st.subheader("🗂️ 文件管理")
        for file in list(st.session_state.loaded_files):
            col_f1, col_f2 = st.columns([4, 1])
            with col_f1: st.text(f"📄 {file[:18]}..." if len(file)>20 else f"📄 {file}")
            with col_f2:
                if st.button("🗑️", key=f"del_{file}"):
                    st.session_state.loaded_files.remove(file)
                    st.session_state.all_chunks = [c for c in st.session_state.all_chunks if c.metadata.get('source_paper') != file]
                    if user_api_key: rebuild_index_from_chunks(user_api_key)
                    st.rerun()
        
        if st.button("🗑️ 清空全部", type="primary"):
            st.session_state.db = None
            st.session_state.loaded_files = []
            st.session_state.all_chunks = []
            st.session_state.chat_history = []
            st.rerun()

    st.subheader("⚙️ 研读模式")
    reading_mode = st.radio("选择模式:", ["🟢 快速问答", "📖 逐段精读 (公式修复版)"], index=1)

    if st.session_state.loaded_files:
        st.markdown("---")
        if st.button("🪄 一键生成综述对比表"):
            if user_api_key and st.session_state.db:
                with st.spinner("分析中..."):
                    llm = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.1)
                    aggregated_context = ""
                    for filename in st.session_state.loaded_files:
                        sub_docs = st.session_state.db.similarity_search("Abstract conclusion main contribution", k=3, filter={"source_paper": filename})
                        if sub_docs: aggregated_context += f"\n=== {filename} ===\n" + "\n".join([d.page_content for d in sub_docs]) + "\n"
                    res = llm.invoke(f"阅读以下摘要，生成 Markdown 表格(列：论文名|创新点|方法|结论)：\n{aggregated_context}")
                    st.session_state.chat_history.append({"role": "assistant", "content": res.content})
                    st.rerun()

        scope_options = ["🌐 对比所有论文"] + st.session_state.loaded_files
        st.session_state.selected_scope = st.selectbox("👁️ 专注范围", scope_options)

    st.markdown("---")
    st.subheader("📥 上传论文")
    uploaded_file = st.file_uploader("拖入 PDF", type="pdf")
    if uploaded_file and user_api_key and st.button("确认加载"):
        with st.spinner("解析中..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                path = tmp.name
            process_and_add_to_db(path, uploaded_file.name, user_api_key)
            os.remove(path)
            st.rerun()

# ================= 6. 主界面 =================
tab_search, tab_chat = st.tabs(["🔍 文献调研 (引用增强)", "💬 研读空间"])

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
        with st.spinner("正在检索并同步 Semantic Scholar 引用数据..."):
            try:
                # ArXiv 排序参数映射
                arxiv_sort = arxiv.SortCriterion.Relevance
                if "时间" in sort_mode: arxiv_sort = arxiv.SortCriterion.SubmittedDate
                
                # 自动优化布尔查询
                refined_query = search_query
                if " " in search_query and "AND" not in search_query and '"' not in search_query:
                    refined_query = " AND ".join([f'(ti:{w} OR abs:{w})' for w in search_query.split()])

                search = arxiv.Search(query=refined_query, max_results=max_results, sort_by=arxiv_sort)
                raw_results = list(search.results())
                
                # 引用数补全
                results_with_cite = []
                progress_bar = st.progress(0)
                for idx, res in enumerate(raw_results):
                    cites = fetch_citations(res.entry_id)
                    results_with_cite.append({'obj': res, 'citations': cites})
                    progress_bar.progress((idx + 1) / len(raw_results))
                
                # 引用排序处理
                if "引用量" in sort_mode:
                    results_with_cite.sort(key=lambda x: x['citations'], reverse=True)
                
                st.session_state.search_results = results_with_cite
                st.success(f"✅ 完成！已获取 {len(results_with_cite)} 篇论文。")
            except Exception as e:
                st.error(f"检索失败: {e}")
                
    if st.session_state.search_results:
        # 新增图谱显示区域
        if st.session_state.focus_paper_id:
            st.markdown("---")
            st.subheader("📊 文献关联图谱 (Connected Graph)")
            col_graph, col_info = st.columns([3, 1])
            with col_graph:
                g_data = fetch_graph_data(st.session_state.focus_paper_id)
                render_connected_graph(g_data)
            with col_info:
                st.caption("🟢 绿色: Citations (引用本文)")
                st.caption("🔵 蓝色: References (参考文献)")
                if st.button("❌ 关闭图谱"):
                    st.session_state.focus_paper_id = None
                    st.rerun()
            st.markdown("---")

        for i, item in enumerate(st.session_state.search_results):
            res = item['obj']
            cites = item['citations']
            with st.expander(f"#{i+1} 📄 {res.title} ({res.published.year})"):
                st.markdown(f"**👨‍🏫 作者**: {', '.join([a.name for a in res.authors])} | **📅 发表**: {res.published.strftime('%Y-%m-%d')}")
                st.markdown(f"**🔥 引用数 (Semantic Scholar)**: <span class='cite-badge'>{cites}</span>", unsafe_allow_html=True)
                
                st.markdown(f'<div class="abstract-box"><b>📝 摘要：</b><br>{res.summary.replace("\n", " ")}</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1: st.markdown(f"[🔗 ArXiv 原文]({res.entry_id})")
                with col2:
                    if st.button(f"⬇️ 下载分析", key=f"dl_search_{i}"):
                        if user_api_key:
                            with st.spinner("下载解析中..."):
                                try:
                                    pdf_path = res.download_pdf(dirpath=tempfile.gettempdir())
                                    process_and_add_to_db(pdf_path, res.title, user_api_key)
                                    st.success("入库成功！")
                                except Exception as e: st.error(f"失败: {e}")
                        else: st.error("请填入 API Key")
                with col3:
                    if st.button(f"🕸️ 关联图谱", key=f"btn_graph_{i}"):
                        st.session_state.focus_paper_id = res.entry_id
                        st.rerun()

with tab_chat:
    if st.session_state.loaded_files:
        st.caption(f"📚 模式：{reading_mode} | 范围：{st.session_state.selected_scope}")

    for msg in st.session_state.chat_history:
        if msg["role"] == "system_notice": st.info(msg["content"])
        else:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("输入问题..."):
        if not st.session_state.db: st.warning("🧠 请先添加论文")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)
            with st.chat_message("assistant"):
                try:
                    search_k = 15 if "精读" in reading_mode else 8
                    current_scope = st.session_state.get("selected_scope", "🌐 对比所有论文")
                    filter_dict = {"source_paper": current_scope} if current_scope != "🌐 对比所有论文" else None

                    docs = st.session_state.db.max_marginal_relevance_search(prompt, k=search_k, fetch_k=20, lambda_mult=0.6, filter=filter_dict)
                    if not docs: st.warning("未找到相关内容。")
                    else:
                        context = "\n\n".join([f"📄【{d.metadata.get('source_paper','?')} P{d.metadata.get('page',0)+1}】:\n{d.page_content}" for d in docs])
                        sys_prompt = f"你是一位科研助手。基于资料回答问题：\n资料：{context}\n问题：{prompt}\n要求：公式用 $ 包裹。"
                        llm = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.1)
                        response = llm.invoke(sys_prompt)
                        final_content = fix_latex_errors(response.content)
                        st.write(final_content)
                        st.session_state.chat_history.append({"role": "assistant", "content": final_content})
                except Exception as e: st.error(f"生成出错: {e}")
