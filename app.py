import streamlit as st
import sys
import os
import time
import tempfile
import arxiv
import requests
import re
from collections import Counter

# ================= 1. 环境自检 =================
try:
    import zhipuai
    import langchain_community
    import fitz  # pymupdf
except ImportError as e:
    st.error(f"🚑 核心环境缺失 -> {e.name}。请执行: pip install zhipuai langchain_community pymupdf requests arxiv")
    st.stop()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= 2. 页面配置与 CSS 样式表 (完整还原) =================
st.set_page_config(page_title="AI 深度研读助手 (全功能终极版)", layout="wide", page_icon="🎓")
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .abstract-box {
        background-color: #f1f3f5; padding: 20px; border-radius: 12px;
        border-left: 6px solid #28a745; font-size: 0.98em; line-height: 1.8;
        margin-bottom: 15px; color: #343a40; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .cite-badge {
        background-color: #dc3545; color: white; padding: 4px 14px;
        border-radius: 20px; font-size: 0.85em; font-weight: bold;
    }
    .topic-tag {
        display: inline-block; background-color: #e7f3ff; color: #007bff;
        padding: 5px 12px; border-radius: 6px; margin: 5px;
        font-size: 0.88em; border: 1px solid #cce5ff; font-weight: 500;
    }
    .metric-card {
        background-color: white; padding: 15px; border-radius: 10px;
        border: 1px solid #dee2e6; text-align: center; box-shadow: 1px 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)
st.title("📖 AI 深度研读助手 (全功能恢复版)")

# ================= 3. 全局状态初始化 =================
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "db" not in st.session_state: st.session_state.db = None
if "loaded_files" not in st.session_state: st.session_state.loaded_files = []
if "all_chunks" not in st.session_state: st.session_state.all_chunks = []
if "suggested_query" not in st.session_state: st.session_state.suggested_query = ""
if "search_results" not in st.session_state: st.session_state.search_results = []
if "selected_scope" not in st.session_state: st.session_state.selected_scope = "🌐 对比所有论文"

# ================= 4. 核心功能函数集 =================

def fetch_citations(arxiv_id):
    """从 Semantic Scholar 实时调取引用量数据"""
    try:
        clean_id = arxiv_id.split('/')[-1].split('v')[0]
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields=citationCount"
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            return response.json().get('citationCount', 0)
    except Exception:
        pass
    return 0

def extract_top_topics(results):
    """学术热点词分析 (Google 检索逻辑)"""
    all_text = ""
    for item in results:
        res = item['obj']
        all_text += f" {res.title} {res.summary}"
    words = re.findall(r'\b\w{5,}\b', all_text.lower())
    stop_words = {'learning', 'robotics', 'education', 'research', 'paper', 'approach', 'system', 'based', 'using', 'results', 'provide', 'model', 'analysis', 'method'}
    meaningful_words = [w for w in words if w not in stop_words]
    return Counter(meaningful_words).most_common(10)

def fix_latex_errors(text):
    """深度修复 LaTeX 渲染标识问题"""
    if not text: return text
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    return text

def generate_html_report(chat_history):
    """导出带 MathJax 渲染支持的专业 HTML 报告"""
    html = """<!DOCTYPE html><html><head><meta charset="UTF-8">
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 900px; margin: 0 auto; padding: 40px; line-height: 1.7; color: #333; background-color: #f9f9f9; }
        h1 { color: #1b5e20; border-bottom: 3px solid #4caf50; padding-bottom: 12px; }
        .message { margin-bottom: 30px; padding: 25px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        .user { background-color: #e3f2fd; border-left: 8px solid #1976d2; }
        .assistant { background-color: #f1f8e9; border-left: 8px solid #43a047; }
        .system { background-color: #fff3e0; border-left: 8px solid #fb8c00; font-style: italic; color: #666; }
        .role-title { font-weight: bold; display: block; margin-bottom: 12px; text-transform: uppercase; font-size: 0.85em; color: #555; }
        pre { background: #eee; padding: 10px; border-radius: 5px; overflow-x: auto; }
    </style></head><body><h1>🎓 AI 深度研读笔记报告</h1>"""
    for msg in chat_history:
        role_label = "🧑‍💻 我" if msg['role'] == 'user' else "🤖 AI 研究员" if msg['role'] == 'assistant' else "🔔 系统系统通知"
        content_formatted = msg['content'].replace('\n', '<br>')
        html += f'<div class="message {msg["role"]}"><span class="role-title">{role_label}</span>{content_formatted}</div>'
    html += "</body></html>"
    return html

def rebuild_index_from_chunks(api_key):
    """物理删除文档后重构向量索引 (含 Batch 保护)"""
    if not st.session_state.all_chunks:
        st.session_state.db = None
        return
    embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
    chunks = st.session_state.all_chunks
    batch_size = 32
    st.session_state.db = FAISS.from_documents(chunks[:batch_size], embeddings)
    for i in range(batch_size, len(chunks), batch_size):
        st.session_state.db.add_documents(chunks[i:i+batch_size])
        time.sleep(0.1)

def process_and_add_to_db(file_path, file_name, api_key):
    """解析 PDF 并执行分批次向量化 (彻底修复 1214 错误)"""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source_paper'] = file_name
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=200, separators=["\n\n", "\n", "。", ".", " ", ""])
        new_chunks = splitter.split_documents(docs)
        valid_new = [c for c in new_chunks if len(c.page_content.strip()) > 30]
        
        embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
        
        # --- 分批处理核心逻辑 ---
        batch_size = 32
        total_len = len(valid_new)
        with st.spinner(f"正在向量化《{file_name}》，共 {total_len} 个片段，分批上传中..."):
            if st.session_state.db is None:
                st.session_state.db = FAISS.from_documents(valid_new[:batch_size], embeddings)
                current_start = batch_size
            else:
                current_start = 0
            
            for i in range(current_start, total_len, batch_size):
                batch_data = valid_new[i : i + batch_size]
                st.session_state.db.add_documents(batch_data)
                time.sleep(0.2) # 防止 API 频率限制
        
        st.session_state.all_chunks.extend(valid_new)
        if file_name not in st.session_state.loaded_files:
            st.session_state.loaded_files.append(file_name)
        st.session_state.chat_history.append({"role": "system_notice", "content": f"📚 **库更新**：成功加载并索引文档《{file_name}》"})
    except Exception as e:
        st.error(f"解析失败: {str(e)}")

# ================= 5. 侧边栏：控制面板 (完整还原功能) =================
with st.sidebar:
    st.header("🎛️ 科研工作台")
    user_api_key = st.text_input("智谱 API Key", type="password", help="请填写 GLM-4 有效 Key")
    st.markdown("---")
    
    if st.session_state.loaded_files:
        st.subheader("🗂️ 文献库管理")
        for file in list(st.session_state.loaded_files):
            col_f, col_d = st.columns([4, 1])
            with col_f: st.caption(f"📄 {file[:25]}...")
            with col_d:
                if st.button("🗑️", key=f"del_{file}"):
                    st.session_state.loaded_files.remove(file)
                    st.session_state.all_chunks = [c for c in st.session_state.all_chunks if c.metadata.get('source_paper') != file]
                    if user_api_key: rebuild_index_from_chunks(user_api_key)
                    st.rerun()

        st.markdown("---")
        # 核心功能 1: 综述对比表
        if st.button("🪄 一键生成综述对比表", type="primary"):
            if user_api_key and st.session_state.db:
                with st.spinner("横向扫描文献中..."):
                    llm = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.1)
                    comparison_ctx = ""
                    for paper in st.session_state.loaded_files:
                        top_sub = st.session_state.db.similarity_search("Abstract methodology contribution", k=3, filter={"source_paper": paper})
                        comparison_ctx += f"\n[文章: {paper}]\n" + "\n".join([d.page_content for d in top_sub])
                    prompt = f"对比以下科研文献，生成一个 Markdown 表格，包含：论文名、核心创新点、主要方法、研究局限。内容如下：\n{comparison_ctx}"
                    response = llm.invoke(prompt)
                    st.session_state.chat_history.append({"role": "assistant", "content": response.content})
                    st.rerun()

        st.markdown("---")
        st.session_state.selected_scope = st.selectbox("👁️ 对话专注范围", ["🌐 对比所有论文"] + st.session_state.loaded_files)

        # 核心功能 2: 挖掘关联论文
        if st.button(f"🔍 挖掘【{st.session_state.selected_scope[:6]}】关联词"):
            if user_api_key and st.session_state.db:
                with st.spinner("AI 深度特征提炼中..."):
                    scope = st.session_state.selected_scope
                    f_dict = {"source_paper": scope} if scope != "🌐 对比所有论文" else None
                    key_docs = st.session_state.db.similarity_search("Introduction future work research gap", k=4, filter=f_dict)
                    llm = ChatZhipuAI(model="glm-4", api_key=user_api_key)
                    prompt = f"基于以下文本片段，提炼 2 个最精准的英文学术搜索词组，用于进一步检索相关文献（只输出词组）：\n" + "\n".join([d.page_content for d in key_docs])
                    new_q = llm.invoke(prompt).content.strip()
                    st.session_state.suggested_query = new_q
                    st.success(f"新搜索词已生成！")
                    st.rerun()

    st.markdown("---")
    if st.session_state.chat_history:
        st.download_button("💾 下载研读报告 (HTML)", generate_html_report(st.session_state.chat_history), "study_report.html", "text/html")
    
    st.subheader("📥 导入本地文献")
    uploaded_pdf = st.file_uploader("上传 PDF 论文", type="pdf")
    if uploaded_pdf and user_api_key and st.button("开始解析"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.getvalue())
            process_and_add_to_db(tmp.name, uploaded_pdf.name, user_api_key)
            os.remove(tmp.name)
            st.rerun()

# ================= 6. 主界面 =================
tab_search, tab_chat = st.tabs(["🔍 文献调研引擎 (Google Logic)", "💬 深度研读对话"])

with tab_search:
    st.subheader("🌍 学术大数据联合调研")
    col_q, col_s, col_n = st.columns([3, 1.2, 0.8])
    with col_q:
        input_q = st.text_input("关键词/挖掘词", value=st.session_state.suggested_query, placeholder="例如: robotics education human-robot interaction")
    with col_s:
        sort_mode = st.selectbox("排序逻辑", ["🔥 相关性优先", "📅 时间最新", "📈 引用量之王"])
    with col_n:
        n_results = st.number_input("篇数", 5, 50, 15)

    if st.button("🚀 启动深度检索") and input_q:
        with st.spinner("正在检索并同步 Citation 数据..."):
            try:
                arxiv_sort = arxiv.SortCriterion.Relevance
                if "时间" in sort_mode: arxiv_sort = arxiv.SortCriterion.SubmittedDate
                
                # 布尔检索优化
                processed_q = input_q if ("AND" in input_q) else " AND ".join([f"(ti:{w} OR abs:{w})" for w in input_q.split()])
                search_client = arxiv.Search(query=processed_q, max_results=n_results, sort_by=arxiv_sort)
                results_with_meta = []
                for res in list(search_client.results()):
                    results_with_meta.append({'obj': res, 'citations': fetch_citations(res.entry_id)})
                    time.sleep(0.1)
                
                if "引用量" in sort_mode:
                    results_with_meta.sort(key=lambda x: x['citations'], reverse=True)
                st.session_state.search_results = results_with_meta
            except Exception as e:
                st.error(f"检索出错: {e}")

    if st.session_state.search_results:
        # 补全功能 3: 领域关键词热度分布
        topics = extract_top_topics(st.session_state.search_results)
        st.write("📊 **当前调研热点统计** (有助于识别研究偏向):")
        topic_cols = st.columns(len(topics))
        for i, (word, count) in enumerate(topics):
            topic_cols[i].markdown(f"<div class='topic-tag'>{word} ({count})</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        for i, item in enumerate(st.session_state.search_results):
            res, cites = item['obj'], item['citations']
            is_precise = any(w.lower() in res.title.lower() for w in input_q.split()[:1])
            with st.expander(f"{'🎯' if is_precise else '📄'} #{i+1} [{cites} 引用] {res.title} ({res.published.year})"):
                clean_abs = res.summary.replace('\n', ' ')
                st.markdown(f"<div class='abstract-box'><b>Abstract:</b><br>{clean_abs}</div>", unsafe_allow_html=True)
                col1, col2 = st.columns([1, 1])
                with col1: st.markdown(f"[🔗 ArXiv 地址]({res.entry_id})")
                with col2:
                    if st.button(f"⬇️ 下载并研读此篇", key=f"dl_btn_{i}"):
                        if user_api_key:
                            with st.spinner("正在分批索引..."):
                                pdf_path = res.download_pdf(dirpath=tempfile.gettempdir())
                                process_and_add_to_db(pdf_path, res.title, user_api_key)
                                st.success("已完成！")
                        else: st.error("请先在侧边栏填写 API Key")

with tab_chat:
    if st.session_state.loaded_files:
        st.caption(f"📚 研读范围: {st.session_state.selected_scope}")
        for msg in st.session_state.chat_history:
            if msg["role"] == "system_notice": st.info(msg["content"])
            else:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("针对选中论文进行深度提问..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)
            with st.chat_message("assistant"):
                try:
                    scope = st.session_state.selected_scope
                    f_dict = {"source_paper": scope} if scope != "🌐 对比所有论文" else None
                    # MMR 检索逻辑补全
                    search_docs = st.session_state.db.max_marginal_relevance_search(
                        prompt, k=7, fetch_k=20, lambda_mult=0.75, filter=f_dict
                    )
                    context_str = "\n\n".join([f"📄【{d.metadata.get('source_paper','?')} P{d.metadata.get('page',0)+1}】:\n{d.page_content}" for d in search_docs])
                    
                    llm_chat = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.1)
                    full_prompt = f"你是资深科研专家。基于以下论文资料回答：\n{context_str}\n\n问题：{prompt}\n要求：严谨引用，数学公式务必用 $ 包裹。"
                    ans_res = llm_chat.invoke(full_prompt)
                    final_ans = fix_latex_errors(ans_res.content)
                    st.write(final_ans)
                    st.session_state.chat_history.append({"role": "assistant", "content": final_ans})
                except Exception as e:
                    st.error(f"生成失败: {e}")
    else:
        st.info("💡 研读库为空。请先通过检索下载论文，或手动上传 PDF 文件。")
