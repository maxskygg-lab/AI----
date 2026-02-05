import streamlit as st
import sys
import os
import time
import tempfile
import arxiv

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
st.set_page_config(page_title="AI 深度研读助手 (全信息版)", layout="wide", page_icon="🎓")
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
</style>
""", unsafe_allow_html=True)
st.title("📖 AI 深度研读助手 (全信息版)")

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
# --- 修复 NameError 的关键点：初始化默认范围 ---
if "selected_scope" not in st.session_state:
    st.session_state.selected_scope = "🌐 对比所有论文"

# ================= 4. 核心逻辑函数 =================

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
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AI 研究笔记</title>
        <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; color: #333; }
            h1 { border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
            .message { margin-bottom: 20px; padding: 15px; border-radius: 8px; }
            .user { background-color: #e3f2fd; border-left: 5px solid #2196F3; }
            .assistant { background-color: #f1f8e9; border-left: 5px solid #4CAF50; }
            .system { background-color: #fff3e0; border-left: 5px solid #ff9800; font-style: italic; }
        </style>
    </head>
    <body>
        <h1>🎓 AI 深度研读笔记</h1>
        <p>导出时间：""" + time.strftime('%Y-%m-%d %H:%M') + """</p>
    """
    for msg in chat_history:
        role_class = msg['role'] if msg['role'] in ['user', 'assistant'] else 'system'
        role_name = "🧑‍💻 我" if msg['role'] == 'user' else "🤖 AI 研究员" if msg['role'] == 'assistant' else "🔔 系统"
        content_html = msg['content'].replace('\n', '<br>')
        html += f"""
        <div class="message {role_class}">
            <span class="role-label">{role_name}</span>
            <div>{content_html}</div>
        </div>
        """
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
            with col_f1:
                st.text(f"📄 {file[:18]}..." if len(file)>20 else f"📄 {file}")
            with col_f2:
                if st.button("🗑️", key=f"del_{file}", help=f"删除 {file}"):
                    st.session_state.loaded_files.remove(file)
                    st.session_state.all_chunks = [
                        c for c in st.session_state.all_chunks 
                        if c.metadata.get('source_paper') != file
                    ]
                    if user_api_key:
                        with st.spinner("正在重组知识库..."):
                            rebuild_index_from_chunks(user_api_key)
                            st.rerun()
                    else:
                        st.error("需要 API Key 来重组数据库")
        
        if st.button("🗑️ 清空全部", type="primary"):
            st.session_state.db = None
            st.session_state.loaded_files = []
            st.session_state.all_chunks = []
            st.session_state.chat_history = []
            st.rerun()
        st.markdown("---")

    st.subheader("⚙️ 研读模式")
    reading_mode = st.radio("选择模式:", ["🟢 快速问答", "📖 逐段精读 (公式修复版)"], index=1)

    st.markdown("---")

    if st.session_state.loaded_files:
        if st.button("🪄 一键生成综述对比表"):
            if not user_api_key:
                st.error("需要 API Key")
            elif not st.session_state.db:
                st.warning("数据库为空")
            else:
                with st.spinner(f"正在分析..."):
                    try:
                        llm = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.1)
                        aggregated_context = ""
                        for filename in st.session_state.loaded_files:
                            sub_docs = st.session_state.db.similarity_search("Abstract conclusion main contribution", k=3, filter={"source_paper": filename})
                            if sub_docs:
                                file_content = "\n".join([d.page_content for d in sub_docs])
                                aggregated_context += f"\n=== {filename} ===\n{file_content}\n"
                        prompt = f"阅读以下论文摘要，生成 Markdown 对比表格(列：论文名|创新点|方法|结论)：\n{aggregated_context}"
                        res = llm.invoke(prompt)
                        st.session_state.chat_history.append({"role": "assistant", "content": res.content})
                        st.rerun()
                    except Exception as e:
                        st.error(f"生成失败: {e}")

        scope_options = ["🌐 对比所有论文"] + st.session_state.loaded_files
        # 将选择的结果存入 session_state 避免丢失
        st.session_state.selected_scope = st.selectbox("👁️ 专注范围", scope_options)
        
        if st.button(f"🔍 基于【{st.session_state.selected_scope[:5]}...】挖掘新论文"):
            if not user_api_key:
                st.error("请填入 API Key")
            else:
                with st.spinner("🤖 AI 正在深度分析文本，提炼搜索词..."):
                    try:
                        if st.session_state.selected_scope == "🌐 对比所有论文":
                            docs = st.session_state.db.similarity_search("Abstract Future Work limitation", k=5)
                        else:
                            docs = st.session_state.db.similarity_search("Abstract Introduction related work", k=4, filter={"source_paper": st.session_state.selected_scope})
                        content_snippet = "\n".join([d.page_content for d in docs])
                        llm = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.5)
                        prompt = f"""
                        任务：你是一个专业的科研助理。根据以下论文片段，识别核心研究问题。
                        目标：生成 1 个能在 ArXiv 获得高质量、高相关性结果的搜索词组。
                        要求：
                        1. 只输出关键词，不要解释。
                        2. 关键词应该是 2-3 个核心概念的组合。
                        片段：
                        {content_snippet[:2000]}
                        """
                        generated_query = llm.invoke(prompt).content.strip().replace('"', '').replace("'", "")
                        st.session_state.suggested_query = generated_query
                        st.success(f"已生成关键词：{generated_query}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"挖掘失败: {e}")

        st.markdown("---")
        st.subheader("📝 笔记导出")
        if st.session_state.chat_history:
            html_content = generate_html_report(st.session_state.chat_history)
            st.download_button("📄 下载笔记 HTML", html_content, "research_notes.html", "text/html")

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
tab_search, tab_chat = st.tabs(["🔍 ArXiv 搜索", "💬 研读空间"])

with tab_search:
    st.subheader("🌍 ArXiv 智能搜索 (Deep Search)")
    col1, col2 = st.columns([4, 1])
    with col1:
        default_query = st.session_state.get("suggested_query", "")
        search_query = st.text_input("输入关键词", value=default_query, placeholder="例如: education robot")
    with col2:
        max_results = st.number_input("数量 (Max 300)", min_value=5, max_value=300, value=20, step=10)
        
    if st.button("🚀 搜索") and search_query:
        with st.spinner(f"正在深度检索 {max_results} 篇论文..."):
            try:
                if " " in search_query and "AND" not in search_query and '"' not in search_query:
                    words = search_query.split()
                    refined_query = " AND ".join([f'(ti:{w} OR abs:{w})' for w in words])
                else:
                    refined_query = search_query

                search = arxiv.Search(
                    query=refined_query, 
                    max_results=max_results, 
                    sort_by=arxiv.SortCriterion.Relevance
                )
                results_list = list(search.results())
                st.session_state.search_results = results_list
                st.success(f"✅ 已针对“{refined_query}”找到 {len(results_list)} 篇论文")
            except Exception as e:
                st.error(f"搜索失败: {e}")
                
    if "search_results" in st.session_state:
        total = len(st.session_state.search_results)
        if total > 0:
            st.caption(f"当前显示 {total} 条高相关结果")
        
        for i, res in enumerate(st.session_state.search_results):
            with st.expander(f"#{i+1} 📄 {res.title} ({res.published.year})"):
                all_authors = ', '.join([a.name for a in res.authors])
                st.markdown(f"**👨‍🏫 作者**: {all_authors}")
                
                clean_summary = res.summary.replace('\n', ' ')
                st.markdown(f"""
                <div class="abstract-box">
                    <b>📝 摘要：</b><br>
                    {clean_summary}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"[🔗 原文链接]({res.entry_id})")
                if st.button(f"⬇️ 下载并研读", key=f"dl_{res.entry_id}_{i}"):
                    if not user_api_key:
                        st.error("请先配置 API Key")
                    else:
                        with st.spinner("下载中..."):
                            try:
                                pdf_path = res.download_pdf(dirpath=tempfile.gettempdir())
                                process_and_add_to_db(pdf_path, res.title, user_api_key)
                                st.success("入库成功！转到“研读空间”即可对话")
                            except Exception as e:
                                st.error(f"下载失败: {e}")

with tab_chat:
    if st.session_state.loaded_files:
        # 使用 st.session_state.selected_scope 替代局部变量，确保全局可用
        st.caption(f"📚 模式：{reading_mode} | 范围：{st.session_state.selected_scope}")

    for msg in st.session_state.chat_history:
        if msg["role"] == "system_notice":
            st.info(msg["content"])
        else:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("输入问题..."):
        if not st.session_state.db:
            st.warning("🧠 请先添加论文")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                try:
                    search_k = 15 if "精读" in reading_mode else 8
                    
                    # 范围过滤逻辑也改为使用 session_state
                    current_scope = st.session_state.get("selected_scope", "🌐 对比所有论文")
                    if current_scope != "🌐 对比所有论文":
                        filter_dict = {"source_paper": current_scope} 
                    else:
                        filter_dict = None

                    docs = st.session_state.db.max_marginal_relevance_search(
                        prompt, 
                        k=search_k, 
                        fetch_k=20,
                        lambda_mult=0.6,
                        filter=filter_dict
                    )

                    if not docs:
                        st.warning("未找到相关内容。")
                        st.stop()

                    context_parts = []
                    for d in docs:
                        source = d.metadata.get('source_paper', '未知')
                        page = d.metadata.get('page', 0) + 1
                        context_parts.append(f"📄【{source} P{page}】:\n{d.page_content}")

                    full_context = "\n\n".join(context_parts)
                    
                    if "精读" in reading_mode:
                        system_prompt = f"""你是一位严谨的科研助手。基于资料回答问题。
资料：{full_context}
问题：{prompt}
要求：
1. 必须使用 $...$ 包裹数学公式。
2. 尽可能引用多个不同片段的信息来回答。
3. 忽略参考文献列表。
"""
                    else:
                        system_prompt = f"""你是一个助手。请简要回答。
资料：{full_context}
问题：{prompt}
要求：公式必须用 $...$ 包裹。
"""
                    llm = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.1)
                    response = llm.invoke(system_prompt)
                    final_content = fix_latex_errors(response.content)

                    st.write(final_content)
                    st.session_state.chat_history.append({"role": "assistant", "content": final_content})

                except Exception as e:
                    st.error(f"生成出错: {e}")
