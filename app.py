import streamlit as st
import sys
import os
import time
import tempfile
import arxiv

# ================= 🏥 环境听诊器 =================
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
st.set_page_config(page_title="AI 深度研读助手 (高精版)", layout="wide", page_icon="🎓")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px;}
    .reportview-container { margin-top: -2em; }
</style>
""", unsafe_allow_html=True)
st.title("📖 AI 深度研读助手 (高精度内核版)")

# ================= 3. 状态初始化 =================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db" not in st.session_state:
    st.session_state.db = None
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []
if "suggested_query" not in st.session_state:
    st.session_state.suggested_query = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = []

# ================= 4. 核心逻辑函数 =================

def fix_latex_errors(text):
    if not text: return text
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    return text

def process_and_add_to_db(file_path, file_name, api_key):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source_paper'] = file_name
        
        # ⚡️ 核心升级 1：针对学术论文的更细致切分策略
        # 减小 chunk_size 以聚焦具体定义，增加 overlap 保证上下文连续
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,       # 缩小块大小，提高检索定位精度
            chunk_overlap=200,    # 增加重叠，防止关键句子被切断
            separators=["\n\n", "\n", "。", ".", " ", ""] # 优先按段落切分
        )
        chunks = splitter.split_documents(docs)
        valid_chunks = [c for c in chunks if len(c.page_content.strip()) > 20]
        
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
    st.subheader("⚙️ 研读模式")
    reading_mode = st.radio("选择模式:", ["🟢 快速问答", "📖 逐段精读 (公式修复版)"], index=1)

    st.markdown("---")

    if st.session_state.loaded_files:
        st.success(f"已加载 {len(st.session_state.loaded_files)} 篇论文")
        
        # 1. 综述生成
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
                            # 增大上下文获取量，保证总结更准
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

        # 2. 挖掘关联论文
        scope_options = ["🌐 对比所有论文"] + st.session_state.loaded_files
        selected_scope = st.selectbox("👁️ 专注范围", scope_options)
        
        if st.button(f"🔍 基于【{selected_scope[:5]}...】挖掘新论文"):
            if not user_api_key:
                st.error("请填入 API Key")
            else:
                with st.spinner("🤖 AI 正在深度分析文本，提炼搜索词..."):
                    try:
                        if selected_scope == "🌐 对比所有论文":
                            docs = st.session_state.db.similarity_search("Abstract Future Work limitation", k=5)
                        else:
                            docs = st.session_state.db.similarity_search("Abstract Introduction related work", k=4, filter={"source_paper": selected_scope})
                        
                        content_snippet = "\n".join([d.page_content for d in docs])
                        
                        llm = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.5)
                        # ⚡️ 核心升级 2：更专业的搜索词 Prompt
                        prompt = f"""
                        任务：你是一个专业的科研助理。根据以下论文片段，识别核心研究问题。
                        目标：生成 1 个能在 ArXiv 获得高质量结果的英文搜索 Query。
                        要求：
                        1. 尽量使用组合关键词（如 "Large Language Model" AND "Reasoning"）。
                        2. 排除过于宽泛的词（如 "AI"）。
                        3. 只输出 Query 字符串本身，不要包含其他解释。
                        
                        片段：
                        {content_snippet[:2000]}
                        """
                        generated_query = llm.invoke(prompt).content.strip().replace('"', '').replace("'", "")
                        
                        st.session_state.suggested_query = generated_query
                        
                        search = arxiv.Search(query=generated_query, max_results=5, sort_by=arxiv.SortCriterion.Relevance)
                        st.session_state.search_results = list(search.results())
                        
                        st.success(f"高精搜索词：{generated_query}")
                        st.info("👈 请点击主界面的 '🔍 ArXiv 搜索' 标签页查看结果")
                    except Exception as e:
                        st.error(f"挖掘失败: {e}")

        if st.button("🗑️ 清空知识库"):
            st.session_state.db = None
            st.session_state.loaded_files = []
            st.session_state.chat_history = []
            st.rerun()

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
    st.subheader("🌍 ArXiv 智能搜索")
    col1, col2 = st.columns([4, 1])
    with col1:
        default_query = st.session_state.get("suggested_query", "")
        search_query = st.text_input("输入关键词", value=default_query, placeholder="支持布尔搜索: LLM AND Agent")
    with col2:
        max_results = st.number_input("数量", min_value=5, max_value=50, value=10, step=5)
        
    if st.button("🚀 搜索") and search_query:
        with st.spinner(f"正在检索 ArXiv (Top {max_results})..."):
            try:
                search = arxiv.Search(query=search_query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
                st.session_state.search_results = list(search.results())
                st.success(f"找到 {len(st.session_state.search_results)} 篇相关论文")
            except Exception as e:
                st.error(f"搜索出错: {e}")
                
    if "search_results" in st.session_state:
        for res in st.session_state.search_results:
            with st.expander(f"📄 {res.title} ({res.published.year})"):
                st.write(f"**作者**: {', '.join([a.name for a in res.authors[:3]])}...")
                st.write(f"**摘要**: {res.summary[:300]}...")
                st.markdown(f"[原文链接]({res.entry_id})")
                if st.button(f"⬇️ 下载并研读", key=res.entry_id):
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
        st.caption(f"📚 模式：{reading_mode} | 范围：{selected_scope}")

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
                    try:
                        if selected_scope != "🌐 对比所有论文":
                            filter_dict = {"source_paper": selected_scope} 
                        else:
                            filter_dict = None
                    except:
                        filter_dict = None

                    # ⚡️ 核心升级 3：使用 MMR (Maximal Marginal Relevance) 算法
                    # 作用：不仅要像（Relevance），还要多样（Marginal）。
                    # fetch_k=20: 先找 20 个最像的
                    # lambda_mult=0.6: 0.6的权重给相关性，0.4给多样性。防止 AI 总是引用同一段话。
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
2. 尽可能引用多个不同片段的信息来回答，不要只盯着一段。
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
