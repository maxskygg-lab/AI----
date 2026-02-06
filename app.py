import streamlit as st
import sys
import os
import time
import tempfile
import arxiv
import requests
import re
from collections import Counter

# ================= 1. 环境与依赖检查 =================
try:
    import zhipuai
    import langchain_community
    import fitz  # pymupdf
except ImportError as e:
    st.error(f"🚑 环境缺失库 -> {e.name}")
    st.stop()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= 2. 页面美化与样式 =================
st.set_page_config(page_title="AI 深度研读助手 (全功能终极版)", layout="wide", page_icon="🎓")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px;}
    .abstract-box {
        background-color: #f8f9fa; padding: 18px; border-radius: 10px;
        border-left: 5px solid #4CAF50; font-size: 0.95em; line-height: 1.7;
        margin-bottom: 12px; color: #2c3e50;
    }
    .cite-badge {
        background-color: #ff4b4b; color: white; padding: 3px 12px;
        border-radius: 15px; font-size: 0.85em; font-weight: bold;
    }
    .topic-tag {
        display: inline-block; background-color: #e3f2fd; color: #1976d2;
        padding: 4px 10px; border-radius: 4px; margin: 4px;
        font-size: 0.85em; border: 1px solid #bbdefb;
    }
</style>
""", unsafe_allow_html=True)
st.title("📖 AI 深度研读助手 (全功能终极版)")

# ================= 3. 严格的状态初始化 =================
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "db" not in st.session_state: st.session_state.db = None
if "loaded_files" not in st.session_state: st.session_state.loaded_files = []
if "all_chunks" not in st.session_state: st.session_state.all_chunks = []
if "suggested_query" not in st.session_state: st.session_state.suggested_query = ""
if "search_results" not in st.session_state: st.session_state.search_results = []
if "selected_scope" not in st.session_state: st.session_state.selected_scope = "🌐 对比所有论文"

# ================= 4. 核心功能函数集 =================

def fetch_citations(arxiv_id):
    """接入 Semantic Scholar 引用流 (带异常处理)"""
    try:
        clean_id = arxiv_id.split('/')[-1].split('v')[0]
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields=citationCount"
        response = requests.get(api_url, timeout=4)
        if response.status_code == 200:
            return response.json().get('citationCount', 0)
    except: pass
    return 0

def extract_top_topics(results):
    """Google 式热点词提取逻辑"""
    all_text = " ".join([f"{r['obj'].title} {r['obj'].summary}" for r in results])
    words = re.findall(r'\b\w{5,}\b', all_text.lower())
    stop_words = {'learning', 'robotics', 'education', 'research', 'paper', 'approach', 'system', 'based', 'using', 'results', 'study', 'performance', 'model'}
    meaningful_words = [w for w in words if w not in stop_words]
    return Counter(meaningful_words).most_common(10)

def fix_latex_errors(text):
    """全量 Latex 修复逻辑，防止渲染失败"""
    if not text: return text
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    return text

def generate_html_report(chat_history):
    """高阶 HTML 导出，保留 MathJax 与专业排版"""
    html = """<!DOCTYPE html><html><head><meta charset="UTF-8">
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px; line-height: 1.6; color: #333; }
        h1 { border-bottom: 3px solid #4CAF50; padding-bottom: 10px; color: #2e7d32; }
        .message { margin-bottom: 25px; padding: 20px; border-radius: 12px; }
        .user { background-color: #e3f2fd; border-left: 6px solid #2196F3; }
        .assistant { background-color: #f1f8e9; border-left: 6px solid #4CAF50; }
        .system { background-color: #fff3e0; border-left: 6px solid #ff9800; font-style: italic; }
        .role { font-weight: bold; display: block; margin-bottom: 10px; text-transform: uppercase; font-size: 0.8em; color: #666; }
    </style></head><body><h1>🎓 AI 深度研读笔记</h1><p>导出日期：""" + time.strftime('%Y-%m-%d %H:%M') + """</p>"""
    for msg in chat_history:
        role_label = "🧑‍💻 Me" if msg['role'] == 'user' else "🤖 AI Researcher" if msg['role'] == 'assistant' else "🔔 System"
        content_html = msg['content'].replace('\n', '<br>')
        html += f'<div class="message {msg["role"]}"><span class="role">{role_label}</span>{content_html}</div>'
    html += "</body></html>"
    return html

def rebuild_index_from_chunks(api_key):
    """删除文件后物理重构 FAISS"""
    if not st.session_state.all_chunks:
        st.session_state.db = None
        return
    embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
    st.session_state.db = FAISS.from_documents(st.session_state.all_chunks, embeddings)

def process_and_add_to_db(file_path, file_name, api_key):
    """全信息解析逻辑"""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs: doc.metadata['source_paper'] = file_name
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200, separators=["\n\n", "\n", "。", ".", " ", ""])
        chunks = splitter.split_documents(docs)
        valid_chunks = [c for c in chunks if len(c.page_content.strip()) > 20]
        st.session_state.all_chunks.extend(valid_chunks)
        embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
        if st.session_state.db is None:
            st.session_state.db = FAISS.from_documents(valid_chunks, embeddings)
        else:
            st.session_state.db.add_documents(valid_chunks)
        if file_name not in st.session_state.loaded_files:
            st.session_state.loaded_files.append(file_name)
        st.session_state.chat_history.append({"role": "system_notice", "content": f"📚 **通知**：已成功解析并加载《{file_name}》"})
    except Exception as e: st.error(f"解析失败: {e}")

# ================= 5. 侧边栏：多功能控制台 =================
with st.sidebar:
    st.header("🎛️ 科研控制面板")
    user_api_key = st.text_input("智谱 API Key", type="password")
    st.markdown("---")
    
    if st.session_state.loaded_files:
        st.subheader("🗂️ 文献库")
        for file in list(st.session_state.loaded_files):
            c1, c2 = st.columns([4, 1])
            with c1: st.caption(f"📄 {file[:22]}...")
            with c2:
                if st.button("🗑️", key=f"del_{file}"):
                    st.session_state.loaded_files.remove(file)
                    st.session_state.all_chunks = [c for c in st.session_state.all_chunks if c.metadata.get('source_paper') != file]
                    if user_api_key: rebuild_index_from_chunks(user_api_key)
                    st.rerun()

        # --- 补全功能 1: 一键生成综述对比表 ---
        if st.button("🪄 一键生成综述对比表", type="primary"):
            if user_api_key and st.session_state.db:
                with st.spinner("AI 正在扫描全库进行横向对比..."):
                    llm = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.1)
                    aggregated_ctx = ""
                    for name in st.session_state.loaded_files:
                        sub = st.session_state.db.similarity_search("Abstract method contribution", k=3, filter={"source_paper": name})
                        aggregated_ctx += f"\n[Paper: {name}]\n" + "\n".join([d.page_content for d in sub])
                    res = llm.invoke(f"分析以下论文片段，生成 Markdown 对比表格(列：论文名|核心创新|方法论|主要结论)：\n{aggregated_ctx}")
                    st.session_state.chat_history.append({"role": "assistant", "content": res.content})
                    st.rerun()

        st.markdown("---")
        st.session_state.selected_scope = st.selectbox("👁️ 研读专注范围", ["🌐 对比所有论文"] + st.session_state.loaded_files)

        # --- 补全功能 2: 深度挖掘新关键词 ---
        if st.button(f"🔍 挖掘【{st.session_state.selected_scope[:5]}...】关联论文"):
            if user_api_key and st.session_state.db:
                with st.spinner("挖掘文本深层特征中..."):
                    scope = st.session_state.selected_scope
                    f_dict = {"source_paper": scope} if scope != "🌐 对比所有论文" else None
                    docs = st.session_state.db.similarity_search("Introduction future work", k=4, filter=f_dict)
                    llm = ChatZhipuAI(model="glm-4", api_key=user_api_key)
                    prompt = f"基于以下论文片段，提炼 2 个最能代表其研究深度的英文学术搜索词组(只输出词组)：\n" + "\n".join([d.page_content for d in docs])
                    st.session_state.suggested_query = llm.invoke(prompt).content.strip()
                    st.success(f"新搜索词已生成！")
                    st.rerun()

    st.markdown("---")
    st.subheader("📝 导出与导入")
    if st.session_state.chat_history:
        st.download_button("💾 导出研读笔记 HTML", generate_html_report(st.session_state.chat_history), "research_notes.html", "text/html")
    
    uploaded_file = st.file_uploader("导入本地 PDF", type="pdf")
    if uploaded_file and user_api_key and st.button("确认识别并分析"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            process_and_add_to_db(tmp.name, uploaded_file.name, user_api_key)
            os.remove(tmp.name)
            st.rerun()

# ================= 6. 主界面：调研与对话 =================
tab_search, tab_chat = st.tabs(["🔍 文献调研 (Google 逻辑增强)", "💬 深度研读空间"])

with tab_search:
    st.subheader("🌍 学术大数据挖掘引擎")
    col_q, col_s, col_n = st.columns([3, 1.2, 0.8])
    with col_q:
        search_q = st.text_input("关键词/挖掘词", value=st.session_state.suggested_query, placeholder="如: education robot K-12")
    with col_s:
        sort_mode = st.selectbox("Google 排序权重", ["🔥 相关性优先", "📅 最新发表", "📈 引用量之王"])
    with col_n:
        max_n = st.number_input("获取数量", 5, 50, 15)

    if st.button("🚀 执行多维检索") and search_q:
        with st.spinner("同步 ArXiv 与 Semantic Scholar 数据中..."):
            try:
                arxiv_sort = arxiv.SortCriterion.Relevance
                if "最新" in sort_mode: arxiv_sort = arxiv.SortCriterion.SubmittedDate
                # 自动布尔优化
                final_q = search_q if ("AND" in search_q) else " AND ".join([f"(ti:{w} OR abs:{w})" for w in search_q.split()])
                search = arxiv.Search(query=final_q, max_results=max_n, sort_by=arxiv_sort)
                results_with_meta = []
                for res in list(search.results()):
                    results_with_meta.append({'obj': res, 'citations': fetch_citations(res.entry_id)})
                    time.sleep(0.1) # 频率保护
                if "引用量" in sort_mode: results_with_meta.sort(key=lambda x: x['citations'], reverse=True)
                st.session_state.search_results = results_with_meta
            except Exception as e: st.error(f"检索失败: {e}")

    if st.session_state.search_results:
        # --- 补全功能 3: 领域热点关键词分布 ---
        topics = extract_top_topics(st.session_state.search_results)
        st.write("📊 **调研热点图谱分析** (协助快速扫盲):")
        t_cols = st.columns(len(topics))
        for i, (word, count) in enumerate(topics):
            t_cols[i].markdown(f"<div class='topic-tag'>{word} ({count})</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        for i, item in enumerate(st.session_state.search_results):
            res, cites = item['obj'], item['citations']
            is_match = all(w.lower() in res.title.lower() for w in search_q.split()[:1])
            with st.expander(f"{'🎯' if is_match else '📄'} #{i+1} [{cites} Cites] {res.title} ({res.published.year})"):
                st.markdown(f"<div class='abstract-box'><b>Abstract:</b><br>{res.summary.replace(chr(10), ' ')}</div>", unsafe_allow_html=True)
                col1, col2 = st.columns([1, 1])
                with col1: st.markdown(f"[🔗 ArXiv 链接]({res.entry_id})")
                with col2:
                    if st.button(f"⬇️ 加入我的研读库", key=f"dl_final_{i}"):
                        if user_api_key:
                            with st.spinner("正在同步至向量索引..."):
                                path = res.download_pdf(dirpath=tempfile.gettempdir())
                                process_and_add_to_db(path, res.title, user_api_key)
                                st.success("已完成！转到‘研读空间’对话")
                        else: st.error("请先在侧边栏填写 API Key")

with tab_chat:
    if st.session_state.loaded_files:
        st.caption(f"📚 模式: {reading_mode} | 范围: {st.session_state.selected_scope}")
        for msg in st.session_state.chat_history:
            if msg["role"] == "system_notice": st.info(msg["content"])
            else:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("输入科研问题..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)
            with st.chat_message("assistant"):
                try:
                    scope = st.session_state.selected_scope
                    f_dict = {"source_paper": scope} if scope != "🌐 对比所有论文" else None
                    # 恢复多样性 MMR 检索逻辑
                    docs = st.session_state.db.max_marginal_relevance_search(prompt, k=8, fetch_k=20, lambda_mult=0.7, filter=f_dict)
                    context = "\n\n".join([f"📄【{d.metadata.get('source_paper','?')} P{d.metadata.get('page',0)+1}】:\n{d.page_content}" for d in docs])
                    llm = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.1)
                    res = llm.invoke(f"你是科研导师。基于资料回答：\n{context}\n\n问题：{prompt}\n要求：数学公式用 $ 包裹。")
                    ans = fix_latex_errors(res.content)
                    st.write(ans)
                    st.session_state.chat_history.append({"role": "assistant", "content": ans})
                except Exception as e: st.error(f"生成失败: {e}")
    else:
        st.info("💡 库中暂无论文。请在左侧上传或通过调研引擎‘下载’论文。")
