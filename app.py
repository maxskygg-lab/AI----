import streamlit as st
import sys
import os
import time
import tempfile
import arxiv
import requests
import re
from collections import Counter

# ================= 1. 核心库检查 =================
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

# ================= 2. 增强型页面配置 =================
st.set_page_config(page_title="AI 深度研读助手 (科研版)", layout="wide", page_icon="🎓")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px;}
    .abstract-box {
        background-color: #f8f9fa;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        font-size: 0.95em;
        line-height: 1.7;
        margin-bottom: 12px;
        color: #2c3e50;
    }
    .cite-badge {
        background-color: #ff4b4b;
        color: white;
        padding: 3px 12px;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .topic-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 4px 10px;
        border-radius: 4px;
        margin: 4px;
        font-size: 0.85em;
        border: 1px solid #bbdefb;
    }
</style>
""", unsafe_allow_html=True)
st.title("📖 AI 深度研读助手")

# ================= 3. 全局状态初始化 =================
# 确保所有变量都存在，防止切换 Tab 时报错
state_keys = {
    "chat_history": [],
    "db": None,
    "loaded_files": [],
    "all_chunks": [],
    "suggested_query": "",
    "search_results": [],
    "selected_scope": "🌐 对比所有论文"
}
for key, default in state_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ================= 4. 核心功能函数 =================

def fetch_citations(arxiv_id):
    """接入 Semantic Scholar 数据流"""
    try:
        clean_id = arxiv_id.split('/')[-1].split('v')[0]
        # 增加延迟防止被封，增加 influentialCitationCount (高影响力引用)
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields=citationCount"
        response = requests.get(api_url, timeout=4)
        if response.status_code == 200:
            return response.json().get('citationCount', 0)
    except:
        pass
    return 0

def extract_top_topics(results):
    """模拟谷歌搜索的关键词热度提取"""
    all_text = ""
    for item in results:
        res = item['obj']
        all_text += f" {res.title} {res.summary}"
    
    # 清洗文本：只保留长于 5 的单词
    words = re.findall(r'\b\w{5,}\b', all_text.lower())
    stop_words = {'learning', 'robotics', 'education', 'research', 'paper', 'approach', 'system', 'based', 'using', 'results', 'study', 'provide', 'performance'}
    meaningful_words = [w for w in words if w not in stop_words]
    return Counter(meaningful_words).most_common(10)

def fix_latex_errors(text):
    """保留完整的 LaTeX 修复逻辑"""
    if not text: return text
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    return text

def generate_html_report(chat_history):
    """恢复完整的 HTML 导出逻辑，带 MathJax 支持"""
    html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>AI 研究笔记</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body { font-family: 'Segoe UI', sans-serif; max-width: 850px; margin: 0 auto; padding: 30px; line-height: 1.6; background-color: #fcfcfc; }
        h1 { color: #2e7d32; border-bottom: 2px solid #2e7d32; padding-bottom: 10px; }
        .message { margin-bottom: 25px; padding: 20px; border-radius: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .user { background-color: #e3f2fd; border-left: 6px solid #2196f3; }
        .assistant { background-color: #f1f8e9; border-left: 6px solid #4caf50; }
        .system { background-color: #fff3e0; border-left: 6px solid #ff9800; font-style: italic; }
        .role-label { font-weight: bold; margin-bottom: 8px; display: block; text-transform: uppercase; font-size: 0.8em; }
    </style></head><body><h1>🎓 AI 深度研读笔记</h1>"""
    for msg in chat_history:
        role = msg['role']
        label = "🧑‍💻 我" if role == 'user' else "🤖 AI 研究员" if role == 'assistant' else "🔔 系统通知"
        content = msg['content'].replace('\n', '<br>')
        html += f'<div class="message {role}"><span class="role-label">{label}</span>{content}</div>'
    html += "</body></html>"
    return html

def rebuild_index_from_chunks(api_key):
    """删除文件后重构数据库"""
    if not st.session_state.all_chunks:
        st.session_state.db = None
        return
    embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
    st.session_state.db = FAISS.from_documents(st.session_state.all_chunks, embeddings)

def process_and_add_to_db(file_path, file_name, api_key):
    """保留完整的 PDF 解析逻辑"""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source_paper'] = file_name
        
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
        
        st.session_state.chat_history.append({"role": "system_notice", "content": f"📚 已成功加载《{file_name}》"})
    except Exception as e:
        st.error(f"处理失败: {e}")

# ================= 5. 侧边栏 =================
with st.sidebar:
    st.header("🎛️ 控制台")
    user_api_key = st.text_input("智谱 API Key", type="password")
    st.markdown("---")
    
    if st.session_state.loaded_files:
        st.subheader("🗂️ 文献库管理")
        for file in list(st.session_state.loaded_files):
            col_f1, col_f2 = st.columns([4, 1])
            with col_f1: st.caption(f"📄 {file[:20]}...")
            with col_f2:
                if st.button("🗑️", key=f"del_{file}"):
                    st.session_state.loaded_files.remove(file)
                    st.session_state.all_chunks = [c for c in st.session_state.all_chunks if c.metadata.get('source_paper') != file]
                    if user_api_key: rebuild_index_from_chunks(user_api_key)
                    st.rerun()
        
        if st.button("🗑️ 清空全部", type="primary"):
            st.session_state.db, st.session_state.loaded_files, st.session_state.all_chunks, st.session_state.chat_history = None, [], [], []
            st.rerun()

    st.subheader("⚙️ 模式设置")
    reading_mode = st.radio("阅读模式:", ["🟢 快速回答", "📖 逐段精读 (公式增强)"], index=1)

    if st.session_state.loaded_files:
        st.markdown("---")
        st.session_state.selected_scope = st.selectbox("👁️ 研读范围", ["🌐 对比所有论文"] + st.session_state.loaded_files)
        
        if st.button("📄 导出研读笔记"):
            html_content = generate_html_report(st.session_state.chat_history)
            st.download_button("下载 HTML 笔记", html_content, "research_notes.html", "text/html")

    st.markdown("---")
    uploaded_file = st.file_uploader("本地 PDF 导入", type="pdf")
    if uploaded_file and user_api_key and st.button("执行加载"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            process_and_add_to_db(tmp.name, uploaded_file.name, user_api_key)
            os.remove(tmp.name)
            st.rerun()

# ================= 6. 主界面 =================
tab_search, tab_chat = st.tabs(["🔍 文献调研 (引用与热点)", "💬 智能研读空间"])

with tab_search:
    st.subheader("🌍 跨库学术调研引擎")
    col_q, col_sort, col_n = st.columns([3, 1.2, 0.8])
    with col_q:
        q = st.text_input("关键词 (支持英文)", value=st.session_state.suggested_query, placeholder="例如: robotics education K-12")
    with col_sort:
        sort_rule = st.selectbox("排序规则", ["🔥 相关性", "📅 时间最新", "📈 引用量优先"])
    with col_n:
        n = st.number_input("获取篇数", 5, 50, 15)

    if st.button("🚀 启动深度检索") and q:
        with st.spinner("正在检索并分析学术元数据..."):
            try:
                arxiv_sort = arxiv.SortCriterion.Relevance
                if "时间" in sort_rule: arxiv_sort = arxiv.SortCriterion.SubmittedDate
                
                # 自动布尔优化
                refined_q = q if ("AND" in q or '"' in q) else " AND ".join([f"(ti:{w} OR abs:{w})" for w in q.split()])
                
                search = arxiv.Search(query=refined_q, max_results=n, sort_by=arxiv_sort)
                raw_results = list(search.results())
                
                results_with_meta = []
                progress = st.progress(0)
                for idx, res in enumerate(raw_results):
                    cites = fetch_citations(res.entry_id)
                    results_with_meta.append({'obj': res, 'citations': cites})
                    progress.progress((idx + 1) / len(raw_results))
                    time.sleep(0.1) # 安全延迟
                
                if "引用量" in sort_rule:
                    results_with_meta.sort(key=lambda x: x['citations'], reverse=True)
                
                st.session_state.search_results = results_with_meta
            except Exception as e:
                st.error(f"检索中断: {e}")

    if st.session_state.search_results:
        # 谷歌式热点词提取
        topics = extract_top_topics(st.session_state.search_results)
        st.write("📊 **领域热点图谱** (辅助识别研究方向):")
        topic_cols = st.columns(len(topics))
        for i, (word, count) in enumerate(topics):
            topic_cols[i].markdown(f"<div class='topic-tag'>{word} ({count})</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        for i, item in enumerate(st.session_state.search_results):
            res, cites = item['obj'], item['citations']
            # 强化精准匹配视觉
            is_high = all(w.lower() in res.title.lower() for w in q.split()[:2])
            
            with st.expander(f"{'🎯' if is_high else '📄'} #{i+1} {res.title} ({res.published.year})"):
                st.markdown(f"**🔥 引用数:** <span class='cite-badge'>{cites}</span> | **主作者:** {res.authors[0].name}", unsafe_allow_html=True)
                # 清洗摘要换行符
                clean_abs = res.summary.replace('\n', ' ')
                st.markdown(f"<div class='abstract-box'><b>摘要预览:</b><br>{clean_abs}</div>", unsafe_allow_html=True)
                
                c1, c2 = st.columns([1, 1])
                with c1: st.markdown(f"[🔗 ArXiv 原文]({res.entry_id})")
                with c2:
                    if st.button(f"⬇️ 加入研读库", key=f"dl_main_{i}"):
                        if user_api_key:
                            with st.spinner("同步至向量库..."):
                                pdf_path = res.download_pdf(dirpath=tempfile.gettempdir())
                                process_and_add_to_db(pdf_path, res.title, user_api_key)
                                st.success("已就绪，转到对话 Tab 即可提问")
                        else: st.error("请在侧边栏填写 API Key")

with tab_chat:
    if st.session_state.loaded_files:
        st.caption(f"📚 研读模式: {reading_mode} | 当前论文: {st.session_state.selected_scope}")

    for msg in st.session_state.chat_history:
        if msg["role"] == "system_notice": st.info(msg["content"])
        else:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("基于已选文献提问..."):
        if not st.session_state.db: st.warning("请先加载至少一篇论文")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)
            with st.chat_message("assistant"):
                try:
                    scope = st.session_state.selected_scope
                    f_dict = {"source_paper": scope} if scope != "🌐 对比所有论文" else None
                    # MMR 搜索保证检索内容的多样性
                    docs = st.session_state.db.max_marginal_relevance_search(prompt, k=8, fetch_k=20, lambda_mult=0.7, filter=f_dict)
                    
                    context = "\n\n".join([f"📄【{d.metadata.get('source_paper','?')} P{d.metadata.get('page',0)+1}】:\n{d.page_content}" for d in docs])
                    
                    llm = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.1)
                    res = llm.invoke(f"你是一位资深科研专家。请基于以下片段回答：\n\n{context}\n\n问题：{prompt}\n要求：严谨准确，公式务必使用 $ 包裹。")
                    final_ans = fix_latex_errors(res.content)
                    st.write(final_ans)
                    st.session_state.chat_history.append({"role": "assistant", "content": final_ans})
                except Exception as e: st.error(f"生成失败: {e}")
