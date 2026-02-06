import streamlit as st
import sys
import os
import time
import tempfile
import arxiv
import requests
import re
from collections import Counter

# ================= 1. 环境检查 =================
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

# ================= 2. 页面配置 =================
st.set_page_config(page_title="AI 深度研读助手 (Google-Style 调研版)", layout="wide", page_icon="🎓")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 8px;}
    .abstract-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4CAF50;
        font-size: 0.95em;
        line-height: 1.6;
        margin-bottom: 10px;
    }
    .cite-badge {
        background-color: #e74c3c;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 10px;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)
st.title("📖 AI 深度研读助手 (专业科研引擎)")

# ================= 3. 状态初始化 =================
for key in ["chat_history", "loaded_files", "all_chunks", "suggested_query", "search_results"]:
    if key not in st.session_state:
        st.session_state[key] = []
if "db" not in st.session_state:
    st.session_state.db = None
if "selected_scope" not in st.session_state:
    st.session_state.selected_scope = "🌐 对比所有论文"

# ================= 4. 核心逻辑工具 =================

def fetch_citations(arxiv_id):
    """接入 Semantic Scholar 引用关系数据"""
    try:
        clean_id = arxiv_id.split('/')[-1].split('v')[0]
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields=citationCount"
        response = requests.get(api_url, timeout=3)
        if response.status_code == 200:
            return response.json().get('citationCount', 0)
    except:
        pass
    return 0

def extract_top_topics(results):
    """学习谷歌的关键词提取，用于辅助判断调研方向"""
    all_text = ""
    for item in results:
        res = item['obj']
        all_text += f" {res.title} {res.summary}"
    
    words = re.findall(r'\b\w{5,}\b', all_text.lower())
    stop_words = {'learning', 'robotics', 'education', 'research', 'paper', 'approach', 'system', 'based', 'using', 'results'}
    meaningful_words = [w for w in words if w not in stop_words]
    return Counter(meaningful_words).most_common(8)

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
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
        if st.session_state.db is None:
            st.session_state.db = FAISS.from_documents(chunks, embeddings)
        else:
            st.session_state.db.add_documents(chunks)
        if file_name not in st.session_state.loaded_files:
            st.session_state.loaded_files.append(file_name)
        st.session_state.chat_history.append({"role": "system_notice", "content": f"📚 已加载《{file_name}》"})
    except Exception as e:
        st.error(f"解析失败: {e}")

# ================= 5. 侧边栏 =================
with st.sidebar:
    st.header("⚙️ 科研控制台")
    user_api_key = st.text_input("API Key (智谱)", type="password")
    
    if st.session_state.loaded_files:
        st.subheader("🗂️ 本地文献库")
        for file in list(st.session_state.loaded_files):
            col_f1, col_f2 = st.columns([4, 1])
            with col_f1: st.caption(f"📄 {file[:20]}...")
            with col_f2:
                if st.button("🗑️", key=f"del_{file}"):
                    st.session_state.loaded_files.remove(file)
                    st.rerun()
        
        st.session_state.selected_scope = st.selectbox("👁️ 专注范围", ["🌐 对比所有论文"] + st.session_state.loaded_files)

    st.markdown("---")
    st.subheader("📥 导入文献")
    uploaded_file = st.file_uploader("上传 PDF", type="pdf")
    if uploaded_file and user_api_key and st.button("开始分析"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            process_and_add_to_db(tmp.name, uploaded_file.name, user_api_key)
            os.remove(tmp.name)
            st.rerun()

# ================= 6. 主界面 =================
tab_search, tab_chat = st.tabs(["🔍 文献调研引擎", "💬 交互研读空间"])

with tab_search:
    st.subheader("🌍 ArXiv + Semantic Scholar 联合搜索")
    col_q, col_sort, col_n = st.columns([3, 1.2, 0.8])
    with col_q:
        q = st.text_input("关键词", value=st.session_state.suggested_query, placeholder="如: education robot review")
    with col_sort:
        sort_rule = st.selectbox("排序规则", ["🔥 相关性", "📅 时间", "📈 引用量"])
    with col_n:
        n = st.number_input("数量", 5, 50, 15)

    if st.button("🔍 深度检索") and q:
        with st.spinner("谷歌式多维检索中..."):
            try:
                # 1. 执行 ArXiv 检索
                arxiv_sort = arxiv.SortCriterion.Relevance
                if "时间" in sort_rule: arxiv_sort = arxiv.SortCriterion.SubmittedDate
                
                # 布尔逻辑自动增强
                refined_q = q
                if " " in q and "AND" not in q:
                    refined_q = " AND ".join([f"(ti:{w} OR abs:{w})" for w in q.split()])
                
                search = arxiv.Search(query=refined_q, max_results=n, sort_by=arxiv_sort)
                raw_results = list(search.results())
                
                # 2. 补全引用数据
                results_with_meta = []
                for res in raw_results:
                    cites = fetch_citations(res.entry_id)
                    results_with_meta.append({'obj': res, 'citations': cites})
                
                # 3. 引用排序
                if "引用量" in sort_rule:
                    results_with_meta.sort(key=lambda x: x['citations'], reverse=True)
                
                st.session_state.search_results = results_with_meta
            except Exception as e:
                st.error(f"检索出错: {e}")

    # --- 渲染调研指标 (Google Knowledge Graph 逻辑) ---
    if st.session_state.search_results:
        topics = extract_top_topics(st.session_state.search_results)
        st.markdown("---")
        st.write("📊 **当前搜索结果领域热点统计** (有助于识别研究偏向):")
        cols = st.columns(8)
        for i, (word, count) in enumerate(topics):
            cols[i].markdown(f"<div class='metric-card'><b>{word}</b><br>{count}次</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        # --- 渲染列表 ---
        for i, item in enumerate(st.session_state.search_results):
            res = item['obj']
            cites = item['citations']
            
            # 判断是否高度相关（标题匹配）
            title_match = any(word in res.title.lower() for word in q.lower().split())
            
            with st.expander(f"{'🎯' if title_match else '📄'} #{i+1} {res.title} ({res.published.year})"):
                st.markdown(f"**🔥 引用量:** <span class='cite-badge'>{cites}</span> | **作者:** {res.authors[0].name} 等", unsafe_allow_html=True)
                st.markdown(f"<div class='abstract-box'>{res.summary.replace(chr(10), ' ')}</div>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])
                with col1: st.markdown(f"[🔗 原文地址]({res.entry_id})")
                with col2:
                    if st.button(f"⬇️ 研读此篇", key=f"dl_{i}"):
                        if user_api_key:
                            with st.spinner("入库中..."):
                                path = res.download_pdf(dirpath=tempfile.gettempdir())
                                process_and_add_to_db(path, res.title, user_api_key)
                                st.success("已加入研读空间！")
                        else: st.error("请配置 API Key")

with tab_chat:
    if st.session_state.loaded_files:
        st.caption(f"📚 专注论文: {st.session_state.selected_scope}")

    for msg in st.session_state.chat_history:
        if msg["role"] == "system_notice": st.info(msg["content"])
        else:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("针对已选论文提问..."):
        if not st.session_state.db: st.warning("请先在搜索结果中点击‘研读此篇’或上传 PDF")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)
            with st.chat_message("assistant"):
                try:
                    scope = st.session_state.selected_scope
                    f_dict = {"source_paper": scope} if scope != "🌐 对比所有论文" else None
                    docs = st.session_state.db.similarity_search(prompt, k=8, filter=f_dict)
                    context = "\n\n".join([f"[{d.metadata.get('source_paper','?')}] {d.page_content}" for d in docs])
                    
                    llm = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.1)
                    res = llm.invoke(f"背景资料：\n{context}\n\n问题：{prompt}\n要求：学术严谨，公式用 $ 包裹。")
                    ans = fix_latex_errors(res.content)
                    st.write(ans)
                    st.session_state.chat_history.append({"role": "assistant", "content": ans})
                except Exception as e: st.error(f"对话异常: {e}")
