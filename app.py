import streamlit as st
import sys
import os
import time
import tempfile
import arxiv
import requests
import re
from collections import Counter

# ================= 1. 环境自检与核心导入 =================
try:
    import zhipuai
    import langchain_community
    import fitz  # pymupdf
except ImportError as e:
    st.error(f"🚑 环境缺失核心库 -> {e.name}。请执行: pip install zhipuai langchain_community pymupdf requests arxiv")
    st.stop()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= 2. 页面配置与谷歌式专业 CSS (全量恢复) =================
st.set_page_config(page_title="AI 深度研读助手 (全功能全量版)", layout="wide", page_icon="🎓")
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; transition: 0.3s; }
    .search-card {
        background-color: white; padding: 22px; border-radius: 10px;
        margin-bottom: 18px; border: 1px solid #dfe1e5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .search-card:hover { box-shadow: 0 4px 12px rgba(32,33,36,0.18); border-color: rgba(223,225,229,0); }
    .paper-title { color: #1a0dab; font-size: 1.25em; text-decoration: none; font-weight: 500; display: block; margin-bottom: 4px; }
    .paper-url { color: #006621; font-size: 0.88em; margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .snippet { color: #4d5156; font-size: 0.92em; line-height: 1.6; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
    .cite-count { color: #70757a; font-size: 0.85em; font-weight: bold; background: #f8f9fa; padding: 3px 10px; border-radius: 5px; border: 1px solid #f1f3f4; }
    .abstract-box {
        background-color: #f8f9fa; padding: 20px; border-radius: 12px;
        border-left: 6px solid #28a745; font-size: 0.98em; line-height: 1.8;
        margin-bottom: 15px; color: #3c4043;
    }
    .topic-tag { 
        display: inline-block; background-color: #f1f3f4; color: #3c4043; 
        padding: 5px 14px; border-radius: 20px; margin: 5px; font-size: 0.88em; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)
st.title("📖 AI 深度研读助手 (Google Logic 全功能版)")

# ================= 3. 全局状态严格初始化 =================
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

# ================= 4. 核心功能函数集 (全量逻辑恢复) =================

def clean_query_for_arxiv(raw_query):
    """【谷歌逻辑 1】将长词组精准拆解为布尔检索式，解决搜索不出论文的问题"""
    words = re.sub(r'[^\w\s]', '', raw_query).split()
    stops = {'the', 'a', 'of', 'and', 'in', 'on', 'with', 'for', 'research', 'paper', 'study', 'impact'}
    important_words = [w for w in words if w.lower() not in stops and len(w) > 2]
    if not important_words: return raw_query
    # 优先搜索标题，摘要兜底，取前 4 个核心词组
    query_parts = [f"(ti:{w} OR abs:{w})" for w in important_words[:4]]
    return " AND ".join(query_parts)

def fetch_citations(arxiv_id):
    """从 Semantic Scholar 获取实时引用量数据"""
    try:
        clean_id = arxiv_id.split('/')[-1].split('v')[0]
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields=citationCount"
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            return response.json().get('citationCount', 0)
    except Exception: pass
    return 0

def extract_top_topics(results):
    """学术热点词分析逻辑"""
    all_text = " ".join([f"{r['obj'].title} {r['obj'].summary}" for r in results])
    words = re.findall(r'\b\w{5,}\b', all_text.lower())
    stop_words = {'learning', 'robotics', 'education', 'research', 'paper', 'approach', 'system', 'based', 'using', 'results', 'model'}
    meaningful = [w for w in words if w not in stop_words]
    return Counter(meaningful).most_common(10)

def fix_latex_errors(text):
    """LaTeX 公式深度修复逻辑"""
    if not text: return text
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    return text

def generate_html_report(chat_history):
    """【恢复全量逻辑】导出带样式与 MathJax 的 HTML 笔记"""
    html = """<!DOCTYPE html><html><head><meta charset="UTF-8">
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; max-width: 900px; margin: 0 auto; padding: 40px; line-height: 1.7; color: #333; background-color: #fdfdfd; }
        h1 { color: #1b5e20; border-bottom: 3px solid #4caf50; padding-bottom: 15px; }
        .message { margin-bottom: 30px; padding: 25px; border-radius: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
        .user { background-color: #e3f2fd; border-left: 8px solid #1976d2; }
        .assistant { background-color: #f1f8e9; border-left: 8px solid #43a047; }
        .system { background-color: #fff3e0; border-left: 8px solid #fb8c00; font-style: italic; color: #666; }
        .role { font-weight: bold; display: block; margin-bottom: 12px; text-transform: uppercase; font-size: 0.85em; color: #555; }
    </style></head><body><h1>🎓 AI 深度研读笔记</h1>"""
    for msg in chat_history:
        role_label = "🧑‍💻 USER" if msg['role'] == 'user' else "🤖 AI RESEARCHER" if msg['role'] == 'assistant' else "🔔 SYSTEM"
        html += f'<div class="message {msg["role"]}"><span class="role">{role_label}</span>{msg["content"].replace(chr(10), "<br>")}</div>'
    html += "</body></html>"
    return html

def rebuild_index_from_chunks(api_key):
    """物理重构向量索引 (带 Batch 保护)"""
    if not st.session_state.all_chunks:
        st.session_state.db = None
        return
    embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
    chunks = st.session_state.all_chunks
    batch_size = 30
    st.session_state.db = FAISS.from_documents(chunks[:batch_size], embeddings)
    for i in range(batch_size, len(chunks), batch_size):
        st.session_state.db.add_documents(chunks[i:i+batch_size])
        time.sleep(0.1)

def process_and_add_to_db(file_path, file_name, api_key):
    """【彻底修复 1214 错误】显式循环分批处理逻辑，不再简写"""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs: doc.metadata['source_paper'] = file_name
        splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=200)
        new_chunks = splitter.split_documents(docs)
        valid_new = [c for c in new_chunks if len(c.page_content.strip()) > 30]
        
        embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
        batch_size = 30
        total = len(valid_new)
        
        with st.spinner(f"正在分批向量化《{file_name}》，绕过接口限制..."):
            if st.session_state.db is None:
                st.session_state.db = FAISS.from_documents(valid_new[:batch_size], embeddings)
                start_idx = batch_size
            else: start_idx = 0
            
            for i in range(start_idx, total, batch_size):
                st.session_state.db.add_documents(valid_new[i : i + batch_size])
                time.sleep(0.2)
        
        st.session_state.all_chunks.extend(valid_new)
        if file_name not in st.session_state.loaded_files: st.session_state.loaded_files.append(file_name)
        st.session_state.chat_history.append({"role": "system_notice", "content": f"📚 已成功入库: {file_name}"})
    except Exception as e: st.error(f"解析失败: {e}")

# ================= 5. 侧边栏：控制面板 (全量功能复原) =================
with st.sidebar:
    st.header("🎛️ 科研控制台")
    api_key_input = st.text_input("智谱 API Key", type="password")
    st.markdown("---")
    
    if st.session_state.loaded_files:
        st.subheader("🗂️ 文献库管理")
        for f in list(st.session_state.loaded_files):
            c1, c2 = st.columns([4, 1])
            with c1: st.caption(f"📄 {f[:22]}...")
            with c2:
                if st.button("🗑️", key=f"del_{f}"):
                    st.session_state.loaded_files.remove(f)
                    st.session_state.all_chunks = [c for c in st.session_state.all_chunks if c.metadata.get('source_paper') != f]
                    if api_key_input: rebuild_index_from_chunks(api_key_input)
                    st.rerun()

        if st.button("🪄 一键生成综述对比表", type="primary"):
            if api_key_input and st.session_state.db:
                with st.spinner("深度扫描文献特征中..."):
                    llm = ChatZhipuAI(model="glm-4", api_key=api_key_input, temperature=0.1)
                    agg_ctx = ""
                    for name in st.session_state.loaded_files:
                        subs = st.session_state.db.similarity_search("Abstract methodology", k=2, filter={"source_paper": name})
                        agg_ctx += f"\n[Paper: {name}]\n" + "\n".join([d.page_content for d in subs])
                    res = llm.invoke(f"分析以下文献片段，生成 Markdown 对比表格(包含：论文名、核心创新点、研究方法、结论)：\n{agg_ctx}")
                    st.session_state.chat_history.append({"role": "assistant", "content": res.content})
                    st.rerun()

        st.markdown("---")
        st.session_state.selected_scope = st.selectbox("👁️ 对话专注范围", ["🌐 对比所有论文"] + st.session_state.loaded_files)

    st.markdown("---")
    if st.session_state.chat_history:
        st.download_button("💾 下载研读报告 (HTML全样式)", generate_html_report(st.session_state.chat_history), "research_notes.html", "text/html")
    
    up_pdf = st.file_uploader("导入 PDF 论文", type="pdf")
    if up_pdf and api_key_input and st.button("开始识别入库"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t:
            t.write(up_pdf.getvalue())
            process_and_add_to_db(t.name, up_pdf.name, api_key_input)
            os.remove(t.name)
            st.rerun()

# ================= 6. 主界面：谷歌式检索与研读 =================
tab_search, tab_chat = st.tabs(["🔍 谷歌式科研调研", "💬 深度研读对话"])

with tab_search:
    st.subheader("🌍 谷歌逻辑加权检索引擎")
    col_q, col_s, col_n = st.columns([3, 1.2, 0.8])
    with col_q: 
        q_input = st.text_input("关键词", value=st.session_state.suggested_query, placeholder="输入课题关键词")
    with col_s: 
        sort_rule = st.selectbox("谷歌排序权重", ["综合排序 (谷歌模式)", "引用量优先", "最新发布优先"])
    with col_n: 
        n_count = st.number_input("获取篇数", 5, 50, 15)

    if st.button("🚀 执行多维检索") and q_input:
        with st.spinner("同步跨库数据并计算权重中..."):
            try:
                arxiv_sort = arxiv.SortCriterion.Relevance
                if "最新" in sort_rule: arxiv_sort = arxiv.SortCriterion.SubmittedDate
                
                # 【谷歌逻辑】布尔转换
                search_q = clean_query_for_arxiv(q_input)
                search_client = arxiv.Search(query=search_q, max_results=n_count, sort_by=arxiv_sort)
                
                final_results = []
                for res in list(search_client.results()):
                    cite_count = fetch_citations(res.entry_id)
                    # 【谷歌优点 2】排序算法：标题匹配权重 + 引用权重
                    title_weight = 100 if any(w.lower() in res.title.lower() for w in q_input.split()) else 0
                    score = cite_count * 2.5 + title_weight
                    final_results.append({'obj': res, 'cite': cite_count, 'score': score})
                    time.sleep(0.1)
                
                if "综合" in sort_rule: final_results.sort(key=lambda x: x['score'], reverse=True)
                elif "引用" in sort_rule: final_results.sort(key=lambda x: x['cite'], reverse=True)
                
                st.session_state.search_results = final_results
            except Exception as e: st.error(f"检索失败: {e}")

    if st.session_state.search_results:
        # 【谷歌优点 3】热点分布摘要
        topics = extract_top_topics(st.session_state.search_results)
        st.write("📊 **当前调研热点聚类统计:**")
        tp_cols = st.columns(len(topics))
        for i, (w, c) in enumerate(topics): tp_cols[i].markdown(f"<div class='topic-tag'>{w} ({c})</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        for i, item in enumerate(st.session_state.search_results):
            res, cite = item['obj'], item['cite']
            # 【谷歌优点 4】模拟谷歌卡片 UI，显示摘要片段
            st.markdown(f"""
            <div class="search-card">
                <a class="paper-title" href="{res.entry_id}" target="_blank">{res.title}</a>
                <div class="paper-url">{res.entry_id}</div>
                <div class="snippet">{res.summary[:350].replace(chr(10), ' ')}...</div>
                <div style="margin-top:12px;">
                    <span class="cite-count">📈 {cite} 引用</span>
                    <span style="margin-left:15px; color:#70757a; font-size:0.85em;">📅 {res.published.year} | {res.authors[0]} 等</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"⬇️ 加入研读库", key=f"dl_btn_{i}"):
                if api_key_input:
                    with st.spinner("同步向量化中..."):
                        p_path = res.download_pdf(dirpath=tempfile.gettempdir())
                        process_and_add_to_db(p_path, res.title, api_key_input)
                        st.success("入库成功！")
                else: st.error("请填入 API Key")

with tab_chat:
    if st.session_state.loaded_files:
        st.caption(f"📚 专注范围: {st.session_state.selected_scope}")
        for msg in st.session_state.chat_history:
            if msg["role"] == "system_notice": st.info(msg["content"])
            else:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if p_input := st.chat_input("基于文献提问..."):
            st.session_state.chat_history.append({"role": "user", "content": p_input})
            with st.chat_message("user"): st.write(p_input)
            with st.chat_message("assistant"):
                try:
                    scope = st.session_state.selected_scope
                    f_dict = {"source_paper": scope} if scope != "🌐 对比所有论文" else None
                    # MMR 深度检索逻辑还原
                    docs = st.session_state.db.max_marginal_relevance_search(p_input, k=8, fetch_k=20, lambda_mult=0.7, filter=f_dict)
                    ctx = "\n\n".join([f"📄【{d.metadata.get('source_paper','?')}】:\n{d.page_content}" for d in docs])
                    llm = ChatZhipuAI(model="glm-4", api_key=api_key_input, temperature=0.1)
                    full_res = llm.invoke(f"你是一位严谨的科研专家。基于资料回答：\n{ctx}\n问题：{p_input}\n要求：学术严谨，公式务必用 $ 包裹。")
                    final_txt = fix_latex_errors(full_res.content)
                    st.write(final_txt)
                    st.session_state.chat_history.append({"role": "assistant", "content": final_txt})
                except Exception as e: st.error(f"对话异常: {e}")

        # 挖掘功能按钮还原
        if st.button("🔍 挖掘当前课题的关联新论文"):
            if api_key_input and st.session_state.chat_history:
                with st.spinner("AI 正在解析语义特征..."):
                    llm = ChatZhipuAI(model="glm-4", api_key=api_key_input)
                    context_bits = str(st.session_state.chat_history[-2:])
                    # 【谷歌逻辑补全】引导 AI 生成简短的、检索友好的词组
                    prompt = f"根据以下研读记录，提取 2 个简练的英文学术搜索词组，用于进一步调研（严禁长句，只输出词组）：\n{context_bits}"
                    st.session_state.suggested_query = llm.invoke(prompt).content.strip()
                    st.success(f"已生成谷歌式词组：{st.session_state.suggested_query}，请去调研 Tab 搜索。")
    else:
        st.info("💡 研读库为空。请先通过调研引擎下载论文，或手动上传 PDF。")
