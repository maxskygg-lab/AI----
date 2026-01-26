import streamlit as st
import tempfile
import os
from backend import ResearchEngine
from utils import fix_latex_errors, generate_html_report

# 页面配置
st.set_page_config(page_title="AI Research Assistant", layout="wide", page_icon="🎓")
st.title("📖 AI 深度研读助手 (Engineering Edition)")

# Session State 初始化
if "engine" not in st.session_state:
    st.session_state.engine = ResearchEngine()  # 实例化后端引擎
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 侧边栏 ---
with st.sidebar:
    st.header("🎛️ 控制台")
    api_key = st.text_input("智谱 API Key", type="password")

    # 动态更新引擎的 Key
    if api_key:
        st.session_state.engine.api_key = api_key

    st.markdown("---")
    # 综述功能
    if st.button("🪄 生成综述对比表"):
        if not api_key:
            st.error("No API Key")
        else:
            with st.spinner("Analyzing..."):
                try:
                    res = st.session_state.engine.generate_summary()
                    st.session_state.chat_history.append({"role": "assistant", "content": res})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # 导出
    if st.session_state.chat_history:
        html = generate_html_report(st.session_state.chat_history)
        st.download_button("📄 导出笔记", html, "notes.html", "text/html")

    # 上传
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("加载"):
        if not api_key:
            st.error("No API Key")
        else:
            with st.spinner("Processing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    path = tmp.name

                # 调用后端
                try:
                    count = st.session_state.engine.process_pdf(path, uploaded_file.name)
                    st.success(f"Loaded {count} chunks!")
                    st.session_state.chat_history.append(
                        {"role": "system_notice", "content": f"Loaded {uploaded_file.name}"})
                except Exception as e:
                    st.error(str(e))
                finally:
                    os.remove(path)

# --- 主界面 ---
tab1, tab2 = st.tabs(["Search", "Chat"])

with tab1:
    query = st.text_input("ArXiv Keywords")
    if st.button("Search") and query:
        results = st.session_state.engine.search_arxiv(query)
        for res in results:
            st.markdown(f"**{res.title}**")
            st.write(res.summary[:200] + "...")
            st.markdown("---")

with tab2:
    for msg in st.session_state.chat_history:
        role = msg["role"]
        if role == "system_notice":
            st.info(msg["content"])
        else:
            with st.chat_message(role):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ask something..."):
        if not st.session_state.engine.db:
            st.warning("Please upload a paper first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # 构建 Prompt 调用后端
            # (这里为了简化，直接在UI层做了简单的检索逻辑，你也可以移到后端)
            docs = st.session_state.engine.db.similarity_search(prompt, k=4)
            context = "\n".join([d.page_content for d in docs])
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}"

            with st.chat_message("assistant"):
                try:
                    response = st.session_state.engine.query_bot(full_prompt, context)
                    final = fix_latex_errors(response)
                    st.write(final)
                    st.session_state.chat_history.append({"role": "assistant", "content": final})
                except Exception as e:
                    st.error(str(e))
