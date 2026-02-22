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

# ================= 新增：内置API密钥 =================
# 替换这里的密钥为你的实际密钥，后续无需手动输入
USER_API_KEY = "8SwYzCFlra3KhzLD4A0KM2ejrtpz4FsGiGVx7xCb"  # 智谱API密钥
SS_API_KEY = ""  # 如果你有Semantic Scholar密钥，填这里，没有则留空

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

# ================= 4. 核心逻辑函数 (SS API 增强版) =================

def get_pure_arxiv_id(url):
    """从 URL 中精准提取 ArXiv ID"""
    match = re.search(r'(\d{4}\.\d{4,5})', url)
    if match:
        return match.group(1)
    return url.split('/')[-1].split('v')[0]

def fetch_citations(arxiv_id, ss_key=None):
    """获取引用数 (已启用 API Key 加速)"""
    try:
        clean_id = get_pure_arxiv_id(arxiv_id)
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields=citationCount"
        # 使用内置的SS密钥
        headers = {"x-api-key": ss_key or SS_API_KEY} if (ss_key or SS_API_KEY) else {}
        
        # 优化点：有 Key 时降低延迟，没 Key 时保持慢速
        if not (ss_key or SS_API_KEY):
            time.sleep(1.0) 
        else:
            time.sleep(0.02) 
            
        response = requests.get(api_url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json().get('citationCount', 0)
    except:
        pass
    return 0

@st.cache_data(ttl=3600)
def fetch_graph_data(arxiv_id, ss_key=None):
    """获取图谱数据 (已增强摘要字段获取)"""
    clean_id = get_pure_arxiv_id(arxiv_id)
    # 关键修改：在 references 和 citations 后面都加上了 .abstract
    fields = "paperId,title,year,citationCount,abstract,references.paperId,references.title,references.citationCount,references.year,references.abstract,citations.paperId,citations.title,citations.citationCount,citations.year,citations.abstract"
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields={fields}"
    # 使用内置的SS密钥
    headers = {"x-api-key": ss_key or SS_API_KEY} if (ss_key or SS_API_KEY) else {}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(api_url, headers=headers, timeout=12)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                time.sleep((attempt + 1) * 2)
                continue
            else: return None
        except:
            if attempt == max_retries - 1: return None
    return None

def render_connected_graph(data):
    """
    深度重构：Connected Papers 风格图谱
    - 节点大小：正比于 citationCount (对数缩放)
    - 颜色深度：代表年份新旧 (2025/26 为深绿/深蓝)
    """
    if not data: return None, {}
    
    nodes, edges = [], []
    paper_details = {} 
    
    # 获取当前年份以计算颜色梯度
    current_year = 2026 
    
    def get_color(year, rel_type):
        if not year or year == 'Unknown': return "#94a3b8"
        # 计算年份差距，越近颜色越亮/深
        age = max(0, current_year - int(year))
        if rel_type == 'seed': return "#FF4B4B" # 种子节点始终红色
        if rel_type == 'cite': # 引用(绿色系)
            return "#059669" if age < 2 else "#10b981" if age < 5 else "#6ee7b7"
        else: # 被引(蓝色系)
            return "#2563eb" if age < 2 else "#3b82f6" if age < 5 else "#93c5fd"

    seed_id = data.get('paperId', 'root')
    seed_title = data.get('title', 'Seed Paper')
    paper_details[seed_id] = {
        "title": seed_title,
        "abstract": data.get('abstract') or "无摘要",
        "year": data.get('year', 'Unknown'),
        "cites": data.get('citationCount', 0),
        "url": f"https://www.semanticscholar.org/paper/{seed_id}"
    }
    
    # 种子节点（中心）
    nodes.append(Node(id=seed_id, label="THIS PAPER", size=35, color=get_color(data.get('year'), 'seed')))

    seen_ids = set([seed_id])
    # 扩大展示范围至 20 篇，更接近 Connected Papers 的密集感
    combined = []
    for p in data.get('references', [])[:15]: 
        p['rel_type'] = 'ref'; combined.append(p)
    for p in data.get('citations', [])[:15]: 
        p['rel_type'] = 'cite'; combined.append(p)

    for item in combined:
        p_id = item.get('paperId')
        if not p_id or p_id in seen_ids: continue
        seen_ids.add(p_id)
        
        title = item.get('title', 'Unknown')
        year = item.get('year')
        cites = item.get('citationCount', 0)
        
        paper_details[p_id] = {
            "title": title,
            "abstract": item.get('abstract') or "暂无详细摘要。",
            "year": year,
            "cites": cites,
            "url": f"https://www.semanticscholar.org/paper/{p_id}"
        }

        # 模拟 Connected Papers 的大小逻辑：min 15px, 根据引用量增长
        node_size = 15 + (math.log(cites + 1) * 3.5)
        node_color = get_color(year, item['rel_type'])
        
        nodes.append(Node(id=p_id, label=f"{title[:20]}...", size=node_size, color=node_color))
        
        # 连线
        if item['rel_type'] == 'cite':
            edges.append(Edge(source=p_id, target=seed_id, color="#d1d5db", width=1, dashed=True))
        else:
            edges.append(Edge(source=seed_id, target=p_id, color="#94a3b8", width=1.5))

    # 配置物理引擎：让节点分布更均匀，避免重叠
    config = Config(
        width="100%", 
        height=650, 
        directed=True, 
        physics=True, 
        nodeHighlightBehavior=True, 
        highlightColor="#F7D154",
        collapsible=False,
        staticGraph=False,
        # 物理引擎微调
        d3={'alphaTarget': 0.05, 'gravity': -250, 'linkLength': 150, 'linkStrength': 0.1}
    )
    
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

# ================= 5. 侧边栏 =================
with st.sidebar:
    st.header("🎛️ 控制台")
    # 注释掉手动输入框，使用内置密钥
    # user_api_key = st.text_input("智谱 API Key", type="password")
    # ss_api_key = st.text_input("Semantic Scholar API Key", type="password", help="在此填入你的 SS 密钥。")
    user_api_key = USER_API_KEY  # 使用内置智谱密钥
    ss_api_key = SS_API_KEY      # 使用内置SS密钥
    
    if ss_api_key:
        st.success("🚀 高速调研模式已激活")
    else:
        st.caption("🐢 处于匿名限速模式")
    
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
        with st.spinner("正在检索并同步引用数据..."):
            try:
                arxiv_sort = arxiv.SortCriterion.Relevance
                if "时间" in sort_mode: arxiv_sort = arxiv.SortCriterion.SubmittedDate
                refined_query = search_query
                if " " in search_query and "AND" not in search_query and '"' not in search_query:
                    refined_query = " AND ".join([f'(ti:{w} OR abs:{w})' for w in search_query.split()])

                search = arxiv.Search(query=refined_query, max_results=max_results, sort_by=arxiv_sort)
                raw_results = list(search.results())
                
                results_with_cite = []
                progress_bar = st.progress(0)
                for idx, res in enumerate(raw_results):
                    # 关键修改：传入内置的ss_api_key
                    cites = fetch_citations(res.entry_id, ss_key=ss_api_key)
                    results_with_cite.append({'obj': res, 'citations': cites})
                    progress_bar.progress((idx + 1) / len(raw_results))
                
                if "引用量" in sort_mode:
                    results_with_cite.sort(key=lambda x: x['citations'], reverse=True)
                
                st.session_state.search_results = results_with_cite
                st.success(f"✅ 完成！已获取 {len(results_with_cite)} 篇论文。")
            except Exception as e:
                st.error(f"检索失败: {e}")
                
    if st.session_state.search_results:
        if st.session_state.focus_paper_id:
            st.markdown("---")
            st.subheader("📊 文献关联图谱 (Connected Graph)")
            
            with st.spinner("正在请求图谱数据..."):
                # 关键修改：传入内置的ss_api_key
                g_data = fetch_graph_data(st.session_state.focus_paper_id, ss_key=ss_api_key)
            
            if not g_data:
                st.warning("⚠️ 暂时无法获取图谱。请检查 Key 或稍后再试。")
            else:
                col_graph, col_info = st.columns([2.5, 1])
                with col_graph:
                    clicked_node_id, all_details = render_connected_graph(g_data)
                
                # 修复：将 col_info 逻辑放在正确的作用域内
                with col_info:
                    if clicked_node_id and clicked_node_id in all_details:
                        info = all_details[clicked_node_id]
                        st.markdown(f"### 📑 文献详情")
                        st.markdown(f"**{info['title']}**")
                        
                        c1, c2 = st.columns(2)
                        c1.metric("📅 年份", info['year'])
                        c2.metric("🔥 引用", info['cites'])
                        
                        st.markdown("---")
                        st.markdown(f"**摘要**: \n\n <div style='font-size:0.85em; color:#444; height:300px; overflow-y:auto;'>{info['abstract']}</div>", unsafe_allow_html=True)
                        st.markdown("---")
                        st.markdown("""
<div style="display: flex; gap: 20px; justify-content: center; padding: 10px; background: #f8fafc; border-radius: 10px;">
    <span style="color: #FF4B4B;">● 当前论文</span>
    <span style="color: #10b981;">● 引用本文 (Citations)</span>
    <span style="color: #3b82f6;">● 本文引用 (References)</span>
    <span style="color: #94a3b8; font-size: 0.8em;">(节点大小 = 引用量)</span>
</div>
""", unsafe_allow_html=True)

                        # 增加跳转功能
                        st.link_button("🌐 在 Semantic Scholar 中查看", info['url'], use_container_width=True)
                        
                        if st.button("🔬 将此文加入研读队列", use_container_width=True):
                            st.info("此功能可结合 ArXiv 下载逻辑实现自动入库")
                    else:
                        st.info("🎯 **图谱交互指南**\n\n- **滚动鼠标**：缩放图谱\n- **拖拽节点**：固定位置\n- **点击圆点**：查看深度摘要和引用分析\n\n*图谱颜色越深代表年份越近，节点越大代表影响力（引用量）越高。*")
        
        # 检索结果列表展示
        for i, item in enumerate(st.session_state.search_results):
            res = item['obj']
            cites = item['citations']
            with st.expander(f"#{i+1} 📄 {res.title} ({res.published.year})"):
                st.markdown(f"**👨‍🏫 作者**: {', '.join([a.name for a in res.authors])} | **📅 发表**: {res.published.strftime('%Y-%m-%d')}")
                st.markdown(f"**🔥 引用数**: <span class='cite-badge'>{cites}</span>", unsafe_allow_html=True)
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
                        else: st.error("请填入 智谱 API Key")
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
                except Exception as e: 
                    st.error(f"生成出错: {e}")
