import streamlit as st
import os, time, tempfile, re, math, uuid
import sqlite3   
import arxiv, requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_agraph import agraph, Node, Edge, Config

# ================= 1. 环境检查 =================
try:
    import zhipuai, langchain_community, fitz
except ImportError as e:
    st.error(f"🚑 环境缺失库 -> {e.name}"); st.stop()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= 2. 页面配置 =================
st.set_page_config(page_title="AI 深度研读助手", layout="wide", page_icon="🎓")
st.markdown("""
<style>
    .stButton>button { width:100%; border-radius:8px; }
    .abstract-box {
        background:#f0f2f6; padding:12px; border-radius:8px;
        border-left:5px solid #4CAF50; font-size:.9em;
        line-height:1.6; margin-bottom:6px;
    }
    .contribution-box {
        background:linear-gradient(90deg,#fffbeb,#fef3c7);
        border-left:4px solid #f59e0b; padding:7px 12px;
        border-radius:6px; font-size:.85em; color:#78350f;
        margin-bottom:8px; font-weight:500;
    }
    .cite-badge {
        background:#ff4b4b; color:white; padding:2px 7px;
        border-radius:12px; font-size:.78em; font-weight:bold;
    }
    .cite-loading { color:#94a3b8; font-size:.78em; font-style:italic; }
    .topic-badge {
        display:inline-block; background:#6366f1; color:white;
        padding:2px 10px; border-radius:20px; font-size:.78em;
        font-weight:600; margin-right:4px;
    }
    .gap-box {
        background:#fef2f2; border:1px solid #fca5a5;
        border-left:4px solid #ef4444; border-radius:8px;
        padding:12px 16px; margin:10px 0;
    }
    .note-card {
        background:#f8fafc; border:1px solid #e2e8f0;
        border-radius:10px; padding:14px 18px; margin-bottom:10px;
    }
    .note-tag {
        display:inline-block; background:#e0e7ff; color:#3730a3;
        padding:1px 8px; border-radius:12px; font-size:.75em;
        margin-right:4px; font-weight:500;
    }
    .chat-panel {
        height:520px; overflow-y:auto; border:1px solid #e2e8f0;
        border-radius:10px; padding:12px; background:#fafafa; margin-bottom:10px;
    }
    .chat-user { background:#dbeafe; border-radius:8px; padding:8px 12px; margin:6px 0; font-size:.9em; }
    .chat-bot  { background:#f0fdf4; border-radius:8px; padding:8px 12px; margin:6px 0; font-size:.9em; }
    .chat-notice { color:#6366f1; font-size:.82em; font-style:italic; margin:4px 0; }
    .section-divider {
        font-size:.72em; text-transform:uppercase; letter-spacing:2px;
        color:#94a3b8; margin:18px 0 8px;
    }
    .perf-badge {
        display:inline-block; background:#dcfce7; color:#166534;
        border:1px solid #86efac; padding:2px 8px; border-radius:12px;
        font-size:.75em; font-weight:600; margin-left:6px;
    }
    .tracker-card {
        background:#f8fafc; border:1px solid #e2e8f0;
        border-radius:10px; padding:14px 16px; margin-bottom:14px;
    }
    .tracker-new-badge {
        display:inline-block; background:#f59e0b; color:#fff;
        padding:2px 9px; border-radius:12px; font-size:.75em; font-weight:700; margin-left:6px;
    }
    .new-paper-card {
        background:#fffbeb; border:1px solid #fde68a;
        border-left:4px solid #f59e0b; border-radius:8px;
        padding:12px 16px; margin:8px 0; font-size:.88em; line-height:1.65;
    }
</style>
""", unsafe_allow_html=True)

st.title("📖 AI 深度研读助手 v5")

# ================= API Key =================
USER_API_KEY = st.secrets["ZHIPU_API_KEY"]
SS_API_KEY   = st.secrets["SS_API_KEY"]

# ================= 3. 状态初始化 =================
defaults = {
    "search_results":         [],
    "citations_loaded":       False,
    "citations_global_cache": {},
    "suggested_query":        "",
    "focus_paper_id":         None,
    "contributions_cache":    {},
    "chat_history":           [],
    "topics":                 {"默认主题": {"files": [], "chunks": [], "db": None}},
    "active_topic":           "默认主题",
    "selected_scope":         "🌐 对比所有论文",
    "notes":                  [],
    "pending_note":           None,
    "graph_references_cache": [],
    "preload_done_ids":       set(),
    "trackers":               {},   # { kw: {check_interval_h, last_checked, seen_ids, new_papers} }
    "tracker_total_new":      0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================= 4. 工具函数 =================

def active_topic_data():
    return st.session_state.topics[st.session_state.active_topic]

def get_pure_arxiv_id(url_or_id):
    m = re.search(r'(\d{4}\.\d{4,5})', url_or_id)
    return m.group(1) if m else url_or_id.split('/')[-1].split('v')[0]

# ── 引用数批量 API ──
@st.cache_data(ttl=1800)
def fetch_citations_batch_cached(arxiv_ids_tuple: tuple, ss_key=None) -> dict:
    clean_ids = [f"ArXiv:{get_pure_arxiv_id(aid)}" for aid in arxiv_ids_tuple]
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    headers = {"x-api-key": ss_key} if ss_key else {}
    try:
        r = requests.post(url, headers=headers,
                          params={"fields": "citationCount,externalIds"},
                          json={"ids": clean_ids}, timeout=15)
        if r.status_code == 200:
            out = {}
            for item in r.json():
                if item and item.get("externalIds"):
                    aid = item["externalIds"].get("ArXiv","")
                    if aid:
                        out[aid] = item.get("citationCount", 0)
            return out
    except Exception as e:
        st.warning(f"批量引用数获取异常，降级: {e}")
    return {}

def fetch_one_citation(args):
    arxiv_id, ss_key = args
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{get_pure_arxiv_id(arxiv_id)}?fields=citationCount"
        headers = {"x-api-key": ss_key} if ss_key else {}
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code == 200:
            return arxiv_id, r.json().get('citationCount', 0)
    except: pass
    return arxiv_id, 0

def fetch_citations_parallel(results, ss_key=None):
    args_list = [(item['obj'].entry_id, ss_key) for item in results]
    out = {}
    with ThreadPoolExecutor(max_workers=8 if ss_key else 3) as pool:
        for future in as_completed({pool.submit(fetch_one_citation, a): a[0] for a in args_list}):
            aid, count = future.result()
            out[aid] = count
    return out

def smart_fetch_citations(results, ss_key=None):
    cache = st.session_state.citations_global_cache
    missing = [item for item in results if get_pure_arxiv_id(item['obj'].entry_id) not in cache]
    hits = len(results) - len(missing)
    if hits:
        st.caption(f"⚡ {hits} 篇命中缓存，{len(missing)} 篇需请求")
    if missing:
        ids = tuple(item['obj'].entry_id for item in missing)
        new_data = fetch_citations_batch_cached(ids, ss_key)
        if not new_data:
            new_data = {get_pure_arxiv_id(k): v
                        for k, v in fetch_citations_parallel(missing, ss_key).items()}
        cache.update(new_data)
    return {item['obj'].entry_id: cache.get(get_pure_arxiv_id(item['obj'].entry_id), 0)
            for item in results}

# ── 图谱 ──
def preload_top_graphs(results, ss_key=None, top_n=3):
    done = st.session_state.preload_done_ids
    to_do = [item for item in sorted(results, key=lambda x: x.get("citations") or 0, reverse=True)[:top_n]
             if item['obj'].entry_id not in done]
    if not to_do: return
    ph = st.empty()
    ph.caption(f"🔄 后台预加载 Top {len(to_do)} 图谱…")
    for item in to_do:
        fetch_graph_data(item['obj'].entry_id, ss_key=ss_key)
        done.add(item['obj'].entry_id)
        time.sleep(0.3)
    ph.caption("✅ 图谱预加载完成")

@st.cache_data(ttl=3600)
def fetch_graph_data(arxiv_id, ss_key=None):
    clean_id = get_pure_arxiv_id(arxiv_id)
    fields = (
        "paperId,title,year,citationCount,abstract,"
        "references.paperId,references.title,references.citationCount,"
        "references.year,references.abstract,references.externalIds,"
        "citations.paperId,citations.title,citations.citationCount,"
        "citations.year,citations.abstract,citations.externalIds"
    )
    url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields={fields}"
    headers = {"x-api-key": ss_key} if ss_key else {}
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=12)
            if r.status_code == 200: return r.json()
            elif r.status_code == 429: time.sleep((attempt+1)*2)
        except:
            if attempt == 2: return None
    return None

def get_one_line_contribution(abstract, title, api_key):
    key = title[:60]
    if key in st.session_state.contributions_cache:
        return st.session_state.contributions_cache[key]
    try:
        llm = ChatZhipuAI(model="glm-4-flash", api_key=api_key, temperature=0.0)
        res = llm.invoke(
            f"请用一句话（不超过40个汉字或20个英文单词）总结这篇论文的核心创新贡献。"
            f"只输出这一句话，不要前缀或解释。\n\n标题：{title}\n摘要：{abstract[:600]}"
        )
        result = res.content.strip()
    except: result = "（生成失败）"
    st.session_state.contributions_cache[key] = result
    return result

def fix_latex(text):
    if not text: return text
    return text.replace(r"\(","$").replace(r"\)","$").replace(r"\[","$$").replace(r"\]","$$")

def process_and_add_to_topic(file_path, file_name, api_key, topic_name=None):
    topic_name = topic_name or st.session_state.active_topic
    t = st.session_state.topics[topic_name]
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source_paper'] = file_name
            doc.metadata['topic'] = topic_name
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=200,
            separators=["\n\n","\n","。","."," ",""]
        )
        chunks = [c for c in splitter.split_documents(docs) if len(c.page_content.strip()) > 20]
        t["chunks"].extend(chunks)
        embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
        batch = 10
        if t["db"] is None:
            t["db"] = FAISS.from_documents(chunks[:batch], embeddings)
            for i in range(batch, len(chunks), batch):
                t["db"].add_documents(chunks[i:i+batch]); time.sleep(0.1)
        else:
            for i in range(0, len(chunks), batch):
                t["db"].add_documents(chunks[i:i+batch]); time.sleep(0.1)
        if file_name not in t["files"]:
            t["files"].append(file_name)
        st.session_state.chat_history.append({
            "role": "system_notice",
            "content": f"📚 《{file_name}》已加入主题「{topic_name}」。"
        })
        return True
    except Exception as e:
        st.error(f"处理失败: {e}"); return False

def rebuild_topic_index(topic_name, api_key):
    t = st.session_state.topics[topic_name]
    if not t["chunks"]: t["db"] = None; return
    embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)
    t["db"] = FAISS.from_documents(t["chunks"], embeddings)

def detect_knowledge_gap(answer_text, docs):
    sigs = ["资料不足","没有找到","无法回答","未提及","不清楚","没有相关","cannot find","not mentioned"]
    if len(docs) < 3: return True
    for s in sigs:
        if s.lower() in answer_text.lower(): return True
    return False

def get_gap_recommendations():
    loaded = set()
    for t in st.session_state.topics.values(): loaded.update(t["files"])
    return [r for r in st.session_state.graph_references_cache
            if not any(r.get("title","")[:20].lower() in f.lower() for f in loaded)][:4]

# ── 关键词追踪 ──
def tracker_check_one(keyword: str, since_date: str | None = None) -> list:
    try:
        cutoff = datetime.fromisoformat(since_date) if since_date else datetime.now() - timedelta(days=7)
        refined = keyword
        if " " in keyword and "AND" not in keyword and '"' not in keyword:
            refined = " AND ".join([f'(ti:{w} OR abs:{w})' for w in keyword.split()])
        results = list(arxiv.Search(
            query=refined, max_results=30,
            sort_by=arxiv.SortCriterion.SubmittedDate
        ).results())
        out = []
        for r in results:
            if r.published.replace(tzinfo=None) > cutoff:
                out.append({
                    "title":     r.title,
                    "authors":   ", ".join([a.name for a in r.authors]),  # 完整作者
                    "published": r.published.strftime("%Y-%m-%d"),
                    "summary":   r.summary,                               # 完整摘要
                    "entry_id":  r.entry_id,
                    "obj":       r,
                })
        return out
    except Exception as e:
        st.warning(f"追踪「{keyword}」时出错: {e}"); return []

def tracker_run_all(force=False):
    if not st.session_state.trackers: return
    now = datetime.now()
    total = 0
    for kw, data in st.session_state.trackers.items():
        last = data.get("last_checked")
        ih   = data.get("check_interval_h", 12)
        if not force and last:
            elapsed = (now - datetime.fromisoformat(last)).total_seconds() / 3600
            if elapsed < ih:
                total += len(data.get("new_papers",[])); continue
        new = tracker_check_one(kw, since_date=data.get("last_checked"))
        seen = set(data.get("seen_ids",[]))
        truly_new = [p for p in new if p["entry_id"] not in seen]
        data["new_papers"]   = truly_new + data.get("new_papers",[])
        data["last_checked"] = now.isoformat(timespec="seconds")
        total += len(data["new_papers"])
    st.session_state.tracker_total_new = total

def tracker_mark_read(keyword: str):
    data = st.session_state.trackers.get(keyword, {})
    for p in data.get("new_papers",[]): data.setdefault("seen_ids",[]).append(p["entry_id"])
    data["new_papers"] = []
    st.session_state.tracker_total_new = sum(
        len(d.get("new_papers",[])) for d in st.session_state.trackers.values()
    )

# 启动时自动静默检查
if st.session_state.trackers:
    tracker_run_all(force=False)

# ================= 5. 图谱渲染 =================
def render_connected_graph(data, min_cite_filter=0):
    if not data: return None, {}
    nodes, edges, details = [], [], {}
    cur_year = 2026

    def color(year, rel):
        if not year or year == 'Unknown': return "#94a3b8"
        age = max(0, cur_year - int(year))
        if rel == 'seed': return "#FF4B4B"
        if rel == 'cite': return "#059669" if age<2 else "#10b981" if age<5 else "#6ee7b7"
        return "#2563eb" if age<2 else "#3b82f6" if age<5 else "#93c5fd"

    seed = data.get('paperId','root')
    details[seed] = {
        "title":    data.get('title','Seed Paper'),
        "abstract": data.get('abstract') or "无摘要",
        "year":     data.get('year','Unknown'),
        "cites":    data.get('citationCount',0),
        "url":      f"https://www.semanticscholar.org/paper/{seed}",
        "arxiv_id": None,
    }
    nodes.append(Node(id=seed, label="THIS PAPER", size=35, color=color(data.get('year'),'seed')))
    seen = {seed}
    refs_for_gap = []

    combined = []
    for p in data.get('references',[])[:20]: p['rel_type']='ref'; combined.append(p)
    for p in data.get('citations',[])[:20]:  p['rel_type']='cite'; combined.append(p)

    for item in combined:
        pid   = item.get('paperId')
        cites = item.get('citationCount',0) or 0
        if not pid or pid in seen or cites < min_cite_filter: continue
        seen.add(pid)
        title    = item.get('title','Unknown')
        year     = item.get('year')
        ext      = item.get('externalIds') or {}
        arxiv_id = ext.get('ArXiv')
        details[pid] = {
            "title":    title,
            "abstract": item.get('abstract') or "暂无摘要",
            "year":     year, "cites": cites,
            "url":      f"https://www.semanticscholar.org/paper/{pid}",
            "arxiv_id": arxiv_id,
        }
        if item['rel_type']=='ref' and arxiv_id:
            refs_for_gap.append({"title":title,"arxiv_id":arxiv_id,"abstract":item.get('abstract','')})
        sz = 15 + math.log(cites+1)*3.5
        nodes.append(Node(id=pid, label=f"{title[:20]}…", size=sz, color=color(year, item['rel_type'])))
        if item['rel_type']=='cite':
            edges.append(Edge(source=pid, target=seed, color="#d1d5db", width=1, dashed=True))
        else:
            edges.append(Edge(source=seed, target=pid, color="#94a3b8", width=1.5))

    st.session_state.graph_references_cache = refs_for_gap
    cfg = Config(width="100%", height=560, directed=True, physics=True,
                 nodeHighlightBehavior=True, highlightColor="#F7D154",
                 d3={'alphaTarget':0.05,'gravity':-250,'linkLength':150,'linkStrength':0.1})
    clicked = agraph(nodes=nodes, edges=edges, config=cfg)
    return clicked, details

# ================= 6. 侧边栏 =================
with st.sidebar:
    st.header("🎛️ 控制台")
    user_api_key = USER_API_KEY
    ss_api_key   = SS_API_KEY
    st.success("🚀 高速调研模式已激活")

    cache_sz = len(st.session_state.citations_global_cache)
    if cache_sz: st.info(f"⚡ 引用数缓存：{cache_sz} 篇")

    st.markdown("---")
    st.subheader("🗂️ 研究主题")
    tnames = list(st.session_state.topics.keys())
    aidx   = tnames.index(st.session_state.active_topic) if st.session_state.active_topic in tnames else 0
    chosen = st.selectbox("当前主题", tnames, index=aidx)
    if chosen != st.session_state.active_topic:
        st.session_state.active_topic = chosen
        st.session_state.selected_scope = "🌐 对比所有论文"
        st.rerun()

    cn, ca = st.columns([3,1])
    with cn: new_tn = st.text_input("新建主题", placeholder="输入名称", label_visibility="collapsed")
    with ca:
        if st.button("➕") and new_tn.strip():
            nm = new_tn.strip()
            if nm not in st.session_state.topics:
                st.session_state.topics[nm] = {"files":[],"chunks":[],"db":None}
                st.session_state.active_topic = nm; st.rerun()

    if len(st.session_state.topics) > 1:
        if st.button(f"🗑️ 删除「{st.session_state.active_topic}」"):
            del st.session_state.topics[st.session_state.active_topic]
            st.session_state.active_topic = list(st.session_state.topics.keys())[0]; st.rerun()

    ts = active_topic_data()
    if ts["files"]:
        st.markdown(f"**已入库（{len(ts['files'])}篇）**")
        for f in list(ts["files"]):
            c1,c2 = st.columns([4,1])
            with c1: st.text(f"📄 {f[:16]}..." if len(f)>18 else f"📄 {f}")
            with c2:
                if st.button("🗑️", key=f"del_{f}"):
                    ts["files"].remove(f)
                    ts["chunks"] = [c for c in ts["chunks"] if c.metadata.get('source_paper')!=f]
                    rebuild_topic_index(st.session_state.active_topic, user_api_key); st.rerun()
        if st.button("🗑️ 清空主题", type="primary"):
            ts["files"],ts["chunks"],ts["db"] = [],[],None
            st.session_state.chat_history = []; st.rerun()

    st.markdown("---")
    st.subheader("📥 上传 PDF")
    uploaded_file = st.file_uploader("拖入 PDF", type="pdf")
    if uploaded_file and st.button("确认加载"):
        with st.spinner("解析中..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue()); path = tmp.name
            process_and_add_to_topic(path, uploaded_file.name, user_api_key)
            os.remove(path); st.rerun()

# ================= 7. 主界面 =================
_n_new = st.session_state.tracker_total_new
_track_label = f"🔔 追踪提醒 ({_n_new} 新)" if _n_new > 0 else "🔔 关键词追踪"

# 关键修复点：将 tab_main 改为 tab_search
tab_search, tab_read, tab_track, tab_notes = st.tabs([
    "🔍 学术检索 & 图谱", "📖 研读空间", _track_label, "📌 我的笔记"
])

# ══════════════════════════════════════════
# Tab 1：学术检索 & 全量信息流 (微博模式)
# ══════════════════════════════════════════
with tab_search:
    # --- 数据库初始化 ---
    def get_db_conn():
        conn = sqlite3.connect("academic_feed.db")
        conn.row_factory = sqlite3.Row  # 允许通过名称访问列
        return conn

    conn = get_db_conn()
    conn.execute('''CREATE TABLE IF NOT EXISTS feed 
                 (id TEXT PRIMARY KEY, title TEXT, summary TEXT, 
                  authors TEXT, date TEXT, topic TEXT, link TEXT)''')
    conn.commit()

    # --- 检索控制区 ---
    st.subheader("🚀 学术全量信息流")
    sc1, sc2, sc3 = st.columns([3, 1, 1])
    with sc1:
        query_input = st.text_input("输入科研关键词", placeholder="如：Large Language Models", key="full_search_input")
    with sc2:
        max_total = st.number_input("抓取总量", min_value=10, max_value=2000, value=200, step=50)
    with sc3:
        st.write("") # 占位
        if st.button("🔥 开始全量抓取", use_container_width=True):
            if query_input.strip():
                with st.spinner(f"正在从 ArXiv 抽取 {max_total} 篇论文..."):
                    client = arxiv.Client()
                    found_new = 0
                    # 循环翻页抓取逻辑 (每次100篇)
                    for offset in range(0, max_total, 100):
                        search = arxiv.Search(
                            query=query_input,
                            max_results=100,
                            offset=offset,
                            sort_by=arxiv.SortCriterion.Relevance
                        )
                        try:
                            results = list(client.results(search))
                            if not results: break
                            for p in results:
                                conn.execute("INSERT OR IGNORE INTO feed VALUES (?,?,?,?,?,?,?)",
                                          (p.entry_id, p.title, p.summary, 
                                           ", ".join(a.name for a in p.authors),
                                           p.published.strftime("%Y-%m-%d"), 
                                           st.session_state.active_topic, p.entry_id))
                                if conn.total_changes > 0: found_new += 1
                            conn.commit()
                        except Exception: break
                    st.success(f"抓取完成！新入库 {found_new} 篇动态。")
                    st.rerun()

    st.markdown("---")

    # --- 微博卡片流渲染 ---
    # 分页设置
    if "feed_page" not in st.session_state: st.session_state.feed_page = 1
    items_per_page = 15
    offset = (st.session_state.feed_page - 1) * items_per_page

    # 从本地数据库读取
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM feed WHERE topic=? ORDER BY date DESC LIMIT ? OFFSET ?", 
                (st.session_state.active_topic, items_per_page, offset))
    rows = cur.fetchall()

    if not rows:
        st.info("💡 这里的学术朋友圈还是空的。在上方输入关键词并点击“全量抓取”来填充它！")
    else:
        for row in rows:
            # 渲染卡片样式
            st.markdown(f"""
            <div style="background: #f8fafc; border-radius: 12px; padding: 20px; 
                        margin-bottom: 15px; border: 1px solid #e2e8f0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="background: #3b82f6; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem;">学术动态</span>
                    <span style="color: #64748b; font-size: 0.85rem;">📅 发布于 {row['date']}</span>
                </div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-bottom: 8px; line-height: 1.4;">
                    {row['title']}
                </div>
                <div style="color: #475569; font-size: 0.9rem; margin-bottom: 12px;">
                    👤 {row['authors']}
                </div>
                <details style="cursor: pointer; color: #334155; font-size: 0.95rem;">
                    <summary style="color: #3b82f6; font-weight: 600;">展开阅读摘要</summary>
                    <div style="padding-top: 10px; line-height: 1.6;">{row['summary']}</div>
                </details>
            </div>
            """, unsafe_allow_html=True)
            
            # 卡片下方的交互按钮
            b1, b2, b3, _ = st.columns([1, 1, 1, 4])
            with b1:
                st.markdown(f"[🔗 原文]({row['link']})")
            with b2:
                if st.button("📥 入库", key=f"feed_dl_{row['id']}"):
                    # 此处调用你原有的下载并处理函数
                    with st.spinner("正在入库..."):
                        try:
                            paper_obj = next(arxiv.Search(id_list=[row['id'].split('/')[-1]]).results())
                            path = paper_obj.download_pdf(dirpath=tempfile.gettempdir())
                            process_and_add_to_topic(path, row['title'], user_api_key)
                            st.toast("✅ 已成功加入研读空间")
                        except Exception as e: st.error(str(e))
            with b3:
                if st.button("🗑️ 隐藏", key=f"feed_hide_{row['id']}"):
                    conn.execute("DELETE FROM feed WHERE id=?", (row['id'],))
                    conn.commit()
                    st.rerun()

        # --- 分页控制 ---
        st.markdown("---")
        p1, p2, p3 = st.columns([1, 2, 1])
        with p1:
            if st.session_state.feed_page > 1:
                if st.button("⬅️ 上一页"):
                    st.session_state.feed_page -= 1
                    st.rerun()
        with p2:
            st.write(f"<p style='text-align:center'>第 {st.session_state.feed_page} 页</p>", unsafe_allow_html=True)
        with p3:
            if len(rows) == items_per_page:
                if st.button("下一页 ➡️"):
                    st.session_state.feed_page += 1
                    st.rerun()
    conn.close()

# ══════════════════════════════════════════
# Tab 2：研读空间
# ══════════════════════════════════════════
with tab_read:
    t = active_topic_data()
    st.markdown(
        f'<div class="section-divider">💬 研读空间 — '
        f'<span class="topic-badge">🗂️ {st.session_state.active_topic}</span> · {len(t["files"])} 篇入库</div>',
        unsafe_allow_html=True
    )
    rc1, rc2 = st.columns([2, 3])
    with rc1:
        reading_mode = st.radio("模式", ["🟢 快速问答", "📖 逐段精读"], horizontal=True)
    with rc2:
        if t["files"]:
            st.session_state.selected_scope = st.selectbox("专注范围", ["🌐 对比所有论文"] + t["files"])

    if not t["files"]:
        st.info("📥 请先在「学术检索 & 图谱」标签页下载论文，或在左侧侧边栏上传 PDF。")
    else:
        # 知识漏洞推荐逻辑
        if st.session_state.pending_note and st.session_state.pending_note.get("has_gap"):
            recs = get_gap_recommendations()
            if recs:
                st.markdown('<div class="gap-box"><b>🔍 知识漏洞推荐</b></div>', unsafe_allow_html=True)
                for r in recs:
                    rx1, rx2 = st.columns([5, 1])
                    with rx1: st.caption(r['title'])
                    with rx2:
                        if r.get('arxiv_id') and st.button("⬇️", key=f"gap_{r['arxiv_id']}"):
                            with st.spinner("下载..."):
                                try:
                                    paper = next(arxiv.Search(id_list=[r['arxiv_id']]).results())
                                    pdf_path = paper.download_pdf(dirpath=tempfile.gettempdir())
                                    process_and_add_to_topic(pdf_path, r['title'], user_api_key)
                                    st.success("已入库！"); st.rerun()
                                except Exception as e: st.error(str(e))

        # 保存笔记逻辑
        if st.session_state.pending_note and st.session_state.pending_note.get("content"):
            with st.expander("📌 保存为笔记", expanded=False):
                note_tags_raw = st.text_input("标签（逗号分隔）", placeholder="方法论, Transformer", key="note_tags_input")
                if st.button("💾 保存", type="primary"):
                    tags = [tg.strip() for tg in note_tags_raw.split(",") if tg.strip()]
                    st.session_state.notes.append({
                        "id": str(uuid.uuid4())[:8],
                        "content": st.session_state.pending_note["content"],
                        "question": st.session_state.pending_note.get("question", ""),
                        "tags": tags,
                        "topic": st.session_state.active_topic,
                        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    })
                    st.session_state.pending_note = None
                    st.success("✅ 已保存到「我的笔记」"); st.rerun()

        # 渲染聊天记录
        chat_html = ""
        for msg in st.session_state.chat_history[-20:]:
            if msg["role"] == "system_notice":
                chat_html += f'<div class="chat-notice">📢 {msg["content"]}</div>'
            elif msg["role"] == "user":
                chat_html += f'<div class="chat-user">🧑 {msg["content"]}</div>'
            else:
                chat_html += f'<div class="chat-bot">🤖 {msg["content"].replace(chr(10),"<br>")}</div>'
        st.markdown(f'<div class="chat-panel">{chat_html}</div>', unsafe_allow_html=True)

        # 输入框
        ci1, ci2 = st.columns([6, 1])
        with ci1:
            user_input = st.text_input("提问", placeholder="输入问题（如：对比 A 论文和 B 论文的方法论差异）...",
                                       label_visibility="collapsed", key="chat_input_box")
        with ci2:
            send_btn = st.button("发送 ➤", use_container_width=True)

        if send_btn and user_input.strip():
            prompt = user_input.strip()
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.spinner("深度检索资料并对比中..."):
                try:
                    sk = 15 if "精读" in reading_mode else 10
                    scope = st.session_state.selected_scope
                    
                    docs = []
                    # 核心改进：针对“对比”场景优化检索
                    if scope == "🌐 对比所有论文":
                        # 1. 首先进行全局 MMR 检索
                        docs = t["db"].max_marginal_relevance_search(prompt, k=sk, fetch_k=30, lambda_mult=0.5)
                        
                        # 2. 增强逻辑：如果论文数量 > 1，且用户提问包含对比倾向，则确保每篇论文都有内容被检出
                        if len(t["files"]) > 1:
                            existing_sources = set(d.metadata.get('source_paper') for d in docs)
                            # 如果有论文在检索中“掉队”了，为掉队的论文补齐最相关的片段
                            for paper_name in t["files"]:
                                if paper_name not in existing_sources:
                                    extra_docs = t["db"].similarity_search(prompt, k=2, filter={"source_paper": paper_name})
                                    docs.extend(extra_docs)
                    else:
                        # 单篇论文检索
                        fd = {"source_paper": scope}
                        docs = t["db"].max_marginal_relevance_search(prompt, k=sk, fetch_k=20, lambda_mult=0.6, filter=fd)

                    if not docs:
                        answer = "未找到相关内容，请尝试换个问法。"
                    else:
                        # 构建上下文，强调来源标识
                        context_list = []
                        for d in docs:
                            src = d.metadata.get('source_paper', '未知来源')
                            pg = d.metadata.get('page', 0) + 1
                            context_list.append(f"【来源文件：{src} | 第 {pg} 页】\n内容：{d.page_content}")
                        
                        context = "\n\n---\n\n".join(context_list)
                        
                        sys_p = (
                            "你是一位资深科研助理。请基于以下提供的多篇论文片段进行回答。\n"
                            "### 任务要求：\n"
                            "1. 如果用户要求对比，请清晰地列出不同论文在观点、方法或结果上的【相同点】和【不同点】。\n"
                            "2. 回答必须严格基于资料，引用时请标注来源（如：据[论文A]所述）。\n"
                            "3. 数学公式使用 $...$ 格式。\n"
                            f"4. 如果资料中没有提到相关信息，请直接回答【资料不足】。\n\n"
                            f"### 检索到的资料：\n{context}\n\n"
                            f"### 用户问题：\n{prompt}"
                        )
                        
                        llm = ChatZhipuAI(model="glm-4", api_key=user_api_key, temperature=0.1)
                        answer = fix_latex(llm.invoke(sys_p).content)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.session_state.pending_note = {
                        "content": answer, 
                        "question": prompt,
                        "has_gap": detect_knowledge_gap(answer, docs if docs else [])
                    }
                    st.rerun()
                except Exception as e: 
                    st.error(f"生成出错: {e}")

# ══════════════════════════════════════════
# Tab 3：关键词追踪
# ══════════════════════════════════════════
with tab_track:
    st.subheader("🔔 关键词追踪")
    st.caption("添加关键词后，App 每次启动自动检查 arXiv，有新论文时 Tab 标题显示数量提醒。")

    add1,add2,add3 = st.columns([3,1.2,1])
    with add1:
        new_kw = st.text_input("关键词", placeholder="例如: diffusion model",
                               label_visibility="collapsed", key="tracker_new_kw")
    with add2:
        ih = st.selectbox("检查间隔",[6,12,24,72],
                          format_func=lambda x:f"每 {x}h",
                          label_visibility="collapsed", key="tracker_interval")
    with add3:
        if st.button("➕ 添加追踪", use_container_width=True) and new_kw.strip():
            kw = new_kw.strip()
            if kw not in st.session_state.trackers:
                st.session_state.trackers[kw] = {
                    "check_interval_h": ih, "last_checked": None,
                    "seen_ids": [], "new_papers": [],
                }
                with st.spinner("首次检查中…"): tracker_run_all(force=True)
                st.rerun()
            else: st.warning("该关键词已在追踪列表中")

    if st.session_state.trackers:
        ga1,ga2 = st.columns([3,1])
        with ga1:
            nn = st.session_state.tracker_total_new
            bdg = (f"<span class='tracker-new-badge'>🆕 {nn} 篇未读</span>" if nn > 0
                   else "<span style='color:#94a3b8;font-size:.85em'>暂无未读</span>")
            st.markdown(f"共追踪 **{len(st.session_state.trackers)}** 个关键词 · {bdg}", unsafe_allow_html=True)
        with ga2:
            if st.button("🔄 立即全部刷新", use_container_width=True):
                with st.spinner("检查中…"): tracker_run_all(force=True)
                st.rerun()

    st.markdown("---")

    if not st.session_state.trackers:
        st.info("还没有追踪任何关键词，在上方添加第一个吧！")
    else:
        for kw, data in list(st.session_state.trackers.items()):
            new_papers = data.get("new_papers",[])
            last_chk   = data.get("last_checked","从未")
            n_new      = len(new_papers)
            badge      = (f"<span class='tracker-new-badge'>🆕 {n_new} 篇新论文</span>"
                          if n_new > 0 else "<span style='color:#94a3b8;font-size:.8em'>暂无新论文</span>")

            st.markdown('<div class="tracker-card">', unsafe_allow_html=True)
            th1,th2,th3,th4 = st.columns([3,2,1,1])
            with th1: st.markdown(f"**🔑 {kw}** {badge}", unsafe_allow_html=True)
            with th2: st.caption(f"🕐 上次: {last_chk[:16] if last_chk != '从未' else '从未'}")
            with th3:
                if st.button("✅ 标记已读", key=f"read_{kw}", use_container_width=True, disabled=(n_new==0)):
                    tracker_mark_read(kw); st.rerun()
            with th4:
                if st.button("🗑️ 删除", key=f"del_track_{kw}", use_container_width=True):
                    del st.session_state.trackers[kw]
                    st.session_state.tracker_total_new = sum(
                        len(d.get("new_papers",[])) for d in st.session_state.trackers.values()
                    ); st.rerun()

            if new_papers:
                for paper in new_papers:
                    # 完整标题、完整作者、完整摘要，不截断
                    st.markdown(
                        f"""
                        <div class="new-paper-card">
                            <div style="font-weight:700;color:#1e293b;font-size:.93em;
                                        margin-bottom:6px;line-height:1.4;">
                                📄 {paper['title']}
                            </div>
                            <div style="color:#64748b;font-size:.83em;margin-bottom:10px;">
                                👤 {paper['authors']} &nbsp;·&nbsp; 📅 {paper['published']}
                            </div>
                            <div style="color:#475569;font-size:.85em;line-height:1.7;">
                                {paper['summary']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True,
                    )
                    pb1,pb2,pb3 = st.columns([1,1,4])
                    with pb1: st.markdown(f"[🔗 ArXiv]({paper['entry_id']})")
                    with pb2:
                        if st.button("⬇️ 入库", key=f"tr_dl_{paper['entry_id']}"):
                            with st.spinner("下载中…"):
                                try:
                                    obj = paper.get("obj") or next(
                                        arxiv.Search(id_list=[get_pure_arxiv_id(paper['entry_id'])]).results()
                                    )
                                    pdf_path = obj.download_pdf(dirpath=tempfile.gettempdir())
                                    process_and_add_to_topic(pdf_path, paper['title'], user_api_key)
                                    st.success("已入库！")
                                except Exception as e: st.error(str(e))
                    with pb3:
                        if st.button("🕸️ 查图谱", key=f"tr_graph_{paper['entry_id']}"):
                            st.session_state.focus_paper_id = paper['entry_id']; st.rerun()
            else:
                st.caption(f"暂无新论文 · 检查间隔：每 {data.get('check_interval_h',12)}h")

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("")

# ══════════════════════════════════════════
# Tab 4：我的笔记
# ══════════════════════════════════════════
with tab_notes:
    st.subheader("📌 我的笔记库")
    if not st.session_state.notes:
        st.info("还没有笔记。在「研读空间」提问后点击「保存为笔记」即可积累。")
    else:
        all_tags   = sorted(set(tag for n in st.session_state.notes for tag in n["tags"]))
        all_topics = sorted(set(n["topic"] for n in st.session_state.notes))
        f1,f2,f3 = st.columns([2,2,2])
        with f1: filter_tag   = st.selectbox("按标签",["全部"]+all_tags)
        with f2: filter_topic = st.selectbox("按主题",["全部"]+all_topics)
        with f3: search_note  = st.text_input("关键词", placeholder="搜索笔记")

        filtered = st.session_state.notes.copy()
        if filter_tag   != "全部": filtered = [n for n in filtered if filter_tag in n["tags"]]
        if filter_topic != "全部": filtered = [n for n in filtered if n["topic"] == filter_topic]
        if search_note:            filtered = [n for n in filtered if search_note.lower() in (n["content"]+n.get("question","")).lower()]

        st.caption(f"共 {len(filtered)} 条")
        for note in reversed(filtered):
            st.markdown('<div class="note-card">', unsafe_allow_html=True)
            h1,h2,h3 = st.columns([3,2,1])
            with h1:
                st.markdown(f'<span class="topic-badge">🗂️ {note["topic"]}</span>', unsafe_allow_html=True)
                for tag in note["tags"]:
                    st.markdown(f'<span class="note-tag">#{tag}</span>', unsafe_allow_html=True)
            with h2: st.caption(f"🕐 {note['ts']}")
            with h3:
                if st.button("🗑️", key=f"delnote_{note['id']}"):
                    st.session_state.notes = [n for n in st.session_state.notes if n["id"] != note["id"]]
                    st.rerun()
            if note.get("question"):
                st.markdown(f"**❓ {note['question']}**")
            st.markdown(note["content"])   # 完整内容，不截断
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑️ 清空所有笔记", type="secondary"):
            st.session_state.notes = []; st.rerun()






