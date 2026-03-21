import streamlit as st
import pandas as pd
import os, time, tempfile, re, math, uuid, itertools, io
import arxiv, requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_agraph import agraph, Node, Edge, Config

# ================= 1. 环境检查与导入 =================
try:
    import langchain_community, fitz
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError as e:
    st.error(f"🚑 环境缺失库 -> {e.name}. 请运行: pip install langchain-openai sentence-transformers pymupdf pypdf")
    st.stop()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= 2. 页面配置 =================
st.set_page_config(page_title="AI 深度研读助手 (DeepSeek版)", layout="wide", page_icon="🎓")
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

st.title("📖 AI 深度研读助手 v5 (DeepSeek Kernel)")

# ================= API Key =================
try:
    USER_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except:
    USER_API_KEY = ""

SS_API_KEY = st.secrets.get("SS_API_KEY", "")

# ================= 3. 状态初始化 =================
defaults = {
    "search_results":         [],
    "search_generator":       None,
    "citations_loaded":       False,
    "citations_global_cache": {},
    "suggested_query":        "",
    "focus_paper_id":         None,
    "contributions_cache":    {},
    "score_cache":            {}, 
    "chat_history":           [],
    "topics":                 {"默认主题": {"files": [], "chunks": [], "db": None}},
    "active_topic":           "默认主题",
    "selected_scope":         "🌐 对比所有论文",
    "notes":                  [],
    "pending_note":           None,
    "graph_references_cache": [],
    "preload_done_ids":       set(),
    "trackers":               {},
    "tracker_total_new":      0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================= 4. 工具函数 =================

# --- 修改点 1：新增 Mock 包装类，无缝将 SS 数据伪装成原来的 ArXiv 格式，防止破坏下层 UI 渲染 ---
class MockArxivResult:
    def __init__(self, ss_data):
        self.title = ss_data.get('title') or '无标题'
        # 伪装 authors 列表，让 a.name 可以正常调用
        self.authors = [type('Author', (), {'name': a.get('name', '未知作者')}) for a in ss_data.get('authors', [])]
        
        # 处理时间格式兼容
        pub_date = ss_data.get('publicationDate')
        if pub_date:
            try:
                self.published = datetime.strptime(pub_date, "%Y-%m-%d")
            except:
                self.published = datetime(ss_data.get('year') or 2000, 1, 1)
        else:
            self.published = datetime(ss_data.get('year') or 2000, 1, 1)
            
        self.summary = ss_data.get('abstract') or "该论文暂无摘要信息。"
        
        # 链接处理：优先用 ArXiv 原生链接，没有就用 SS 链接
        ext_ids = ss_data.get('externalIds', {})
        arxiv_id = ext_ids.get('ArXiv')
        if arxiv_id:
            self.entry_id = f"https://arxiv.org/abs/{arxiv_id}"
        else:
            self.entry_id = ss_data.get('url') or f"https://www.semanticscholar.org/paper/{ss_data.get('paperId')}"
            
        self.ss_paper_id = ss_data.get('paperId')
        self.open_access_pdf = ss_data.get('openAccessPdf') or {}
        self.citation_count = ss_data.get('citationCount', 0)

# --- 修改点 2：新增 SS API 专属生成器，完全替代原来的 arxiv.Search().results() ---
def ss_search_generator(query, api_key, sort_mode):
    offset = 0
    limit = 50
    while True:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        headers = {"x-api-key": api_key} if api_key else {}
        params = {
            "query": query,
            "offset": offset,
            "limit": limit,
            "fields": "title,authors,year,publicationDate,abstract,externalIds,url,openAccessPdf,citationCount"
        }
        # SS 排序映射
        if "最新" in sort_mode:
            params["sort"] = "publicationDate:desc"
        elif "引用量" in sort_mode:
            params["sort"] = "citationCount:desc"
            
        try:
            r = requests.get(url, headers=headers, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json().get('data', [])
                if not data: break
                for item in data:
                    yield MockArxivResult(item)
                
                # 如果超出了总数或 API 最大便宜限制，停止
                if offset + limit >= r.json().get('total', 0) or offset >= 9900:
                    break
                offset += limit
            elif r.status_code == 429:
                st.warning("⚠️ 请求过于频繁，请稍后再试（拉取下一页已暂停）。")
                break
            else:
                break
        except Exception as e:
            st.warning(f"搜索发生异常: {e}")
            break

# --- 修改点 3：增强 PDF 下载逻辑，保留原来 ArXiv 直链的同时，支持 SS 的免费 PDF 链接 ---
def download_arxiv_pdf_direct(url_or_id, open_access_url=None):
    # 如果是 ArXiv 的，继续严格按照历史要求拼接原生直链
    if "arxiv.org" in url_or_id or re.match(r'^\d{4}\.\d{4,5}', url_or_id):
        clean_id = get_pure_arxiv_id(url_or_id)
        pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
        pdf_path = os.path.join(tempfile.gettempdir(), f"{clean_id}.pdf")
        r = requests.get(pdf_url, timeout=15)
        with open(pdf_path, 'wb') as f:
            f.write(r.content)
        return pdf_path
    # 如果不是 ArXiv，尝试使用 SS 给的免费开源下载链接
    elif open_access_url:
        pdf_path = os.path.join(tempfile.gettempdir(), f"ss_pdf_{uuid.uuid4().hex[:8]}.pdf")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        r = requests.get(open_access_url, headers=headers, timeout=20)
        with open(pdf_path, 'wb') as f:
            f.write(r.content)
        return pdf_path
    else:
        raise Exception("该论文既非 ArXiv 论文，也无 Semantic Scholar 开源 PDF 链接，暂不支持一键下载。")

def get_deepseek_llm(api_key, temperature=0.1):
    if not api_key:
        st.error("请先配置 DEEPSEEK_API_KEY"); st.stop()
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=temperature
    )

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def active_topic_data():
    return st.session_state.topics[st.session_state.active_topic]

def get_pure_arxiv_id(url_or_id):
    m = re.search(r'(\d{4}\.\d{4,5})', url_or_id)
    return m.group(1) if m else url_or_id.split('/')[-1].split('v')[0]

def convert_to_excel(results):
    data = []
    for item in results:
        res = item['obj']
        contrib = st.session_state.contributions_cache.get(res.title[:60], "未生成")
        data.append({
            "标题": res.title,
            "作者": ", ".join([a.name for a in res.authors]),
            "年份": res.published.year,
            "引用数": item.get('citations', 0),
            "核心贡献 (AI)": contrib,
            "链接": res.entry_id,
            "摘要": res.summary.replace('\n', ' ')
        })
    
    df = pd.DataFrame(data)
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='检索结果')
        workbook  = writer.book
        worksheet = writer.sheets['检索结果']
        
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        cell_fmt   = workbook.add_format({'border': 1, 'valign': 'top', 'text_wrap': True})
        num_fmt    = workbook.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter'})
        
        worksheet.set_column('A:A', 40, cell_fmt)
        worksheet.set_column('B:B', 20, cell_fmt)
        worksheet.set_column('C:D', 10, num_fmt)
        worksheet.set_column('E:E', 50, cell_fmt)
        worksheet.set_column('F:F', 30, cell_fmt)
        worksheet.set_column('G:G', 60, cell_fmt)
        
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_fmt)
            
    return output.getvalue()

def auto_batch_contributions(results, api_key, limit=50):
    to_process = results[:limit]
    pending = [p for p in to_process if p['obj'].title[:60] not in st.session_state.contributions_cache]
    
    if not pending:
        return
    
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(get_one_line_contribution, p['obj'].summary, p['obj'].title, api_key): p 
            for p in pending
        }
        for future in as_completed(futures):
            try:
                future.result()
            except:
                pass

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

@st.cache_data(ttl=3600)
def fetch_ss_paper_details_by_title(title, ss_key=None):
    clean_title = title.replace(".pdf", "").strip()
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={clean_title}&limit=1&fields=title,year,authors,citationCount,influentialCitationCount,tldr,venue"
    headers = {"x-api-key": ss_key} if ss_key else {}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                return data[0]
    except: pass
    return None

def get_one_line_contribution(abstract, title, api_key):
    key = title[:60]
    if key in st.session_state.contributions_cache:
        return st.session_state.contributions_cache[key]
    try:
        llm = get_deepseek_llm(api_key, temperature=0.0)
        res = llm.invoke(
            f"请用一句话（不超过40个汉字或20个英文单词）总结这篇论文的核心创新贡献。"
            f"只输出这一句话，不要前缀或解释。\n\n标题：{title}\n摘要：{abstract[:600]}"
        )
        result = res.content.strip()
    except Exception as e: 
        result = "（生成失败）"
    st.session_state.contributions_cache[key] = result
    return result

def get_paper_score(arxiv_id, title, abstract, api_key, ss_key):
    key = title[:60]
    if key in st.session_state.score_cache:
        return st.session_state.score_cache[key]
    try:
        clean_id = get_pure_arxiv_id(arxiv_id)
        url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields=tldr,influentialCitationCount"
        headers = {"x-api-key": ss_key} if ss_key else {}
        ss_info = ""
        try:
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                tldr = data.get("tldr", {}).get("text", "无") if data.get("tldr") else "无"
                inf_cites = data.get("influentialCitationCount", 0)
                ss_info = f"\n\nSemantic Scholar真实数据：\n- 极具影响力引用数: {inf_cites}\n- TLDR极简摘要: {tldr}"
        except: pass

        llm = get_deepseek_llm(api_key, temperature=0.1)
        prompt = (
            f"请根据这篇论文的信息，结合其学术价值、潜力和提供的真实影响力指标，对其进行综合打分（满分100分）。\n"
            f"请严格按照此格式输出：【xx分】一句话理由（不超过30字）。\n\n"
            f"标题：{title}\n摘要：{abstract[:600]}{ss_info}"
        )
        res = llm.invoke(prompt)
        result = res.content.strip()
    except Exception as e:
        result = "（打分失败）"
    st.session_state.score_cache[key] = result
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
        
        embeddings = get_embeddings_model()
        
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
    
    embeddings = get_embeddings_model()
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
# --- 修改点 4：追踪器也替换为底层调用 SS 接口 ---
def tracker_check_one(keyword: str, since_date: str | None = None) -> list:
    try:
        cutoff = datetime.fromisoformat(since_date) if since_date else datetime.now() - timedelta(days=7)
        
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        headers = {"x-api-key": st.secrets.get("SS_API_KEY", "")}
        params = {
            "query": keyword,
            "limit": 30,
            "fields": "title,authors,year,publicationDate,abstract,externalIds,url,openAccessPdf",
            "sort": "publicationDate:desc"
        }
        r = requests.get(url, headers=headers, params=params, timeout=15)
        out = []
        if r.status_code == 200:
            for item in r.json().get('data', []):
                mock_res = MockArxivResult(item)
                if mock_res.published.replace(tzinfo=None) > cutoff:
                    out.append({
                        "title":     mock_res.title,
                        "authors":   ", ".join([a.name for a in mock_res.authors]),
                        "published": mock_res.published.strftime("%Y-%m-%d"),
                        "summary":   mock_res.summary,
                        "entry_id":  mock_res.entry_id,
                        "obj":       mock_res,
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

if st.session_state.trackers:
    tracker_run_all(force=False)

# ================= 5. 图谱渲染 =================
def render_connected_graph(data, min_cite_filter=0):
    if not data: return None, {}
    nodes, edges, details = [], [], {}
    cur_year = 2026

    def color(year, rel):
        if not year or year == 'Unknown': return "#94a3b8"
        try:
            age = max(0, cur_year - int(year))
        except:
            age = 5
        if rel == 'seed': return "#FF4B4B"
        if rel == 'cite': return "#059669" if age<2 else "#10b981" if age<5 else "#6ee7b7"
        return "#2563eb" if age<2 else "#3b82f6" if age<5 else "#93c5fd"

    # 确保 ID 是字符串，避免 streamlit-agraph 报错
    seed = str(data.get('paperId','root'))
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
        pid   = str(item.get('paperId'))
        cites = item.get('citationCount',0) or 0
        if not pid or pid == 'None' or pid in seen or cites < min_cite_filter: continue
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
        # size 必须为 int，不能为 float
        sz = int(15 + math.log(cites+1)*3.5)
        nodes.append(Node(id=pid, label=f"{title[:20]}…", size=sz, color=color(year, item['rel_type'])))
        if item['rel_type']=='cite':
            # 移除 dashed=True，因为部分版本的 Edge 不支持此参数且会导致 TypeError
            edges.append(Edge(source=pid, target=seed, color="#d1d5db", width=1))
        else:
            edges.append(Edge(source=seed, target=pid, color="#94a3b8", width=1.5))

    st.session_state.graph_references_cache = refs_for_gap
    # width 必须为 int 类型，不可传 "100%" 字符串，否则会触发 Python 与 JS 桥接层的 TypeError
    cfg = Config(width=800, height=560, directed=True, physics=True,
                 nodeHighlightBehavior=True, highlightColor="#F7D154",
                 d3={'alphaTarget':0.05,'gravity':-250,'linkLength':150,'linkStrength':0.1})
    
    unique_key = f"graph_{seed}_{min_cite_filter}_{len(nodes)}"
    clicked = agraph(nodes=nodes, edges=edges, config=cfg, key=unique_key)
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

tab_main, tab_read, tab_track, tab_notes = st.tabs([
    "🔍 学术检索 & 图谱", "📖 研读空间", _track_label, "📌 我的笔记"
])

# ══════════════════════════════════════════
# Tab 1：学术检索 & 图谱
# ══════════════════════════════════════════
with tab_main:
    st.markdown('<div class="section-divider">🌍 学术检索</div>', unsafe_allow_html=True)
    
    sq1,sq2 = st.columns([4,2])
    with sq1:
        search_query = st.text_input("关键词", value=st.session_state.suggested_query,
                                     placeholder="例如: education robot", label_visibility="collapsed")
    with sq2:
        sort_mode = st.selectbox("排序",["🔥 相关性","📅 最新","📈 引用量"], label_visibility="collapsed")

    if st.button("🚀 检索", use_container_width=True) and search_query:
        with st.spinner("检索论文中 (使用 Semantic Scholar)..."):
            try:
                # --- 修改点 5：彻底使用刚刚写好的 SS 生成器，丢弃 arxiv.Search ---
                raw_gen = ss_search_generator(search_query, ss_api_key, sort_mode)
                st.session_state.search_generator = raw_gen
                raw = list(itertools.islice(raw_gen, 50))
                
                # 因为 SS API 已经携带了 citationCount，我们可以直接填入
                st.session_state.search_results = [{"obj":r, "citations": r.citation_count} for r in raw]
                st.session_state.citations_loaded = False
                st.session_state.contributions_cache = {}
                st.session_state.focus_paper_id = None
            except Exception as e: st.error(f"检索失败: {e}")

        if st.session_state.search_results:
            t0 = time.time()
            with st.spinner("处理数据展示..."):
                # --- 修改点 6：不删除你的老代码，通过注释安全跳过。因为 SS 自带引用数，无需再二次请求了 ---
                # id2c = smart_fetch_citations(st.session_state.search_results, ss_key=ss_api_key)
                # for item in st.session_state.search_results:
                #     item["citations"] = id2c.get(item['obj'].entry_id, 0)
                
                if "引用量" in sort_mode:
                    st.session_state.search_results.sort(key=lambda x: x["citations"] or 0, reverse=True)
                
                st.session_state.citations_loaded = True
            
            st.success(f"✅ 首批 {len(st.session_state.search_results)} 篇完成，耗时 {time.time()-t0:.1f}s")
            
    # ── 图谱区 ──
    if st.session_state.focus_paper_id:
        st.markdown('<div class="section-divider">📊 文献关联图谱</div>', unsafe_allow_html=True)
        min_cf = st.slider("最低引用数过滤", 0, 200, 5, step=1, key="graph_cite_filter")
        with st.spinner("加载图谱…"):
            g_data = fetch_graph_data(st.session_state.focus_paper_id, ss_key=ss_api_key)

        if not g_data:
            st.warning("⚠️ 暂时无法获取图谱，请稍后再试。")
        else:
            gc_graph, gc_info = st.columns([1.6, 1])
            with gc_graph:
                clicked_id, all_details = render_connected_graph(g_data, min_cite_filter=min_cf)
                sid = g_data.get('paperId','root')
                if sid in all_details:
                    all_details[sid]['arxiv_id'] = get_pure_arxiv_id(st.session_state.focus_paper_id)
                if not (clicked_id and clicked_id in all_details):
                    st.caption("👆 点击节点 → 右侧看完整详情 | 🔴 当前  🟢 引用本文  🔵 本文引用")

            with gc_info:
                if clicked_id and clicked_id in all_details:
                    info = all_details[clicked_id]
                    st.markdown(
                        f"""
                        <div style="background:#f8fafc;border:1px solid #e2e8f0;
                                    border-left:4px solid #6366f1;border-radius:10px;padding:14px 16px;">
                            <div style="font-size:.93em;font-weight:700;color:#1e293b;
                                        margin-bottom:10px;line-height:1.4;">
                                📑 {info['title']}
                            </div>
                            <div style="display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap;">
                                <span style="background:#e0e7ff;color:#3730a3;padding:2px 10px;
                                             border-radius:12px;font-size:.8em;font-weight:600;">
                                    📅 {info['year'] or '年份未知'}
                                </span>
                                <span style="background:#fee2e2;color:#991b1b;padding:2px 10px;
                                             border-radius:12px;font-size:.8em;font-weight:600;">
                                    🔥 {info['cites']} 引用
                                </span>
                            </div>
                            <div style="font-size:.82em;color:#475569;
                                        max-height:320px;overflow-y:auto;line-height:1.65;">
                                {info['abstract']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True,
                    )
                    st.markdown("")
                    target_topic = st.selectbox(
                        "加入主题", list(st.session_state.topics.keys()),
                        index=list(st.session_state.topics.keys()).index(st.session_state.active_topic),
                        key="graph_topic_sel", label_visibility="collapsed"
                    )
                    arxiv_id = info.get('arxiv_id')
                    ga,gb,gc = st.columns(3)
                    with ga:
                        if arxiv_id and st.button("⬇️ 入库", type="primary", use_container_width=True, key="ginfo_dl"):
                            with st.spinner("下载中..."):
                                try:
                                    pdf_path = download_arxiv_pdf_direct(arxiv_id)
                                    if process_and_add_to_topic(pdf_path, info['title'], user_api_key, topic_name=target_topic):
                                        st.success("✅ 入库成功！"); st.balloons()
                                except Exception as e: st.error(str(e))
                        elif not arxiv_id: st.caption("暂无全文")
                    with gb:
                        st.link_button("🌐 SS", info['url'], use_container_width=True)
                    with gc:
                        if info.get('arxiv_id') and st.button("🕸️ 展开", use_container_width=True, key="ginfo_expand"):
                            st.session_state.focus_paper_id = info['arxiv_id']; st.rerun()
                else:
                    st.markdown(
                        """<div style="background:#f1f5f9;border:1px dashed #cbd5e1;border-radius:10px;
                                      padding:40px 16px;text-align:center;color:#94a3b8;font-size:.88em;
                                      min-height:260px;display:flex;align-items:center;justify-content:center;">
                            ← 点击左侧节点<br>查看完整详情</div>""",
                        unsafe_allow_html=True,
                    )

    # ── 检索结果列表 ──
    if st.session_state.search_results:
        st.markdown(
            f'<div class="section-divider">📋 检索结果（已加载 {len(st.session_state.search_results)} 篇）'
            f'<span class="perf-badge">⚡ 缓存 {len(st.session_state.citations_global_cache)} 篇</span></div>',
            unsafe_allow_html=True
        )

        st.download_button(
            label="📥 点击下载当前已加载论文 (Excel)",
            data=convert_to_excel(st.session_state.search_results),
            file_name=f"ArXiv_Search_{datetime.now().strftime('%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.markdown("---")
        
        for i, item in enumerate(st.session_state.search_results):
            res   = item['obj']
            cites = item['citations']
            cite_html = (f"<span class='cite-badge'>{cites}</span>" if cites is not None
                         else "<span class='cite-loading'>加载中…</span>")
            with st.expander(f"#{i+1} {res.title} ({res.published.year})"):
                st.markdown(
                    f"**{', '.join([a.name for a in res.authors])}** | "
                    f"{res.published.strftime('%Y-%m-%d')} | 引用：{cite_html}",
                    unsafe_allow_html=True
                )
                ck = res.title[:60]
                cc, cg, cs = st.columns([4,1,1])
                with cc:
                    box_content = ""
                    if ck in st.session_state.contributions_cache:
                        box_content += f"💡 <b>贡献:</b> {st.session_state.contributions_cache[ck]}<br>"
                    else:
                        box_content += "💡 <span style='color:#aaa;'>点击右侧 ✨ 生成核心贡献摘要</span><br>"
                        
                    if ck in st.session_state.score_cache:
                        box_content += f"⭐ <b>评分:</b> {st.session_state.score_cache[ck]}"
                    else:
                        box_content += "⭐ <span style='color:#aaa;'>点击右侧 ⭐ 获取SS综合打分</span>"
                        
                    st.markdown(f'<div class="contribution-box">{box_content}</div>', unsafe_allow_html=True)
                with cg:
                    if st.button("✨ 摘要", key=f"contrib_{i}"):
                        with st.spinner("分析..."): get_one_line_contribution(res.summary, res.title, user_api_key)
                        st.rerun()
                with cs:
                    if st.button("⭐ 打分", key=f"score_{i}"):
                        with st.spinner("获取SS数据并打分..."): get_paper_score(res.entry_id, res.title, res.summary, user_api_key, ss_api_key)
                        st.rerun()
                st.markdown(f'<div class="abstract-box"><b>摘要：</b>{res.summary.replace(chr(10)," ")}</div>', unsafe_allow_html=True)
                b1,b2,b3 = st.columns(3)
                # --- 修改点 7：把写死的 ArXiv 文字改成了通用“来源”，因为部分来自 SS 纯净库 ---
                with b1: st.markdown(f"[🔗 原文来源]({res.entry_id})")
                with b2:
                    if st.button("⬇️ 下载入库", key=f"dl_{i}"):
                        with st.spinner("下载解析..."):
                            try:
                                # --- 修改点 8：将 SS 的 open_access 链接也传进去，如果不是 ArXiv 也能下载 ---
                                pdf_path = download_arxiv_pdf_direct(res.entry_id, res.open_access_pdf.get('url'))
                                process_and_add_to_topic(pdf_path, res.title, user_api_key)
                                st.success("入库成功！")
                            except Exception as e: st.error(str(e))
                with b3:
                    lbl = "🕸️ 图谱 ⚡" if res.entry_id in st.session_state.preload_done_ids else "🕸️ 图谱"
                    if st.button(lbl, key=f"graph_{i}"):
                        st.session_state.focus_paper_id = res.entry_id; st.rerun()
        
    if st.session_state.search_generator:
        st.markdown("---")
        if st.button("🔽 加载更多 50 篇...", use_container_width=True):
            with st.spinner("正在拉取新论文摘要..."):
                more_raw = list(itertools.islice(st.session_state.search_generator, 50))
                
                if more_raw:
                    # --- 修改点 9：加载更多时，也只需取生成器吐出来的 citation_count ---
                    new_results = [{"obj": r, "citations": r.citation_count} for r in more_raw]
                    st.session_state.search_results.extend(new_results)
                    
                    if "引用量" in sort_mode:
                        st.session_state.search_results.sort(key=lambda x: x["citations"] or 0, reverse=True)
                    
                    st.rerun()
                else:
                    st.info("✨ 到底啦，没有更多匹配的论文了。")
# ══════════════════════════════════════════
# Tab 2：研读空间
# ══════════════════════════════════════════
with tab_read:
    t = active_topic_data()
    
    c_ctrl, c_info = st.columns([2, 3])
    with c_ctrl:
        scope_options = ["🌐 全库综合 (对比/综述)"] + t["files"]
        selected_scope = st.selectbox(
            "📚 阅读范围 (Scope)", 
            scope_options,
            index=0,
            help="选择“全库”进行跨论文对比；选择“单篇”将只检索该论文内容，更省 Token 且更精准。"
        )
    with c_info:
        if selected_scope == "🌐 全库综合 (对比/综述)":
            st.caption(f"🚀 当前模式：检索所有 {len(t['files'])} 篇论文")
        else:
            st.caption(f"🎯 当前模式：**专注研读** (已屏蔽其他论文干扰)")

    st.divider()

    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown(f"👋 我是你的 DeepSeek 研读助手。当前主题库中有 **{len(t['files'])}** 篇论文。")
            if t["files"]:
                st.markdown("你可以问我：\n- *这篇论文的核心方法是什么？* (建议选择单篇范围)\n- *对比 Transformer 和 CNN 在这些论文中的观点差异* (建议选择全库范围)")
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("输入问题..."):
        if not t["db"]:
            st.error("请先在左侧上传 PDF 或从检索页下载论文入库！")
            st.stop()

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                search_kwargs = {}
                filter_rule = None
                
                if selected_scope != "🌐 全库综合 (对比/综述)":
                    filter_rule = {"source_paper": selected_scope}
                    search_kwargs = {"k": 20, "filter": filter_rule}
                    status_text = f"🔍 正在深度扫描论文《{selected_scope}》..."
                else:
                    search_kwargs = {"k": 15, "fetch_k": 50, "lambda_mult": 0.6}
                    status_text = f"🔍 正在全库 {len(t['files'])} 篇论文中检索..."

                with st.spinner(status_text):
                    if filter_rule:
                        docs = t["db"].similarity_search(prompt, **search_kwargs)
                        ss_data = fetch_ss_paper_details_by_title(selected_scope, ss_api_key)
                    else:
                        docs = t["db"].max_marginal_relevance_search(prompt, **search_kwargs)
                        ss_data = None

                if not docs:
                    full_response = "⚠️ 未在文档中检索到相关信息，请尝试更换关键词或检查文档是否完整。"
                    message_placeholder.markdown(full_response)
                else:
                    context_text = ""
                    refs = []
                    for i, d in enumerate(docs):
                        src = d.metadata.get('source_paper', '未知')
                        page = d.metadata.get('page', 0) + 1 
                        snippet = d.page_content.replace('\n', ' ')
                        context_text += f"[资料{i+1} | {src} (P{page})]: {snippet}\n\n"
                        refs.append(f"**[{i+1}] {src} (P{page})**: {snippet[:100]}...")
                        
                    ss_context = ""
                    if ss_data:
                        tldr = ss_data.get('tldr', {}).get('text', '无') if ss_data.get('tldr') else '无'
                        authors = ", ".join([a.get('name', '') for a in ss_data.get('authors', [])])
                        ss_context = (
                            f"\n\n【Semantic Scholar 真实元数据补充】\n"
                            f"- 标题: {ss_data.get('title')}\n"
                            f"- 作者: {authors}\n"
                            f"- 发表年份/会议: {ss_data.get('year')} / {ss_data.get('venue')}\n"
                            f"- 总引用数: {ss_data.get('citationCount')} (其中极具影响力引用: {ss_data.get('influentialCitationCount')})\n"
                            f"- 官方TLDR摘要: {tldr}\n"
                            f"**指令**：你在回答背景、影响力和核心结论时，必须优先以这部分的真实元数据为准，不要编造不存在的事实。\n"
                        )

                    if selected_scope != "🌐 全库综合 (对比/综述)":
                        sys_prompt = (
                            f"你正在辅助用户精读论文《{selected_scope}》。\n"
                            "请利用提供的[资料片段]回答问题。\n"
                            "要求：\n"
                            "1. 回答要深入、具体，多引用数据或具体算法步骤。\n"
                            "2. 如果资料中包含公式描述，请还原为 LaTeX 格式。\n"
                            "3. 严禁编造资料中不存在的内容。"
                            f"{ss_context}"
                        )
                    else:
                        sys_prompt = (
                            "你是一名学术顾问。请综合提供的多篇论文资料回答问题。\n"
                            "要求：\n"
                            "1. 必须明确指出不同观点分别来自哪篇论文（如：‘Paper A 提出了...而 Paper B 则认为...’）。\n"
                            "2. 如果涉及对比，请使用 Markdown 表格形式展示。\n"
                            "3. 保持客观中立。"
                        )

                    with st.expander("📚 查看 AI 参考的原文片段 (Sources)", expanded=False):
                        st.markdown("\n\n".join(refs))

                    llm = get_deepseek_llm(USER_API_KEY, temperature=0.3)
                    
                    messages = [
                        {"role": "system", "content": f"{sys_prompt}\n\n### 检索到的资料：\n{context_text}"},
                        {"role": "user", "content": prompt}
                    ]
                    
                    stream = llm.stream(messages)
                    
                    for chunk in stream:
                        if chunk.content:
                            full_response += chunk.content
                            message_placeholder.markdown(full_response + "▌") 
                    
                    message_placeholder.markdown(full_response) 

                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
                st.session_state.pending_note = {
                    "content": full_response, 
                    "question": prompt,
                    "has_gap": detect_knowledge_gap(full_response, docs)
                }

            except Exception as e:
                st.error(f"发生错误: {str(e)}")

    if st.session_state.pending_note:
        st.markdown("---")
        c_note_1, c_note_2 = st.columns([5, 1])
        with c_note_1:
            note_tags = st.text_input("给刚才的回答加个标签？(可选)", placeholder="例如: 核心算法, 实验结果", key="quick_note_tag")
        with c_note_2:
            st.write("") 
            if st.button("📌 存笔记"):
                tags = [t.strip() for t in note_tags.split(",")] if note_tags else []
                st.session_state.notes.append({
                    "id": str(uuid.uuid4())[:8],
                    "content": st.session_state.pending_note["content"],
                    "question": st.session_state.pending_note["question"],
                    "tags": tags,
                    "topic": st.session_state.active_topic,
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
                })
                st.session_state.pending_note = None
                st.toast("笔记保存成功！", icon="🎉")
                st.rerun()
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
                    # --- 修改点 10：这里的 tracker 来源也改为了通用文字 ---
                    with pb1: st.markdown(f"[🔗 原文来源]({paper['entry_id']})")
                    with pb2:
                        if st.button("⬇️ 入库", key=f"tr_dl_{paper['entry_id']}"):
                            with st.spinner("下载中…"):
                                try:
                                    # --- 修改点 11：支持传递 SS 的开放下载链接，同时兼容你的 ArXiv 直链逻辑 ---
                                    obj = paper.get("obj")
                                    open_access_url = obj.open_access_pdf.get('url') if obj else None
                                    pdf_path = download_arxiv_pdf_direct(paper['entry_id'], open_access_url)
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

# ================= 4. 我的笔记 =================
with tab_notes:
    st.subheader("📌 我的笔记库")
    if not st.session_state.notes:
        st.info("还没有笔记。在「研读空间」提问后点击「保存为笔记」即可积累。")
    else:
        all_tags   = sorted(set(tag for n in st.session_state.notes for tag in n["tags"]))
        all_topics = sorted(set(n["topic"] for n in st.session_state.notes))
        f1,f2,f3 = st.columns([2,2,2])
        with f1: filter_tag    = st.selectbox("按标签",["全部"]+all_tags)
        with f2: filter_topic  = st.selectbox("按主题",["全部"]+all_topics)
        with f3: search_note   = st.text_input("关键词", placeholder="搜索笔记")

        filtered = st.session_state.notes.copy()
        if filter_tag    != "全部": filtered = [n for n in filtered if filter_tag in n["tags"]]
        if filter_topic  != "全部": filtered = [n for n in filtered if n["topic"] == filter_topic]
        if search_note:             filtered = [n for n in filtered if search_note.lower() in (n["content"]+n.get("question","")).lower()]

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
            st.markdown(note["content"]) 
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑️ 清空所有笔记", type="secondary"):
            st.session_state.notes = []; st.rerun()
