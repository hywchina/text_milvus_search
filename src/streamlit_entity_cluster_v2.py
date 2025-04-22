import streamlit as st
import requests
import os
import numpy as np

# —— 页面配置 —— 
st.set_page_config(page_title="Milvus Text Search & Recommend", layout="wide")
st.title("Milvus Text Search & Recommend")

# —— 初始化 Session State —— 
for key, default in [
    ("liked_docs", set()),
    ("disliked_docs", set()),
    ("query_history", []),
    ("doc_embeddings", {}),
    ("search_results", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

@st.cache_data(show_spinner=False)
def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def highlight_text(query, text):
    idx = text.lower().find(query.lower())
    if idx == -1:
        return text
    before = text[:idx]
    match = text[idx:idx+len(query)]
    after = text[idx+len(query):]
    return f"{before}<span style='color:red'>{match}</span>{after}"

def search_milvus(query, search_type, limit, sparse_weight, dense_weight):
    payload = {
        "query": query,
        "limit": limit,
        "sparse_weight": sparse_weight,
        "dense_weight": dense_weight,
    }
    endpoints = {
        "dense": f"{api_url}/dense_search/",
        "sparse": f"{api_url}/sparse_search/",
        "hybrid": f"{api_url}/hybrid_search/",
    }
    try:
        r = requests.post(endpoints[search_type], json=payload, timeout=10)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        st.error(f"检索接口调用失败：{e}")
        return []

def aggregate_results(results):
    agg = {}
    for r in results:
        p = r["path"]
        if p not in agg or r["score"] > agg[p]["score"]:
            agg[p] = r
    return list(agg.values())

def fetch_doc_embedding(path):
    """
    读取整个文件内容，调用 POST /embed_dense 获取 1024 维稠密向量并缓存。
    """
    if path in st.session_state.doc_embeddings:
        return st.session_state.doc_embeddings[path]

    # 1) 先读文本
    try:
        text = load_file(path)
    except Exception as e:
        st.warning(f"读取文件失败，无法获取 embedding: {e}")
        return None

    # 2) 调用 embed_dense 接口
    EMBEDDING_API_URL = "http://localhost:8001"
    try:
        resp = requests.post(
            f"{EMBEDDING_API_URL}/embed_dense",
            json={"text": text},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        dense = data.get("dense")
        if not isinstance(dense, list):
            st.warning(f"接口发回格式异常: {data}")
            return None
        emb = np.array(dense, dtype=float)
        # 校验维度
        if emb.shape[0] != 1024:
            st.warning(f"警告: 返回向量维度 {emb.shape[0]}，预期 1024")
        st.session_state.doc_embeddings[path] = emb
        return emb

    except Exception as e:
        st.warning(f"获取 embedding 失败: {e}")
        return None

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0

def get_recommendations(top_k: int = 5):
    liked = list(st.session_state.liked_docs)
    if not liked:
        return []
    embs = [fetch_doc_embedding(p) for p in liked]
    embs = [e for e in embs if e is not None]
    if not embs:
        return []
    user_vec = np.mean(embs, axis=0)
    candidates = {
        p: emb for p, emb in st.session_state.doc_embeddings.items()
        if p not in st.session_state.liked_docs and p not in st.session_state.disliked_docs
    }
    scored = [(cosine_similarity(user_vec, emb), p) for p, emb in candidates.items()]
    scored.sort(reverse=True, key=lambda x: x[0])
    return [p for _, p in scored[:top_k]]

# —— UI：FastAPI 地址 & 参数 —— 
api_url = st.text_input(
    "FastAPI 服务地址（含端口）",
    value="http://localhost:8002",
    help="用于检索的服务和 embed_dense 服务都应挂在此地址下"
).rstrip("/")

query = st.text_input("查询文本", "")
search_type = st.selectbox("检索类型", ("dense", "sparse", "hybrid"))
limit = st.slider("返回结果数", 1, 20, 5)
sparse_weight = st.slider("稀疏权重 (hybrid 有效)", 0.0, 1.0, 1.0)
dense_weight = st.slider("密集权重 (hybrid 有效)", 0.0, 1.0, 1.0)

# —— 搜索按钮 —— 
if st.button("Search"):
    if not query.strip():
        st.error("请输入查询文本后再检索。")
    else:
        raw = search_milvus(query, search_type, limit, sparse_weight, dense_weight)
        docs = aggregate_results(raw)
        st.session_state.search_results = docs
        if query not in st.session_state.query_history:
            st.session_state.query_history.append(query)
        # 预取 embedding
        for doc in docs:
            fetch_doc_embedding(doc["path"])

# —— 主区：展示搜索结果 —— 
if st.session_state.search_results:
    st.success(f"共命中 {len(st.session_state.search_results)} 篇文档")
    for doc in st.session_state.search_results:
        title = os.path.basename(doc["path"])
        score = doc.get("score", 0.0)
        with st.expander(f"{title} (score: {score:.4f})"):
            snippet = highlight_text(query, doc.get("text", ""))
            st.markdown(snippet, unsafe_allow_html=True)
            st.write(f"**Path:** {doc['path']}")
            st.write(f"**Date:** {doc.get('date','N/A')}")
            if os.path.isfile(doc["path"]):
                try:
                    content = load_file(doc["path"])
                    st.text_area("原文内容", content, height=300)
                except Exception as e:
                    st.error(f"读取原文失败：{e}")
            else:
                st.warning("原文文件不存在或路径无效")
            c1, c2 = st.columns([1,1])
            with c1:
                if st.button("👍 喜欢", key=f"like_{doc['path']}"):
                    st.session_state.liked_docs.add(doc["path"])
                    st.session_state.disliked_docs.discard(doc["path"])
            with c2:
                if st.button("👎 不喜欢", key=f"dislike_{doc['path']}"):
                    st.session_state.disliked_docs.add(doc["path"])
                    st.session_state.liked_docs.discard(doc["path"])

# —— 侧边栏：猜你喜欢 + 历史查询 —— 
with st.sidebar:
    st.subheader("🧠 猜你喜欢")
    recs = get_recommendations()
    if recs:
        for p in recs:
            st.markdown(f"- **{os.path.basename(p)}**  \n`{p}`")
    else:
        st.write("暂无推荐，先标记一些“喜欢”的文档吧！")

    st.markdown("---")
    st.subheader("🔍 历史查询")
    for q in reversed(st.session_state.query_history):
        st.write(f"- {q}")
    if st.button("🧹 清除历史记录"):
        st.session_state.query_history.clear()
