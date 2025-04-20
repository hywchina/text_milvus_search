# streamlit_milvus_demo.py

import streamlit as st
import requests
import os

st.title("Milvus Text Search Demo")

# FastAPI 服务地址输入框
api_url = st.text_input(
    "FastAPI 服务地址（含端口）",
    value="http://localhost:8001",
    help="例如：http://localhost:8001，需和 uvicorn 启动时 --port 保持一致"
).rstrip("/")

# 参数输入区
query = st.text_input("查询文本", "")
search_type = st.selectbox("检索类型", ("dense", "sparse", "hybrid"))
limit = st.slider("返回结果数", 1, 20, 5)
sparse_weight = st.slider("稀疏权重 (只有 hybrid 生效)", 0.0, 1.0, 1.0)
dense_weight = st.slider("密集权重 (只有 hybrid 生效)", 0.0, 1.0, 1.0)

def search_milvus(query, search_type, limit, sparse_weight, dense_weight):
    payload = {
        "query": query,
        "limit": limit,
        "sparse_weight": sparse_weight,
        "dense_weight": dense_weight,
    }
    endpoint_map = {
        "dense": f"{api_url}/dense_search/",
        "sparse": f"{api_url}/sparse_search/",
        "hybrid": f"{api_url}/hybrid_search/",
    }
    try:
        resp = requests.post(endpoint_map[search_type], json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def aggregate_results(results):
    agg = {}
    for r in results:
        p = r["path"]
        if p not in agg or r["score"] > agg[p]["score"]:
            agg[p] = r
    return list(agg.values())

def highlight_text(query, text):
    h = text
    idx = h.lower().find(query.lower())
    if idx == -1:
        return h
    # 只做一次高亮，为了性能
    before, match, after = h[:idx], h[idx:idx+len(query)], h[idx+len(query):]
    return f"{before}<span style='color:red'>{match}</span>{after}"

@st.cache_data(show_spinner=False)
def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


if st.button("Search"):
    if not query:
        st.error("请输入查询文本后再检索。")
    else:
        res = search_milvus(query, search_type, limit, sparse_weight, dense_weight)
        if "error" in res:
            st.error(f"接口调用失败: {res['error']}")
        else:
            docs = aggregate_results(res["results"])
            st.success(f"共命中 {len(docs)} 篇文档")
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Result {i} — {os.path.basename(doc['path'])} (score: {doc['score']:.4f})"):
                    snippet = highlight_text(query, doc["text"])
                    st.markdown(snippet, unsafe_allow_html=True)
                    st.write(f"**Filename:** {doc.get('filename','N/A')}")
                    st.write(f"**Path:** {doc['path']}")
                    st.write(f"**Date:** {doc.get('date','N/A')}")

                    # 如果文件存在，使用 checkbox 来显示原文
                    file_path = doc.get("path")
                    if os.path.isfile(file_path):
                        try:
                            content = load_file(file_path)
                            st.text_area("原文内容", content, height=300)
                        except Exception as e:
                            st.error(f"读取原文失败：{e}")
                    else:
                        st.warning("原文文件不存在或路径无效")
                # with st.expander(f"Result {i} — {os.path.basename(doc['path'])} (score: {doc['score']:.4f})"):
                #     # 高亮片段
                #     snippet = highlight_text(query, doc["text"])
                #     st.markdown(snippet, unsafe_allow_html=True)
                #     st.write(f"**Filename:** {doc.get('filename','N/A')}")
                #     st.write(f"**Path:** {doc['path']}")
                #     st.write(f"**Date:** {doc.get('date','N/A')}")
                    
                #     # 原文嵌套 Expander
                #     if doc.get("path") and os.path.isfile(doc["path"]):
                #         with st.expander("查看原文"):
                #             try:
                #                 content = open(doc["path"], "r", encoding="utf-8").read()
                #             except Exception as e:
                #                 st.error(f"读取文件失败：{e}")
                #             else:
                #                 st.text_area("原文内容", content, height=300)
                #     else:
                #         st.warning("原文文件不存在或路径无效")
