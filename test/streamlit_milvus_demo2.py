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
    """调用 FastAPI 接口并做统一的异常处理，返回 dict"""
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

    url = endpoint_map[search_type]
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": f"请求失败：{e}"}

    try:
        return resp.json()
    except ValueError:
        return {"error": f"无法解析返回值为 JSON：\n{resp.text}"}

def aggregate_results(results):
    """根据路径（path）聚合同一文档的多个 chunk，保留得分最高的 chunk"""
    aggregated = {}
    for result in results:
        path = result['path']
        if path not in aggregated:
            aggregated[path] = result
        else:
            # 比较得分，保留得分更高的 chunk
            if result['score'] > aggregated[path]['score']:
                aggregated[path] = result
    return list(aggregated.values())

def highlight_text(query, text):
    """高亮查询结果中的文本"""
    highlighted = text
    start = 0
    # 查找每次出现查询词的位置并替换成高亮显示的HTML标签
    while start < len(highlighted):
        start = highlighted.find(query, start)
        if start == -1:
            break
        end = start + len(query)
        highlighted = highlighted[:start] + "<span style='color:red'>" + highlighted[start:end] + "</span>" + highlighted[end:]
        start = end
    return highlighted

# 点击检索，展示结果或错误
if st.button("Search"):
    if not query:
        st.error("请输入查询文本后再检索。")
    else:
        result = search_milvus(query, search_type, limit, sparse_weight, dense_weight)

        if "error" in result:
            st.error(result["error"])
        elif "results" in result:
            docs = result["results"]

            # 根据文档路径聚合结果，保留每篇文章的得分最高的 chunk
            aggregated_docs = aggregate_results(docs)
            
            st.success(f"共返回 {len(aggregated_docs)} 条")

            # 展示每篇文档的内容和高亮显示
            for i, doc in enumerate(aggregated_docs, 1):
                with st.expander(f"Result {i} (score: {doc['score']:.4f})"):
                    # 高亮显示文本内容
                    highlighted_text = highlight_text(query, doc['text'])
                    st.markdown(f"- **Text**: {highlighted_text}", unsafe_allow_html=True)
                    st.markdown(f"- **Filename**: {doc.get('filename','N/A')}")
                    st.markdown(f"- **Path**: {doc.get('path','N/A')}")
                    st.markdown(f"- **Date**: {doc.get('date','N/A')}")
                    
                    # 查看原文按钮
                    file_path = doc.get('path', None)
                    if file_path:
                        # 检查文件是否存在
                        if os.path.exists(file_path):
                            if st.button(f"查看原文: {doc['filename']}", key=f"view_{i}"):
                                with open(file_path, 'r', encoding='utf-8') as file:
                                    full_text = file.read()
                                full_text = "测试内容"
                                st.text_area("原文内容", full_text, height=300)
                        else:
                            st.error(f"无法找到文件：{file_path}")
                    else:
                        st.warning("该文档没有原文路径")
        else:
            st.error("接口返回格式不符合预期，请检查日志。")
