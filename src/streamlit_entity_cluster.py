import streamlit as st
import json
import requests
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt

# 设置中文字体，解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# Streamlit 应用配置
st.set_page_config(page_title="Entity Disambiguation", layout="wide")
st.title("Entity Disambiguation")

# 侧边栏参数：始终显示
eps = st.sidebar.slider("DBSCAN eps", min_value=0.01, max_value=1.0, value=0.1)
min_samples = st.sidebar.slider("DBSCAN 最小样本数", min_value=1, max_value=10, value=2)

# 默认 JSON 示例
default_json = {
  "1": {"name": "中国", "type": "country", "attributes": {"area": "960万平方公里"}},
  "2": {"name": "China", "type": "country", "attributes": {"area": "9600000 square kilometers"}},
  "3": {"name": "北京", "type": "city", "attributes": {"population": "21 million"}}
}

# 输入框
input_text = st.text_area(
    "在此粘贴或编辑实体 JSON：", value=json.dumps(default_json, ensure_ascii=False, indent=2), height=300
)

# # 显示合并前实体 JSON
# st.subheader("合并前实体 JSON")
# try:
#     raw = json.loads(input_text)
#     st.json(raw)
# except Exception as e:
#     st.error(f"JSON 格式错误: {e}")

# 添加实体消歧按钮
if st.button("实体消歧"):
    try:
        raw = json.loads(input_text)
        input_data = {int(k): v for k, v in raw.items()}

        # 构建待嵌入文本
        def build_texts(entities):
            texts, ids = [], []
            for eid, info in entities.items():
                name = info.get("name", "")
                etype = info.get("type", "")
                attrs = "; ".join([f"{k}:{v}" for k, v in info.get("attributes", {}).items()])
                texts.append(f"Name: {name}; Type: {etype}; Attributes: {attrs}")
                ids.append(eid)
            return ids, texts

        entity_ids, texts = build_texts(input_data)

        # 缓存获取嵌入向量
        @st.cache_data(show_spinner=False)
        def get_dense_embeddings(text_list):
            BASE_URL = "http://localhost:8001"
            all_embeds = []
            for i in range(0, len(text_list), 20):
                batch = text_list[i:i+20]
                resp = requests.post(f"{BASE_URL}/embed_batch_dense", json={"texts": batch})
                resp.raise_for_status()
                all_embeds.extend(resp.json().get("dense_vectors"))
            return np.array(all_embeds)

        embeddings = get_dense_embeddings(texts)

        @st.cache_data(show_spinner=False)
        def compute_pca(embeds):
            pca = PCA(n_components=2)
            return pca.fit_transform(embeds)

        emb_2d = compute_pca(embeddings)

        # 聚类
        dist_matrix = cosine_distances(embeddings)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = clusterer.fit_predict(dist_matrix)

        # 合并实体
        merged = {}
        centroids = []
        next_mid = 1
        for label in set(labels):
            members = [entity_ids[i] for i, lab in enumerate(labels) if lab == label]
            if label == -1:
                for eid in members:
                    info = input_data[eid]
                    merged[next_mid] = {"name": info["name"], "type": info["type"], "attributes": info["attributes"], "cluster": [eid]}
                    centroids.append(emb_2d[entity_ids.index(eid)])
                    next_mid += 1
            else:
                main = input_data[members[0]]
                merged_attrs = {}
                coords = []
                for eid in members:
                    merged_attrs.update(input_data[eid].get("attributes", {}))
                    coords.append(emb_2d[entity_ids.index(eid)])
                centroid = np.mean(coords, axis=0)
                merged[next_mid] = {"name": main["name"], "type": main["type"], "attributes": merged_attrs, "cluster": members}
                centroids.append(centroid)
                next_mid += 1
        centroids = np.array(centroids)

        # 可视化
        fig_before, ax_before = plt.subplots()
        ax_before.scatter(emb_2d[:,0], emb_2d[:,1], s=30, alpha=0.7)
        ax_before.set_title(f"Entity distribution before clustering (entity num: {len(emb_2d)})")
        ax_before.set_xlabel("PC1")
        ax_before.set_ylabel("PC2")

        fig_after, ax_after = plt.subplots()
        ax_after.scatter(centroids[:,0], centroids[:,1], s=50, marker='X')
        ax_after.set_title(f"Entity distribution after clustering (entity num: {len(centroids)})")
        ax_after.set_xlabel("PC1")
        ax_after.set_ylabel("PC2")

        # 并排显示图像
        st.subheader("聚类可视化")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_before)
        with col2:
            st.pyplot(fig_after)

        # 并排显示原始和合并后 JSON
        st.subheader("实体 JSON 对比")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**合并前实体 JSON**")
            st.json(raw)
        with col2:
            st.markdown("**合并后实体 JSON**")
            st.json({str(k): v for k, v in merged.items()})

    except Exception as e:
        st.error(f"错误: {e}")