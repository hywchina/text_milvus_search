import json
import requests
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -----------------------------
# 1. 从文件读取输入数据（50条实体）
# -----------------------------
# 输入文件: input.json，格式:
# {
#   "1": {"name": ..., "type": ..., "attributes": {...}},
#   "2": {...},
#   ...
# }
with open('data/entities_input.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)
# 将字符串 key 转为 int
input_data = {int(k): v for k, v in raw.items()}

# -----------------------------
# 2. 构造待嵌入的文本
# -----------------------------
def build_texts(entities):
    texts, ids = [], []
    for eid, info in entities.items():
        name = info.get("name", "")
        etype = info.get("type", "")
        attrs = "; ".join([f"{k}:{v}" for k, v in info.get("attributes", {}).items()])
        text = f"Name: {name}; Type: {etype}; Attributes: {attrs}"
        texts.append(text)
        ids.append(eid)
    return ids, texts

entity_ids, texts = build_texts(input_data)

# -----------------------------
# 3. 调用本地嵌入服务获取稠密向量
# -----------------------------
BASE_URL = "http://localhost:8001"
def get_dense_embeddings(text_list, batch_size=20):
    all_embeds = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        resp = requests.post(f"{BASE_URL}/embed_batch_dense", json={"texts": batch})
        resp.raise_for_status()
        data = resp.json()
        all_embeds.extend(data.get("dense_vectors"))
    return np.array(all_embeds)

embeddings = get_dense_embeddings(texts)

# -----------------------------
# 4. 可视化：聚类前实体在2D空间中的分布
# -----------------------------
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)
plt.figure()
plt.scatter(emb_2d[:,0], emb_2d[:,1], s=30, alpha=0.7)
plt.title('Entities before Clustering (2D PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('before_clustering.png')

# -----------------------------
# 5. 聚类：使用 DBSCAN
# -----------------------------
eps = 0.1  # 根据数据规模和相似度分布调整
min_samples = 2

dist_matrix = cosine_distances(embeddings)
clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
labels = clusterer.fit_predict(dist_matrix)

# -----------------------------
# 6. 合并簇内实体，并收集簇中心（合并后实体位置）
# -----------------------------
merged = {}
centroids = []
next_mid = 1
for label in set(labels):
    members = [entity_ids[i] for i, lab in enumerate(labels) if lab == label]
    if label == -1:
        for eid in members:
            info = input_data[eid]
            merged[next_mid] = {
                "name": info["name"],
                "type": info["type"],
                "attributes": info["attributes"],
                "cluster": [eid]
            }
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
        merged[next_mid] = {
            "name": main["name"],
            "type": main["type"],
            "attributes": merged_attrs,
            "cluster": members
        }
        centroids.append(centroid)
        next_mid += 1

# -----------------------------
# 7. 可视化：聚类后合并实体在2D空间中的位置
# -----------------------------
centroids = np.array(centroids)
plt.figure()
plt.scatter(centroids[:,0], centroids[:,1], s=50, marker='X')
plt.title('Merged Entities after Clustering (2D PCA Centroids)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('after_clustering.png')

# -----------------------------
# 8. 输出合并结果到文件
# -----------------------------
with open('data/entities_output.json', 'w', encoding='utf-8') as f:
    json.dump({str(k):v for k,v in merged.items()}, f, ensure_ascii=False, indent=2)

print("Clustering complete. Results saved to output.json, plots saved as PNG files.")