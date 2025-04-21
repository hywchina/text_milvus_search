import json
import requests
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

# -----------------------------
# 配置本地嵌入服务地址
# -----------------------------
BASE_URL = "http://localhost:8001"

# -----------------------------
# 输入示例：读取 JSON 文件或直接定义
# -----------------------------
input_data = {
    1: {"name": "中国", "type": "country", "attributes": {"area": "960万平方公里"}},
    2: {"name": "China", "type": "country", "attributes": {"area": "9600000 square kilometers"}},
    3: {"name": "北京", "type": "city", "attributes": {"population": "21 million"}}
}

# -----------------------------
# 1. 构造待嵌入的文本
# -----------------------------
def build_texts(entities):
    texts = []
    ids = []
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
# 2. 调用本地嵌入服务获取稠密向量
# -----------------------------
def get_dense_embeddings(text_list):
    # 批量请求
    payload = {"texts": text_list}
    resp = requests.post(f"{BASE_URL}/embed_batch_dense", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return np.array(data.get("dense_vectors"))

embeddings = get_dense_embeddings(texts)

# -----------------------------
# 3. 聚类：使用 DBSCAN
# -----------------------------
# 这里选择余弦距离度量
# eps: 邻域半径，min_samples: 最小簇大小

eps = 0.1  # 根据数据可调
min_samples = 2

dist_matrix = cosine_distances(embeddings)
clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
labels = clusterer.fit_predict(dist_matrix)

# -----------------------------
# 4. 合并簇内实体
# -----------------------------
merged = {}
next_mid = 1
for label in set(labels):
    members = [entity_ids[i] for i, lab in enumerate(labels) if lab == label]
    if label == -1:
        # 噪声点单独成实体
        for eid in members:
            info = input_data[eid]
            merged[next_mid] = {
                "name": info["name"],
                "type": info["type"],
                "attributes": info["attributes"],
                "cluster": [eid]
            }
            next_mid += 1
    else:
        # 合并簇内所有实体
        main = input_data[members[0]]
        merged_attrs = {}
        for eid in members:
            merged_attrs.update(input_data[eid].get("attributes", {}))
        merged[next_mid] = {
            "name": main["name"],
            "type": main["type"],
            "attributes": merged_attrs,
            "cluster": members
        }
        next_mid += 1

# -----------------------------
# 5. 输出结果
# -----------------------------
output = merged
print(json.dumps(output, ensure_ascii=False, indent=2))