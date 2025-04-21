import os
import json
import hashlib
import datetime
import requests
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# ---------------------------------------------------------------------
# 配置项
# ---------------------------------------------------------------------
DATA_DIR = "./data"
METADATA_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
BASE_EMBEDDING_URL = "http://localhost:8001"

# 分块设置（中文检索，约500字符并重叠50字符）
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Milvus Standalone 配置
COLLECTION_NAME = "hybrid_demo"
MILVUS_URI = "http://localhost:19530"

# ---------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------
def load_metadata():
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    metadata[entry["path"]] = entry
                except:
                    continue
    return metadata


def save_metadata(metadata: dict):
    orig = []
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    orig.append(json.loads(line.strip()))
                except:
                    continue
    seen = set()
    updated = []
    for e in orig:
        p = e.get("path")
        if p in metadata:
            e.update({"md5": metadata[p]["md5"], "inserted": metadata[p]["inserted"]})
            if "chunks" in metadata[p]:
                e["chunks"] = metadata[p]["chunks"]
            seen.add(p)
        updated.append(e)
    for p, m in metadata.items():
        if p not in seen:
            updated.append(m)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        for e in updated:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def file_md5(path: str) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def chunk_text(text: str) -> list:
    """
    将文本分块，每块最多 CHUNK_SIZE 字符，且每块间重叠 CHUNK_OVERLAP 字符，同时确保每块字节数不超过 max_bytes。
    """
    max_bytes = 1024
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        # 计算初始结束位置，不能超过文本长度
        end = min(start + CHUNK_SIZE, L)
        # 如果块字节数超过限制，则逐步缩减 end
        while end > start and len(text[start:end].encode("utf-8")) > max_bytes:
            end -= 1
        chunk = text[start:end]
        chunks.append(chunk)
        # 如果已到达文本末尾，则退出
        if end >= L:
            break
        # 下一个块起始位置：在 end 之上回退 overlap，保证重叠覆盖
        start = max(0, end - CHUNK_OVERLAP)
    return chunks

# ---------------------------------------------------------------------
# 初始化 Milvus Collection
# ---------------------------------------------------------------------
def init_collection():
    connections.connect("default", uri=MILVUS_URI)
    EMBEDDING_DIM = 1024
    if not utility.has_collection(COLLECTION_NAME):
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=20),
        ]
        schema = CollectionSchema(fields, description="Hybrid demo collection with sparse and dense vectors")
        col = Collection(COLLECTION_NAME, schema, consistency_level="Strong")
        col.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
        col.create_index("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})
    else:
        col = Collection(COLLECTION_NAME)
    col.load()
    return col

# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
def main():
    metadata = load_metadata()
    col = init_collection()

    to_ingest = []
    for fname in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(path) or not fname.endswith(".txt"):
            continue
        m = file_md5(path)
        meta = metadata.get(path)
        date_str = meta.get("date") if meta else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not meta or meta.get("md5") != m:
            metadata[path] = {"filename": fname, "path": path, "date": date_str, "md5": m, "inserted": False}
            to_ingest.append(path)

    if not to_ingest:
        print("No new or updated files to ingest.")
        return

    for path in to_ingest:
        fname = os.path.basename(path)
        print(f"Processing {fname}...")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)

        dense_list = []
        sparse_list = []
        batch_size = 50

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            # 获取稠密向量
            resp_d = requests.post(f"{BASE_EMBEDDING_URL}/embed_batch_dense", json={"texts": batch})
            resp_d.raise_for_status()
            dens = resp_d.json()["dense_vectors"]
            # 获取稀疏权重
            resp_s = requests.post(f"{BASE_EMBEDDING_URL}/embed_batch_sparse", json={"texts": batch})
            resp_s.raise_for_status()
            sparse_weights = resp_s.json()["lexical_weights"]

            for idx in range(len(batch)):
                dense_list.append(dens[idx])
                sparse_list.append(sparse_weights[idx])

        filenames = [metadata[path]["filename"]] * len(chunks)
        paths = [metadata[path]["path"]] * len(chunks)
        dates = [metadata[path]["date"]] * len(chunks)

        # 插入 Milvus：text, sparse_vector, dense_vector, filename, path, date
        entities = [
            chunks,
            sparse_list,
            dense_list,
            filenames,
            paths,
            dates,
        ]
        col.insert(entities)

        # 插入完成后立即更新元数据并保存
        metadata[path]["inserted"] = True
        metadata[path]["chunks"] = len(chunks)
        save_metadata(metadata)
        print(f"Inserted {len(chunks)} chunks for {fname}. Metadata updated.")

    print("Ingestion complete.")

if __name__ == "__main__":
    main()
