#!/usr/bin/env python3
# test_milvus.py

import sys
import requests
from pymilvus import (
    connections,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)
from transformers import AutoTokenizer

# -----------------------------
# 配置项
# -----------------------------
BASE_URL = "http://localhost:8001"
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "hybrid_demo"
DENSE_DIM = 1024

# -----------------------------
# 1. Embedding 接口调用
# -----------------------------
def get_dense_embedding(text: str) -> list:
    resp = requests.post(f"{BASE_URL}/embed_dense", json={"text": text})
    resp.raise_for_status()
    return resp.json()["dense"]

def get_sparse_embedding(text: str) -> dict:
    resp = requests.post(f"{BASE_URL}/embed_sparse", json={"text": text})
    resp.raise_for_status()
    return resp.json()["lexical_weights"]

# -----------------------------
# 2. Milvus 连接与加载
# -----------------------------
def connect_milvus(uri: str = MILVUS_URI):
    connections.connect("default", uri=uri)

def load_collection(name: str = COLLECTION_NAME) -> Collection:
    col = Collection(name)
    col.load()
    return col

# -----------------------------
# 3. 搜索函数
# -----------------------------
def dense_search(col: Collection, dense_emb: list, limit: int = 10) -> list:
    search_params = {"metric_type": "IP", "params": {}}
    hits = col.search(
        [dense_emb],
        anns_field="dense_vector",
        param=search_params,
        limit=limit,
        output_fields=["text",'filename', 'path', 'date'],
    )[0]

    return [
        {
            "text": hit.entity.get("text"),
            "filename": hit.entity.get("filename"),  
            "path": hit.entity.get("path"),          
            "date": hit.entity.get("date"),
            "score": hit.score,               # <-- 新增相似度       
        }
        for hit in hits
    ]

def sparse_search(col: Collection, sparse_emb: dict, limit: int = 10) -> list:
    search_params = {"metric_type": "IP", "params": {}}
    hits = col.search(
        [sparse_emb],
        anns_field="sparse_vector",
        param=search_params,
        limit=limit,
        output_fields=["text",'filename', 'path', 'date'],
    )[0]

    return [
        {
            "text": hit.entity.get("text"),
            "filename": hit.entity.get("filename"),  
            "path": hit.entity.get("path"),          
            "date": hit.entity.get("date"),
            "score": hit.score,               # <-- 新增相似度               
        }
        for hit in hits
    ]

def hybrid_search(
    col: Collection,
    dense_emb: list,
    sparse_emb: dict,
    sparse_weight: float = 1.0,
    dense_weight: float = 1.0,
    limit: int = 10,
) -> list:
    # 构造两个 AnnSearchRequest
    dense_req = AnnSearchRequest(
        data=[dense_emb],
        anns_field="dense_vector",
        param={"metric_type": "IP", "params": {}},
        limit=limit,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_emb],
        anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {}},
        limit=limit,
    )
    # WeightedRanker 接受稀疏和密集的权重
    rerank = WeightedRanker(sparse_weight, dense_weight)
    # 正确调用 hybrid_search：只传入 reqs 和 rerank 两个位置参数
    hits = col.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=rerank,
        limit=limit,
        output_fields=["text",'filename', 'path', 'date'],
    )[0]

    return [
        {
            "text": hit.entity.get("text"),
            "filename": hit.entity.get("filename"),  
            "path": hit.entity.get("path"),          
            "date": hit.entity.get("date"),
            "score": hit.score,               # <-- 新增相似度               
        }
        for hit in hits
    ]

# -----------------------------
# 4. 结果格式化 & 高亮
# -----------------------------
# 使用中文 BERT Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=True)

def doc_text_formatting(tokenizer, query: str, docs: list) -> list:
    # 编码 query，获取 token ids 和 offsets
    q_enc = tokenizer.encode_plus(
        query,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    q_ids = q_enc["input_ids"][1:-1]
    q_toks = set(tokenizer.convert_ids_to_tokens(q_ids))

    formatted_texts = []
    for doc in docs:
        enc = tokenizer.encode_plus(
            doc,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])[1:-1]
        offsets = enc["offset_mapping"][1:-1]
        spans = []
        for tok, (st, ed) in zip(tokens, offsets):
            if tok in q_toks:
                spans.append((st, ed))
        # 合并相邻区间
        merged = []
        for st, ed in sorted(spans):
            if not merged or st > merged[-1][1]:
                merged.append([st, ed])
            else:
                merged[-1][1] = max(merged[-1][1], ed)
        # 插入 HTML 标签
        res = ""
        last = 0
        for st, ed in merged:
            res += doc[last:st]
            res += "<span style='color:red'>"
            res += doc[st:ed]
            res += "</span>"
            last = ed
        res += doc[last:]
        formatted_texts.append(res)
    return formatted_texts


def highlight_text(html: str) -> str:
    # 将 <span> 转为 ANSI 颜色
    return html.replace(
        "<span style='color:red'>", "\033[31m"
    ).replace("</span>", "\033[0m")


def display_search_results(title: str, results: list, query: str):
    print(f"\n=== {title} (top {len(results)}) ===")
    for result in results:
        # Extract fields from each result
        text = result.get('text', '')
        filename = result.get('filename', 'N/A')
        path = result.get('path', 'N/A')
        date = result.get('date', 'N/A')
        score = result["score"]
        
        # Format the text with highlights
        html = doc_text_formatting(tokenizer, query, [text])[0]  # Highlighting the text
        
        # Print the results in a structured format
        print(f"Filename: {filename}")
        print(f"Path: {path}")
        print(f"Date: {date}")
        print(f"Score: {score:.4f}")   # 这里格式化到小数点后四位
        print("Text: ")
        print(highlight_text(html))
        print("-" * 80)  # Separator between results


# -----------------------------
# 5. 主测试流程
# -----------------------------
def test_search():
    connect_milvus()
    col = load_collection()

    query = "混沌未分天地乱，茫茫渺渺无人见。"
    dense_emb = get_dense_embedding(query)
    sparse_emb = get_sparse_embedding(query)

    # Dense Search
    print("\n=== Dense Search ===")
    dense_results = dense_search(col, dense_emb, limit=5)
    display_search_results("Dense Search Results", dense_results, query)

    # Sparse Search
    print("\n=== Sparse Search ===")
    sparse_results = sparse_search(col, sparse_emb, limit=5)
    display_search_results("Sparse Search Results", sparse_results, query)

    # Hybrid Search
    print("\n=== Hybrid Search ===")
    hybrid_results = hybrid_search(
        col,
        dense_emb,
        sparse_emb,
        sparse_weight=0.7,
        dense_weight=1.0,
        limit=5,
    )
    display_search_results("Hybrid Search Results", hybrid_results, query)

if __name__ == "__main__":
    test_search()
