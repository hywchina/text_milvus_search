#!/usr/bin/env python3
# stream_milvus.py
# 一个交互式的 Milvus 搜索演示脚本，支持密集、稀疏和混合搜索

import argparse
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
def dense_search(col: Collection, dense_emb: list, limit: int = 5) -> list:
    search_params = {"metric_type": "IP", "params": {}}
    hits = col.search(
        [dense_emb],
        anns_field="dense_vector",
        param=search_params,
        limit=limit,
        output_fields=["text"],
    )[0]
    return [hit.entity.get("text") for hit in hits]

def sparse_search(col: Collection, sparse_emb: dict, limit: int = 5) -> list:
    search_params = {"metric_type": "IP", "params": {}}
    hits = col.search(
        [sparse_emb],
        anns_field="sparse_vector",
        param=search_params,
        limit=limit,
        output_fields=["text"],
    )[0]
    return [hit.entity.get("text") for hit in hits]

def hybrid_search(col: Collection, dense_emb: list, sparse_emb: dict,
                  sparse_weight: float, dense_weight: float, limit: int = 5) -> list:
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
    rerank = WeightedRanker(sparse_weight, dense_weight)
    hits = col.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=rerank,
        limit=limit,
        output_fields=["text"],
    )[0]
    return [hit.entity.get("text") for hit in hits]

# -----------------------------
# 4. 文本高亮（ANSI 颜色）
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=True)

def doc_text_formatting(tokenizer, query: str, docs: list) -> list:
    enc_q = tokenizer.encode_plus(query, return_offsets_mapping=True, add_special_tokens=True)
    q_ids = enc_q["input_ids"][1:-1]
    q_tokens = set(tokenizer.convert_ids_to_tokens(q_ids))

    formatted = []
    for doc in docs:
        enc = tokenizer.encode_plus(doc, return_offsets_mapping=True, add_special_tokens=True)
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"])[1:-1]
        offs = enc["offset_mapping"][1:-1]
        spans = [(st, ed) for tok, (st, ed) in zip(toks, offs) if tok in q_tokens]
        # 合并相邻区间
        merged = []
        for st, ed in sorted(spans):
            if not merged or st > merged[-1][1]:
                merged.append([st, ed])
            else:
                merged[-1][1] = max(merged[-1][1], ed)
        # 插入 ANSI 红色
        out = ""
        last = 0
        for st, ed in merged:
            out += doc[last:st] + "\033[31m" + doc[st:ed] + "\033[0m"
            last = ed
        out += doc[last:]
        formatted.append(out)
    return formatted

# -----------------------------
# 5. CLI & 流式交互
# -----------------------------
def interactive_loop(col: Collection, args):
    prompt = "\n输入查询（直接回车退出）： "
    while True:
        query = input(prompt).strip()
        if not query:
            print("退出。")
            break

        # 1) 获取嵌入
        dense_emb = get_dense_embedding(query)
        sparse_emb = get_sparse_embedding(query)

        # 2) 分别执行三种搜索
        print("\n>>> 密集搜索结果:")
        dense_hits = dense_search(col, dense_emb, limit=args.limit)
        for line in doc_text_formatting(tokenizer, query, dense_hits):
            print(line)

        print("\n>>> 稀疏搜索结果:")
        sparse_hits = sparse_search(col, sparse_emb, limit=args.limit)
        for line in doc_text_formatting(tokenizer, query, sparse_hits):
            print(line)

        print(f"\n>>> 混合搜索结果 (sparse:{args.sparse_weight}, dense:{args.dense_weight}):")
        hybrid_hits = hybrid_search(col, dense_emb, sparse_emb,
                                    sparse_weight=args.sparse_weight,
                                    dense_weight=args.dense_weight,
                                    limit=args.limit)
        for line in doc_text_formatting(tokenizer, query, hybrid_hits):
            print(line)

def main():
    parser = argparse.ArgumentParser(description="Milvus 流式搜索 Demo")
    parser.add_argument("--limit", type=int, default=5, help="每种搜索返回的 top k")
    parser.add_argument("--sparse-weight", type=float, default=0.7, help="稀疏权重")
    parser.add_argument("--dense-weight", type=float, default=1.0, help="密集权重")
    args = parser.parse_args()

    connect_milvus()
    col = load_collection()
    print(f"已连接 Milvus 集合: {COLLECTION_NAME}")
    interactive_loop(col, args)

if __name__ == "__main__":
    main()
