# search_milvus_api.py

from fastapi import FastAPI
from pydantic import BaseModel
from pymilvus import (
    connections,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)
import requests
from transformers import AutoTokenizer

app = FastAPI()

# Configuration
BASE_URL = "http://localhost:8001"
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "hybrid_demo"
DENSE_DIM = 1024

# Connect to Milvus
def connect_milvus(uri: str = MILVUS_URI):
    connections.connect("default", uri=uri)

def load_collection(name: str = COLLECTION_NAME) -> Collection:
    col = Collection(name)
    col.load()
    return col

# Request Models
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    sparse_weight: float = 1.0
    dense_weight: float = 1.0

# Embedding Methods
def get_dense_embedding(text: str) -> list:
    resp = requests.post(f"{BASE_URL}/embed_dense", json={"text": text})
    resp.raise_for_status()
    return resp.json()["dense"]

def get_sparse_embedding(text: str) -> dict:
    resp = requests.post(f"{BASE_URL}/embed_sparse", json={"text": text})
    resp.raise_for_status()
    return resp.json()["lexical_weights"]

# Search Functions
def dense_search(col: Collection, dense_emb: list, limit: int = 10) -> list:
    search_params = {"metric_type": "IP", "params": {}}
    hits = col.search(
        [dense_emb],
        anns_field="dense_vector",
        param=search_params,
        limit=limit,
        output_fields=["text", 'filename', 'path', 'date'],
    )[0]

    return [
        {
            "text": hit.entity.get("text"),
            "filename": hit.entity.get("filename"),
            "path": hit.entity.get("path"),
            "date": hit.entity.get("date"),
            "score": hit.score,
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
        output_fields=["text", 'filename', 'path', 'date'],
    )[0]

    return [
        {
            "text": hit.entity.get("text"),
            "filename": hit.entity.get("filename"),
            "path": hit.entity.get("path"),
            "date": hit.entity.get("date"),
            "score": hit.score,
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
        output_fields=["text", 'filename', 'path', 'date'],
    )[0]

    return [
        {
            "text": hit.entity.get("text"),
            "filename": hit.entity.get("filename"),
            "path": hit.entity.get("path"),
            "date": hit.entity.get("date"),
            "score": hit.score,
        }
        for hit in hits
    ]

# API Endpoints
@app.post("/dense_search/")
async def dense_search_api(request: SearchRequest):
    connect_milvus()
    col = load_collection()
    dense_emb = get_dense_embedding(request.query)
    results = dense_search(col, dense_emb, limit=request.limit)
    return {"results": results}

@app.post("/sparse_search/")
async def sparse_search_api(request: SearchRequest):
    connect_milvus()
    col = load_collection()
    sparse_emb = get_sparse_embedding(request.query)
    results = sparse_search(col, sparse_emb, limit=request.limit)
    return {"results": results}

@app.post("/hybrid_search/")
async def hybrid_search_api(request: SearchRequest):
    connect_milvus()
    col = load_collection()
    dense_emb = get_dense_embedding(request.query)
    sparse_emb = get_sparse_embedding(request.query)
    results = hybrid_search(
        col,
        dense_emb,
        sparse_emb,
        sparse_weight=request.sparse_weight,
        dense_weight=request.dense_weight,
        limit=request.limit,
    )
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_search_milvus:app", host="0.0.0.0", port=8002, reload=True)

