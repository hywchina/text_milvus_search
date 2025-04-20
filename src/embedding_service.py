import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from FlagEmbedding import BGEM3FlagModel

# ---------------------------------------------------------------------
# 模型加载：使用 BGE-M3 多功能模型
# ---------------------------------------------------------------------
# 可根据实际情况调整 use_fp16 和 device
model = BGEM3FlagModel(
    model_name_or_path="BAAI/bge-m3",
    use_fp16=False,
    device="cuda" if (os.getenv("CUDA_VISIBLE_DEVICES") or False) else "cpu"
)

# ---------------------------------------------------------------------
# FastAPI 服务定义
# ---------------------------------------------------------------------
app = FastAPI(
    title="BGE-M3 Embedding Service",
    description="同时支持稠密检索和稀疏检索的文本嵌入生成服务",
    version="1.0.0"
)

# 输入模型
class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

# 稠密向量响应
class DenseResponse(BaseModel):
    dense: List[float]

class BatchDenseResponse(BaseModel):
    dense_vectors: List[List[float]]

# 稀疏向量响应
class SparseResponse(BaseModel):
    lexical_weights: Dict[int, float]  # Token ID -> 权重映射

class BatchSparseResponse(BaseModel):
    lexical_weights: List[Dict[int, float]]

@app.post("/embed_dense", response_model=DenseResponse)
def embed_dense(req: TextRequest):
    """单条文本的稠密嵌入接口"""
    if not req.text:
        raise HTTPException(status_code=400, detail="文本为空")
    # 只返回稠密向量
    output = model.encode(
        [req.text],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )
    dense_vec = output["dense_vecs"][0]
    return DenseResponse(dense=dense_vec.tolist())

@app.post("/embed_batch_dense", response_model=BatchDenseResponse)
def embed_batch_dense(req: BatchRequest):
    """批量文本的稠密嵌入接口"""
    if not req.texts:
        raise HTTPException(status_code=400, detail="文本列表为空")
    output = model.encode(
        req.texts,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )
    dense_list = output["dense_vecs"]
    return BatchDenseResponse(dense_vectors=[vec.tolist() for vec in dense_list])

@app.post("/embed_sparse", response_model=SparseResponse)
def embed_sparse(req: TextRequest):
    """单条文本的稀疏嵌入接口"""
    if not req.text:
        raise HTTPException(status_code=400, detail="文本为空")
    # 只返回稀疏权重字典
    output = model.encode(
        [req.text],
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False
    )
    weights = output.get("lexical_weights")
    if weights is None or len(weights) == 0:
        raise HTTPException(status_code=500, detail="未能生成稀疏向量")
    return SparseResponse(lexical_weights=weights[0])

@app.post("/embed_batch_sparse", response_model=BatchSparseResponse)
def embed_batch_sparse(req: BatchRequest):
    """批量文本的稀疏嵌入接口"""
    if not req.texts:
        raise HTTPException(status_code=400, detail="文本列表为空")
    output = model.encode(
        req.texts,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False
    )
    weights_list = output.get("lexical_weights") or []
    if not weights_list:
        raise HTTPException(status_code=500, detail="未能生成稀疏向量列表")
    return BatchSparseResponse(lexical_weights=weights_list)

if __name__ == "__main__":
    import uvicorn
    # 在生产环境，将 reload=False, debug=False
    uvicorn.run("embedding_service:app", host="0.0.0.0", port=8001, reload=True)
