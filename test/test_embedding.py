import requests

BASE_URL = "http://localhost:8001"


def test_embed_dense():
    text = "你好，世界！这是测试稠密嵌入。"
    resp = requests.post(f"{BASE_URL}/embed_dense", json={"text": text})
    assert resp.status_code == 200, f"[Embed Dense] 请求失败: {resp.status_code}"
    data = resp.json()
    dense = data.get("dense")
    assert isinstance(dense, list), "稠密向量返回格式不正确"
    assert len(dense) == 1024, f"稠密向量维度应为 1024，实际为 {len(dense)}"
    print("✅ 单条稠密嵌入测试通过！")


def test_batch_dense():
    texts = [
        "今天天气不错，出去走走。",
        "机器学习改变了世界。",
        "测试批量稠密接口。"
    ]
    resp = requests.post(f"{BASE_URL}/embed_batch_dense", json={"texts": texts})
    assert resp.status_code == 200, f"[Batch Embed Dense] 请求失败: {resp.status_code}"
    data = resp.json()
    dense_list = data.get("dense_vectors")
    assert isinstance(dense_list, list), "批量稠密向量返回格式不正确"
    assert len(dense_list) == len(texts), "返回条数与输入文本数不一致"
    for i, vec in enumerate(dense_list):
        assert isinstance(vec, list), f"第 {i} 条稠密向量格式错误"
        assert len(vec) == 1024, f"第 {i} 条稠密向量维度应为 1024，实际为 {len(vec)}"
    print("✅ 批量稠密嵌入测试通过！")


def test_embed_sparse():
    text = "你好，世界！这是测试稀疏嵌入。"
    resp = requests.post(f"{BASE_URL}/embed_sparse", json={"text": text})
    assert resp.status_code == 200, f"[Embed Sparse] 请求失败: {resp.status_code}"
    data = resp.json()
    weights = data.get("lexical_weights")
    assert isinstance(weights, dict), "稀疏向量返回格式不正确"
    assert len(weights) > 0, "稀疏向量字典为空"
    for k, v in weights.items():
        # JSON 中键会以字符串形式返回
        print(k, v)
        assert isinstance(k, (str,)), f"稀疏键类型应为字符串，实际为 {type(k)}"
        assert isinstance(v, float), f"稀疏权重类型应为 float，实际为 {type(v)}"
    print("✅ 单条稀疏嵌入测试通过！")


def test_batch_sparse():
    texts = [
        "今天天气不错，出去走走。",
        "机器学习改变了世界。",
        "测试批量稀疏接口。"
    ]
    resp = requests.post(f"{BASE_URL}/embed_batch_sparse", json={"texts": texts})
    assert resp.status_code == 200, f"[Batch Embed Sparse] 请求失败: {resp.status_code}"
    data = resp.json()
    weights_list = data.get("lexical_weights")
    assert isinstance(weights_list, list), "批量稀疏向量返回格式不正确"
    assert len(weights_list) == len(texts), "返回条数与输入文本数不一致"
    for i, weights in enumerate(weights_list):
        assert isinstance(weights, dict), f"第 {i} 条稀疏向量格式错误"
        assert len(weights) > 0, f"第 {i} 条稀疏向量字典为空"
        for k, v in weights.items():
            assert isinstance(k, (str,)), f"第 {i} 条稀疏向量键类型错误: {type(k)}"
            assert isinstance(v, float), f"第 {i} 条稀疏向量权重类型错误: {type(v)}"
    print("✅ 批量稀疏嵌入测试通过！")


if __name__ == "__main__":
    test_embed_dense()
    test_batch_dense()
    test_embed_sparse()
    test_batch_sparse()
    print("🎉 所有测试通过！")
