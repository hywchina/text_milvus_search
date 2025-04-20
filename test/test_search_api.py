# test_search_api.py

import requests

BASE_URL = "http://localhost:8002"  # FastAPI service URL

def test_dense_search(query: str):
    response = requests.post(f"{BASE_URL}/dense_search/", json={"query": query, "limit": 5})
    assert response.status_code == 200
    print(response.json())

def test_sparse_search(query: str):
    response = requests.post(f"{BASE_URL}/sparse_search/", json={"query": query, "limit": 5})
    assert response.status_code == 200
    print(response.json())

def test_hybrid_search(query: str):
    response = requests.post(f"{BASE_URL}/hybrid_search/", json={"query": query, "limit": 5, "sparse_weight": 0.7, "dense_weight": 1.0})
    assert response.status_code == 200
    print(response.json())

if __name__ == "__main__":
    query = "混沌未分天地乱，茫茫渺渺无人见。"
    test_dense_search(query)
    test_sparse_search(query)
    test_hybrid_search(query)
