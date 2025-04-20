# streamlit_milvus_demo.py

import streamlit as st
import requests

BASE_URL = "http://localhost:8002"  # FastAPI service URL

# Function to query the FastAPI API
def search_milvus(query, search_type, limit=5, sparse_weight=1.0, dense_weight=1.0):
    payload = {"query": query, "limit": limit, "sparse_weight": sparse_weight, "dense_weight": dense_weight}
    if search_type == "dense":
        response = requests.post(f"{BASE_URL}/dense_search/", json=payload)
    elif search_type == "sparse":
        response = requests.post(f"{BASE_URL}/sparse_search/", json=payload)
    elif search_type == "hybrid":
        response = requests.post(f"{BASE_URL}/hybrid_search/", json=payload)
    return response.json()

# Streamlit UI
st.title("Milvus Text Search Demo")

query = st.text_input("Enter query text", "")

search_type = st.selectbox("Select search type", ("dense", "sparse", "hybrid"))
limit = st.slider("Number of results", min_value=1, max_value=20, value=5)

sparse_weight = st.slider("Sparse Weight", min_value=0.0, max_value=1.0, value=1.0)
dense_weight = st.slider("Dense Weight", min_value=0.0, max_value=1.0, value=1.0)

if st.button("Search"):
    if query:
        results = search_milvus(query, search_type, limit, sparse_weight, dense_weight)
        if "results" in results:
            st.write(f"Found {len(results['results'])} results")
            for idx, result in enumerate(results["results"]):
                st.write(f"Result {idx+1}")
                st.write(f"Score: {result['score']}")
                st.write(f"Text: {result['text']}")
                st.write(f"Filename: {result['filename']}")
                st.write(f"Path: {result['path']}")
                st.write(f"Date: {result['date']}")
        else:
            st.error("No results found")
    else:
        st.error("Please enter a query to search.")
