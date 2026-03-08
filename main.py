# main.py
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from semantic_cache import SemanticCache

# --------------------------
# Load documents, embeddings, fuzzy clusters
# --------------------------
with open("cleaned_documents.pkl", "rb") as f:
    documents = pickle.load(f)

with open("document_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

with open("fuzzy_clusters.pkl", "rb") as f:
    fuzzy_memberships = pickle.load(f)

print(f"Loaded {len(documents)} documents, embeddings, and fuzzy clusters.")

# --------------------------
# Initialize semantic cache
# --------------------------
cache = SemanticCache(similarity_threshold=0.8)

# --------------------------
# FastAPI setup
# --------------------------
app = FastAPI(title="Semantic Search API")

class QueryRequest(BaseModel):
    query: str

# --------------------------
# POST /query endpoint
# --------------------------
@app.post("/query")
def query_endpoint(request: QueryRequest):
    query_text = request.query

    # Check semantic cache
    hit, matched_query, similarity, result = cache.get(query_text)
    
    if hit:
        dominant_cluster = None
        # Find dominant cluster if cached result exists
        if result in documents:
            doc_index = documents.index(result)
            dominant_cluster = int(np.argmax(fuzzy_memberships[doc_index]))
        return {
            "query": query_text,
            "cache_hit": True,
            "matched_query": matched_query,
            "similarity_score": float(similarity),
            "result": result,
            "dominant_cluster": dominant_cluster
        }

    # On miss: compute similarity with all embeddings
    model = cache.model
    query_embedding = model.encode([query_text])[0]
    sims = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_doc = documents[best_idx]
    dominant_cluster = int(np.argmax(fuzzy_memberships[best_idx]))

    # Add to cache
    cache.add(query_text, best_doc)

    return {
        "query": query_text,
        "cache_hit": False,
        "matched_query": query_text,
        "similarity_score": best_score,
        "result": best_doc,
        "dominant_cluster": dominant_cluster
    }

# --------------------------
# GET /cache/stats endpoint
# --------------------------
@app.get("/cache/stats")
def cache_stats():
    return cache.stats()

# --------------------------
# DELETE /cache endpoint
# --------------------------
@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"message": "Cache cleared!"}