Semantic Search Project

Overview:
A lightweight semantic search system built on the 20 Newsgroups dataset (~20k posts) with:

Fuzzy clustering (documents can belong to multiple clusters)

Semantic cache (avoids recomputation for similar queries)

FastAPI service exposing a live API endpoint

Setup:

1. Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

2. Install dependencies:
pip install -r requirements.txt

3. Place these large files in the project folder:
cleaned_documents.pkl
document_embeddings.pkl
fuzzy_clusters.pkl

Run the API:
uvicorn main:app --reload

Access the docs at: http://127.0.0.1:8000/docs

Endpoints:
POST /query – query text → returns closest document, similarity score, and dominant cluster
GET /cache/stats – returns cache stats (entries, hit/miss, hit rate)
DELETE /cache – clears the cache

Notes:
Ensure .pkl files are in the same folder as main.py.
Cache helps speed up repeated or similar queries.
Fuzzy clusters handle overlapping topics in documents.
