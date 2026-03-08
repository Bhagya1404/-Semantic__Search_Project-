from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded successfully!")

# Documents
documents = [
    "The hockey playoffs were exciting this year.",
    "The Penguins defeated the Devils in the match.",
    "Python is a great programming language.",
    "Artificial intelligence is transforming industries.",
    "The new GPU improves machine learning performance."
]

print("Number of documents:", len(documents))

# Convert documents to embeddings
doc_embeddings = model.encode(documents)

print("Embedding vector length:", len(doc_embeddings[0]))

# Query
query = "hockey game result"
print("\nQuery:", query)

# Convert query to embedding
query_embedding = model.encode(query)

# Compute similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

print("\nTop matching documents:\n")

threshold = 0.3

results = list(zip(documents, scores))
results = sorted(results, key=lambda x: x[1], reverse=True)

for doc, score in results:
    if score > threshold:
        print(doc, "(Score:", score, ")")