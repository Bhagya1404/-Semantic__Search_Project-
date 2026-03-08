# embeddings.py
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Step 1: Load cleaned documents
with open("cleaned_documents.pkl", "rb") as f:
    cleaned_documents = pickle.load(f)

print("Number of documents:", len(cleaned_documents))

# Step 2: Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

# Step 3: Generate embeddings
# Convert to list to ensure picklable format
embeddings = model.encode(cleaned_documents, batch_size=32, show_progress_bar=True)
embeddings = embeddings.tolist()  # ensure it can be saved with pickle

print("Number of embeddings:", len(embeddings))
print("Embedding vector length:", len(embeddings[0]))

# Step 4: Save embeddings
with open("document_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Embeddings saved! Total documents:", len(embeddings))