from data_loader import load_dataset
from embeddings import generate_embeddings

docs, labels = load_dataset()

# test only first 10 documents
embeddings = generate_embeddings(docs[:10])

print("Number of embeddings:", len(embeddings))
print("Embedding vector length:", len(embeddings[0]))