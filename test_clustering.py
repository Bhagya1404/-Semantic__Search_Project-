from data_loader import load_dataset
from embeddings import generate_embeddings
from clustering import perform_clustering

docs, labels = load_dataset()

embeddings = generate_embeddings(docs[:100])

gmm, cluster_probs = perform_clustering(embeddings)

print("Documents clustered:", len(cluster_probs))
print("Clusters per document:", len(cluster_probs[0]))

print("\nCluster probability example:\n")
print(cluster_probs[0])