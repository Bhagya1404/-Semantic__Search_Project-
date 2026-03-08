from data_loader import load_dataset

docs, labels = load_dataset()

print("Number of documents:", len(docs))
print("Number of labels:", len(labels))

print("\nSample document:\n")
print(docs[0][:500])