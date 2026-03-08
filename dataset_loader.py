from sklearn.datasets import fetch_20newsgroups

# Load dataset
newsgroups = fetch_20newsgroups(subset='all')

documents = newsgroups.data
labels = newsgroups.target
label_names = newsgroups.target_names

print("Total documents:", len(documents))
print("Total categories:", len(label_names))

print("\nSample document:\n")
print(documents[0][:500])  # print first 500 characters