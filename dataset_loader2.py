# dataset_loader2.py
import re
import pickle
from sklearn.datasets import fetch_20newsgroups

# Step 1: Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = newsgroups.data
labels = newsgroups.target

print("Total documents:", len(documents))
print("Total categories:", len(set(labels)))
print("\nSample document:\n")
print(documents[0][:500])

# Step 2: Clean the text
def clean_text(text):
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    return text.strip()

# Apply cleaning to all documents
cleaned_documents = [clean_text(doc) for doc in documents]

print("\nSample cleaned document:\n")
print(cleaned_documents[0][:500])

# Step 3: Save cleaned documents for later use
with open("cleaned_documents.pkl", "wb") as f:
    pickle.dump(cleaned_documents, f)

print("\nCleaned documents saved! Total documents:", len(cleaned_documents))