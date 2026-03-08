from sklearn.datasets import fetch_20newsgroups


def load_dataset():
    
    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes")
    )

    documents = data.data
    labels = data.target

    return documents, labels