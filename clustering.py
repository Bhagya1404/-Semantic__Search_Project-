from sklearn.mixture import GaussianMixture


def perform_clustering(embeddings, n_clusters=10):

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="tied",
        random_state=42
    )

    gmm.fit(embeddings)

    cluster_probabilities = gmm.predict_proba(embeddings)

    return gmm, cluster_probabilities