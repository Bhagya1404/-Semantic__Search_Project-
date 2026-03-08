# semantic_cache.py
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from fuzzy_clustering import fuzzy_memberships  # load your saved fuzzy memberships

class SemanticCache:
    def __init__(self, similarity_threshold=0.8):
        """
        similarity_threshold: minimum cosine similarity to consider a query a 'cache hit'
        """
        self.similarity_threshold = similarity_threshold
        self.cache = {}  # stores {query_text: (embedding, result)}
        self.hit_count = 0
        self.miss_count = 0
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Semantic Cache initialized!")

    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def get(self, query):
        """Check if a query is in cache (or similar enough)"""
        query_embedding = self.model.encode([query])[0]

        for cached_query, (embedding, result) in self.cache.items():
            similarity = self.cosine_similarity(query_embedding, embedding)
            if similarity >= self.similarity_threshold:
                self.hit_count += 1
                return True, cached_query, similarity, result

        self.miss_count += 1
        return False, query, 0.0, None

    def add(self, query, result):
        """Add a new query-result pair to cache"""
        query_embedding = self.model.encode([query])[0]
        self.cache[query] = (query_embedding, result)

    def stats(self):
        total = len(self.cache)
        hit_rate = self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0.0
        return {
            "total_entries": total,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def clear(self):
        """Clear the cache completely"""
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
        print("Cache cleared!")


# --------------------------
# Example usage / test
# --------------------------
if __name__ == "__main__":
    # Initialize cache
    cache = SemanticCache(similarity_threshold=0.8)

    # Example query
    query_text = "hockey game result"

    # Fake result for testing (normally this would come from your semantic search)
    result_text = "The Penguins defeated the Devils in the match."

    # Check cache first (should be a miss)
    hit, matched_query, sim_score, cached_result = cache.get(query_text)
    print("Before adding to cache:", hit, sim_score, cached_result)

    # Add the result to cache
    cache.add(query_text, result_text)

    # Check again (should now be a hit)
    hit, matched_query, sim_score, cached_result = cache.get(query_text)
    print("After adding to cache:", hit, sim_score, cached_result)

    # Print cache stats
    print("Cache stats:", cache.stats())