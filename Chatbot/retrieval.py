from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

class HybridRetriever:
    def __init__(self, nodes):
        self.nodes = nodes
        
        # Better preprocessing
        self.docs = [self._clean(n.title + " " + n.text) for n in nodes]
        self.tokenized = [doc.split() for doc in self.docs]
        
        self.bm25 = BM25Okapi(self.tokenized)
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(self.docs, normalize_embeddings=True)
        
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def _clean(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text

    def _normalize(self, scores):
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s == 0:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def retrieve(self, query, top_k=3, candidate_k=10):
        query_clean = self._clean(query)
        query_tokens = query_clean.split()

        bm25_scores = self.bm25.get_scores(query_tokens)

        query_vec = self.model.encode(query_clean, normalize_embeddings=True)
        sim_scores = cosine_similarity([query_vec], self.embeddings)[0]

        bm25_norm = self._normalize(bm25_scores)
        sim_norm = self._normalize(sim_scores)

        alpha = 0.7
        hybrid_scores = alpha * bm25_norm + (1 - alpha) * sim_norm

        candidate_idx = np.argsort(hybrid_scores)[::-1][:candidate_k]

        pairs = [(query, self.docs[i]) for i in candidate_idx]
        rerank_scores = self.reranker.predict(pairs)

        reranked_idx = [
            candidate_idx[i]
            for i in np.argsort(rerank_scores)[::-1][:top_k]
        ]

        results = [self.nodes[i] for i in reranked_idx]
        top_score = float(np.max(rerank_scores))

        return results, top_score