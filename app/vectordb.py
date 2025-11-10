import os, json, math
from typing import List

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    has_st = True
except Exception:
    has_st = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

class VectorDB:
    def __init__(self, storage_path='server/data/memory.json'):
        self.storage_path = storage_path
        self._load()
        if has_st:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self.model = None
        else:
            self.model = None
            self.tfidf = None

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.store = json.load(f)
            except Exception:
                self.store = {}
        else:
            self.store = {}

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.store, f, indent=2)

    def upsert(self, user_id: str, text: str):
        user = self.store.get(user_id, [])
        user.append(text)
        self.store[user_id] = user[-200:]  # keep last 200 entries
        self._save()

    def retrieve(self, user_id: str, top_k=5) -> List[str]:
        texts = self.store.get(user_id, [])
        if not texts:
            return []
        # If embedding model available, compute similarities; otherwise simple substring heuristic
        if self.model:
            try:
                emb = self.model.encode([" ".join(texts)])
                # naive return last top_k
                return texts[-top_k:][::-1]
            except Exception:
                return texts[-top_k:][::-1]
        # fallback
        return texts[-top_k:][::-1]
