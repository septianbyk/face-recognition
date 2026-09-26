import faiss
import numpy as np
import pickle
from pathlib import Path


class FaceStore:
    def __init__(self, dim=512, index_path="face_index.faiss", meta_path="face_meta.pkl"):
        self.dim = dim
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.names = []
        self.index = faiss.IndexFlatIP(dim)
        self.load()

    def load(self):
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "rb") as f:
                self.names = pickle.load(f)

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.names, f)

    def add(self, name: str, embedding: np.ndarray, persist: bool = True):
        vec = embedding.reshape(1, -1).astype("float32")
        self.index.add(vec)
        self.names.append(name)
        if persist:
            self.save()

    def search(self, embedding: np.ndarray, threshold=0.5):
        if self.index.ntotal == 0:
            return None, None

        vec = embedding.reshape(1, -1).astype("float32")
        similarities, indices = self.index.search(vec, k=1)
        best_similarity = float(similarities[0][0])
        best_idx = int(indices[0][0])

        if best_similarity >= threshold:
            return self.names[best_idx], best_similarity
        return None, best_similarity
