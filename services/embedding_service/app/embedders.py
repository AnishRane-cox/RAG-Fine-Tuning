from abc import ABC, abstractmethod
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError("Embedder must implement embed")

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Embedder must implement embed_batch")

class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        vec = self.model.encode(text, show_progress_bar=True)
        return np.array(vec, dtype=np.float32).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        return [np.array(v, dtype=np.float32).tolist() for v in vecs]