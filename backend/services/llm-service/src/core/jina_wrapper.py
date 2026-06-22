from langchain_core.embeddings import Embeddings
import numpy as np

class JinaLangChainWrapper(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Jina's encode returns a numpy array or tensor; FAISS needs a list of floats
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        # Embed a single string for search queries
        embeddings = self.model.encode([text])
        return embeddings[0].tolist()
