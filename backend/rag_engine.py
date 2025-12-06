# backend/rag_engine.py
import os
import glob
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


class RAGEngine:
    def __init__(self, docs_path: str = "backend/documents", model_name: str = "all-MiniLM-L6-v2"):
        self.docs_path = docs_path
        self.model = SentenceTransformer(model_name)
        self.text_chunks: List[str] = []
        self.embeddings: np.ndarray | None = None

    def load_documents(self) -> None:
        """Load all .txt files and split into chunks."""
        files = glob.glob(os.path.join(self.docs_path, "*.txt"))
        chunks: List[str] = []

        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()

            # split by empty line paragraphs
            for paragraph in text.split("\n\n"):
                cleaned = paragraph.strip()
                if len(cleaned) > 50:  # ignore tiny lines
                    chunks.append(cleaned)

        self.text_chunks = chunks

    def build_index(self) -> None:
        """Create embeddings for all chunks."""
        if not self.text_chunks:
            self.load_documents()
        if not self.text_chunks:
            print("⚠️ No documents found in documents folder.")
            self.embeddings = None
            return

        self.embeddings = self.model.encode(
            self.text_chunks,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        # Normalize for cosine similarity
        self.embeddings = self._normalize(self.embeddings)
        print(f"✅ Built RAG index with {len(self.text_chunks)} chunks.")

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        return vectors / norms

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Return top-k relevant text chunks for a given query."""
        if self.embeddings is None or not self.text_chunks:
            return []

        query_emb = self.model.encode([query], convert_to_numpy=True)
        query_emb = self._normalize(query_emb)[0]

        scores = np.dot(self.embeddings, query_emb)
        top_k_idx = np.argsort(scores)[::-1][:k]

        results: List[Tuple[str, float]] = []
        for idx in top_k_idx:
            results.append((self.text_chunks[idx], float(scores[idx])))

        return results
