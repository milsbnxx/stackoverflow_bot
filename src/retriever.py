from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class EmbeddingRetriever:
    def __init__(self, index_dir: Path | str, model_name: str | None = None):
        self.index_dir = Path(index_dir)

        embeddings_path = self.index_dir / "question_embeddings.npy"
        metadata_path = self.index_dir / "metadata.json"
        config_path = self.index_dir / "index_config.json"

        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.embeddings = np.load(embeddings_path).astype(np.float32)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        config: dict[str, Any] = {}
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))

        self.model_name = model_name or config.get("model_name", DEFAULT_MODEL_NAME)

        if len(self.embeddings) != len(self.metadata):
            raise ValueError(
                "Embeddings and metadata have different lengths. Rebuild index."
            )

        self.model = SentenceTransformer(self.model_name)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        top_k = max(1, min(top_k, len(self.metadata)))

        query_vec = self.model.encode(query, convert_to_numpy=True)
        query_vec = self._normalize(query_vec.astype(np.float32))

        scores = self.embeddings @ query_vec

        candidate_ids = np.argpartition(scores, -top_k)[-top_k:]
        sorted_ids = candidate_ids[np.argsort(scores[candidate_ids])[::-1]]

        results: list[dict[str, Any]] = []
        for idx in sorted_ids:
            row = self.metadata[int(idx)]
            results.append(
                {
                    "score": float(scores[idx]),
                    "question_id": int(row["question_id"]),
                    "answer_id": int(row["answer_id"]),
                    "question_title": row["question_title"],
                    "question_text": row["question_text"],
                    "answer_text": row["answer_text"],
                }
            )

        return results
