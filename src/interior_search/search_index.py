from pathlib import Path

import numpy as np


class SearchIndex:
    """Small cosine-similarity index backed by NumPy arrays."""

    def __init__(self, image_paths: list[str], embeddings: np.ndarray) -> None:
        if len(image_paths) != len(embeddings):
            raise ValueError("image_paths and embeddings must have the same length")

        self.image_paths = image_paths
        self.embeddings = embeddings.astype(np.float32)

    def search(self, query_embedding: np.ndarray, top_k: int = 12) -> list[dict]:
        if len(self.image_paths) == 0:
            return []

        query = query_embedding.astype(np.float32)
        scores = self.embeddings @ query
        top_indices = np.argsort(-scores)[:top_k]

        return [
            {
                "path": self.image_paths[index],
                "score": float(scores[index]),
            }
            for index in top_indices
        ]

    def save(self, output_path: str | Path) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            image_paths=np.array(self.image_paths),
            embeddings=self.embeddings,
        )

    @classmethod
    def load(cls, index_path: str | Path) -> "SearchIndex":
        data = np.load(index_path, allow_pickle=False)
        return cls(
            image_paths=data["image_paths"].astype(str).tolist(),
            embeddings=data["embeddings"],
        )

