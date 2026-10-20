"""
vectorstores/chroma_store.py
-----------------------------
ChromaDB wrapper.
Role: MEMORY PERSISTENCE — stores past (query, answer, feedback, score).
NOT used for primary retrieval — that is FAISS's job.

Why Chroma here, not FAISS?
  - Rich metadata filtering (by session, feedback type, date)
  - Persistent across restarts without manual serialisation
  - Built-in collection management for multiple memory stores
"""

from typing import Any, List, Optional

import chromadb
from chromadb.config import Settings

from utils.logger import get_logger

import os
import shutil
import tempfile

logger = get_logger(__name__)

DEFAULT_CHROMA_PATH = "data/processed/chroma_db"


def _resolve_chroma_path() -> str:
    path = os.getenv("CHROMA_PATH", DEFAULT_CHROMA_PATH)
    is_serverless = os.getenv("VERCEL") == "1" or os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None
    if is_serverless:
        tmp_path = os.path.join(tempfile.gettempdir(), "chroma_db")
        if not os.path.exists(tmp_path) and os.path.exists(path):
            try:
                shutil.copytree(path, tmp_path, dirs_exist_ok=True)
            except Exception as e:
                logger.warning("chroma_copy_to_tmp_failed", error=str(e))
        return tmp_path
    return path


class ChromaStore:
    """
    Manages a ChromaDB collection for episodic memory.

    Each record stores:
      - document:  the original query text (used for similarity search)
      - metadata:  answer, chunks_used, feedback, score, session_id, timestamp
      - embedding: provided externally (sentence-transformer)
    """

    def __init__(self, collection_name: str = "aura_memory") -> None:
        target_path = _resolve_chroma_path()
        try:
            self._client = chromadb.PersistentClient(
                path=target_path,
                settings=Settings(anonymized_telemetry=False),
            )
        except Exception as exc:
            logger.warning(
                "chroma_persistent_client_fallback_ephemeral",
                path=target_path,
                error=str(exc),
            )
            self._client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "chroma_ready",
            collection=collection_name,
            count=self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store_interaction(
        self,
        interaction_id: str,
        query: str,
        query_embedding: List[float],
        answer: str,
        chunks_used: List[str],
        feedback: str = "",
        feedback_category: str = "",
        confidence: float = 0.0,
        session_id: str = "",
    ) -> None:
        self._collection.upsert(
            ids=[interaction_id],
            documents=[query],
            embeddings=[query_embedding],
            metadatas=[
                {
                    "answer":             answer[:2000],  # Chroma metadata size limit
                    "chunks_used":        " ||| ".join(chunks_used[:3]),
                    "feedback":           feedback,
                    "feedback_category":  feedback_category,
                    "confidence":         confidence,
                    "session_id":         session_id,
                }
            ],
        )
        logger.info("memory_stored", interaction_id=interaction_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query_memory(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        similarity_threshold: float = 0.75,
        filter_by: Optional[dict] = None,
    ) -> List[dict]:
        """
        Retrieve past interactions similar to the current query.
        Applies similarity_threshold gate — only inject high-quality memory.

        Returns list of dicts with keys: query, answer, confidence, distance
        """
        if self._collection.count() == 0:
            return []

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results":        min(top_k, self._collection.count()),
            "include":          ["documents", "metadatas", "distances"],
        }
        if filter_by:
            kwargs["where"] = filter_by

        results = self._collection.query(**kwargs)

        memories = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Chroma cosine distance: 0 = identical, 1 = orthogonal
            # Convert to similarity: sim = 1 - distance
            similarity = 1.0 - float(dist)
            if similarity >= similarity_threshold:
                memories.append(
                    {
                        "query":      doc,
                        "answer":     meta.get("answer", ""),
                        "confidence": meta.get("confidence", 0.0),
                        "similarity": round(similarity, 3),
                        "feedback_category": meta.get("feedback_category", ""),
                    }
                )
        logger.debug(
            "memory_retrieved",
            candidates=len(results["documents"][0]),
            above_threshold=len(memories),
            threshold=similarity_threshold,
        )
        return memories

    def count(self) -> int:
        return self._collection.count()
