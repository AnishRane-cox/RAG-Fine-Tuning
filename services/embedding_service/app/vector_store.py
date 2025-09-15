import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import certifi
from pymongo import MongoClient, ReplaceOne, errors
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

class VectorStore(ABC):
    @abstractmethod
    def upsert(self, doc_id: str, text: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def bulk_upsert(self, docs: List[Dict[str, Any]]) -> int:
        raise NotImplementedError()

    @abstractmethod
    def search(self, embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError()

    @abstractmethod
    def delete_stale(self, older_than_days: int) -> int:
        raise NotImplementedError()

class MongoVectorStore(VectorStore):
    """
    MongoDB Atlas Vector Store Helper
    """
    def __init__(self, mongo_uri: str, db_name: str, collection_name: str):
        try:
            self.client = MongoClient(
                mongo_uri,
                tls = True,
                tlsAllowInvalidCertificates=True,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS = 10000  # 10-second timeout
            )
            # Trigger server selection to catch DNS/network issues early
            self.client.admin.command('ping')

            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            print(f"Connected to MongoDB: {db_name}/{collection_name}")

        except errors.ServerSelectionTimeoutError as e:
            print("Error: Could not connect to MongoDB Atlas.")
            print("Check your network, firewall, IP whitelist, and URI.")
            print(str(e))
            raise
        self._coll: Collection = self.client[db_name][collection_name]
        # index recommendations
        try:
            self._coll.create_index("updated_at")
        except errors.OperationFailure as exc:
            logger.debug("Index creation failed (operation error): %s", exc)
        except errors.ConfigurationError as exc:
            logger.debug("Index creation failed (configuration error): %s", exc)

    def upsert(self, doc_id: str, text: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> bool:
        doc = {
            "_id" : doc_id,
            "text" : text,
            "embedding" : embedding,
            "metadata" : metadata or {},
            "updated_at" : datetime.now(timezone.utc)
        }
        self._coll.replace_one({"_id": doc_id}, doc, upsert=True)
        return True

    def bulk_upsert(self, docs: List[Dict[str, Any]]) -> int | None:
        ops = []
        for doc in docs:
            # Use "id" as "_id" if it exists
            doc_id = doc.get("_id") or doc.get("id")
            if not doc_id:
                continue  # skip if no ID

            # Optional metadata field
            metadata = doc.get("metadata", {})

            ops.append(
                ReplaceOne(
                    {"_id": doc_id},
                    {
                        "_id": doc_id,
                        "text": doc["text"],
                        "embedding": doc["embedding"],
                        "metadata": metadata,
                        "updated_at": datetime.now(timezone.utc)
                    },
                    upsert=True
                )
            )

        if not ops:
            return 0

        result = self._coll.bulk_write(ops, ordered=False)
        return int(getattr(result, "upserted_count", 0) + getattr(result, "modified_count", 0))

    def search(self, embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Uses Atlas Search knnBeta operator. Requires a vector index created in Atlas on 'Embedding'
        Example aggregate:
        [
          { "$search": { "knnBeta": { "vector": <embedding>, "path": "embedding", "k": k } } },
          { "$project": { "_id": 1, "text": 1, "score": { "$meta": "searchScore" } } }
        ]
        """
        pipeline = [
            {
                "$search":{
                    "knnBeta":{
                        "vector": embedding,
                        "path": "embedding",
                        "k": k
                    }
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "metadata": 1,
                    "score": {
                        "$meta": "searchScore"
                    }
                }
            }
        ]
        try:
            cursor = self._coll.aggregate(pipeline)
            results = []
            for doc in cursor:
                results.append(
                    {
                        "id" : str(doc["_id"]),
                        "text" : doc.get("text"),
                        "metadata" : doc.get("metadata", {}),
                        "score" : doc.get("score")
                    }
                )
            return results
        except Exception as exc:
            # If Atlas search isn't available (local Mongo), fallback to a naive linear scan
            logger.warning("Atlas Search failed, falling back to naive search: %s", exc)
            return self._naive_search(embedding, k)

    def _naive_search(self,
                      embedding: List[float],
                      k:int) -> List[Dict[str, Any]]:
        k = int(k)
        import numpy as np
        rows = list(self._coll.find({}, {"embedding": 1, "text":1, "metadata": 1}))
        if not rows:
            return []
        emb_array = np.array([r["embedding"] for r in rows], dtype=np.float32)
        query = np.array(embedding, dtype=np.float32)
        # Cosine Similarity
        emb_array = emb_array / (np.linalg.norm(emb_array, axis=1, keepdims=True) + 1e-12)
        eps = np.float32(1e-12)
        q_norm = query/(np.linalg.norm(query) + eps)
        sims = (emb_array @ q_norm).tolist() # Dot Product
        scored = []
        for r, s in zip (rows, sims):
            scored.append({
                "id": str(r["_id"]),
                "text": r.get("text"),
                "metadata": r.get("metadata", {}),
                "score": float(s)
            })
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:k]

    def delete_stale(self, older_than_days: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        result = self._coll.delete_many({"updated_at": {"$lt": cutoff}})
        return int(result.deleted_count)