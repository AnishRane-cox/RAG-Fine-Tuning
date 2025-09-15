from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import List
from config import get_settings
from dotenv import load_dotenv
from embedders import SentenceTransformerEmbedder
from models import (
EmbedRequest, BulkEmbedRequest, EmbedResponse,
BulkEmbedResponse, SearchResponse, SearchRequest,
DeleteStaleRequest, DeleteStaleResponse
)
from vector_store import MongoVectorStore
load_dotenv()
settings = get_settings()

app = FastAPI(title="Embedding Service")
DOCUMENTS = []


embedder = SentenceTransformerEmbedder(settings.MODEL_NAME)

try:
    store = MongoVectorStore(
        settings.MONGO_URI,
        settings.MONGO_DB,
        settings.MONGO_COLLECTION
    )
except Exception as e:
    print("Failed to initialize MongoVectorStore:", e)
    store = None

def flatten_vector(vec):
    return vec if isinstance(vec[0], float) else vec[0]

class IngestRequest(BaseModel):
    documents: List[dict]

@app.get("/")
def root():
    return {"status": "Embedding Service is running"}

@app.post("/ingest")
def ingest(req: IngestRequest):
    DOCUMENTS.extend(req.documents)
    return {"status": "ok", "count": len(DOCUMENTS)}

@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    """Return the embedding for a single document."""

    if store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        embedding: List[float]  = embedder.embed(req.text)
        return EmbedResponse(id=req.id,
                             embedding=embedding)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/bulk_embed", response_model=BulkEmbedResponse)
async def bulk_embed(req: BulkEmbedRequest):
    """Embed multiple documents and store them in Mongo Atlas."""
    if store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        texts = [doc.text for doc in req.docs]
        ids = [doc.id for doc in req.docs]

        vectors: List[List[float]] = embedder.embed_batch(texts)
        # Ensure no accidental nesting
        vectors = [flatten_vector(v) for v in vectors]
        docs = [{"id": i, "text": t, "embedding": v} for i, t, v in zip(ids, texts, vectors)]
        count = store.bulk_upsert(docs)
        return BulkEmbedResponse(upserted=count, processed=len(req.docs))

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Search for similar documents given a query string."""
    embedding: List[float] = embedder.embed(req.query)
    results = store.search(embedding, k = req.k)
    return JSONResponse({"results": results})

@app.post("/delete_stale", response_model=DeleteStaleResponse)
async def delete_stale(req: DeleteStaleRequest):
    """Delete documents older than N days (housekeeping)."""
    deleted = store.delete_stale(req.days)
    return {"deleted": deleted}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)