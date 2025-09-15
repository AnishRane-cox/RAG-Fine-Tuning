README.md
# LoRA + Embedding Service Demo

This repository demonstrates a **LoRA-augmented language model** with a **document embedding & vector search service**, enabling context-aware question answering.

---

## Architecture
```
         +--------------------+
         | LoRA Model Service |
         |  (FastAPI, 8080)  |
         +--------------------+
                   |
          /generate, /load_adapter
                   |
                   v
         +------------------------+
         |  Inference Adapter     |
         |  (PEFT, HuggingFace)  |
         +------------------------+
                   |
                   v
         +--------------------+
         | Embedding Service  |
         |  (FastAPI, 8081)  |
         |  Mongo Vector DB   |
         +--------------------+
                   |
 /ingest, /embed, /search, /delete_stale
                   |
                   v
        Context-aware Question Answering
```


---

## Requirements

- Python 3.10+  
- `torch`, `transformers`, `peft`, `fastapi`, `uvicorn`, `pydantic`, `pymongo`  
- MongoDB instance (local or Atlas)  

---

## Setup & Run Locally

1. **Clone repo and create virtual environment:**

```bash
git clone <repo-url>
cd Lora_RAGII
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Set environment variables (or .env):
```
BASE_MODEL=distilgpt2
ADAPTER_PATH=./inference_adapter
MONGO_URI=mongodb://localhost:27017
MONGO_DB=rag
MONGO_COLLECTION=documents
```

# Run services:
### LoRA Model Service
```
cd services/model_service
uvicorn app:app --host 127.0.0.1 --port 8080 --reload
```
### Embedding Service
```
cd ../embedding_service
uvicorn main:app --host 127.0.0.1 --port 8081 --reload
```
# API Specifications

### LoRA Model Service (8080)
```
Endpoint	Method	Request Body	Response Example
/health	GET	-	{ "status": "ok", "base_model": "distilgpt2" }
/generate	POST	{ "prompt": "Hello AI!", "max_new_tokens": 32 }	{ "answer": "Generated text..." }
/load_adapter	POST	{ "adapter_path": "./inference_adapter" }	{ "status": "ok", "adapter": "./inference_adapter" }
```
### Embedding Service (8081)
```
Endpoint	Method	Request Body	Response Example
/GET	-	{ "status": "Embedding Service is running" }
/ingest	POST	{ "documents": [{"id":"1","text":"..."}, ...] }	{ "status":"ok","count":3 }
/embed	POST	{ "id":"4", "text":"..." }	{ "id":"4","embedding":[...float values...] }
/bulk_embed	POST	{ "docs":[{"id":"1","text":"..."}, ...] }	{ "upserted":3,"processed":3 }
/search	POST	{ "query":"ML and FastAPI", "k":3 }	{ "results":[{"id":"1","score":0.92,"text":"..."}] }
/delete_stale	POST	{ "days":30 }	{ "deleted":2 }
```
# Demo Script (PowerShell / curl)
### Ingest Documents
```
curl -X POST http://127.0.0.1:8081/ingest `
-H "Content-Type: application/json" `
-d '{
  "documents":[
    {"id":"1","text":"FastAPI is great for ML services"},
    {"id":"2","text":"MongoDB vector search example"}
  ]
}'
```
### Generate Context-Aware Answer
```
curl -X POST http://127.0.0.1:8080/generate `
-H "Content-Type: application/json" `
-d '{"prompt":"Explain vector search in simple terms","max_new_tokens":64}'
```

### Load LoRA Adapter
```
curl -X POST http://127.0.0.1:8080/load_adapter `
-H "Content-Type: application/json" `
-d '{"adapter_path":"./inference_adapter"}'
```
### Search Documents
```
curl -X POST http://127.0.0.1:8081/search `
-H "Content-Type: application/json" `
-d '{"query":"ML and FastAPI","k":3}'
```

# Tradeoffs, Scaling, Cost & Latency

- LoRA adapter: lightweight fine-tuning, minimal GPU memory, fast to load.
- Generation latency: small with DistilGPT2 base, can be slower for large models.
- Embedding & vector search: Mongo Atlas scales horizontally; latency grows with index size.
- Tradeoff: Using Mongo Atlas vs. FAISS in-memory — Atlas persists and scales, FAISS is faster but ephemeral.
- Batch embedding reduces DB write overhead.