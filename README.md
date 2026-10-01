# 🧩 RAG + LoRA Microservices — Retrieval-Augmented Chat with a Fine-Tuned Adapter

![Python](https://img.shields.io/badge/Python-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-Express%20orchestrator-339933?style=flat-square&logo=nodedotjs&logoColor=white)
![React](https://img.shields.io/badge/React-Demo%20UI-61DAFB?style=flat-square&logo=react&logoColor=black)
![HuggingFace](https://img.shields.io/badge/HF-PEFT%20%2F%20LoRA-FFD21E?style=flat-square)
![MongoDB](https://img.shields.io/badge/MongoDB%20Atlas-Vector%20Search-47A248?style=flat-square&logo=mongodb&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)

> A small but **production-minded** RAG system split into independent services: a **sentence-transformer embedding + vector-search API**, a **LoRA-adapted language model API**, a **Node.js orchestrator**, and a **React chat UI** — all containerised.

The focus is **system design**: clean service boundaries, swappable components (SOLID interfaces), and the trade-offs of fine-tuning vs. retrieval.

---

## 🏗️ Architecture

```mermaid
flowchart LR
    UI[React demo UI<br/>rag_demo] -->|POST /chat| O[Orchestrator<br/>Node.js · Express]
    O -->|POST /search| E[Embedding service<br/>FastAPI]
    E --> ST[all-MiniLM-L6-v2<br/>sentence-transformers]
    E <--> DB[(MongoDB Atlas<br/>vector search · knn)]
    O -->|POST /generate<br/>prompt + retrieved context| M[Model service<br/>FastAPI]
    M --> L[distilgpt2 + LoRA adapter<br/>PEFT]
    O -->|answer + sources| UI
```

**Request flow:** question → embed & retrieve top-k documents → build an augmented prompt (*context + question*) → generate with the LoRA-adapted model → return the answer **with its source passages**.

## 🧱 Services

| Service | Stack | Key endpoints | Responsibility |
|---|---|---|---|
| `embedding_service` | FastAPI, sentence-transformers, PyMongo | `/embed`, `/bulk_embed`, `/search`, `/delete_stale` | Embeds text, upserts vectors in batches, k-NN search, TTL-style housekeeping |
| `model_service` | FastAPI, Transformers, PEFT | `/health`, `/generate`, `/load_adapter` | Serves `distilgpt2` + LoRA adapter; **hot-swaps adapters** at runtime |
| `orchestrator_service` | Node.js, Express | `/chat`, `/health` | Retrieval → prompt building → generation; error handling per downstream call |
| `rag_demo` | React | — | Minimal chat UI that shows the answer and its sources |

`train_adapter.py` trains a **LoRA adapter** with PEFT on a small Q&A set (engineering + ML questions), CPU-friendly, and saves it for the model service to load.

## 🧠 Design Choices & Trade-offs

- **Abstract `Embedder` and `VectorStore` interfaces** → swap sentence-transformers for OpenAI embeddings, or MongoDB for FAISS/pgvector, without touching the API layer.
- **LoRA vs. full fine-tuning** – LoRA trains only a small fraction of the weights, uses little memory and adapters load in seconds; retrieval supplies fresh facts while the adapter shapes tone and domain vocabulary.
- **MongoDB Atlas vs. FAISS** – Atlas persists and scales horizontally; FAISS is faster in-memory but ephemeral.
- **Batch embedding** reduces round-trips and DB write overhead.
- **Latency** – `distilgpt2` keeps generation fast on CPU; a larger base model would improve answer quality at higher latency/cost.

## 🚀 Run Locally

**1 · Configure** — create `.env` files (never commit them):

```bash
# services/embedding_service/app/.env
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>/   # your own Atlas URI
MONGO_DB=rag
MONGO_COLLECTION=documents
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# services/orchestrator_service/.env
PORT=3000
EMBEDDING_URL=http://localhost:8080
MODEL_URL=http://localhost:5000/generate
```

Create an Atlas **vector search index** on the `embedding` field.

**2 · Start the services**

```bash
# Embedding service
cd services/embedding_service/app
pip install -r ../requirments fastapi uvicorn sentence-transformers pymongo certifi pydantic-settings python-dotenv
uvicorn main:app --port 8080

# Model service (train an adapter first, or use the base model)
python train_adapter.py
cd ../../model_service && uvicorn app:app --port 5000

# Orchestrator
cd ../orchestrator_service && npm install && npm start

# UI
cd ../rag_demo && npm install && npm start
```

**3 · Try it**

```bash
curl -X POST http://localhost:8080/bulk_embed -H "Content-Type: application/json" \
  -d '{"docs":[{"id":"1","text":"FastAPI is great for ML services"},{"id":"2","text":"MongoDB supports vector search"}]}'

curl -X POST http://localhost:3000/chat -H "Content-Type: application/json" \
  -d '{"query":"Which database supports vector search?","k":2}'
```

A `docker-compose.yml` is included for running the stack in containers.

## 📁 Repository Structure

```
services/
├── embedding_service/app/   # FastAPI: embedders.py, vector_store.py, models.py, train_adapter.py
├── model_service/           # FastAPI: LoRA inference (app.py, inference_adapter.py)
├── orchestrator_service/    # Express: controllers/, services/ (embedding & model clients)
├── rag_demo/                # React UI
└── sample_doc.json          # Sample documents to ingest
docker-compose.yml
```

## 🔭 Roadmap

- Move all secrets to environment variables only; add `.env.example` files.
- Align Docker Compose build paths/ports and add a Dockerfile for the model service.
- Add retrieval evaluation (hit-rate / MRR) and response streaming.
- Replace `distilgpt2` with an instruction-tuned small model (e.g. Qwen / Llama 3.2 1B) + QLoRA.

---

## 👤 Author

**Anish Rane** — Data & AI Engineer · MSc Machine Learning & AI (LJMU) · Mechanical Engineer

[![Portfolio](https://img.shields.io/badge/Portfolio-1D9E75?style=flat-square&logo=githubpages&logoColor=white)](https://anishrane-cox.github.io/Portfolio/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anish-rane/)
[![GitHub](https://img.shields.io/badge/GitHub-AnishRane--cox-181717?style=flat-square&logo=github)](https://github.com/AnishRane-cox)

⭐ If you found this useful, consider starring the repo.
