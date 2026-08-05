<div align="center">

# 🧠 ditdev-rag

**RAG (Retrieval-Augmented Generation) service for CHANGLI-AI**  
*The knowledge backbone of [ditdev.kyuzenstudio.com](https://ditdev.kyuzenstudio.com)*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange?style=flat)](https://trychroma.com)
[![Workers AI](https://img.shields.io/badge/Workers%20AI-bge--m3-F38020?style=flat&logo=cloudflare&logoColor=white)](https://developers.cloudflare.com/workers-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🌟 What is this?

**ditdev-rag** is a lightweight RAG service that powers CHANGLI-AI - the shrine maiden AI guide of Rahmat Aditya's pixel art portfolio.

Instead of stuffing all portfolio data into the LLM system prompt (expensive & slow), this service:
1. **Embeds** structured portfolio data into a local vector database
2. **Retrieves** only the most relevant chunks for each user query
3. **Injects** that context into the prompt before sending to the LLM

This keeps token usage minimal while keeping CHANGLI-AI's answers accurate and grounded in real data.

Embeddings come from **Cloudflare Workers AI** (`@cf/baai/bge-m3`). The vector store
is still local: only the embedding call leaves the machine.

---

## ✨ Features

- 🔍 **Semantic search** - finds relevant data by meaning, not just keywords
- ⚡ **Incremental indexing** - add, update, delete single chunks without full rebuild
- 🗄️ **Persistent vector store** - ChromaDB stores embeddings locally on disk
- 🪶 **No local model** - no torch, no 470 MB of weights, no cold start; embeddings are one HTTPS call
- 🛡️ **Graceful fallback** - if RAG is down, the LLM still responds from its base persona
- 🔄 **Real-time sync** - portfolio data stays in sync with PostgreSQL via backend index hooks
- 🧹 **Self-healing index** - startup compares the index against Postgres and rebuilds on drift (or on an embedding dimension change), so a hook lost while this service was down repairs itself
- 🚫 **Off-topic filtering** - a cosine distance cutoff means `found: false` is a real answer, not a formality

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Rust Backend (axum)
    │
    ├── POST /api/chat ──────────────────────────────────┐
    │                                                    │
    │   1. POST /retrieve with the last 2 user turns      │
    │   2. Inject context into system prompt             │
    │   3. Send to Cerebras LLM                          │
    │   4. Return response to user                       │
    │                                                    │
    └── Admin CRUD ──────────────────────────────────────┤
        │                                                │
        ├── Project/Cert Created → POST /index/add       │
        ├── Project/Cert Updated → POST /index/update    │
        ├── Project/Cert Deleted → POST /index/delete    │
        └── any of the above    → POST /index/refresh-derived
                                  (recompute totals + list chunks)
                                                         │
                                    ditdev-rag (this) ◄──┘
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                      Cloudflare Workers AI      ChromaDB
                        (@cf/baai/bge-m3)       (persistent)
                        embeddings, 1024-d      vector store
```

The backend never formats chunk text for whole-DB summaries; it just asks for a
refresh. That template lives here only, so the two sides cannot drift.

`embeddings.py` is the only module that knows how a vector is produced. Swapping
providers means adding one `EmbeddingProvider` subclass and injecting it into
`RAGEngine(embedder=...)` - retrieval, scoring and the store stay untouched.

---

## 📦 Data Structure

Portfolio data is split into semantic chunks across categories:

| Category | Source | Example |
|----------|--------|---------|
| `skill` | `skills_data.json` | Unity (Advanced), React (Intermediate) |
| `project` | PostgreSQL (dynamic) | Game projects, web apps |
| `certificate` | PostgreSQL (dynamic) | Certificates earned |
| `education` | `skills_data.json` | SMK Negeri 4 Payakumbuh |
| `about` | `skills_data.json` | Background, location, links |
| `contact` | `skills_data.json` | Availability, open for work |
| `stats` | PostgreSQL (derived) | Authoritative project/certificate totals |

Three chunks summarise the whole corpus - `stats_summary`, `projects_summary` and
`skills_summary`. Retrieval returns 3-5 chunks, so "list all your skills" can
never be answered by walking the individual `skill` chunks; the summary chunk
carries the complete list in one hit.

The coding duration is **not** stored in the indexed text. It is recomputed from
`coding_start` on every `/retrieve`, because a baked-in month count goes stale the
moment a month passes with no admin CRUD - while the same chunk tells the LLM the
number is authoritative.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL database (for dynamic data)
- A Cloudflare account with Workers AI, and an API token with **Workers AI: Read**

### Installation

```bash
# Clone the repo
git clone https://github.com/rillToMe/ditdev-rag.git
cd ditdev-rag

# Create virtual environment
python -m venv rag-env
source rag-env/bin/activate  # Windows: rag-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

No model download step: embeddings are an HTTPS call to Workers AI. If you are
upgrading from the local-model version, `pip uninstall sentence-transformers torch`
reclaims a couple of GB.

### Configuration

```bash
cp env.example .env
```

Edit `.env`:
```env
DATABASE_URL=postgresql://user:password@host/dbname
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_API_TOKEN=your_workers_ai_token
RAG_PORT=8765
# Generate one. /rebuild stays disabled while this is empty.
RAG_REBUILD_SECRET=your_secret_here
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `CLOUDFLARE_ACCOUNT_ID` | *(required)* | Workers AI account. Startup fails without it |
| `CLOUDFLARE_API_TOKEN` | *(required)* | Token with `Workers AI: Read`. Never logged |
| `CLOUDFLARE_EMBEDDING_MODEL` | `@cf/baai/bge-m3` | Embedding model; swap without code changes |
| `RAG_HOST` | `127.0.0.1` | Bind address. Anything non-local requires `RAG_API_SECRET` |
| `RAG_PORT` | `8765` | Listen port |
| `RAG_API_SECRET` | *(empty)* | Required on every mutating route once set; sent by the backend as `X-RAG-Secret` |
| `RAG_REBUILD_SECRET` | *(empty)* | `/rebuild` is disabled while unset - no fallback default |
| `RAG_DISTANCE_THRESHOLD` | `0.21` | Cosine cutoff for "relevant". **Model-specific** - re-measure after changing the model |
| `RAG_LOG_LEVEL` | `INFO` | Log verbosity |

### Run

```bash
python main.py
```

Run it this way rather than through `uvicorn` directly: `__main__` refuses a
non-local bind without `RAG_API_SECRET`, because `/index/*` text lands verbatim in
a public chatbot's system prompt.

Single process only. The query cache lives in memory, so extra workers mean N
caches that never agree.

On first run, the service will automatically:
1. Load all chunks from `skills_data.json` and PostgreSQL
2. Embed them through Workers AI in one batched request
3. Store vectors in `chroma_store/` (created automatically)

On every later start it compares the index against Postgres and rebuilds on drift -
in chunk count, or in embedding dimension after a model swap - unless the DB is
unreachable, in which case it keeps the existing index rather than replacing it
with a static-only one.

### Test

```bash
python test_retrieve.py
```

Pure-logic checks always run and need no credentials: the embedding provider is
exercised against a mocked transport (batching, retries, malformed responses).
The integration section self-skips when credentials or the index are missing, and
prints the min-distance spread per query so
`RAG_DISTANCE_THRESHOLD` can be tuned by eye: on-topic queries should sit well
below it, off-topic ones above. The 0.10-0.19 on-topic / 0.23-0.25 off-topic figures
behind the current default were measured on the **old local model**, so re-run this
with credentials set and re-tune before trusting `found: false` from bge-m3.

---

## 📡 API Endpoints

Every mutating route (`/index/*`, `/rebuild`, `/cache/clear`) requires the
`X-RAG-Secret` header once `RAG_API_SECRET` is set.

Any route that has to embed returns **503** if Workers AI is unreachable after
retries - never a wrong answer built from a failed embedding.

### `GET /health`
Service status, chunk census, last known DB state and the model that built the index.
`degraded` means the index is empty, Postgres was unreachable, or the authoritative
`stats` chunk is missing.

```json
{
  "status": "ok",
  "chunks": 31,
  "by_type": { "skill": 12, "project": 8, "certificate": 4, "stats": 1 },
  "db_ok": true,
  "embed_model": "@cf/baai/bge-m3",
  "cache": { "size": 3, "maxsize": 128, "ttl": 300 }
}
```

### `POST /retrieve`
Semantic search - returns most relevant chunks for a query.

```json
// Request. Omit top_k and the service sizes it from the query (3-5).
{ "query": "what is adit's unity skill level?" }

// Response
{
  "context": "[REALM DATA]\n\n[DATA]\ntype: skill\nid: Unity\nrelevance: 0.87\ncontent: Skill: Unity | Category: Game Dev | Level: Advanced | ...",
  "found": true
}
```

`relevance` is absolute (`1 - cosine distance`), not normalised against the best
hit, so the LLM can tell a strong match from a weak one. Queries past the distance
cutoff return `{"context": "", "found": false}`.

### `POST /index/add`
Add a new chunk (called automatically on project/cert creation).

```json
{ "chunk_id": "project_32", "text": "Project by Adit-san: ...", "metadata": {} }
```

### `POST /index/update`
Update an existing chunk (called automatically on edit). Same upsert as `/index/add`.

### `POST /index/delete`
Delete a chunk (called automatically on deletion).

```json
{ "chunk_id": "project_32" }
```

### `POST /index/refresh-derived`
Recompute the whole-DB summary chunks (`stats_summary`, `projects_summary`,
`skills_summary`) from Postgres. Called by the backend after any create/update/delete.
Returns `503` if the DB is down, so a stale total surfaces as an error instead of
being served as authoritative.

```json
{ "status": "refreshed", "chunk_ids": ["stats_summary", "projects_summary", "skills_summary"] }
```

### `POST /rebuild`
Full index rebuild - for emergencies only. Disabled unless `RAG_REBUILD_SECRET` is set.

```json
{ "secret": "your_rebuild_secret" }
```

### `GET /cache/stats` · `POST /cache/clear`
Inspect or drop the query cache (128 entries, 300s TTL). Index writes invalidate it
automatically; the manual clear is for when you edit `skills_data.json` by hand.

---

## 🔧 Incremental Indexing

This is the key design decision. Instead of rebuilding the entire index on every change:

```
INSERT project → embed 1 chunk → upsert to ChromaDB  (~50ms)
UPDATE project → re-embed 1 chunk → upsert to ChromaDB  (~50ms)
DELETE project → delete by chunk_id from ChromaDB  (~5ms)

vs.

Full rebuild → embed ALL chunks → store ALL  (~5-30s)
```

The backend's index hooks are fire-and-forget: a failed hook is logged and dropped,
never propagated to the admin request. Startup reconciliation is what makes that
safe - a chunk missed while this service was down is repaired on its next boot.

---

## 📁 Project Structure

```
ditdev-rag/
├── main.py            # FastAPI app & endpoints
├── rag_engine.py      # Core RAG logic (embed, score, retrieve, index writes)
├── embeddings.py      # EmbeddingProvider + Cloudflare Workers AI implementation
├── data_loader.py     # Static + dynamic chunk loading; owns all chunk templates
├── test_retrieve.py   # Logic checks + distance probe
├── skills_data.json   # Static portfolio data (skills, education, contact)
├── requirements.txt   # Python dependencies
├── env.example        # Environment template
└── chroma_store/      # ChromaDB vector store (gitignored)
```

---

## 🔁 Migrating from the local model

If you are updating an existing deployment that ran `multilingual-e5-small` locally:

1. **Set the new env vars.** `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`
   (token needs `Workers AI: Read`). Startup fails fast without them instead of
   erroring on the first user query. `RAG_EMBED_MODEL` is gone - use
   `CLOUDFLARE_EMBEDDING_MODEL`.
2. **Let the first boot rebuild the index.** The old store holds 384-d vectors and
   bge-m3 emits 1024-d; Chroma rejects a mismatched query. Startup probes the live
   dimension, sees the drift and rebuilds automatically - no manual `rm -rf
   chroma_store/`. Requires Postgres to be reachable, or it keeps the old index and
   reports `degraded`.
3. **Re-tune `RAG_DISTANCE_THRESHOLD`.** The `0.21` default was measured on e5-small.
   Run `python test_retrieve.py` with credentials set and read the printed
   min-distance table: pick a cutoff above every on-topic query and below every
   off-topic one. Until you do, `found: false` is not trustworthy.
4. **Optional cleanup.** `pip uninstall sentence-transformers torch` reclaims a
   couple of GB. `onnxruntime` may stay as a ChromaDB transitive dependency; it is
   never invoked, since every embedding is passed in explicitly.

Nothing else changed: chunk templates, scoring, retrieval, the store and every API
route are the same. `embeddings.py` is the only module that knows how a vector is
made.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| Embedding Model | `@cf/baai/bge-m3` via Cloudflare Workers AI (1024-d) |
| HTTP Client | httpx (one pooled `Client`, retries with backoff) |
| Vector Database | ChromaDB (persistent, local, cosine space) |
| LLM | Cerebras API - `gpt-oss-120b` |
| Database | Neon PostgreSQL |

The model is multilingual because the queries are: CHANGLI-AI is asked things like
"berapa total project adit" far more often than the English equivalent. bge-m3 is
prefix-free - the `query:`/`passage:` prefixes the previous e5 model required are
gone, and prepending them now would only add noise.

---

## 🤝 Integration

This service is designed to work with the [ditdev portfolio backend](https://github.com/rillToMe) (private).
The Rust (axum) backend calls `/retrieve` on every chat message, `/index/*` on every
admin CRUD operation, and `/index/refresh-derived` afterwards so the totals and list
chunks stay correct.

---

## 📄 License

MIT © [Rahmat Aditya](https://github.com/rillToMe)