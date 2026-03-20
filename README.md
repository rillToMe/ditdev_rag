<div align="center">

# 🧠 ditdev-rag

**Offline RAG (Retrieval-Augmented Generation) service for CHANGLI-AI**  
*The knowledge backbone of [ditdev.kyuzenstudio.com](https://ditdev.kyuzenstudio.com)*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange?style=flat)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🌟 What is this?

**ditdev-rag** is a lightweight, fully offline RAG service that powers CHANGLI-AI - the shrine maiden AI guide of Rahmat Aditya's pixel art portfolio.

Instead of stuffing all portfolio data into the LLM system prompt (expensive & slow), this service:
1. **Embeds** structured portfolio data into a local vector database
2. **Retrieves** only the most relevant chunks for each user query
3. **Injects** that context into the prompt before sending to the LLM

This keeps token usage minimal while keeping CHANGLI-AI's answers accurate and grounded in real data.

---

## ✨ Features

- 🔍 **Semantic search** - finds relevant data by meaning, not just keywords
- ⚡ **Incremental indexing** - add, update, delete single chunks without full rebuild
- 🗄️ **Persistent vector store** - ChromaDB stores embeddings locally on disk
- 🛡️ **Graceful fallback** - if RAG is down, the LLM still responds from its base persona
- 🔄 **Real-time sync** - portfolio data stays in sync with PostgreSQL via Node.js hooks

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Node.js Backend (Express)
    │
    ├── POST /api/chat ──────────────────────────────────┐
    │                                                    │
    │   1. Fetch relevant chunks from RAG                │
    │   2. Inject context into system prompt             │
    │   3. Send to Cerebras LLM (qwen-3-235b)            │
    │   4. Return response to user                       │
    │                                                    │
    └── Admin CRUD ──────────────────────────────────────┤
        │                                                │
        ├── Project/Cert Created → POST /index/add       │
        ├── Project/Cert Updated → POST /index/update    │
        └── Project/Cert Deleted → POST /index/delete    │
                                                         │
                                    ditdev-rag (this) ◄──┘
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                       sentence-transformers     ChromaDB
                       (all-MiniLM-L6-v2)     (persistent)
                       embedding model          vector store
```

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

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PostgreSQL database (for dynamic data)

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

### Configuration

```bash
cp .env.example .env
```

Edit `.env`:
```env
DATABASE_URL=postgresql://user:password@host/dbname
RAG_PORT=8765
RAG_REBUILD_SECRET=your_secret_here
```

### Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8765
```

On first run, the service will automatically:
1. Load all chunks from `skills_data.json` and PostgreSQL
2. Generate embeddings using `all-MiniLM-L6-v2`
3. Store vectors in `chroma_store/` (created automatically)

---

## 📡 API Endpoints

### `GET /health`
Check service status and chunk count.

```json
{ "status": "ok", "chunks": 20 }
```

### `POST /retrieve`
Semantic search - returns most relevant chunks for a query.

```json
// Request
{ "query": "what is adit's unity skill level?", "top_k": 4 }

// Response
{
  "context": "• Skill: Unity | Category: Game Dev | Level: Advanced | ...",
  "found": true
}
```

### `POST /index/add`
Add a new chunk (called automatically on project/cert creation).

```json
{ "chunk_id": "project_32", "text": "Project by Adit-san: ...", "metadata": {} }
```

### `POST /index/update`
Update an existing chunk (called automatically on edit).

### `POST /index/delete`
Delete a chunk (called automatically on deletion).

```json
{ "chunk_id": "project_32" }
```

### `POST /rebuild`
Full index rebuild - for emergencies only.

```json
{ "secret": "your_rebuild_secret" }
```

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

Node.js hooks in the Express backend call these endpoints automatically after every admin CRUD operation - no manual intervention needed.

---

## 📁 Project Structure

```
ditdev-rag/
├── main.py           # FastAPI app & endpoints
├── rag_engine.py     # Core RAG logic (embed, retrieve, CRUD)
├── data_loader.py    # Load static + dynamic data
├── skills_data.json  # Static portfolio data (skills, education, contact)
├── requirements.txt  # Python dependencies
├── .env.example      # Environment template
└── chroma_store/     # ChromaDB vector store (gitignored)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| Embedding Model | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector Database | ChromaDB (persistent, local) |
| LLM | Cerebras API - `qwen-3-235b-a22b-instruct-2507` |
| Database | Neon PostgreSQL |

---

## 🤝 Integration

This service is designed to work with the [ditdev portfolio backend](https://github.com/rillToMe) (private). The Node.js backend calls `/retrieve` on every chat message and `/index/*` on every admin CRUD operation.

---

## 📄 License

MIT © [Rahmat Aditya](https://github.com/rillToMe)