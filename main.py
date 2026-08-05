import logging
import os
import secrets
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from data_loader import DBUnavailable
from embeddings import EmbeddingError
from rag_engine import RAGEngine

load_dotenv()

logging.basicConfig(
    level=os.getenv('RAG_LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
)
log = logging.getLogger('ditdev_rag')

HOST           = os.getenv('RAG_HOST', '127.0.0.1')
PORT           = int(os.getenv('RAG_PORT', '8765'))
API_SECRET     = os.getenv('RAG_API_SECRET', '')
REBUILD_SECRET = os.getenv('RAG_REBUILD_SECRET', '')
LOCAL_HOSTS    = {'127.0.0.1', 'localhost', '::1'}

rag: RAGEngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global rag
    log.info('Starting up...')
    # Embeddings come from Cloudflare Workers AI; a missing CLOUDFLARE_* var raises
    # EmbeddingConfigError here rather than on the first user query.
    rag = RAGEngine()
    log.info('Ready on %s:%s', HOST, PORT)
    yield
    rag.close()
    rag = None


app = FastAPI(title='DitDev RAG Service', version='2.1.0', lifespan=lifespan)


@app.exception_handler(EmbeddingError)
async def embedding_error(_: Request, exc: EmbeddingError):
    log.error('Embedding failed: %s', exc)
    return JSONResponse(status_code=503, content={'detail': f'Embedding provider unavailable: {exc}'})


def get_rag() -> RAGEngine:
    if rag is None:
        raise HTTPException(status_code=503, detail='RAG engine not ready')
    return rag


def require_secret(x_rag_secret: str = Header(default='')):
    if API_SECRET and not secrets.compare_digest(x_rag_secret, API_SECRET):
        raise HTTPException(status_code=401, detail='Invalid or missing X-RAG-Secret')


# Request/response models

class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    # None lets the engine size top_k from the query. The old `= 4` default meant
    # every caller silently pinned it and _dynamic_top_k() was dead code.
    top_k: int | None = Field(default=None, ge=1, le=20)


class RetrieveResponse(BaseModel):
    context: str
    found  : bool


class ChunkUpsertRequest(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=200)
    text    : str = Field(min_length=1, max_length=20_000)
    metadata: dict | None = None


class ChunkDeleteRequest(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=200)


class RebuildRequest(BaseModel):
    secret: str


# Endpoints

@app.get('/health')
def health():
    return get_rag().health()


@app.post('/retrieve', response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    context = get_rag().retrieve(req.query, top_k=req.top_k)
    return RetrieveResponse(context=context, found=bool(context))


# /index/add and /index/update are the same upsert; both kept so the backend
# hooks stay readable.
@app.post('/index/add', dependencies=[Depends(require_secret)])
@app.post('/index/update', dependencies=[Depends(require_secret)])
def index_upsert(req: ChunkUpsertRequest):
    r = get_rag()
    if not r.upsert_chunk(req.chunk_id, req.text, req.metadata):
        raise HTTPException(status_code=500, detail='Failed to upsert chunk')
    return {'status': 'upserted', 'chunk_id': req.chunk_id, 'total': r.collection.count()}


@app.post('/index/delete', dependencies=[Depends(require_secret)])
def index_delete(req: ChunkDeleteRequest):
    r = get_rag()
    if not r.delete_chunk(req.chunk_id):
        raise HTTPException(status_code=500, detail='Failed to delete chunk')
    return {'status': 'deleted', 'chunk_id': req.chunk_id, 'total': r.collection.count()}


@app.post('/index/refresh-derived', dependencies=[Depends(require_secret)])
def index_refresh_derived():
    try:
        refreshed = get_rag().refresh_derived()
    except DBUnavailable as e:
        raise HTTPException(status_code=503, detail=f'Database unavailable: {e}') from e
    return {'status': 'refreshed', 'chunk_ids': refreshed}


@app.post('/rebuild', dependencies=[Depends(require_secret)])
def rebuild(req: RebuildRequest):
    # No hardcoded fallback: the old default ('changli_rebuild') was committed in
    # env.example, so it was effectively public.
    if not REBUILD_SECRET:
        raise HTTPException(status_code=503, detail='RAG_REBUILD_SECRET is not set; rebuild disabled')
    if not secrets.compare_digest(req.secret, REBUILD_SECRET):
        raise HTTPException(status_code=401, detail='Invalid secret')
    try:
        total = get_rag().rebuild_index()
    except DBUnavailable as e:
        raise HTTPException(status_code=503, detail=f'Database unavailable: {e}') from e
    return {'status': 'rebuilt', 'chunks': total}


@app.get('/cache/stats')
def cache_stats():
    return get_rag().cache_stats()


@app.post('/cache/clear', dependencies=[Depends(require_secret)])
def cache_clear():
    get_rag().cache.invalidate()
    return {'status': 'cleared'}


if __name__ == '__main__':
    if HOST not in LOCAL_HOSTS and not API_SECRET:
        raise SystemExit(
            f'Refusing to bind {HOST} without RAG_API_SECRET: /index/* would let anyone '
            f'inject text straight into the chatbot prompt. Set RAG_API_SECRET or keep '
            f'RAG_HOST=127.0.0.1.'
        )
    # Keep this single-process: the index is built at startup and the query cache
    # lives in memory, so extra workers mean N model copies and N caches.
    
    uvicorn.run(
    "main:app",
    host=HOST,
    port=PORT,
    reload=False,
    workers=1,
    loop="uvloop",
    http="httptools",
    )
