from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
from rag_engine import RAGEngine

load_dotenv()

app = FastAPI(title='DitDev RAG Service', version='2.0.0')

rag: RAGEngine = None

@app.on_event('startup')
async def startup():
    global rag
    print('[RAG Service] Starting up...')
    rag = RAGEngine()
    print('[RAG Service] Ready!')

def get_rag() -> RAGEngine:
    if not rag:
        raise HTTPException(status_code=503, detail='RAG engine not ready')
    return rag

#Request/Response models 

class RetrieveRequest(BaseModel):
    query : str
    top_k : int = 4

class RetrieveResponse(BaseModel):
    context: str
    found  : bool

class IndexAddRequest(BaseModel):
    chunk_id: str
    text    : str
    metadata: Optional[dict] = {}

class IndexDeleteRequest(BaseModel):
    chunk_id: str

class RebuildRequest(BaseModel):
    secret: str

# Endpoints

@app.get('/health')
def health():
    r = get_rag()
    return {'status': 'ok', 'chunks': r.collection.count()}


@app.post('/retrieve', response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    r = get_rag()
    if not req.query.strip():
        return RetrieveResponse(context='', found=False)
    context = r.retrieve(req.query.strip(), top_k=req.top_k)
    return RetrieveResponse(context=context, found=bool(context))

#add chunk
@app.post('/index/add')
def index_add(req: IndexAddRequest):
    r = get_rag()
    ok = r.add_chunk(req.chunk_id, req.text, req.metadata or {})
    if not ok:
        raise HTTPException(status_code=500, detail='Failed to add chunk')
    return {'status': 'added', 'chunk_id': req.chunk_id, 'total': r.collection.count()}

#update chunk
@app.post('/index/update')
def index_update(req: IndexAddRequest):
    r = get_rag()
    ok = r.update_chunk(req.chunk_id, req.text, req.metadata or {})
    if not ok:
        raise HTTPException(status_code=500, detail='Failed to update chunk')
    return {'status': 'updated', 'chunk_id': req.chunk_id}

#delete chunk
@app.post('/index/delete')
def index_delete(req: IndexDeleteRequest):
    r = get_rag()
    ok = r.delete_chunk(req.chunk_id)
    if not ok:
        raise HTTPException(status_code=500, detail='Failed to delete chunk')
    return {'status': 'deleted', 'chunk_id': req.chunk_id, 'total': r.collection.count()}

#rebuild chunk
@app.post('/rebuild')
def rebuild(req: RebuildRequest):
    if req.secret != os.getenv('RAG_REBUILD_SECRET', 'changli_rebuild'):
        raise HTTPException(status_code=401, detail='Invalid secret')
    r = get_rag()
    r.rebuild_index()
    return {'status': 'rebuilt', 'chunks': r.collection.count()}

# Cache stats
@app.get('/cache/stats')
def cache_stats():
    r = get_rag()
    return r.cache_stats()

# Cache clear
@app.post('/cache/clear')
def cache_clear():
    r = get_rag()
    r.cache.invalidate()
    return {'status': 'cleared'}


if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('RAG_PORT', 8765))
    uvicorn.run('main:app', host='0.0.0.0', port=port, reload=False)