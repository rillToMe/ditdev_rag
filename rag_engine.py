import os
import re
import time
import hashlib
from collections import OrderedDict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from data_loader import load_all_chunks

COLLECTION_NAME    = 'ditdev_portfolio'
CHROMA_PATH        = os.path.join(os.path.dirname(__file__), 'chroma_store')
EMBED_MODEL        = 'intfloat/multilingual-e5-small'
DISTANCE_THRESHOLD = 0.7

# Cache config
CACHE_SIZE = 128   # max unique queries in-cache
CACHE_TTL  = 300   # 5 menit - Stale after data changes


class LRUCache:
    def __init__(self, maxsize: int, ttl: int):
        self.cache   = OrderedDict()
        self.maxsize = maxsize
        self.ttl     = ttl

    def get(self, key: str):
        if key not in self.cache:
            return None
        value, ts = self.cache[key]
        if time.time() - ts > self.ttl:
            del self.cache[key]
            return None
        self.cache.move_to_end(key)
        return value

    def set(self, key: str, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = (value, time.time())
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

    def invalidate(self):
        """Clear all cache - called when the index changes."""
        self.cache.clear()


class RAGEngine:
    def __init__(self):
        print('[RAG] Loading embedding model...')
        self.embedder = SentenceTransformer(
            EMBED_MODEL,
            device='cpu',
            local_files_only=True
        )

        print('[RAG] Initializing ChromaDB...')
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={'hnsw:space': 'cosine'}
        )

        self.cache = LRUCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)

        if self.collection.count() == 0:
            print('[RAG] Collection empty - building initial index...')
            self._build_index()
        else:
            print(f'[RAG] Collection ready - {self.collection.count()} chunks')

    # Query preprocessing

    def _preprocess_query(self, query: str) -> str:
        q = query.lower().strip()
        q = re.sub(r'\s+', ' ', q)
        q = re.sub(r'[^\w\s\?\.]', '', q)

        # Synonym expansion
        synonyms = {
            r'\bbelajar\b'   : 'coding studying learning',
            r'\blama\b'      : 'duration months years time',
            r'\bproject\b'   : 'projects quests creations',
            r'\bproyek\b'    : 'projects quests creations',
            r'\bskill\b'     : 'skills abilities tech stack',
            r'\bkemampuan\b' : 'skills abilities',
            r'\bsertif\b'    : 'certificate achievement',
            r'\bbuat\b'      : 'built created made',
            r'\bhubungi\b'   : 'contact reach',
            r'\bberapa\b'    : 'how many total count',
            r'\btotal\b'     : 'total count how many',
            r'\bmengenal\b': 'start begin first time early',
            r'\bawal\b': 'start beginning first time',
            r'\bsejak\b': 'since from when',
        }
        for pattern, expansion in synonyms.items():
            if re.search(pattern, q):
                q = q + ' ' + expansion

        return q.strip()

    def _cache_key(self, query: str, top_k: int) -> str:
        raw = f"{query}:{top_k}"
        return hashlib.md5(raw.encode()).hexdigest()

    # Dynamic top_k

    def _dynamic_top_k(self, query: str) -> int:
        words = len(query.split())
        if words <= 4:
            return 2   # short
        elif words <= 8:
            return 3   # medium
        else:
            return 5   # complex


    def retrieve(self, query: str, top_k: int = None) -> str:
        processed = self._preprocess_query(query)

        if top_k is None:
            top_k = self._dynamic_top_k(processed)

        cache_key = self._cache_key(processed, top_k)
        cached = self.cache.get(cache_key)
        if cached is not None:
            print(f'[RAG] Cache hit: {cache_key[:8]}...')
            return cached

        query_embedding = self._embed(processed, is_query=True)
        n = min(top_k * 2, self.collection.count())  

        results = self.collection.query(
            query_embeddings = [query_embedding],
            n_results        = n,
            include          = ['documents', 'distances', 'metadatas']
        )

        docs      = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]

        candidates = [
            (doc, dist, meta)
            for doc, dist, meta in zip(docs, distances, metadatas)
            if dist < DISTANCE_THRESHOLD
        ]

        if not candidates:
            self.cache.set(cache_key, '')
            return ''

        # --- Scoring with type priority + intent boost ---

        TYPE_PRIORITY = {
            'stats'      : 0.3,
            'project'    : 0.2,
            'certificate': 0.2,
            'skill'      : 0.1,
            'education'  : 0.12,
            'about'      : 0.05,
            'contact'    : 0.05,
        }

        def score(item):
            doc, dist, meta = item
            base_score = 1 - dist

            # type-based priority
            base_score += TYPE_PRIORITY.get(meta.get('type', ''), 0)

            # intent boost
            q = processed
            if any(k in q for k in ['berapa', 'total', 'how many', 'count', 'bulan', 'months', 'lama', 'banyak', 'how long', 'duration']):
                if meta.get('type') == 'stats':
                    base_score += 0.3

            if any(k in q for k in ['skill', 'kemampuan', 'bisa', 'pakai']):
                if meta.get('type') == 'skill':
                    base_score += 0.2

            if any(k in q for k in ['project', 'proyek', 'buat', 'build', 'portfolio']):
                if meta.get('type') == 'project':
                    base_score += 0.2
                    
            if any(k in q for k in ['awal', 'mulai', 'mengenal', 'sejak', 'dari kapan', 'background', 'sekolah']):
                if meta.get('type') == 'education':
                    base_score += 0.3

            # keyword overlap micro-priority
            query_words = set(q.split())
            doc_words   = set(doc.lower().split())
            overlap     = len(query_words & doc_words)
            base_score += overlap * 0.01

            return base_score

        candidates.sort(key=score, reverse=True)

        top = candidates[:top_k]

        # --- Structured context output with normalized priority ---

        scores    = [score(item) for item in top]
        max_score = max(scores) if scores else 1.0

        blocks = []
        for (doc, dist, meta), raw_score in zip(top, scores):
            priority = round(raw_score / max_score, 2)
            chunk_type = meta.get('type', 'unknown')
            chunk_id   = meta.get('name') or meta.get('title') or meta.get('db_id') or 'unknown'

            blocks.append(
                f"[DATA]\n"
                f"type: {chunk_type}\n"
                f"id: {chunk_id}\n"
                f"priority: {priority}\n"
                f"content: {doc}"
            )

        context = '[REALM DATA]\n\n' + '\n\n'.join(blocks) if blocks else ''

        self.cache.set(cache_key, context)
        return context

    # Internal

    def _embed(self, text: str, is_query: bool = False) -> list[float]:
        if is_query:
            text = f"query: {text}"
        else:
            text = f"passage: {text}"
        return self.embedder.encode([text]).tolist()[0]

    def _build_index(self):
        chunks = load_all_chunks()
        if not chunks:
            print('[RAG] No chunks to index!')
            return

        ids        = [c['id']               for c in chunks]
        texts      = [c['text']             for c in chunks]
        metadatas  = [c.get('metadata', {}) for c in chunks]
        prefixed   = [f"passage: {t}" for t in texts]
        embeddings = self.embedder.encode(prefixed, show_progress_bar=True).tolist()

        batch = 100
        for i in range(0, len(chunks), batch):
            self.collection.upsert(
                ids        = ids[i:i+batch],
                documents  = texts[i:i+batch],
                embeddings = embeddings[i:i+batch],
                metadatas  = metadatas[i:i+batch],
            )
        print(f'[RAG] Index built: {self.collection.count()} chunks')

    # Incremental ops 

    def add_chunk(self, chunk_id: str, text: str, metadata: dict = {}) -> bool:
        try:
            embedding = self._embed(text)
            self.collection.upsert(
                ids=[chunk_id], documents=[text],
                embeddings=[embedding], metadatas=[metadata],
            )
            self.cache.invalidate()
            print(f'[RAG] Added chunk: {chunk_id}')
            return True
        except Exception as e:
            print(f'[RAG] Error adding chunk {chunk_id}: {e}')
            return False

    def update_chunk(self, chunk_id: str, text: str, metadata: dict = {}) -> bool:
        try:
            embedding = self._embed(text)
            self.collection.upsert(
                ids=[chunk_id], documents=[text],
                embeddings=[embedding], metadatas=[metadata],
            )
            self.cache.invalidate()
            print(f'[RAG] Updated chunk: {chunk_id}')
            return True
        except Exception as e:
            print(f'[RAG] Error updating chunk {chunk_id}: {e}')
            return False

    def delete_chunk(self, chunk_id: str) -> bool:
        try:
            self.collection.delete(ids=[chunk_id])
            self.cache.invalidate()
            print(f'[RAG] Deleted chunk: {chunk_id}')
            return True
        except Exception as e:
            print(f'[RAG] Error deleting chunk {chunk_id}: {e}')
            return False

    def chunk_exists(self, chunk_id: str) -> bool:
        try:
            result = self.collection.get(ids=[chunk_id])
            return len(result['ids']) > 0
        except:
            return False

    # Full rebuild 

    def rebuild_index(self):
        print('[RAG] Full rebuild triggered...')
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={'hnsw:space': 'cosine'}
        )
        self.cache.invalidate()
        self._build_index()
        print(f'[RAG] Rebuild complete: {self.collection.count()} chunks')

    def cache_stats(self) -> dict:
        return {
            'size'   : len(self.cache.cache),
            'maxsize': self.cache.maxsize,
            'ttl'    : self.cache.ttl,
        }