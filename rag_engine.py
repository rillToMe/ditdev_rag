import hashlib
import logging
import os
import re
import threading
import time
from collections import Counter, OrderedDict

import chromadb
from chromadb.config import Settings

from data_loader import DBUnavailable, load_all_chunks, months_since, refresh_derived_chunks
from embeddings import CloudflareEmbeddingProvider, EmbeddingProvider

log = logging.getLogger('ditdev_rag')

COLLECTION_NAME = 'ditdev_portfolio'
CHROMA_PATH     = os.path.join(os.path.dirname(__file__), 'chroma_store')

# Cosine distance cutoff, measured against a specific model - it is NOT portable
# across models. Numbers below were measured on multilingual-e5-small with
# `python test_retrieve.py`: 11 on-topic queries spanned 0.102-0.193, 3 off-topic
# ones 0.234-0.249, so 0.21 split them with ~0.02 of margin.
# bge-m3 has its own spread: re-run test_retrieve.py and re-tune before trusting
# `found: false`. The old 0.7 let literally everything through.
DISTANCE_THRESHOLD = float(os.getenv('RAG_DISTANCE_THRESHOLD', '0.21'))

CACHE_SIZE = 128
CACHE_TTL  = 300   # seconds; writes also invalidate explicitly

# Keyword -> (chunk type to boost, terms appended to the query before embedding).
# Single source of truth: these keywords used to live in two separate tables
# (synonym expansion + intent boost) that had already drifted apart.
# Matching is word-start (`\bkw`) so Indonesian suffixes still hit ("mulainya")
# while embedded substrings do not ("pengalaman" must not match "lama").
INTENT_RULES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (('berapa', 'total', 'banyak', 'jumlah', 'how many', 'count',
      'lama', 'bulan', 'months', 'how long', 'duration'),
     'stats', 'total count how many duration months'),

    (('skill', 'kemampuan', 'bisa', 'pakai', 'stack', 'tech', 'kuasai', 'menguasai'),
     'skill', 'skills abilities tech stack'),

    (('project', 'proyek', 'portfolio', 'karya',
      'buat', 'membuat', 'dibuat', 'bikin', 'build', 'bangun', 'membangun'),
     'project', 'projects built created portfolio'),

    (('awal', 'mulai', 'memulai', 'mengenal', 'sejak', 'kapan',
      'belajar', 'mempelajari', 'background', 'sekolah', 'pendidikan'),
     'education', 'start begin first time school studying'),

    (('sertif', 'certificate', 'achievement', 'badge', 'piagam'),
     'certificate', 'certificate achievement badge'),

    (('hubungi', 'kontak', 'contact', 'email', 'freelance', 'hire', 'sewa'),
     'contact', 'contact reach email hire'),
)

# Boosts are deliberately small. `1 - distance` realistically spans ~0.55-0.90 on
# this corpus, so the old 0.3 type priority + 0.3 intent boost could outrank
# semantic relevance outright and let `stats` squat a slot on every query.
TYPE_PRIORITY = {
    'stats'      : 0.06,
    'project'    : 0.04,
    'certificate': 0.04,
    'education'  : 0.03,
    'skill'      : 0.02,
    'about'      : 0.01,
    'contact'    : 0.01,
}
INTENT_BOOST   = 0.08
OVERLAP_WEIGHT = 0.01

# Precompiled once: `\bkw` per keyword, grouped per rule.
_INTENT_MATCHERS = tuple(
    (tuple(re.compile(r'\b' + re.escape(k)) for k in keywords), chunk_type, expansion)
    for keywords, chunk_type, expansion in INTENT_RULES
)


class LRUCache:

    def __init__(self, maxsize: int, ttl: int):
        self.cache   = OrderedDict()
        self.maxsize = maxsize
        self.ttl     = ttl
        self._lock   = threading.Lock()

    def get(self, key: str):
        with self._lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            value, ts = entry
            if time.time() - ts > self.ttl:
                self.cache.pop(key, None)
                return None
            self.cache.move_to_end(key)
            return value

    def set(self, key: str, value):
        with self._lock:
            self.cache[key] = (value, time.time())
            self.cache.move_to_end(key)
            while len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)

    def invalidate(self):
        with self._lock:
            self.cache.clear()

    def __len__(self):
        with self._lock:
            return len(self.cache)


class RAGEngine:
    def __init__(self, embedder: EmbeddingProvider | None = None):
        self.embedder = embedder or CloudflareEmbeddingProvider.from_env()
        log.info('Embedding provider: %s (%s)', type(self.embedder).__name__, self.embedder.model)

        log.info('Opening ChromaDB at %s', CHROMA_PATH)
        self.client     = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self._open_collection()

        self.cache = LRUCache(CACHE_SIZE, CACHE_TTL)
        self.db_ok = True                      # last known Postgres state, for /health
        self._write_lock = threading.RLock()   # serialises index mutations and rebuilds

        self._reconcile()

    def close(self) -> None:
        self.embedder.close()

    def _open_collection(self):
        return self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={'hnsw:space': 'cosine'},
        )

    def _indexed_dim(self) -> int | None:
        got = self.collection.peek(limit=1).get('embeddings')
        # Chroma hands back a numpy array here, so `or`/truthiness is not safe.
        return len(got[0]) if got is not None and len(got) else None

    def _reconcile(self):
        indexed = self.collection.count()
        chunks, self.db_ok = load_all_chunks()

        if not self.db_ok:
            if indexed:
                log.warning('DB unavailable; keeping the existing %d indexed chunks', indexed)
            else:
                log.error('DB unavailable and index empty - indexing static chunks only')
                self._write(chunks)
            return

        live_dim    = len(self.embedder.embed('dimension probe'))
        indexed_dim = self._indexed_dim() if indexed else None
        if indexed_dim is not None and indexed_dim != live_dim:
            log.warning(
                'Embedding dimension changed (%d indexed vs %d from %s) - rebuilding',
                indexed_dim, live_dim, self.embedder.model,
            )
            self.rebuild_index(chunks)
        elif indexed != len(chunks):
            log.warning('Index drift: %d indexed vs %d expected - rebuilding', indexed, len(chunks))
            self.rebuild_index(chunks)
        else:
            log.info('Collection ready - %d chunks', indexed)

    # Query preprocessing

    @staticmethod
    def _normalize(query: str) -> str:
        q = query.lower().strip()
        q = re.sub(r'\s+', ' ', q)
        # Keep # + . - so "C#", "C++", "Node.js" and "e5-small" survive; the old
        # pattern stripped them and turned "C#" into "c".
        return re.sub(r'[^\w\s\?\.\#\+\-]', '', q).strip()

    @staticmethod
    def _intents(normalized: str) -> set[str]:
        return {
            chunk_type
            for matchers, chunk_type, _ in _INTENT_MATCHERS
            if any(m.search(normalized) for m in matchers)
        }

    @staticmethod
    def _expand(normalized: str, intents: set[str]) -> str:
        extra = [exp for _, chunk_type, exp in INTENT_RULES if chunk_type in intents]
        return f"{normalized} {' '.join(extra)}" if extra else normalized

    @staticmethod
    def _dynamic_top_k(normalized: str) -> int:
        words = len(normalized.split())
        if words <= 4:
            return 3
        if words <= 8:
            return 4
        return 5

    @staticmethod
    def _cache_key(query: str, top_k: int) -> str:
        return hashlib.md5(f'{query}:{top_k}'.encode()).hexdigest()

    # Scoring

    @staticmethod
    def _score(doc: str, dist: float, meta: dict, normalized: str, intents: set[str]) -> float:
        chunk_type = meta.get('type', '')
        score = (1.0 - dist) + TYPE_PRIORITY.get(chunk_type, 0.0)
        if chunk_type in intents:
            score += INTENT_BOOST
        overlap = len(set(normalized.split()) & set(doc.lower().split()))
        return score + overlap * OVERLAP_WEIGHT

    @staticmethod
    def _freshen(doc: str, meta: dict) -> str:
        if meta.get('type') != 'stats':
            return doc
        start  = meta.get('coding_start', '')
        months = months_since(start)
        if months is None:
            return doc
        return (
            f'{doc} Adit-san has been coding for exactly {months} months '
            f'(started {start}). Do NOT say 2 years, 3 years or 4 years - '
            f'the correct answer is {months} months.'
        )

    # Retrieval

    def retrieve(self, query: str, top_k: int | None = None) -> str:
        normalized = self._normalize(query)
        if not normalized:
            return ''

        intents  = self._intents(normalized)
        expanded = self._expand(normalized, intents)
        if top_k is None:
            top_k = self._dynamic_top_k(normalized)

        cache_key = self._cache_key(expanded, top_k)
        cached    = self.cache.get(cache_key)
        if cached is not None:
            log.debug('Cache hit %s', cache_key[:8])
            return cached

        count = self.collection.count()
        if count == 0:
            log.warning('Retrieve on an empty collection')
            return ''

        # No `query: ` prefix any more: that was e5's required instruction format.
        # bge-m3 is prefix-free and prepending one just adds noise to the vector.
        embedding = self.embedder.embed(expanded)

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k * 2, count),
            include=['documents', 'distances', 'metadatas'],
        )
        docs      = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]

        scored = sorted(
            (
                (self._score(doc, dist, meta, normalized, intents), doc, dist, meta)
                for doc, dist, meta in zip(docs, distances, metadatas)
                if dist < DISTANCE_THRESHOLD
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if not scored:
            self.cache.set(cache_key, '')
            return ''

        # `relevance` is absolute (1 - distance). The old `priority` field was
        # normalised against the best hit, so the first block was always 1.0 and
        # told the LLM nothing.
        blocks = [
            '[DATA]\n'
            f'type: {meta.get("type", "unknown")}\n'
            f'id: {meta.get("name") or meta.get("title") or meta.get("db_id") or "unknown"}\n'
            f'relevance: {round(1.0 - dist, 2)}\n'
            f'content: {self._freshen(doc, meta)}'
            for _, doc, dist, meta in scored[:top_k]
        ]
        context = '[REALM DATA]\n\n' + '\n\n'.join(blocks)
        self.cache.set(cache_key, context)
        return context

    # Index writes

    def _write(self, chunks: list[dict]):
        if not chunks:
            log.error('Nothing to index')
            return
        ids   = [c['id']               for c in chunks]
        texts = [c['text']             for c in chunks]
        metas = [c.get('metadata', {}) for c in chunks]

        # One batched call, not one request per chunk. The `passage: ` prefix went
        # with e5; bge-m3 embeds documents and queries the same way.
        embeddings = self.embedder.embed_batch(texts)

        for i in range(0, len(ids), 100):
            self.collection.upsert(
                ids        = ids[i:i + 100],
                documents  = texts[i:i + 100],
                embeddings = embeddings[i:i + 100],
                metadatas  = metas[i:i + 100],
            )
        log.info('Indexed %d chunks', len(ids))

    def upsert_chunk(self, chunk_id: str, text: str, metadata: dict | None = None) -> bool:
        try:
            with self._write_lock:
                self._write([{'id': chunk_id, 'text': text, 'metadata': metadata or {}}])
                self.cache.invalidate()
            return True
        except Exception as e:
            log.error('Upsert failed for %s: %s', chunk_id, e)
            return False

    def delete_chunk(self, chunk_id: str) -> bool:
        try:
            with self._write_lock:
                self.collection.delete(ids=[chunk_id])
                self.cache.invalidate()
            log.info('Deleted chunk %s', chunk_id)
            return True
        except Exception as e:
            log.error('Delete failed for %s: %s', chunk_id, e)
            return False

    def refresh_derived(self) -> list[str]:
        chunks = refresh_derived_chunks()
        with self._write_lock:
            self._write(chunks)
            self.cache.invalidate()
        self.db_ok = True
        return [c['id'] for c in chunks]

    def rebuild_index(self, chunks: list[dict] | None = None) -> int:
        with self._write_lock:
            if chunks is None:
                chunks, self.db_ok = load_all_chunks()
                if not self.db_ok:
                    # Never replace a good index with a static-only one.
                    raise DBUnavailable('refusing to rebuild while the database is down')
            log.info('Full rebuild of %d chunks', len(chunks))
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self._open_collection()
            self.cache.invalidate()
            self._write(chunks)
            total = self.collection.count()
        log.info('Rebuild complete: %d chunks', total)
        return total

    # Introspection

    def cache_stats(self) -> dict:
        return {'size': len(self.cache), 'maxsize': self.cache.maxsize, 'ttl': self.cache.ttl}

    def health(self) -> dict:
        metadatas = self.collection.get(include=['metadatas']).get('metadatas') or []
        by_type = Counter((meta or {}).get('type', 'unknown') for meta in metadatas)

        healthy = bool(metadatas) and self.db_ok and by_type.get('stats', 0) > 0
        return {
            'status' : 'ok' if healthy else 'degraded',
            'chunks' : len(metadatas),
            'by_type': dict(by_type),
            'db_ok'  : self.db_ok,
            # Which model built this index. The top failure mode after the
            # Cloudflare migration is querying a 384-dim index with 1024-dim vectors.
            'embed_model': self.embedder.model,
            'cache'  : self.cache_stats(),
        }
