import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from data_loader import load_all_chunks

COLLECTION_NAME = 'ditdev_portfolio'
CHROMA_PATH     = os.path.join(os.path.dirname(__file__), 'chroma_store')
EMBED_MODEL     = 'all-MiniLM-L6-v2'
TOP_K           = 4
DISTANCE_THRESHOLD = 0.7

class RAGEngine:
    def __init__(self):
        print('[RAG] Loading embedding model...')
        self.embedder = SentenceTransformer(EMBED_MODEL)

        print('[RAG] Initializing ChromaDB...')
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={'hnsw:space': 'cosine'}
        )

        if self.collection.count() == 0:
            print('[RAG] Collection empty — building initial index...')
            self._build_index()
        else:
            print(f'[RAG] Collection ready — {self.collection.count()} chunks')

    # Internal
    def _embed(self, text: str) -> list[float]:
        return self.embedder.encode([text]).tolist()[0]

    def _build_index(self):
        chunks = load_all_chunks()
        if not chunks:
            print('[RAG] No chunks to index!')
            return

        ids        = [c['id']              for c in chunks]
        texts      = [c['text']            for c in chunks]
        metadatas  = [c.get('metadata', {}) for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()

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
                ids        = [chunk_id],
                documents  = [text],
                embeddings = [embedding],
                metadatas  = [metadata],
            )
            print(f'[RAG] Added chunk: {chunk_id}')
            return True
        except Exception as e:
            print(f'[RAG] Error adding chunk {chunk_id}: {e}')
            return False

    def update_chunk(self, chunk_id: str, text: str, metadata: dict = {}) -> bool:
        try:
            embedding = self._embed(text)
            self.collection.upsert(
                ids        = [chunk_id],
                documents  = [text],
                embeddings = [embedding],
                metadatas  = [metadata],
            )
            print(f'[RAG] Updated chunk: {chunk_id}')
            return True
        except Exception as e:
            print(f'[RAG] Error updating chunk {chunk_id}: {e}')
            return False

    def delete_chunk(self, chunk_id: str) -> bool:
        try:
            self.collection.delete(ids=[chunk_id])
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

    # Retrieve 

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        query_embedding = self._embed(query)
        results = self.collection.query(
            query_embeddings = [query_embedding],
            n_results        = min(top_k, self.collection.count()),
            include          = ['documents', 'distances']
        )

        docs      = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0]

        relevant = [
            doc for doc, dist in zip(docs, distances)
            if dist < DISTANCE_THRESHOLD
        ]

        if not relevant:
            return ''

        return '\n'.join(f'• {doc}' for doc in relevant)

    #Full rebuild (fallback)

    def rebuild_index(self):
        print('[RAG] Full rebuild triggered...')
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={'hnsw:space': 'cosine'}
        )
        self._build_index()
        print(f'[RAG] Rebuild complete: {self.collection.count()} chunks')