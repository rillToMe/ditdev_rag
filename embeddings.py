from __future__ import annotations

import logging
import os
import random
import time
from abc import ABC, abstractmethod

import httpx

log = logging.getLogger('ditdev_rag')

CF_API_BASE     = 'https://api.cloudflare.com/client/v4/accounts'
DEFAULT_MODEL   = '@cf/baai/bge-m3'
DEFAULT_TIMEOUT = 20.0
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 0.5
# Workers AI caps a bge batch at 100 texts. The whole corpus is ~30 chunks, so
# this only bites on a full rebuild after the portfolio grows.
DEFAULT_BATCH   = 100
RETRY_STATUS    = frozenset({429, 500, 502, 503, 504})


class EmbeddingError(RuntimeError):
    """An embedding could not be produced. Never raised for a partial result:
    either every requested vector comes back, or this is raised."""


class EmbeddingConfigError(EmbeddingError):
    """Misconfiguration - missing credentials. Raised at construction, so a bad
    deploy fails at startup instead of on the first user query."""


class EmbeddingUnavailable(EmbeddingError):
    """Transient failure: timeout, network error, or retries exhausted on a 5xx.
    Callers may map this to a 503; retrying later is reasonable."""


class EmbeddingProvider(ABC):
    """Turns text into vectors. Implementations must be thread-safe: FastAPI runs
    sync endpoints in a threadpool, so `/retrieve` calls can overlap."""

    model: str

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed one string. Raises EmbeddingError on failure."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many strings, in order. Raises EmbeddingError on failure."""

    def close(self) -> None:
        """Release pooled connections. No-op unless the provider owns a client."""


def _safe_json(res: httpx.Response) -> object | None:
    try:
        return res.json()
    except ValueError:
        return None


def _cf_errors(payload: object) -> str:
    if isinstance(payload, dict):
        errors = payload.get('errors')
        if isinstance(errors, list) and errors:
            return '; '.join(
                f"{e.get('code', '?')}: {e.get('message', e)}" if isinstance(e, dict) else str(e)
                for e in errors
            )
    return 'no error detail returned'


def _retry_after(res: httpx.Response) -> float:
    try:
        return max(0.0, float(res.headers.get('retry-after', 0)))
    except ValueError:
        return 0.0


class CloudflareEmbeddingProvider(EmbeddingProvider):

    def __init__(
        self,
        account_id: str,
        api_token: str,
        model: str = DEFAULT_MODEL,
        *,
        client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
        backoff: float = DEFAULT_BACKOFF,
        batch_size: int = DEFAULT_BATCH,
    ) -> None:
        if not account_id or not api_token:
            raise EmbeddingConfigError('account_id and api_token are both required')
        self.model       = model
        self.timeout     = timeout
        self.max_retries = max_retries
        self.backoff     = backoff
        self.batch_size  = max(1, batch_size)
        self._url        = f'{CF_API_BASE}/{account_id}/ai/run/{model}'
        # Kept out of the client so an injected client still authenticates, and so
        # the token never shows up in a client repr.
        self._headers    = {'Authorization': f'Bearer {api_token}'}
        self._owns_client = client is None
        self._client      = client or httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls, client: httpx.Client | None = None) -> 'CloudflareEmbeddingProvider':
        account = os.getenv('CLOUDFLARE_ACCOUNT_ID', '').strip()
        token   = os.getenv('CLOUDFLARE_API_TOKEN', '').strip()
        missing = [
            name for name, value in
            (('CLOUDFLARE_ACCOUNT_ID', account), ('CLOUDFLARE_API_TOKEN', token))
            if not value
        ]
        if missing:
            raise EmbeddingConfigError(f'missing env var(s): {", ".join(missing)}')
        model = os.getenv('CLOUDFLARE_EMBEDDING_MODEL', '').strip() or DEFAULT_MODEL
        return cls(account, token, model, client=client)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text or not text.strip() for text in texts):
            raise EmbeddingError('refusing to embed a blank string')

        started = time.perf_counter()
        vectors: list[list[float]] = []
        batches = 0
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._post(texts[start:start + self.batch_size]))
            batches += 1

        elapsed = (time.perf_counter() - started) * 1000
        log.info(
            'Embedded %d text(s) in %d request(s) via %s in %.0f ms (dim=%d)',
            len(texts), batches, self.model, elapsed, len(vectors[0]),
        )
        return vectors

    def _post(self, texts: list[str]) -> list[list[float]]:
        reason = 'no attempt made'
        for attempt in range(self.max_retries + 1):
            delay = self.backoff * 2 ** attempt
            try:
                res = self._client.post(
                    self._url, json={'text': texts}, headers=self._headers, timeout=self.timeout,
                )
                if res.status_code not in RETRY_STATUS:
                    if res.is_error:
                        raise EmbeddingError(
                            f'{self.model}: HTTP {res.status_code} from Cloudflare '
                            f'({_cf_errors(_safe_json(res))})'
                        )
                    return self._parse(res, len(texts))
                reason = f'HTTP {res.status_code}'
                delay = max(delay, _retry_after(res))
            except httpx.TimeoutException:
                reason = f'timed out after {self.timeout}s'
            except httpx.HTTPError as e:
                reason = f'network error ({type(e).__name__}: {e})'

            if attempt == self.max_retries:
                break
            log.warning(
                '%s: %s - retry %d/%d in %.1fs', self.model, reason,
                attempt + 1, self.max_retries, delay,
            )
            time.sleep(delay + random.uniform(0, 0.2))   # jitter: avoid lockstep retries

        raise EmbeddingUnavailable(
            f'{self.model}: {reason}, gave up after {self.max_retries + 1} attempt(s)'
        )

    def _parse(self, res: httpx.Response, expected: int) -> list[list[float]]:
        payload = _safe_json(res)
        if payload is None:
            raise EmbeddingError(f'{self.model}: response body was not valid JSON')
        if not isinstance(payload, dict):
            raise EmbeddingError(f'{self.model}: expected a JSON object, got {type(payload).__name__}')
        # Workers AI answers 200 with success:false on some model-side failures.
        if payload.get('success') is False:
            raise EmbeddingError(f'{self.model}: Cloudflare API error: {_cf_errors(payload)}')

        result = payload.get('result')
        data   = result.get('data') if isinstance(result, dict) else None
        if not isinstance(data, list) or len(data) != expected:
            raise EmbeddingError(
                f'{self.model}: expected {expected} vector(s) at result.data, '
                f'got {len(data) if isinstance(data, list) else type(data).__name__}'
            )
        if any(not isinstance(vec, list) or not vec for vec in data):
            raise EmbeddingError(f'{self.model}: result.data contained a non-vector entry')
        return data
