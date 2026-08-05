import json

import httpx

from data_loader import months_since
from embeddings import CloudflareEmbeddingProvider, EmbeddingError
from rag_engine import DISTANCE_THRESHOLD, LRUCache, RAGEngine

E = RAGEngine  # static methods only, no instance needed


# Embedding provider (mocked transport - no network, no credentials)

def _provider(handler, **kwargs) -> CloudflareEmbeddingProvider:
    return CloudflareEmbeddingProvider(
        'acct', 'secret-token',
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        backoff=0, **kwargs,
    )


def _ok(count: int, dim: int = 3) -> httpx.Response:
    return httpx.Response(200, json={
        'success': True,
        'result' : {'shape': [count, dim], 'data': [[0.1] * dim for _ in range(count)]},
    })


def test_embed_batches_instead_of_one_call_per_text():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers['authorization'] == 'Bearer secret-token'
        texts = json.loads(request.content)['text']
        sent.append(texts)
        return _ok(len(texts))

    provider = _provider(handler, batch_size=2)
    assert len(provider.embed_batch(['a', 'b', 'c'])) == 3
    assert sent == [['a', 'b'], ['c']], 'must batch, not one request per text'
    assert provider.embed('solo') == [0.1, 0.1, 0.1]
    assert provider.embed_batch([]) == []


def test_retries_5xx_then_succeeds():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return _ok(1) if len(attempts) > 2 else httpx.Response(503)

    assert _provider(handler, max_retries=3).embed('x') == [0.1, 0.1, 0.1]
    assert len(attempts) == 3


def test_client_errors_are_not_retried():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(401, json={'errors': [{'code': 10000, 'message': 'bad token'}]})

    try:
        _provider(handler, max_retries=3).embed('x')
        raise AssertionError('a 401 must raise')
    except EmbeddingError as e:
        assert 'bad token' in str(e) and 'secret-token' not in str(e)
    assert len(attempts) == 1, 'retrying a bad token just burns the rate limit'


def test_bad_bodies_raise_instead_of_returning_junk():
    cases = (
        httpx.Response(200, text='<html>gateway</html>'),   # invalid JSON
        httpx.Response(200, json={'success': False, 'errors': [{'message': 'model down'}]}),
        httpx.Response(200, json={'result': {'data': []}}),  # fewer vectors than asked for
        httpx.Response(200, json={'result': {'data': [[]]}}),  # empty vector
    )
    for response in cases:
        provider = _provider(lambda request, r=response: r, max_retries=0)
        try:
            provider.embed('x')
            raise AssertionError(f'expected a raise for {response.text[:40]!r}')
        except EmbeddingError:
            pass

    try:
        _provider(lambda request: _ok(1), max_retries=0).embed_batch(['ok', '  '])
        raise AssertionError('blank text must raise')
    except EmbeddingError:
        pass


def test_normalize():
    assert E._normalize('  Berapa   TOTAL  project? ') == 'berapa total project?'
    # symbols that carry meaning must survive; the old regex turned "C#" into "c"
    assert E._normalize('skill C# dan C++ atau Node.js') == 'skill c# dan c++ atau node.js'
    assert E._normalize('halo, "adit"!') == 'halo adit'
    assert E._normalize('   ') == ''


def test_intents():
    assert E._intents('berapa total project adit') == {'stats', 'project'}
    assert E._intents('gimana cara hubungi adit') == {'contact'}
    assert E._intents('sekolah adit dimana') == {'education'}
    assert E._intents('sudah berapa lama belajar coding') == {'stats', 'education'}
    # word-start matching: "pengalaman" contains "lama" but must not boost stats
    assert E._intents('apa pengalaman adit') == set()
    # Indonesian suffixes still hit
    assert 'stats' in E._intents('totalnya berapa')


def test_top_k_uses_raw_query():
    short = 'berapa total project'
    assert len(E._expand(short, E._intents(short)).split()) > 8   # expansion inflates it
    assert E._dynamic_top_k(short) == 3                           # sizing ignores the expansion
    assert E._dynamic_top_k('a b c d e f g h i j k l') == 5


def test_boost_cannot_outrank_relevance():
    intents = {'stats'}
    close   = E._score('irrelevant text', 0.10, {'type': 'about'}, 'q', intents)
    boosted = E._score('irrelevant text', 0.25, {'type': 'stats'}, 'q', intents)
    assert close > boosted, 'a 0.15 distance gap must beat type priority + intent boost'


def test_cache():
    c = LRUCache(maxsize=2, ttl=300)
    c.set('a', '1')
    c.set('b', '2')
    assert c.get('a') == '1'
    c.set('c', '3')            # 'b' is the least recently used
    assert c.get('b') is None
    assert c.get('a') == '1' and c.get('c') == '3'

    expired = LRUCache(maxsize=2, ttl=-1)
    expired.set('a', '1')
    assert expired.get('a') is None
    assert len(expired) == 0


def test_months_are_recomputed_not_stored():
    from datetime import date
    today = date.today()
    assert months_since(today.isoformat()) == 1
    assert months_since(f'{today.year - 1}-{today.month:02d}-01') == 13
    assert months_since('') is None and months_since('2024-13-01') is None

    meta = {'type': 'stats', 'coding_start': '2024-08-28'}
    out  = E._freshen('4 projects.', meta)
    assert f'exactly {months_since("2024-08-28")} months' in out
    assert 'Do NOT say 2 years' in out
    # the guard must not leak into other chunk types
    assert E._freshen('Skill: Unity', {'type': 'skill'}) == 'Skill: Unity'


# Part 2: needs the embedding model + a built index

GOLDEN = (
    ('skill apa aja yang dikuasai adit', 'skill'),
    ('berapa total project yang sudah dibuat', 'stats'),
    ('sudah berapa lama adit belajar coding', 'stats'),
    ('gimana cara hubungi adit', 'contact'),
    ('sekolah adit dimana', 'education'),
    ('sertifikat apa yang dimiliki adit', 'certificate'),
)

OFF_TOPIC = (
    'resep rendang padang yang enak',
    'harga bitcoin hari ini naik atau turun',
    'cara memperbaiki mesin diesel truk',
)

# The false-negative guard for DISTANCE_THRESHOLD. The gap between on- and
# off-topic distances is only ~0.03, so a tighter cutoff must be checked against
# phrasings that differ from GOLDEN's - these only have to return *something*.
ON_TOPIC = (
    'adit tinggal dimana',
    'siapa itu adit',
    'apakah adit bisa dihire untuk freelance',
    'project apa yang paling keren',
    'what programming languages does adit know',
)


def first_type(context: str) -> str | None:
    for line in context.splitlines():
        if line.startswith('type: '):
            return line[6:].strip()
    return None


def probe_distances(engine: RAGEngine):
    print(f'  DISTANCE_THRESHOLD = {DISTANCE_THRESHOLD}  (model: {engine.embedder.model})')
    for query in [q for q, _ in GOLDEN] + list(ON_TOPIC) + list(OFF_TOPIC):
        normalized = engine._normalize(query)
        expanded   = engine._expand(normalized, engine._intents(normalized))
        embedding  = engine.embedder.embed(expanded)
        res = engine.collection.query(
            query_embeddings=[embedding], n_results=3, include=['distances'],
        )
        print(f'    min dist {min(res["distances"][0]):.3f}  {query[:46]}')


def integration() -> int:
    try:
        engine = RAGEngine()
    except Exception as e:
        print(f'  SKIP - no credentials, index or DB available ({type(e).__name__}: {e})')
        return 0

    report = engine.health()
    print(f'  index: {report["chunks"]} chunks, {report["by_type"]}, db_ok={report["db_ok"]}')
    probe_distances(engine)

    failures = 0
    for query, expected in GOLDEN:
        if not report['by_type'].get(expected):
            print(f'  skip  {query!r} - no {expected} chunk indexed')
            continue
        got = first_type(engine.retrieve(query))
        failures += got != expected
        print(f'  {"ok  " if got == expected else "FAIL"} {query!r} -> {got} (want {expected})')

    for query in ON_TOPIC:
        context = engine.retrieve(query)
        failures += not context
        print(f'  {"ok  " if context else "FAIL"} on-topic  {query!r} -> '
              f'{first_type(context) if context else "FILTERED OUT"}')

    for query in OFF_TOPIC:
        context = engine.retrieve(query)
        failures += context != ''
        print(f'  {"ok  " if not context else "FAIL"} off-topic {query!r} -> '
              f'{"filtered" if not context else first_type(context)}')
    return failures


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

    checks = [fn for name, fn in sorted(globals().items()) if name.startswith('test_')]
    for check in checks:
        check()
        print(f'  ok   {check.__name__}')
    print(f'{len(checks)} logic checks passed\n\n--- integration ---')

    raise SystemExit(1 if integration() else 0)



