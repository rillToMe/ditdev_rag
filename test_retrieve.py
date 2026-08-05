"""Runnable checks for the retrieval logic: `python test_retrieve.py`.

Part 1 is pure logic and always runs (no model, no DB, no index).
Part 2 hits the real index and is skipped when the model or store is missing.
"""

from data_loader import months_since
from rag_engine import DISTANCE_THRESHOLD, LRUCache, RAGEngine

E = RAGEngine  # static methods only, no instance needed


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
    """Prints the distance spread so DISTANCE_THRESHOLD can be tuned by eye:
    on-topic queries should sit well below it, off-topic ones above."""
    print(f'  DISTANCE_THRESHOLD = {DISTANCE_THRESHOLD}')
    for query in [q for q, _ in GOLDEN] + list(ON_TOPIC) + list(OFF_TOPIC):
        normalized = engine._normalize(query)
        expanded   = engine._expand(normalized, engine._intents(normalized))
        embedding  = engine.embedder.encode([f'query: {expanded}']).tolist()[0]
        res = engine.collection.query(
            query_embeddings=[embedding], n_results=3, include=['distances'],
        )
        print(f'    min dist {min(res["distances"][0]):.3f}  {query[:46]}')


def integration() -> int:
    try:
        engine = RAGEngine()
    except Exception as e:
        print(f'  SKIP - no model or index available ({type(e).__name__}: {e})')
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



