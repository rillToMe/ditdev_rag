"""Build portfolio chunks from skills_data.json (static) + PostgreSQL (dynamic).

This module owns every chunk text template. The backend used to format
`stats_summary` itself, which silently dropped the anti-hallucination guard the
Python side had added - so the service now exposes /index/refresh-derived and
the backend just asks for a refresh.
"""

import json
import logging
import os
from contextlib import contextmanager
from datetime import date

import psycopg2
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger('ditdev_rag')

BASE_DIR = os.path.dirname(__file__)


class DBUnavailable(RuntimeError):
    """Postgres could not be reached, so dynamic chunks are missing."""


@contextmanager
def _cursor():
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    except Exception as e:
        raise DBUnavailable(f'connect failed: {e}') from e
    try:
        with conn.cursor() as cur:
            yield cur
    except Exception as e:
        raise DBUnavailable(f'query failed: {e}') from e
    finally:
        conn.close()


def months_since(start_iso: str) -> int | None:
    """Whole months between start_iso (YYYY-MM-DD) and today, counting both ends."""
    try:
        year, month = int(start_iso[:4]), int(start_iso[5:7])
    except (TypeError, ValueError, IndexError):
        return None
    if not 1 <= month <= 12:
        return None
    today = date.today()
    return (today.year - year) * 12 + (today.month - month) + 1


# Static chunks (skills_data.json)

def _skill_chunk(skill: dict) -> dict:
    return {
        'id'  : skill['id'],
        'text': (
            f"Skill: {skill['name']} | Category: {skill['category']} | "
            f"Level: {skill['level']} | {skill['description']}"
        ),
        'metadata': {
            'type'    : 'skill',
            'name'    : skill['name'],
            'category': skill['category'],
            'level'   : skill['level'],
        },
    }


def _skills_summary_chunk(skills: list[dict]) -> dict:
    """One chunk holding every skill.

    Retrieval returns at most top_k chunks, so "skill apa aja yang dikuasai?"
    could only ever surface 4 of the 11 skill chunks. This chunk answers the
    enumeration in a single hit.
    """
    listed = '; '.join(f"{s['name']} ({s['category']}, {s['level']})" for s in skills)
    return {
        'id'  : 'skills_summary',
        'text': (
            f"Complete skill list of Adit-san - {len(skills)} skills in total: {listed}. "
            f"Use this list when asked to list, count, or compare all skills."
        ),
        'metadata': {'type': 'skill', 'name': 'All skills summary'},
    }


def _about_chunk(about: dict) -> dict:
    return {
        'id'  : 'about_adit',
        'text': (
            f"About Adit-san: {about.get('description', '')} "
            f"Location: {about.get('location')}. "
            f"Role: {about.get('role')}. "
            f"Available for: {about.get('available_for')}. "
            f"GitHub: {about.get('github')}. "
            f"TikTok: {about.get('tiktok')}. "
            f"Instagram: {about.get('instagram')}."
        ),
        'metadata': {'type': 'about', 'name': about.get('name', '')},
    }


def _education_chunk(edu: dict) -> dict:
    return {
        'id'  : edu['id'],
        'text': (
            f"Education: {edu['level']} at {edu['institution']}, "
            f"{edu['location']}. Period: {edu['period']}. "
            f"Status: {edu['status']}. Focus: {edu['focus']}."
        ),
        'metadata': {
            'type'       : 'education',
            'name'       : edu['institution'],
            'institution': edu['institution'],
            'status'     : edu['status'],
        },
    }


def _contact_chunk(contact: dict) -> dict:
    return {
        'id'  : 'contact_info',
        'text': (
            f"Contact Adit-san: {contact.get('description', '')} "
            f"{contact.get('method', '')} "
            f"Response time: {contact.get('response_time', '')}. "
            f"Open for: {', '.join(contact.get('open_for', []))}."
        ),
        'metadata': {'type': 'contact', 'name': 'Contact info'},
    }


def load_static_data() -> list[dict]:
    with open(os.path.join(BASE_DIR, 'skills_data.json'), encoding='utf-8') as f:
        data = json.load(f)

    skills = data.get('skills', [])
    chunks = [_skill_chunk(s) for s in skills]
    if skills:
        chunks.append(_skills_summary_chunk(skills))
    chunks.append(_about_chunk(data.get('about', {})))
    chunks += [_education_chunk(e) for e in data.get('education', [])]
    chunks.append(_contact_chunk(data.get('contact', {})))
    return chunks


# Dynamic chunks (PostgreSQL)

def _stats_chunk(total_projects: int, total_certs: int, coding_start: str) -> dict:
    """Counts only.

    The month count is deliberately NOT baked into the text: it used to be, and
    went stale the moment a month passed without an admin CRUD. RAGEngine
    recomputes it from `coding_start` on every retrieve instead.
    """
    return {
        'id'  : 'stats_summary',
        'text': (
            f"AUTHORITATIVE REAL-TIME STATS - always use these numbers: "
            f"Adit-san has built {total_projects} projects in total "
            f"and earned {total_certs} certificates."
        ),
        'metadata': {
            'type'          : 'stats',
            'name'          : 'Portfolio stats',
            'total_projects': str(total_projects),
            'total_certs'   : str(total_certs),
            'coding_start'  : coding_start or '',
        },
    }


def _projects_summary_chunk(rows: list[tuple]) -> dict:
    """One chunk listing every project title, for the same reason as skills."""
    titles = '; '.join(str(r[1]) for r in rows)
    return {
        'id'  : 'projects_summary',
        'text': (
            f"Complete project list of Adit-san - {len(rows)} projects in total: {titles}. "
            f"Use this list when asked to list or count all projects."
        ),
        'metadata': {'type': 'project', 'name': 'All projects summary'},
    }


def _project_chunk(row: tuple) -> dict:
    pid, title, description, tags, links = row
    tags_str  = ', '.join(tags) if tags else ''
    links_str = ''.join(
        f" {l['type']}: {l['url']}" for l in (links or []) if l.get('url')
    )
    return {
        'id'  : f'project_{pid}',
        'text': (
            f"Project by Adit-san: {title}. "
            f"Description: {description}. "
            f"Tags/Tech stack: {tags_str}."
            f"{(' Links:' + links_str) if links_str else ''}"
        ),
        'metadata': {
            'type' : 'project',
            'name' : title,
            'title': title,
            'tags' : tags_str,
            'db_id': str(pid),
        },
    }


def _cert_chunk(row: tuple) -> dict:
    cid, title, provider, issue_date, credential_url = row
    date_str = str(issue_date)[:7] if issue_date else 'unknown date'
    return {
        'id'  : f'cert_{cid}',
        'text': (
            f"Certificate earned by Adit-san: {title}. "
            f"Issued by: {provider}. "
            f"Date: {date_str}."
            f"{(' Credential: ' + credential_url) if credential_url else ''}"
        ),
        'metadata': {
            'type'    : 'certificate',
            'name'    : title,
            'title'   : title,
            'provider': provider,
            'db_id'   : str(cid),
        },
    }


PROJECTS_SQL = """
    SELECT p.id, p.title, p.description, p.tags,
           json_agg(json_build_object('type', pl.type, 'url', pl.url))
           FILTER (WHERE pl.id IS NOT NULL) as links
    FROM projects p
    LEFT JOIN project_links pl ON p.id = pl.project_id
    GROUP BY p.id
    ORDER BY p.created_at DESC
"""

CERTS_SQL = """
    SELECT id, title, provider, issue_date, credential_url
    FROM certificates
    ORDER BY created_at DESC
"""


def _coding_start(cur) -> str:
    """ISO date Adit-san started coding, from the stats table. '' when unset."""
    cur.execute("SELECT start_date FROM stats WHERE key = 'months_studying'")
    row = cur.fetchone()
    return row[0].isoformat() if row and row[0] else ''


def _derived_chunks(cur) -> tuple[list[dict], list[tuple]]:
    """(derived chunks, raw project rows) - the rows are reused by the caller."""
    cur.execute('SELECT COUNT(*) FROM projects')
    total_projects = int(cur.fetchone()[0])
    cur.execute('SELECT COUNT(*) FROM certificates')
    total_certs = int(cur.fetchone()[0])

    cur.execute(PROJECTS_SQL)
    projects = cur.fetchall()

    return [
        _stats_chunk(total_projects, total_certs, _coding_start(cur)),
        _projects_summary_chunk(projects),
    ], projects


def refresh_derived_chunks() -> list[dict]:
    """Chunks that summarise the whole DB, so they need a refresh after any CRUD.

    Raises DBUnavailable so the caller can return a real error instead of
    silently indexing nothing.
    """
    with _cursor() as cur:
        derived, _ = _derived_chunks(cur)
    return derived


def load_dynamic_data() -> list[dict]:
    with _cursor() as cur:
        derived, projects = _derived_chunks(cur)
        cur.execute(CERTS_SQL)
        certs = cur.fetchall()

    return derived + [_project_chunk(r) for r in projects] + [_cert_chunk(r) for r in certs]


def load_all_chunks() -> tuple[list[dict], bool]:
    """Returns (chunks, db_ok). db_ok=False means dynamic chunks are missing, so
    the caller must not treat a short index as complete."""
    static = load_static_data()
    try:
        dynamic = load_dynamic_data()
        db_ok   = True
    except DBUnavailable as e:
        log.error('DB unavailable, dynamic chunks skipped: %s', e)
        dynamic = []
        db_ok   = False

    log.info('Loaded %d static + %d dynamic = %d chunks (db_ok=%s)',
             len(static), len(dynamic), len(static) + len(dynamic), db_ok)
    return static + dynamic, db_ok
