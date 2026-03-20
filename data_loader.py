import json
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def load_static_data() -> list[dict]:
    base_dir = os.path.dirname(__file__)
    with open(os.path.join(base_dir, 'skills_data.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = []

    # Skills 
    for skill in data.get('skills', []):
        chunks.append({
            'id'      : skill['id'],
            'type'    : 'skill',
            'text'    : (
                f"Skill: {skill['name']} | Category: {skill['category']} | "
                f"Level: {skill['level']} | {skill['description']}"
            ),
            'metadata': {
                'type'    : 'skill',
                'name'    : skill['name'],
                'category': skill['category'],
                'level'   : skill['level'],
            }
        })

    #About 
    about = data.get('about', {})
    chunks.append({
        'id'  : 'about_adit',
        'type': 'about',
        'text': (
            f"About Adit-san: {about.get('description', '')} "
            f"Location: {about.get('location')}. "
            f"Role: {about.get('role')}. "
            f"Experience: {about.get('experience')}. "
            f"Available for: {about.get('available_for')}. "
            f"GitHub: {about.get('github')}."
        ),
        'metadata': {'type': 'about', 'name': about.get('name', '')}
    })

    #Education 
    for edu in data.get('education', []):
        chunks.append({
            'id'  : edu['id'],
            'type': 'education',
            'text': (
                f"Education: {edu['level']} at {edu['institution']}, "
                f"{edu['location']}. Period: {edu['period']}. "
                f"Status: {edu['status']}. Focus: {edu['focus']}."
            ),
            'metadata': {
                'type'       : 'education',
                'institution': edu['institution'],
                'status'     : edu['status'],
            }
        })

    #Contact 
    contact = data.get('contact', {})
    chunks.append({
        'id'  : 'contact_info',
        'type': 'contact',
        'text': (
            f"Contact Adit-san: {contact.get('description', '')} "
            f"{contact.get('method', '')} "
            f"Response time: {contact.get('response_time', '')}. "
            f"Open for: {', '.join(contact.get('open_for', []))}."
        ),
        'metadata': {'type': 'contact'}
    })

    return chunks


def load_dynamic_data() -> list[dict]:
    """Load projects dan certificates dari PostgreSQL — return list of chunks."""
    chunks = []
    conn   = None

    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        cur  = conn.cursor()

        # Projects 
        cur.execute("""
            SELECT p.id, p.title, p.description, p.tags,
                   json_agg(json_build_object('type', pl.type, 'url', pl.url))
                   FILTER (WHERE pl.id IS NOT NULL) as links
            FROM projects p
            LEFT JOIN project_links pl ON p.id = pl.project_id
            GROUP BY p.id
            ORDER BY p.created_at DESC
        """)
        projects = cur.fetchall()

        for row in projects:
            pid, title, description, tags, links = row
            tags_str  = ', '.join(tags) if tags else ''
            links_str = ''
            if links:
                for link in links:
                    if link.get('url'):
                        links_str += f" {link['type']}: {link['url']}"

            chunks.append({
                'id'  : f'project_{pid}',
                'type': 'project',
                'text': (
                    f"Project by Adit-san: {title}. "
                    f"Description: {description}. "
                    f"Tags/Tech stack: {tags_str}."
                    f"{(' Links:' + links_str) if links_str else ''}"
                ),
                'metadata': {
                    'type'    : 'project',
                    'title'   : title,
                    'tags'    : tags_str,
                    'db_id'   : str(pid),
                }
            })

        # Certificates
        cur.execute("""
            SELECT id, title, provider, issue_date, credential_url
            FROM certificates
            ORDER BY created_at DESC
        """)
        certs = cur.fetchall()

        for row in certs:
            cid, title, provider, issue_date, credential_url = row
            date_str = str(issue_date)[:7] if issue_date else 'unknown date'

            chunks.append({
                'id'  : f'cert_{cid}',
                'type': 'certificate',
                'text': (
                    f"Certificate earned by Adit-san: {title}. "
                    f"Issued by: {provider}. "
                    f"Date: {date_str}."
                    f"{(' Credential: ' + credential_url) if credential_url else ''}"
                ),
                'metadata': {
                    'type'    : 'certificate',
                    'title'   : title,
                    'provider': provider,
                    'db_id'   : str(cid),
                }
            })

        cur.close()

    except Exception as e:
        print(f'[data_loader] DB error: {e}')
    finally:
        if conn:
            conn.close()

    return chunks


def load_all_chunks() -> list[dict]:
    static  = load_static_data()
    dynamic = load_dynamic_data()
    all_chunks = static + dynamic
    print(f'[data_loader] Loaded {len(static)} static + {len(dynamic)} dynamic = {len(all_chunks)} total chunks')
    return all_chunks