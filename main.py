import os
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import feedparser

# Optional: OpenAI for summarization/normalization
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    if OPENAI_API_KEY:
        from openai import OpenAI  # type: ignore
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        openai_client = None
except Exception:
    openai_client = None

app = FastAPI(title="French News API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        # Try to import database module
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Check environment variables
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


CATEGORY_SOURCES: Dict[str, List[str]] = {
    "Politique": [
        "https://www.lemonde.fr/politique/rss_full.xml",
        "https://www.francetvinfo.fr/politique.rss",
    ],
    "Économie": [
        "https://www.lemonde.fr/economie/rss_full.xml",
        "https://www.francetvinfo.fr/economie.rss",
    ],
    "Culture": [
        "https://www.francetvinfo.fr/culture.rss",
        "https://www.lemonde.fr/culture/rss_full.xml",
    ],
    "Sports": [
        "https://www.lequipe.fr/rss/actu_rss.xml",
        "https://www.francetvinfo.fr/sports.rss",
    ],
    "Technologie": [
        "https://www.lemonde.fr/pixels/rss_full.xml",
        "https://www.francetvinfo.fr/internet-et-high-tech.rss",
    ],
    "Environnement": [
        "https://www.lemonde.fr/planete/rss_full.xml",
        "https://www.francetvinfo.fr/monde/environnement.rss",
    ],
}


def normalize_entry(entry: Dict[str, Any], category: str) -> Dict[str, Any]:
    title = entry.get("title", "")
    summary = entry.get("summary", "") or entry.get("description", "")
    link = entry.get("link")
    published = entry.get("published", entry.get("updated"))
    media_thumbnail = None
    # Attempt to pull media content if present
    for key in ("media_content", "media_thumbnail", "links"):
        if key in entry and entry[key]:
            try:
                candidates = entry[key]
                if isinstance(candidates, list):
                    for c in candidates:
                        url = c.get("url") if isinstance(c, dict) else None
                        if url and url.startswith("http"):
                            media_thumbnail = url
                            break
                elif isinstance(candidates, dict) and candidates.get("url"):
                    media_thumbnail = candidates["url"]
            except Exception:
                pass
        if media_thumbnail:
            break

    return {
        "category": category,
        "title": title,
        "summary": summary,
        "link": link,
        "published_at": published,
        "image": media_thumbnail,
        "source": entry.get("source", {}).get("title") if isinstance(entry.get("source"), dict) else None,
    }


async def summarize_items_if_possible(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not openai_client:
        return items
    try:
        # Build a concise prompt in French
        bullets = "\n".join(
            [f"- Catégorie: {it['category']} | Titre: {it['title']} | Résumé: {it['summary'][:280]}" for it in items]
        )
        system = (
            "Tu es un éditeur de presse français. Tu reformules des titres et résumés en français clair et"
            " professionnel, sans inventer de faits. Raccourcis à 1 phrase par résumé (max 200 caractères)."
        )
        user = (
            "Harmonise ces brèves pour un site d’actualité. Ne crée aucune information."
            " Réponds au format JSON d’objets: [{title, summary}].\n\n" + bullets
        )
        # Use Responses API
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        content = resp.choices[0].message.content if resp.choices else None
        import json
        if content:
            # Attempt to extract JSON array from content
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1 and end > start:
                data = json.loads(content[start : end + 1])
                for i, obj in enumerate(data):
                    if i < len(items):
                        items[i]["title"] = obj.get("title", items[i]["title"])[:160]
                        items[i]["summary"] = obj.get("summary", items[i]["summary"])[:220]
    except Exception:
        return items
    return items


@app.get("/api/news")
async def get_news(
    categories: Optional[str] = Query(None, description="Virgule: Politique,Économie,..."),
    limit: int = Query(18, ge=3, le=50),
):
    # Decide which categories to include
    cats = [c.strip() for c in categories.split(",")] if categories else list(CATEGORY_SOURCES.keys())
    items: List[Dict[str, Any]] = []

    for cat in cats:
        sources = CATEGORY_SOURCES.get(cat)
        if not sources:
            continue
        collected = []
        for url in sources:
            try:
                feed = feedparser.parse(url)
                for e in feed.entries[: max(0, limit // len(cats)) or 3]:
                    collected.append(normalize_entry(e, cat))
            except Exception:
                continue
        # Deduplicate by title
        seen = set()
        deduped = []
        for it in collected:
            key = (it["title"], it.get("link"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(it)
        items.extend(deduped[: max(3, limit // len(cats))])

    # Sort by presence of published_at (feeds often already ordered)
    items = items[:limit]

    # Optionally summarize with OpenAI to ensure consistent tone in French
    items = await summarize_items_if_possible(items)

    return {"items": items, "count": len(items), "categories": cats, "summarized": bool(openai_client)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
