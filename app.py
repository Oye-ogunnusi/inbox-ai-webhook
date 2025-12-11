from fastapi import FastAPI, Request
import os
from typing import List, Any, Dict
from uuid import uuid4

from openai import OpenAI
from pinecone import Pinecone

# --------- FastAPI app ----------
app = FastAPI()

# --------- OpenAI client ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------- Pinecone setup ----------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "inbox-memory")
PINECONE_HOST = os.getenv("PINECONE_HOST")

# Option A — Namespace per sender:
#  - One Pinecone namespace per email sender
#  - All reads/writes for that sender go to their namespace
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)


# ---- Embedding helpers (2048-dim with text-embedding-3-large) ----
def embed_text(text: str) -> List[float]:
    """
    Embed a single string into a 2048-dim vector using OpenAI embeddings.
    Make sure your Pinecone index dimension is set to 2048.
    """
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=[text],
        dimensions=2048,
    )
    return resp.data[0].embedding


def derive_namespace(sender: str) -> str:
    """
    Option A — namespace per sender:
    Use the sender email address directly as the namespace.
    """
    sender = (sender or "unknown").strip().lower()
    return sender if sender else "unknown"


def _get_matches(response: Any) -> List[Any]:
    """
    Helper to normalize Pinecone query response for both dict- and attr-style.
    """
    if hasattr(response, "matches"):
        return response.matches or []
    if isinstance(response, dict):
        return response.get("matches", []) or []
    return []


def _get_metadata(match: Any) -> Dict[str, Any]:
    """
    Normalize metadata access for both dict- and attr-style matches.
    """
    if hasattr(match, "metadata"):
        meta = match.metadata or {}
    elif isinstance(match, dict):
        meta = match.get("metadata", {}) or {}
    else:
        meta = {}
    return meta if isinstance(meta, dict) else {}


# --------- Triage endpoint ----------
@app.post("/triage")
async def triage(request: Request):
    data = await request.json()

    subject = data.get("subject", "") or ""
    body = data.get("body_text", "") or ""
    sender = data.get("from_email", "unknown") or "unknown"

    namespace = derive_namespace(sender)

    email_text = f"From: {sender}\nSubject: {subject}\n\n{body}"

    # 1) Retrieve similar memory from Pinecone (namespace = sender)
    try:
        query_embedding = embed_text(email_text)

        query_resp = index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=3,
            include_values=False,
            include_metadata=True,
        )

        matches = _get_matches(query_resp)

        snippets: List[str] = []
        for m in matches:
            meta = _get_metadata(m)
            # We store summary in "summary" (see write step below)
            text = meta.get("summary") or meta.get("text") or ""
            if text:
                snippets.append(text)

        memory_snippets = "\n\n---\n\n".join(snippets) if snippets else ""
    except Exception:
        memory_snippets = ""

    # 2) Build prompt with memory
    system_prompt = (
        "You are an executive assistant. You write short, clear, polite email replies.\n"
        "Use the provided past context only if it is relevant. "
        "Never mention 'memory' or 'Pinecone' to the user.\n"
    )

    user_prompt = f"""
Incoming email:
----------------
{email_text}

Relevant past context (may be empty):
----------------
{memory_snippets}

Task:
----------------
Write a short, professional reply (3–6 sentences).
Be concrete and helpful. Avoid fluff.
Sign off with 'Best,' and no placeholder brackets.
Reply in plain text (no markdown).
"""

    chat_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    reply_text = chat_resp.choices[0].message.content

    # 3) Store a new summary as memory in the sender's namespace
    try:
        summary_prompt = (
            "Summarise this email in 1–2 sentences for future context:\n\n"
            f"{email_text}"
        )
        summary_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": summary_prompt}],
        )
        summary = summary_resp.choices[0].message.content.strip()

        summary_embedding = embed_text(summary)

        vector_id = f"{sender}-{uuid4()}"

        index.upsert(
            vectors=[
                (
                    vector_id,
                    summary_embedding,
                    {
                        "sender": sender,
                        "subject": subject,
                        "summary": summary,
                    },
                )
            ],
            namespace=namespace,
        )
    except Exception:
        # Don't break the reply if memory write fails
        pass

    return {"reply_text": reply_text}
