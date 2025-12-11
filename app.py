from fastapi import FastAPI, Request
import os
from typing import List, Any, Dict, Optional
from uuid import uuid4

from openai import OpenAI
from pinecone import Pinecone
import httpx

# --------- FastAPI app ----------
app = FastAPI()

# --------- OpenAI client ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------- Pinecone setup ----------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "inbox-memory")
PINECONE_HOST = os.getenv("PINECONE_HOST")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

# --------- Telegram + n8n config ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = (
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    if TELEGRAM_BOT_TOKEN
    else None
)

# Your own chat id with the bot (you'll get it from /start, see below)
TELEGRAM_OWNER_CHAT_ID = os.getenv("TELEGRAM_OWNER_CHAT_ID")

# n8n webhook that actually sends the email (POST)
N8N_SEND_EMAIL_URL = os.getenv("N8N_SEND_EMAIL_URL")

# In-memory conversation state (fine for 1-user prototype)
pending_requests: Dict[str, Dict[str, Any]] = {}


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
    Namespace per sender: use the sender email address as the namespace.
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


def generate_reply(
    subject: str,
    body: str,
    sender: str,
    decision_text: Optional[str] = None,
) -> str:
    """
    Core logic used both by /triage and by Telegram flow.
    Uses Pinecone memory + OpenAI to generate a reply.
    decision_text: explanation of what YOU decided (available, reschedule, etc.).
    """
    sender = sender or "unknown"
    body = body or ""
    subject = subject or ""

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
            text = meta.get("summary") or meta.get("text") or ""
            if text:
                snippets.append(text)

        memory_snippets = "\n\n---\n\n".join(snippets) if snippets else ""
    except Exception:
        memory_snippets = ""

    decision_block = ""
    if decision_text:
        decision_block = f"""
Your explicit decision / instructions:
----------------
{decision_text}
"""

    # 2) Build prompt with memory + decision
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
{decision_block}

Task:
----------------
Write a short, professional reply (3–6 sentences).
Be concrete and helpful. Avoid fluff.
Follow the decision/instructions exactly if provided.
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

    return reply_text


# --------- Original triage endpoint (now with optional decision) ----------
@app.post("/triage")
async def triage(request: Request):
    data = await request.json()

    subject = data.get("subject", "") or ""
    body = data.get("body_text", "") or ""
    sender = data.get("from_email", "unknown") or "unknown"
    decision = data.get("decision") or None

    reply_text = generate_reply(subject, body, sender, decision)

    return {"reply_text": reply_text}


# --------- Telegram helpers ----------

async def telegram_send_message(chat_id: str, text: str):
    if not TELEGRAM_API_URL:
        return
    async with httpx.AsyncClient() as client_http:
        await client_http.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )


async def send_to_n8n(email: Dict[str, Any], final_body: str):
    """
    Call your n8n webhook to actually send the Gmail response.
    """
    if not N8N_SEND_EMAIL_URL:
        return
    payload = {
        "to": email.get("from_email"),
        "subject": f"Re: {email.get('subject', '')}",
        "body": final_body,
        "original_message_id": email.get("message_id"),
    }
    async with httpx.AsyncClient() as client_http:
        await client_http.post(N8N_SEND_EMAIL_URL, json=payload)


async def handle_final_decision(chat_id: str, session: Dict[str, Any], decision: Dict[str, Any]):
    email = session["email"]
    decision_type = decision.get("type")
    time_str = decision.get("time")

    # Turn your choice into clear instructions for the AI
    if decision_type == "accept":
        decision_text = "I confirm that I am available for this meeting at the proposed time."
    elif decision_type == "accept_with_time":
        decision_text = f"I am available for this meeting at: {time_str}."
    elif decision_type == "reschedule":
        decision_text = (
            "I am not available at the proposed time. "
            f"Please request to reschedule the meeting to: {time_str}."
        )
    elif decision_type == "decline":
        decision_text = "I am not available and would like to decline this meeting."
    else:
        decision_text = ""

    # Generate reply using the same AI + Pinecone logic
    reply_text = generate_reply(
        subject=email.get("subject", ""),
        body=email.get("body_text", ""),
        sender=email.get("from_email", "unknown"),
        decision_text=decision_text,
    )

    final_body = (
        f"{reply_text}\n\n---\n"
        "This response was generated with the help of AI under my authorization and supervision."
    )

    await telegram_send_message(chat_id, "Got it. I'm sending this reply:\n\n" + final_body)
    await send_to_n8n(email, final_body)


# --------- Endpoint n8n calls when it detects a meeting email ----------
@app.post("/incoming-email")
async def incoming_email(request: Request):
    """
    Called by n8n workflow when IF node says the email is about meeting/interview/etc.
    Expects JSON with at least: subject, body_text, from_email, message_id.
    """
    data = await request.json()

    email = {
        "from_email": data.get("from_email") or data.get("from") or "",
        "subject": data.get("subject") or "",
        "body_text": data.get("body_text") or data.get("body") or "",
        "message_id": data.get("message_id") or data.get("messageId"),
    }

    chat_id = TELEGRAM_OWNER_CHAT_ID
    if not chat_id:
        # If not set, just log and return
        return {"ok": False, "error": "TELEGRAM_OWNER_CHAT_ID not set"}

    pending_requests[chat_id] = {
        "email": email,
        "state": "awaiting_availability",
    }

    preview = (email["body_text"] or "")[:400]

    text = (
        f"New meeting-related email:\n\n"
        f"From: {email['from_email']}\n"
        f"Subject: {email['subject']}\n\n"
        f"{preview}\n\n"
        "Are you available for this meeting? (yes/no)"
    )

    await telegram_send_message(chat_id, text)

    return {"ok": True}


# --------- Telegram webhook endpoint ----------
@app.post("/telegram")
async def telegram_webhook(request: Request):
    update = await request.json()

    message = update.get("message") or update.get("edited_message")
    if not message:
        return {"ok": True}

    chat = message.get("chat", {})
    chat_id = str(chat.get("id"))
    text_raw = message.get("text") or ""
    text = text_raw.strip().lower()

    # /start command: show your chat id
    if text == "/start":
        reply = (
            "Hi! I will help manage your meeting emails.\n\n"
            f"Your chat id is: {chat_id}\n\n"
            "Set this as TELEGRAM_OWNER_CHAT_ID in Render."
        )
        await telegram_send_message(chat_id, reply)
        return {"ok": True}

    session = pending_requests.get(chat_id)

    if not session:
        await telegram_send_message(
            chat_id,
            "I don't have any active meeting request right now. "
            "Wait for a new email to arrive."
        )
        return {"ok": True}

    state = session.get("state")

    if state == "awaiting_availability":
        if "yes" in text:
            # Are we missing a specific time?
            if not session["email"].get("proposed_time"):
                session["state"] = "awaiting_time"
                await telegram_send_message(chat_id, "Great! What time are you available?")
            else:
                session["state"] = "finalizing_accept"
                await handle_final_decision(chat_id, session, {"type": "accept"})
                pending_requests.pop(chat_id, None)
        elif "no" in text:
            session["state"] = "awaiting_reschedule_confirm"
            await telegram_send_message(
                chat_id,
                "Okay, you're not available. Should I ask to reschedule? (yes/no)"
            )
        else:
            await telegram_send_message(
                chat_id,
                'Please reply "yes" if you are available, or "no" if you are not.'
            )

    elif state == "awaiting_time":
        user_time = text_raw.strip()
        session["state"] = "finalizing_accept"
        await handle_final_decision(
            chat_id,
            session,
            {"type": "accept_with_time", "time": user_time},
        )
        pending_requests.pop(chat_id, None)

    elif state == "awaiting_reschedule_confirm":
        if "yes" in text:
            session["state"] = "awaiting_reschedule_time"
            await telegram_send_message(chat_id, "What time would you like me to propose?")
        elif "no" in text:
            session["state"] = "finalizing_decline"
            await handle_final_decision(chat_id, session, {"type": "decline"})
            pending_requests.pop(chat_id, None)
        else:
            await telegram_send_message(chat_id, 'Please answer "yes" or "no".')

    elif state == "awaiting_reschedule_time":
        new_time = text_raw.strip()
        session["state"] = "finalizing_reschedule"
        await handle_final_decision(
            chat_id,
            session,
            {"type": "reschedule", "time": new_time},
        )
        pending_requests.pop(chat_id, None)

    return {"ok": True}
