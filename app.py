from fastapi import FastAPI, Request, BackgroundTasks
import os
from typing import List

from openai import OpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# --------- FastAPI app ----------
app = FastAPI()

# --------- Health check route ----------
@app.get("/")
async def root():
    return {"status": "ok", "message": "Inbox AI webhook is running"}

# --------- OpenAI client ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------- Pinecone setup ----------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "inbox-memory")
PINECONE_HOST = os.getenv("PINECONE_HOST")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)


# ---- Embedding function (2048-dim with text-embedding-3-large) ----
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Takes a list of texts and returns a list of 2048-dim embeddings.
    Uses OpenAI text-embedding-3-large with dimension override=2048.
    """
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
        dimensions=2048,
    )
    return [d.embedding for d in resp.data]


# ---- LangChain Embeddings wrapper for PineconeVectorStore ----
class OpenAI2048Embeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        return embed_texts([text])[0]


embedding_model = OpenAI2048Embeddings()

# ---- LangChain Pinecone vectorstore ----
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_model,
    text_key="text",
)


def store_memory(email_text: str, sender: str, subject: str):
    """Background task: summarise and store in Pinecone."""
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

        doc = Document(
            page_content=summary,
            metadata={
                "sender": sender,
                "subject": subject,
            },
        )

        vectorstore.add_documents([doc])
    except Exception:
        # Ignore background errors
        pass


@app.post("/triage")
async def triage(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()

    subject = data.get("subject", "")
    body = data.get("body_text", "")
    sender = data.get("from_email", "unknown")

    email_text = f"From: {sender}\nSubject: {subject}\n\n{body}"

    # 1) Retrieve similar memory from Pinecone (still in the main path)
    try:
        docs = vectorstore.similarity_search(email_text, k=3)
        memory_snippets = "\n\n---\n\n".join(
            [d.page_content for d in docs]
        ) if docs else ""
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

    # 3) Schedule memory write in the background (non-blocking)
    background_tasks.add_task(store_memory, email_text, sender, subject)

    return {"reply_text": reply_text}
