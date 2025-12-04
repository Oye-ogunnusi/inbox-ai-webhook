from fastapi import FastAPI, Request
import os
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/triage")
async def triage(request: Request):
    data = await request.json()

    subject = data.get("subject", "")
    body = data.get("body_text", "")

    prompt = f"Write a short, professional reply to this email:\n\nSubject: {subject}\n\n{body}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful email assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    reply_text = response.choices[0].message.content
    return {"reply_text": reply_text}
