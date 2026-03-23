import os
from dotenv import load_dotenv
from mistralai import Mistral


def generate_answer(user_text: str) -> str:
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Missing MISTRAL_API_KEY in .env")

    client = Mistral(api_key=api_key)

    resp = client.chat.complete(
        model="mistral-small",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be concise and correct."},
            {"role": "user", "content": user_text},
        ],
        temperature=0.2,
        max_tokens=300,
    )

    return resp.choices[0].message.content.strip()