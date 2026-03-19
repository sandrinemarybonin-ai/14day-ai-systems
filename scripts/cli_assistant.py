import os
import requests
from dotenv import load_dotenv

def build_messages(user_text: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Be concise, correct, and do not invent facts. "
                "If unsure, say you are unsure."
            ),
        },
        {"role": "user", "content": user_text},
    ]

def call_llm(api_key: str, user_text: str) -> str:
    prompts = build_messages(user_text)
        
    endpoint = "https://api.mistral.ai/v1/chat/completions"  # adjust if needed
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-small",
        "messages": prompts,
        "max_tokens": 300,
        "temperature": 0.2
    }

    response = requests.post(endpoint, headers=headers, json=payload)
    
    response.raise_for_status()  # raise exception on HTTP errors
    
    # Parse the returned text
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

def main():
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("Missing MISTRAL_API_KEY. Put it in a .env file.")

    print("CLI AI Assistant (type 'exit' to quit)")
    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not user_text:
            print("Please enter a question.")
            continue

        try:
            answer = call_llm(api_key, user_text)
            print("\nAssistant:", answer)
        except Exception as e:
            print("\nError calling the LLM API:", str(e))

if __name__ == "__main__":
    main()