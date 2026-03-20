import os
import requests
from dotenv import load_dotenv
import json

def build_messages(interview_text: str) -> list[dict]:
    system_prompt = (
        "You are a qualitative research assistant.\n "
        "Return ONLY valid JSON following this schema:\n"
        "{\n"
        "  \"main_topics\": [\n"
        "    {\n"
        "      \"topic\": string,\n"
        "      \"summary\": string,\n"
        "      \"sentiment\": \"positive\" | \"negative\" | \"neutral\" | \"mixed\",\n"
        "      \"key_points\": [string],\n"
        "      \"representative_quotes\": [string]\n"
        "    }\n"
        "  ],\n"
        "  \"overall_summary\": string,\n"
        "  \"confidence\": number\n"
        "}\n"
        "RULES:\n"
        "Only return JSON, no extra text, bullet points or markdown."
        "All strings must be on a single line; escape any internal double quotes with backslashes. Use only the provided input text."
        "Summarize long content to fit within token limits (max 300 tokens per topic)." 
        "Do not truncate JSON; if necessary, reduce number of topics rather than breaking JSON."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": interview_text},
    ]

def call_llm(api_key: str, interview_text: str) -> str:
    prompts = build_messages(interview_text)
        
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

def parse_json_output(raw_output: str) -> dict:
    """
    Try to parse raw LLM output as JSON.
    If parsing fails, raise a clean error.
    """
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError as e:
        # Fail fast, do not try to fix malformed JSON
        raise ValueError(f"Invalid JSON from LLM: {e}")

# Step 8 & 9: Validate fields and types
# -----------------------------
VALID_SENTIMENTS = ["positive", "negative", "neutral", "mixed"]

def validate_qualitative_data(data: dict):
    # required top-level keys
    required_top_keys = ["main_topics", "overall_summary", "confidence"]
    for key in required_top_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    # overall_summary must be string
    if not isinstance(data["overall_summary"], str) or data["overall_summary"] == "":
        raise ValueError("overall_summary must be a non-empty string")

    # confidence must be number 0-1
    if not isinstance(data["confidence"], (int, float)) or not (0 <= data["confidence"] <= 1):
        raise ValueError("confidence must be a number between 0 and 1")

    # main_topics must be a list
    if not isinstance(data["main_topics"], list) or len(data["main_topics"]) == 0:
        raise ValueError("main_topics must be a non-empty list")

    # validate each topic
    for topic in data["main_topics"]:
        required_topic_keys = ["topic", "summary", "sentiment", "key_points", "representative_quotes"]
        for key in required_topic_keys:
            if key not in topic:
                raise ValueError(f"Missing key '{key}' in main_topics")
        if not isinstance(topic["topic"], str) or topic["topic"] == "":
            raise ValueError("topic must be a non-empty string")
        if not isinstance(topic["summary"], str) or topic["summary"] == "":
            raise ValueError("summary must be a non-empty string")
        if topic["sentiment"] not in VALID_SENTIMENTS:
            raise ValueError(f"sentiment must be one of {VALID_SENTIMENTS}")
        if not isinstance(topic["key_points"], list):
            raise ValueError("key_points must be a list")
        if not isinstance(topic["representative_quotes"], list):
            raise ValueError("representative_quotes must be a list")

# -----------------------------
# Step 10: Process and validate
# -----------------------------
def process_llm_output(raw_output: str) -> dict:
    parsed = parse_json_output(raw_output)
    validate_qualitative_data(parsed)
    return parsed

def main():
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("Missing MISTRAL_API_KEY. Put it in a .env file.")

    print("Qualitative CLI AI Assistant (type 'exit' to quit)")
    while True:
        interview_text = input("\nYou: ").strip()
        if interview_text.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not interview_text:
            print("Please enter an interview text.")
            continue

        try:
            raw_output = call_llm(api_key, interview_text)
            print("\nRAW OUTPUT:\n", raw_output)
            validated_data = process_llm_output(raw_output)
            print("\nVALIDATED JSON:\n", json.dumps(validated_data, indent=2))

        except Exception as e:
            print("\nError calling the LLM API:", str(e))

if __name__ == "__main__":
    main()