import os
import json
import time
from typing import Dict, Any
from dotenv import load_dotenv
from mistralai import Mistral

MODEL = "mistral-small"

load_dotenv()

print("DEBUG KEY:", repr(os.getenv("MISTRAL_API_KEY")))

def step1_load_input(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return {"ok": True, "input_path": file_path, "raw_text": text}


def step2_extract_structured(raw_text: str) -> Dict[str, Any]:
    import os, json
    from mistralai import Mistral

    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    # system = (
    #     "Extract structured information from the user text.\n"
    #     "Return ONLY valid JSON with keys:\n"
    #     "topic, requester, urgency (low|medium|high), summary, action_items (array of strings).\n"
    #     "If unknown, use null or empty values.\n"
    #     "Do not include explanations or text outside the JSON object."
    # )
    system = """
    Return ONLY valid JSON.
    No markdown.
    No explanation.
    No text before or after JSON.

    Schema:
    {
    "topic": string or null,
    "requester": string or null,
    "urgency": "low" | "medium" | "high",
    "summary": string or null,
    "action_items": array of strings
    }
    """

    resp = client.chat.complete(
        model="mistral-small",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": raw_text},
        ],
        temperature=0.0,
        max_tokens=500,
    )

    content = (resp.choices[0].message.content or "").strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Return defaults for missing/invalid JSON
        required = ["topic", "requester", "urgency", "summary", "action_items"]
        data = {k: None if k != "action_items" else [] for k in required}
        return {"ok": False, "error": f"Invalid JSON returned", "raw": content}

    # Ensure all keys exist
    required = ["topic", "requester", "urgency", "summary", "action_items"]
    for k in required:
        if k not in data:
            data[k] = None if k != "action_items" else []

    return {"ok": True, "extracted": data}


def step3_classify_and_route(extracted: Dict[str, Any]) -> Dict[str, Any]:
    urgency = (extracted.get("urgency") or "").lower()

    if urgency not in {"low", "medium", "high"}:
        urgency = "medium"

    if urgency == "high":
        route = "priority"
        sla = "4 hours"
    elif urgency == "medium":
        route = "standard"
        sla = "24 hours"
    else:
        route = "low"
        sla = "72 hours"

    return {"ok": True, "route": route, "sla": sla}


def step4_generate_draft_reply(extracted: Dict[str, Any], route: str, sla: str) -> Dict[str, Any]:
    load_dotenv()
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    system = (
        "Write a concise, professional response draft.\n"
        "Use only the provided structured fields.\n"
        "If information is missing, ask one clarifying question.\n"
    )

    user = {
        "route": route,
        "sla": sla,
        "extracted": extracted,
    }

    resp = client.chat.complete(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        temperature=0.2,
        max_tokens=350,
    )

    draft = (resp.choices[0].message.content or "").strip()
    return {"ok": True, "draft_reply": draft}


def step5_save_outputs(out_base: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    os.makedirs(out_base, exist_ok=True)

    json_path = os.path.join(out_base, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    draft_path = os.path.join(out_base, "draft_reply.txt")
    with open(draft_path, "w", encoding="utf-8") as f:
        f.write(payload.get("draft_reply", ""))

    return {"ok": True, "json_path": json_path, "draft_path": draft_path}


def step6_log_run(log_path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    record = dict(record)
    record["ts"] = int(time.time())

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")