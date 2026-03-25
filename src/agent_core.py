import os
import json
from typing import Any, Dict, List, Callable

from dotenv import load_dotenv
from mistralai import Mistral

from src import tools as tool_impl


ToolFn = Callable[..., Dict[str, Any]]

TOOL_REGISTRY: Dict[str, ToolFn] = {
    "list_files": tool_impl.list_files,
    "read_file": tool_impl.read_file,
    "write_file": tool_impl.write_file,
    "search_docs": tool_impl.search_docs,
    "calculator": tool_impl.calculator,
}


# Tool schemas for function calling (Mistral-compatible format)
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files inside the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search local documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_hits": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        },
    },
]


def run_agent(goal: str, max_steps: int = 8) -> Dict[str, Any]:
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Missing MISTRAL_API_KEY in .env")

    client = Mistral(api_key=api_key)

    system = (
        "You are a task-executing AI agent.\n"
        "You have tools to search documents, read/write files, and do calculations.\n"
        "Use tools when needed.\n"
        "Stop when the task is complete and provide a clear final result.\n"
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"GOAL:\n{goal}"},
    ]

    for step in range(1, max_steps + 1):
        resp = client.chat.complete(
            model="mistral-small",
            messages=messages,
            tools=TOOLS_SCHEMA,
            temperature=0.2,
            max_tokens=500,
        )

        msg = resp.choices[0].message

        # Mistral tool calls
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            final_text = (msg.content or "").strip()
            return {"ok": True, "steps": step, "final": final_text}

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": tool_calls
        })

        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")

            if name not in TOOL_REGISTRY:
                tool_result = {"ok": False, "error": f"Unknown tool: {name}"}
            else:
                try:
                    tool_result = TOOL_REGISTRY[name](**args)
                except Exception as e:
                    tool_result = {"ok": False, "error": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(tool_result),
            })

    return {"ok": False, "error": f"Max steps exceeded ({max_steps})."}