# 14 Days to Building AI Systems & Agents

This repository contains hands-on labs and builds from the 14-day course.

## Setup

1. Create and activate a virtual environment
2. Install dependencies (added later)
3. Run scripts in the `scripts/` folder

## Run

```bash
python scripts/hello_ai.py


---

## Step 10: Initialize Git locally
```bash
git init
git add .
git commit -m "Day 2: initial project setup"

## Day 3: CLI Assistant

Create a `.env` file in the project root:

OPENAI_API_KEY=y3P4PISYLPysHxZDOir1qMp8N1mpUdPuL

Install dependencies:
pip install -r requirements.txt

Run:
python scripts/cli_assistant.py

## Day 6: Document-Aware Assistant (RAG)

1) Add `.txt` documents to `data/docs/`
2) Build the index:
python scripts/rag_assistant.py --build-index

3) Ask questions:
python scripts/rag_assistant.py

## Day 7: AI Web App (Local)

Run backend:
uvicorn src.app:app --reload --port 8000

Open:
http://127.0.0.1:8000/

## Day 8: Conversational Memory + Sessions

Run:
uvicorn src.app:app --reload --port 8000

Open:
http://127.0.0.1:8001/

Notes:
- Browser stores session_id in localStorage
- Backend persists sessions to data/sessions.json

## Day 9: Task-Executing AI Agent

Run:
python scripts/agent_cli.py

Tools:
- search_docs (local search over data/docs)
- read_file / write_file (workspace only)
- calculator