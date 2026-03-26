import os
import time
import shutil
from src.workflow_runner import run_workflow

INBOX = "inbox"
OUTBOX = "outbox"
PROCESSED = os.path.join(INBOX, "processed")
LOG_FILE = "logs/workflow.log"
POLL_SECONDS = 3

def main():
    os.makedirs(INBOX, exist_ok=True)
    os.makedirs(OUTBOX, exist_ok=True)
    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    print("Watching inbox for new .txt files... (Ctrl+C to stop)")

    while True:
        for name in os.listdir(INBOX):
            if not name.lower().endswith(".txt"):
                continue
            path = os.path.join(INBOX, name)
            if os.path.isdir(path):
                continue

            print(f"Processing: {path}")
            result = run_workflow(path, OUTBOX, LOG_FILE)
            print(f"Result: {result.get('ok')}")

            # Move to processed
            try:
                dest = os.path.join(PROCESSED, name)
                shutil.move(path, dest)
            except Exception as e:
                print(f"Error moving file: {e}")

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()