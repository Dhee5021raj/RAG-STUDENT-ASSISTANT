import os
import json
import datetime
from typing import List, Dict, Any

HISTORY_FILE = "query_history.json"


def log_query(question: str, mode: str, latency_ms: float, n_sources: int, top_source: str = "N/A"):
    """Appends a query log record to the local JSON history store."""
    record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "mode": mode,
        "latency_ms": latency_ms,
        "n_sources": n_sources,
        "top_source": top_source
    }

    history = get_history_records()
    history.insert(0, record)  # Newest first

    # Retain top 100 entries
    history = history[:100]

    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Failed to log query history: {e}")


def get_history_records(limit: int = 50) -> List[Dict[str, Any]]:
    """Reads query log records from the local JSON file."""
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data[:limit]
    except Exception:
        return []


def clear_history():
    """Clears the query log history."""
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
        except Exception:
            pass
