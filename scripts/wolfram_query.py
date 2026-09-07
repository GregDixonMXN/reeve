#!/usr/bin/env python3
"""
Herald Wolfram|Alpha LLM Bridge

Input (stdin JSON): {"query": "integral of x^2 from 0 to 1"}
Output (stdout):    Plain text result optimized for LLMs

Requires: pip install requests
"""

import json
import os
import sys
import urllib.parse

import requests


def query_wolfram_llm(query: str, app_id: str) -> str:
    # URL-encode the math/data query safely
    safe_query = urllib.parse.quote_plus(query)
    url = (
        f"https://www.wolframalpha.com/api/v1/llm-api?input={safe_query}&appid={app_id}"
    )

    try:
        # Fetch the response with a 15-second timeout
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        # The API natively returns conversational text, so we just pass it straight back!
        return response.text
    except requests.exceptions.RequestException as e:
        return f"[ERROR] Wolfram|Alpha query failed: {e}"


def main():
    raw = sys.stdin.read().strip()
    if not raw:
        print("[ERROR] No input", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"query": raw}

    query = data.get("query", "")
    if not query:
        print("[ERROR] No query", file=sys.stderr)
        sys.exit(1)

    # Pull the App ID from the environment or the JSON payload
    app_id = os.environ.get("WOLFRAM_APP_ID", data.get("app_id", ""))
    if not app_id:
        print("[ERROR] WOLFRAM_APP_ID not set", file=sys.stderr)
        sys.exit(1)

    print(query_wolfram_llm(query, app_id))


if __name__ == "__main__":
    main()
