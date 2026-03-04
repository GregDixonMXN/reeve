#!/usr/bin/env python3
"""
Axiom Web Search Tool (Tool A: "The Eyes — Search")
════════════════════════════════════════════════════
Searches DuckDuckGo and returns clean JSON results.
Called by the Go tool registry via stdin JSON.

Input:  {"query": "rust async web framework 2025"}
Output: JSON array of {title, url, snippet}

Install: pip install duckduckgo-search
"""

import json
import sys


def search(query: str, max_results: int = 5) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return [
            {
                "error": "duckduckgo-search not installed. Run: pip install duckduckgo-search"
            }
        ]

    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as e:
        return [{"error": f"Search failed: {e}"}]

    clean = []
    for r in results:
        clean.append(
            {
                "title": r.get("title", ""),
                "url": r.get("href", r.get("link", "")),
                "snippet": r.get("body", r.get("snippet", ""))[:500],
            }
        )

    return clean


def main():
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "No input"}))
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Treat raw input as query string
        data = {"query": raw}

    query = data.get("query", "")

    # Safely cast to int, fallback to default if it fails
    try:
        max_results = int(data.get("max_results", 5))
    except (ValueError, TypeError):
        max_results = 5

    if not query:
        print(json.dumps({"error": "No query provided"}))
        return

    results = search(query, max_results)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
