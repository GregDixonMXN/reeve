#!/usr/bin/env python3
"""
Axiom Web Scraper Tool (Tool B: "The Eyes — Scrape")
═════════════════════════════════════════════════════
Downloads a webpage and extracts clean article text.
Strips ads, sidebars, nav, CSS, JS — returns only content.

Input:  {"url": "https://example.com/article", "max_chars": 8000}
Output: {"url": "...", "title": "...", "content": "...", "char_count": 1234}

Install: pip install trafilatura
"""

import json
import sys

# Hard safety ceiling — never exceed this regardless of what's asked
ABSOLUTE_MAX_CHARS = 16000

# Default limit fits inside Phi-4's 4096 token window with room for
# system prompt + memory context + tool results
DEFAULT_MAX_CHARS = 8000


def scrape(url: str, max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    try:
        import trafilatura
    except ImportError:
        return {"error": "trafilatura not installed. Run: pip install trafilatura"}

    max_chars = min(max_chars, ABSOLUTE_MAX_CHARS)

    try:
        # Download the page
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return {"error": f"Failed to download: {url}", "url": url}

        # Extract clean text (strips HTML, ads, nav, sidebars)
        content = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=True,
        )

        if not content:
            return {"error": "No extractable content found", "url": url}

        # Extract title separately
        metadata = trafilatura.extract(
            downloaded,
            output_format="json",
            include_comments=False,
        )
        title = ""
        if metadata:
            try:
                meta_dict = json.loads(metadata)
                title = meta_dict.get("title", "")
            except (json.JSONDecodeError, TypeError):
                pass

        # ── GUARDRAIL: Hard character limit ──────────────────────────────
        original_len = len(content)
        if original_len > max_chars:
            content = content[:max_chars]
            content += (
                f"\n\n[TRUNCATED: showing {max_chars} of {original_len} characters]"
            )

        return {
            "url": url,
            "title": title,
            "content": content,
            "char_count": len(content),
            "original_char_count": original_len,
            "truncated": original_len > max_chars,
        }

    except Exception as e:
        return {"error": f"Scrape failed: {e}", "url": url}


def main():
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "No input"}))
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Treat raw input as URL
        data = {"url": raw}

    url = data.get("url", "")

    # Safely cast to int, fallback to default if it fails
    try:
        max_chars = int(data.get("max_chars", DEFAULT_MAX_CHARS))
    except (ValueError, TypeError):
        max_chars = DEFAULT_MAX_CHARS

    if not url:
        print(json.dumps({"error": "No URL provided"}))
        return

    # Basic URL validation
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    result = scrape(url, max_chars)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
