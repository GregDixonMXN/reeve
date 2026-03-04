#!/usr/bin/env python3
"""
Headless browser tool using Playwright.
Usage: python browser.py <url> <css_selector>
"""

import sys
from playwright.sync_api import sync_playwright

def scrape_element(url, selector):
    """Load a page with Playwright and extract text from CSS selector."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            # Navigate to URL
            page.goto(url, wait_until='domcontentloaded', timeout=30000)
            
            # Wait for selector and get text
            page.wait_for_selector(selector, timeout=10000)
            element = page.locator(selector)
            text = element.inner_text()
            
            print(text)
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            browser.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python browser.py <url> <css_selector>")
        print("Example: python browser.py https://example.com h1")
        sys.exit(1)
    
    url = sys.argv[1]
    selector = sys.argv[2]
    
    scrape_element(url, selector)
