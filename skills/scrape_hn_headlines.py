#!/usr/bin/env python3
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://news.ycombinator.com', wait_until='domcontentloaded', timeout=30000)
    page.wait_for_selector('.titleline > a', timeout=10000)
    
    # Get all headline links
    headlines = page.locator('.titleline > a').all()
    texts = [h.inner_text() for h in headlines[:10]]  # Get first 10
    
    print(' | '.join(texts))
    browser.close()
