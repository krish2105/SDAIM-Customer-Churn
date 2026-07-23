#!/usr/bin/env python3
"""Capture the report's evidence screenshots from the real running systems.

Every image is taken from a live page — the running application, the public
GitHub repository, the real workflow runs and the deployed Hugging Face Space.
Nothing here mocks or reconstructs a screen.

Usage::

    python scripts/capture_screenshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "screenshots"

LOCAL_APP = "http://127.0.0.1:8850"
SPACE_APP = "https://krish21may-churn.hf.space"
SPACE_PAGE = "https://huggingface.co/spaces/krish21may/churn"
REPO = "https://github.com/krish2105/SDAIM-Customer-Churn"
CI_RUN = f"{REPO}/actions/runs/30019016553"
DEPLOY_RUN = f"{REPO}/actions/runs/30023117028"
ACTIONS = f"{REPO}/actions"
WORKFLOW_FILE = f"{REPO}/blob/main/.github/workflows/deploy.yml"

VIEWPORT = {"width": 1440, "height": 1000}


def _settle(page, seconds: float = 3.0) -> None:
    page.wait_for_timeout(int(seconds * 1000))


def _streamlit_ready(page, timeout_ms: int = 60_000) -> bool:
    """Wait for the app to finish loading its artifacts."""
    try:
        page.wait_for_function(
            """() => [...document.querySelectorAll('button')]
                     .some(b => b.innerText.trim() === 'Generate churn assessment')""",
            timeout=timeout_ms,
        )
        return True
    except Exception:  # noqa: BLE001
        return False


def _predict(page) -> None:
    page.evaluate(
        """() => {
            const b = [...document.querySelectorAll('button')]
                .find(x => x.innerText.trim() === 'Generate churn assessment');
            if (b) b.click();
        }"""
    )
    page.wait_for_timeout(6000)


def capture_app(page, base_url: str, prefix: str) -> list[str]:
    """Home screen, a completed assessment, contributions, and the batch tab."""
    captured: list[str] = []
    page.goto(base_url, wait_until="domcontentloaded")
    if not _streamlit_ready(page):
        print(f"  ! {prefix}: app did not become ready", file=sys.stderr)
        return captured

    _settle(page, 2)
    name = f"{prefix}-01-home.png"
    page.screenshot(path=str(OUTPUT / name))
    captured.append(name)

    _predict(page)
    name = f"{prefix}-02-assessment.png"
    page.screenshot(path=str(OUTPUT / name))
    captured.append(name)

    # Full-page capture of the whole assessment surface, before switching tabs.
    name = f"{prefix}-04-full-page.png"
    page.screenshot(path=str(OUTPUT / name), full_page=True)
    captured.append(name)

    # Contribution chart. A viewport screenshot at scroll 0 would look identical
    # to the assessment shot regardless of which tab is active, because the chart
    # sits well below the fold — so scroll the chart itself into view first.
    try:
        page.get_by_role("tab", name="Why this score").click(timeout=15_000)
        _settle(page, 4)
        chart = page.locator('[data-testid="stImage"] img, .stImage img').first
        chart.scroll_into_view_if_needed(timeout=15_000)
        _settle(page, 2)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {prefix}: contribution tab — {exc}", file=sys.stderr)
    name = f"{prefix}-03-contributions.png"
    page.screenshot(path=str(OUTPUT / name))
    captured.append(name)

    # Batch scoring tab.
    try:
        page.get_by_role("tab", name="Batch scoring").click(timeout=15_000)
        _settle(page, 4)
        page.evaluate("window.scrollTo(0, 420)")
        _settle(page, 2)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {prefix}: batch tab — {exc}", file=sys.stderr)
    name = f"{prefix}-05-batch.png"
    page.screenshot(path=str(OUTPUT / name))
    captured.append(name)

    # Dark theme, back on the assessment tab.
    try:
        page.get_by_role("tab", name="Single customer assessment").click(timeout=15_000)
        _settle(page, 2)
        page.get_by_role("checkbox", name="Dark mode").click(timeout=15_000)
        _settle(page, 4)
        page.evaluate("window.scrollTo(0, 0)")
        _settle(page, 1.5)
        name = f"{prefix}-06-dark-mode.png"
        page.screenshot(path=str(OUTPUT / name))
        captured.append(name)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {prefix}: dark mode — {exc}", file=sys.stderr)

    return captured


def capture_page(page, url: str, name: str, wait: float = 4.0,
                 full_page: bool = False, scroll: int = 0) -> str | None:
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        _settle(page, wait)
        if scroll:
            page.evaluate(f"window.scrollTo(0, {scroll})")
            _settle(page, 1.5)
        page.screenshot(path=str(OUTPUT / name), full_page=full_page)
        return name
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {name}: {exc}", file=sys.stderr)
        return None


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        page = context.new_page()

        print("Local application…")
        captured += capture_app(page, LOCAL_APP, "app-local")

        print("Deployed Hugging Face Space…")
        captured += capture_app(page, SPACE_APP, "app-space")

        print("GitHub and Hugging Face pages…")
        targets = [
            (REPO, "gh-01-repository.png", 5.0, False, 0),
            (f"{REPO}/tree/main/deploy", "gh-02-deploy-directory.png", 5.0, False, 0),
            (ACTIONS, "gh-03-actions-list.png", 5.0, False, 0),
            (CI_RUN, "gh-04-ci-run.png", 6.0, False, 0),
            (DEPLOY_RUN, "gh-05-deploy-run.png", 6.0, False, 0),
            (WORKFLOW_FILE, "gh-06-deploy-workflow.png", 5.0, False, 0),
            (f"{REPO}/commits/main", "gh-07-commit-history.png", 5.0, False, 0),
            (SPACE_PAGE, "hf-01-space.png", 8.0, False, 0),
            (f"{SPACE_PAGE}/tree/main", "hf-02-space-files.png", 6.0, False, 0),
        ]
        for url, name, wait, full, scroll in targets:
            result = capture_page(page, url, name, wait, full, scroll)
            if result:
                captured.append(result)
                print(f"  {result}")

        browser.close()

    print(f"\n{len(captured)} screenshot(s) -> {OUTPUT}")
    for name in captured:
        size = (OUTPUT / name).stat().st_size // 1024
        print(f"  {name:36} {size:>5} KB")
    return 0 if captured else 1


if __name__ == "__main__":
    raise SystemExit(main())
