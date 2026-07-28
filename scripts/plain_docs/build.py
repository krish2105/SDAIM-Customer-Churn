from pathlib import Path
from playwright.sync_api import sync_playwright

DEST = Path(__file__).resolve().parents[2]
# Some hosts (e.g. this project's CI-adjacent sandboxes) pre-install Chromium
# outside Playwright's own managed browser directory. Prefer it if present;
# otherwise fall back to Playwright's default discovery.
_PRESEEDED_CHROMIUM = Path("/opt/pw-browsers/chromium")
CHROMIUM_PATH = str(_PRESEEDED_CHROMIUM) if _PRESEEDED_CHROMIUM.is_file() else None
jobs = [
    ("explainer.html", "Project_Explained_Simply.pdf",
     "Customer Churn Intelligence — Plain-English Summary"),
    ("script.html", "Presentation_Script_3_Speakers.pdf",
     "Customer Churn Intelligence — Presentation Script"),
]
with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path=CHROMIUM_PATH)
    for src, out, foot in jobs:
        pg = b.new_page()
        pg.goto(Path(src).resolve().as_uri(), wait_until="load")
        pg.wait_for_timeout(1200)
        pg.pdf(
            path=str(DEST / out), format="Letter", print_background=True,
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
        )
        pg.close()
        kb = (DEST / out).stat().st_size // 1024
        print(f"{out}: {kb} KB")
    b.close()
