from pathlib import Path
from playwright.sync_api import sync_playwright

DEST = Path("/Users/krishnamathurm4pro/Desktop/Academics/SDIAM Term 3/SDAIM FINAL PROJECT")
jobs = [
    ("explainer.html", "Project_Explained_Simply.pdf",
     "Customer Churn Intelligence — Plain-English Summary"),
    ("script.html", "Presentation_Script_3_Speakers.pdf",
     "Customer Churn Intelligence — Presentation Script"),
]
with sync_playwright() as pw:
    b = pw.chromium.launch()
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
