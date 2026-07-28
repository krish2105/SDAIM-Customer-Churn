"""Render the report DOCX (via pandoc HTML) to a paginated, styled PDF.

The route is:
  docx  --pandoc-->  HTML (structure)  --inject CSS-->  --Chromium print--> PDF

The CSS restores the navy headings, ruled tables, captions and page breaks that
the flat pandoc HTML loses, and controls pagination so figures do not split.

Run from the repository root: ``python scripts/make_report_pdf.py``. Requires
``pandoc`` on PATH and Playwright's Chromium (``PLAYWRIGHT_BROWSERS_PATH`` /
the default managed install both work; a pre-seeded system Chromium at
``/opt/pw-browsers/chromium`` is used automatically if present).

Known limitation: the DOCX's Table of Contents is a live Word field
(``TOC \\h \\o "1-3"``). pandoc extracts document content, not a Word field's
cached render, so a TOC populated only by opening the DOCX in Word/LibreOffice
will not appear in this PDF. Open the DOCX once in Word (it is configured to
prompt to update fields on open) or press Ctrl+A, F9 to refresh it there.
"""

import re
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "Customer_Churn_Intelligence_Final_Report.docx"
SRC = Path(__file__).with_name("report_base.html")
STYLED = Path(__file__).with_name("report_styled.html")
PDF = ROOT / "Customer_Churn_Intelligence_Final_Report.pdf"
_PRESEEDED_CHROMIUM = Path("/opt/pw-browsers/chromium")
CHROMIUM_PATH = str(_PRESEEDED_CHROMIUM) if _PRESEEDED_CHROMIUM.is_file() else None

CSS = """
<style>
@page { size: Letter; margin: 20mm 18mm 18mm 18mm; }
* { box-sizing: border-box; }
body {
  font-family: "Calibri", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.7pt; line-height: 1.5; color: #0F172A; max-width: none; margin: 0;
}
/* Headings — pandoc emits bold paragraphs, not <h*>, so target strong-leading paras
   by class where possible; also style real headings if present. */
h1, h2, h3 { font-family: "Calibri", Arial, sans-serif; page-break-after: avoid; }
h1 { color:#1E3A8A; font-size:17pt; border-bottom:2px solid #1E3A8A; padding-bottom:4px;
     margin-top:22px; margin-bottom:10px; }
h2 { color:#0F172A; font-size:13.5pt; margin-top:16px; margin-bottom:7px; }
h3 { color:#475569; font-size:12pt; margin-top:12px; margin-bottom:5px; }
p { margin: 0 0 8px 0; orphans: 2; widows: 2; }
strong { color:#0F172A; }
a { color:#1E3A8A; text-decoration:none; }
img { max-width: 100%; height:auto; display:block; margin:12px auto 4px auto;
      border:1px solid #E2E8F0; border-radius:3px; page-break-inside: avoid; }
/* Figure captions — pandoc renders the italic caption run as <em> in its own para */
em { color:#475569; }
table {
  border-collapse: collapse; width:100%; margin:10px 0 16px 0; font-size:9.6pt;
  page-break-inside: avoid;
}
th {
  background:#EEF3FB; color:#1E3A8A; font-weight:700; text-align:left;
  padding:6px 8px; border:1px solid #CBD5E1;
}
td { padding:5px 8px; border:1px solid #E2E8F0; vertical-align:top; }
tr:nth-child(even) td { background:#F7F9FC; }
code, pre {
  font-family:"SFMono-Regular", Consolas, Menlo, monospace; font-size:8.9pt;
}
pre {
  background:#F4F6FA; border-left:4px solid #1E3A8A; border-radius:3px;
  padding:9px 12px; margin:10px 0 14px 0; white-space:pre-wrap; word-break:break-word;
  page-break-inside: avoid; color:#0F172A;
}
code { background:#EEF2F8; padding:1px 4px; border-radius:3px; }
pre code { background:none; padding:0; }
ul, ol { margin:0 0 10px 0; padding-left:22px; }
li { margin-bottom:3px; }
hr { border:none; border-top:1px solid #CBD5E1; margin:14px 0; }
/* keep list items and headings from stranding */
li, blockquote { page-break-inside: avoid; }
</style>
"""


def main() -> int:
    subprocess.run(
        ["pandoc", "-f", "docx", "-t", "html", "--embed-resources", "--standalone",
         str(DOCX), "-o", str(SRC)],
        check=True,
    )
    html = SRC.read_text(encoding="utf-8")

    # Inject CSS just before </head>.
    html = html.replace("</head>", CSS + "</head>", 1)

    # pandoc wraps the whole doc in a narrow container; widen it.
    html = re.sub(r"max-width:\s*\d+\w+;", "max-width:none;", html)

    STYLED.write_text(html, encoding="utf-8")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROMIUM_PATH)
        page = browser.new_page()
        page.goto(STYLED.resolve().as_uri(), wait_until="load")
        page.wait_for_timeout(3500)
        page.pdf(
            path=str(PDF),
            format="Letter",
            print_background=True,
            margin={"top": "20mm", "bottom": "18mm", "left": "18mm", "right": "18mm"},
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=(
                '<div style="width:100%;font-size:8pt;color:#64748B;text-align:center;'
                'font-family:Calibri,Arial,sans-serif;">'
                'Customer Churn Intelligence — SDAIM Term 3 &nbsp;·&nbsp; '
                'Page <span class="pageNumber"></span> of <span class="totalPages"></span>'
                "</div>"
            ),
        )
        browser.close()

    kb = PDF.stat().st_size // 1024
    print(f"Wrote {PDF}")
    print(f"{kb} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
