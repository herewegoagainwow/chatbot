import json
from pathlib import Path
import fitz  # PyMuPDF

'''Opens a PDF, performs a heuristic scan to detect pages likely containing tables,
 and saves the list of such page numbers to a JSON file for downstream processing.'''

# paths
PDF_PATH = Path("input.pdf")
OUTPUT_PATH = Path("data/pdf_name/table_pages.json")

doc = fitz.open(PDF_PATH)

table_pages = []

# simple heuristic scan (cheap pass)
for page_idx, page in enumerate(doc, start=1):
    text = page.get_text("text").lower()

    if "table" in text or "|" in text:
        table_pages.append(page_idx)

print("Table candidate pages:", table_pages)

# persist detector output
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(table_pages, f, indent=2)
