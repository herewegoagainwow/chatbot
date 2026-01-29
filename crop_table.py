import fitz
import json
import os
import re

'''Opens a PDF, detects pages containing tables based on prior analysis,
 extracts and smart-crops the table regions, and saves them as PNG images
 for downstream processing.
'''

PDF_PATH = "input.pdf"
PAGES_JSON = "data/pdf_name/table_pages.json"
OUT_DIR = "/home/jellyboi/Desktop/Metamatic_Stuffs/multimodal_chatbot/data/pdf_name/cropped_tables"

os.makedirs(OUT_DIR, exist_ok=True)

doc = fitz.open(PDF_PATH)

with open(PAGES_JSON, "r") as f:
    table_pages = json.load(f)

def looks_like_table(text):
    if text.count("|") >= 2:
        return True
    if len(re.findall(r"\d+", text)) >= 6:
        return True
    if text.count("\n") >= 3:
        return True
    return False

for page_num in table_pages:
    page = doc[page_num - 1]
    blocks = page.get_text("blocks")

    table_blocks = []

    for b in blocks:
        x0, y0, x1, y1, text, *_ = b
        if looks_like_table(text.lower()):
            table_blocks.append(fitz.Rect(x0, y0, x1, y1))

    if not table_blocks:
        print(f"No table block detected on page {page_num}")
        continue

    # merge all detected table blocks
    table_rect = table_blocks[0]
    for r in table_blocks[1:]:
        table_rect |= r

    # add small padding
    table_rect.x0 -= 10
    table_rect.y0 -= 10
    table_rect.x1 += 10
    table_rect.y1 += 10

    pix = page.get_pixmap(clip=table_rect, dpi=300)
    out_path = os.path.join(OUT_DIR, f"page_{page_num}.png")
    pix.save(out_path)

    print(f"Smart-cropped page {page_num} → {out_path}")
