from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import re

# ============================================================
# CONFIG
# ============================================================

PDF_PATH = "input.pdf"
BASE_DIR = Path("data/pdf_name")

TEXT_DIR = BASE_DIR / "text"
PAGE_IMG_DIR = BASE_DIR / "pages"
CROPPED_TABLES_DIR = BASE_DIR / "cropped_tables"

TEXT_DIR.mkdir(parents=True, exist_ok=True)
PAGE_IMG_DIR.mkdir(exist_ok=True)

# ============================================================
# MODEL (shared for tables + diagrams)
# ============================================================

model_id = "Qwen/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.config.use_cache = False  # 🔴 VRAM critical
model.eval()

# ============================================================
# INSTRUCTIONS
# ============================================================

TABLE_INSTRUCTION = (
    "Analyze the table shown in this image and extract high-level insights.\n"
    "\n"
    "Focus on:\n"
    "- What the table is about\n"
    "- What each column represents conceptually\n"
    "- What kind of information rows encode\n"
    "- Visible structure or patterns\n"
    "\n"
    "Rules:\n"
    "- Do NOT infer unseen information\n"
    "\n"
    "Output:\n"
    "- 1–3 short technical paragraphs\n"
)

DIAGRAM_INSTRUCTION = (
   "Analyze the diagram carefully and extract all information that can be directly observed.\n"
"\n"
"Your goal is to capture whatever is necessary to understand, reason about, or reconstruct the diagram.\n"
"\n"
"Include anything that seems important, such as:\n"
"- structural elements\n"
"- labels and text\n"
"- spatial or geometric relationships\n"
"- quantities, measurements, or scales (if visible)\n"
"- visual conventions (arrows, groupings, styles, symbols)\n"
"\n"
"Organize the output using clear labels or sections of your choice.\n"
"\n"
"Do not write prose or explanations—focus on extraction.\n"
"\n"
"If something is ambiguous, mark it as unclear rather than guessing.\n"
"- Do not add knowledge that is not present in the diagram.\n"
)

# ============================================================
# UTILS
# ============================================================

def vlm_infer(image: Image.Image, instruction: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction},
        ],
    }]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False
        )

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]

    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0].strip()

    del inputs, output_ids
    torch.cuda.empty_cache()

    return output if output else "[EMPTY OUTPUT]"


def update_section(file_path: Path, header: str, new_content: str):
    text = file_path.read_text(encoding="utf-8")

    pattern = rf"--- {re.escape(header)} ---.*?(?=\n--- |\Z)"
    replacement = f"--- {header} ---\n{new_content.strip()}\n"

    updated = re.sub(pattern, replacement, text, flags=re.S)

    file_path.write_text(updated, encoding="utf-8")


# ============================================================
# INGESTION STEP 1 — RAW TEXT + PAGE IMAGE
# ============================================================

doc = fitz.open(PDF_PATH)
print(f"Total pages: {len(doc)}")

for i, page in enumerate(doc):
    page_num = i + 1

    raw_text = page.get_text().strip()

    txt_path = TEXT_DIR / f"page_{page_num}.txt"
    img_path = PAGE_IMG_DIR / f"page_{page_num}.png"

    pix = page.get_pixmap(dpi=200)
    pix.save(img_path)

    txt_path.write_text(
        f"""=== PAGE {page_num} ===

--- RAW TEXT ---
{raw_text if raw_text else "[NO TEXT FOUND]"}

--- TABLES (EXTRACTED) ---
[PENDING]

--- DIAGRAM CONTEXT (EXTRACTED) ---
[PENDING]
""",
        encoding="utf-8"
    )

    print(f"Initialized page {page_num}")

# ============================================================
# INGESTION STEP 2 — TABLE EXTRACTION (CROPPED IMAGES)
# ============================================================

for img_path in sorted(CROPPED_TABLES_DIR.glob("*.png")):
    print(f"Table → {img_path.name}")

    # expected naming: page_3_table_1.png
    match = re.search(r"page_(\d+)", img_path.name)
    if not match:
        continue

    page_num = int(match.group(1))
    page_txt = TEXT_DIR / f"page_{page_num}.txt"

    image = Image.open(img_path).convert("RGB")
    table_summary = vlm_infer(image, TABLE_INSTRUCTION)

    update_section(
        page_txt,
        "TABLES (EXTRACTED)",
        table_summary
    )

# ============================================================
# INGESTION STEP 3 — DIAGRAM EXTRACTION (PAGE IMAGE)
# ============================================================

for img_path in sorted(PAGE_IMG_DIR.glob("page_*.png")):
    page_num = int(re.search(r"page_(\d+)", img_path.name).group(1))
    page_txt = TEXT_DIR / f"page_{page_num}.txt"

    image = Image.open(img_path).convert("RGB")
    diagram_text = vlm_infer(image, DIAGRAM_INSTRUCTION)

    update_section(
        page_txt,
        "DIAGRAM CONTEXT (EXTRACTED)",
        diagram_text
    )

print("✅ FULL INGESTION COMPLETE")
