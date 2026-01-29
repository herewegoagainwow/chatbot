from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

'''Processes cropped table images using a vision-language model to extract
 high-level textual summaries of the tables, saving the results to text files.
'''

# -------------------- paths --------------------
BASE_DIR = Path("/home/jellyboi/Desktop/Metamatic_Stuffs/multimodal_chatbot/data/pdf_name")
CROPPED_DIR = BASE_DIR / "cropped_tables"
OUTPUT_DIR = BASE_DIR / "extracted_tables_text"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------- model --------------------
model_id = "Qwen/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 🔴 critical for VRAM
model.config.use_cache = False
model.eval()

# -------------------- instruction --------------------
INSTRUCTION = (
    "Analyze the table shown in this image and extract high-level insights.\n"
    "\n"
    "Your goal is NOT to reproduce the table.\n"
    "\n"
    "Focus on:\n"
    "- What the table is about and why it exists\n"
    "- What each column represents conceptually\n"
    "- What kind of information the rows encode (categories, ranges, mappings, etc.)\n"
    "- Any visible patterns, structure, or organization (grouping, repetition, hierarchy, progression)\n"
    "\n"
    "Guidelines:\n"
    "- Do NOT list individual rows or values\n"
    "- Do NOT rewrite the table in markdown\n"
    "- If the table is numeric-heavy, summarize the trends or structure instead\n"
    "- If precise values matter, explicitly state that the reader should refer to the original table\n"
    "- Do NOT guess or infer information that is not clearly visible\n"
    "\n"
    "Output:\n"
    "- A concise technical summary (1 to 3 short paragraphs)\n"
    "- No bullet lists unless necessary\n"
    "- No JSON\n"
)

# -------------------- loop --------------------
for img_path in sorted(CROPPED_DIR.glob("*.png")):
    print(f"\nProcessing: {img_path.name}")

    image = Image.open(img_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": INSTRUCTION},
            ],
        }
    ]

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

    output_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0].strip()

    if not output_text:
        print("⚠️ Empty output, skipping")
    else:
        out_file = OUTPUT_DIR / f"{img_path.stem}.txt"
        with open(out_file, "w") as f:
            f.write(output_text)
        print(f"Saved → {out_file.name}")

    # 🔴 critical cleanup
    del inputs, output_ids, image
    torch.cuda.empty_cache()
    