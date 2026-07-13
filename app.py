# app.py -- Smart Document Extraction Pipeline (HF Space entry point)
# Upload a receipt/invoice image -> structured JSON + annotated overlay
# + downloads in JSON / CSV / Excel / Word / PDF
import os, sys, json, time, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download

from pipeline import PathAPipeline
from output_formatter import export_all

# ── Config: your HF model repos ───────────────────────────────────────
KIE_REPO  = "daudshah/layoutlmv3-cord-receipts"
YOLO_REPO = "daudshah/yolo26-doclaynet-layout"

# ── Load models ONCE at startup (module-level singleton) ─────────────
print("Downloading YOLO checkpoint from HF Hub...")
yolo_path = hf_hub_download(YOLO_REPO, "yolo26_doclaynet_best.pt")

print("Loading pipeline (this takes ~60s on first start)...")
PIPE = PathAPipeline(
    yolo_checkpoint = yolo_path,
    kie_checkpoint  = KIE_REPO,     # loads from Hub -- single-slash repo id works
    yolo_device     = "cpu",
    kie_device      = "cpu",
    yolo_conf       = 0.25,
)
print("Pipeline ready")

# Warm-up call so first user request is not slow
try:
    _examples = sorted((Path(__file__).parent / "examples").glob("*.jpg"))
    if _examples:
        PIPE.run(_examples[0])
        print("Warm-up complete")
except Exception as e:
    print(f"Warm-up skipped: {e}")


def process_document(image_path):
    # Main handler: image path in -> (json_str, overlay_img, timing_md, files)
    if image_path is None:
        return "{}", None, "Upload an image first.", []

    work = Path(tempfile.mkdtemp())
    overlay_path = work / "overlay.jpg"
    base = Path(image_path).stem

    result = PIPE.run(
        image_path   = image_path,
        save_overlay = overlay_path,
    )

    data    = result["extraction"].to_dict()
    timings = result["timings"]

    # Export all download formats
    files = export_all(data, work, base, overlay_path=overlay_path)

    timing_md = (
        f"**Pipeline timing**  
"
        f"Layout detection (YOLO26): {timings['yolo_ms']:.0f} ms  
"
        f"OCR (Tesseract): {timings['ocr_ms']:.0f} ms  
"
        f"Field extraction (LayoutLMv3): {timings['kie_ms']:.0f} ms  
"
        f"**Total: {timings['total_ms']:.0f} ms**"
    )

    overlay_img = Image.open(overlay_path) if overlay_path.exists() else None
    return json.dumps(data, indent=2), overlay_img, timing_md, files


# ── UI ─────────────────────────────────────────────────────────────────
EXAMPLES = [[str(p)] for p in sorted((Path(__file__).parent / "examples").glob("*.jpg"))]

with gr.Blocks(title="Smart Document Extraction") as demo:
    gr.Markdown(
        "# Smart Document Extraction Pipeline
"
        "Upload a receipt or invoice image. Get structured JSON, an annotated "
        "overlay, and downloads in 5 formats.  
"
        "**Stack:** YOLO26 layout detection -> Tesseract OCR -> LayoutLMv3 KIE "
        "(fine-tuned on CORD, entity F1 = 0.971) -> Pydantic-validated JSON."
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp     = gr.Image(type="filepath", label="Receipt / Invoice image")
            run_btn = gr.Button("Extract", variant="primary")
            timing  = gr.Markdown()
        with gr.Column(scale=1):
            overlay = gr.Image(label="Annotated overlay")
    with gr.Row():
        json_out = gr.Code(label="Extracted JSON", language="json")
    with gr.Row():
        downloads = gr.Files(label="Download (JSON / CSV / Excel / Word / PDF)")

    if EXAMPLES:
        gr.Examples(examples=EXAMPLES, inputs=inp)

    run_btn.click(
        fn      = process_document,
        inputs  = inp,
        outputs = [json_out, overlay, timing, downloads],
    )

demo.launch()