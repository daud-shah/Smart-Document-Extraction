# app_local.py -- run the pipeline locally with Gradio
import sys, json, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
from PIL import Image

from pipeline import PathAPipeline
from output_formatter import export_all

BASE_DIR   = Path(__file__).parent
YOLO_PATH  = BASE_DIR / "models/yolo26_doclaynet/yolo26_doclaynet_best.pt"
KIE_PATH   = BASE_DIR / "models/layoutlmv3_cord/best"

print("Loading pipeline locally...")
PIPE = PathAPipeline(
    yolo_checkpoint = str(YOLO_PATH),
    kie_checkpoint  = str(KIE_PATH),
    yolo_device     = "cpu",
    kie_device      = "cpu",
    yolo_conf       = 0.25,
)
print("Pipeline ready")


def process_document(image_path):
    if image_path is None:
        return "{}", None, "Upload an image first.", []

    work         = Path(tempfile.mkdtemp())
    overlay_path = work / "overlay.jpg"
    base         = Path(image_path).stem

    result  = PIPE.run(image_path=image_path, save_overlay=overlay_path)
    data    = result["extraction"].to_dict()
    timings = result["timings"]

    files = export_all(data, work, base, overlay_path=overlay_path)

    timing_lines = [
        "**Pipeline timing**",
        "Layout detection (YOLO26): " + str(round(timings["yolo_ms"])) + " ms",
        "OCR (Tesseract): " + str(round(timings["ocr_ms"])) + " ms",
        "Field extraction (LayoutLMv3): " + str(round(timings["kie_ms"])) + " ms",
        "**Total: " + str(round(timings["total_ms"])) + " ms**",
    ]
    timing_md = "  \n".join(timing_lines)

    overlay_img = Image.open(overlay_path) if overlay_path.exists() else None
    return json.dumps(data, indent=2), overlay_img, timing_md, files


EXAMPLES = [[str(p)] for p in sorted((BASE_DIR / "examples").glob("*.jpg"))]

with gr.Blocks(title="Smart Document Extraction (Local)") as demo:
    gr.Markdown("# Smart Document Extraction Pipeline\nRunning locally on CPU.")

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
        downloads = gr.File(label="Downloads", file_count="multiple")

    if EXAMPLES:
        gr.Examples(examples=EXAMPLES, inputs=inp)

    run_btn.click(fn=process_document, inputs=inp, outputs=[json_out, overlay, timing, downloads])

demo.launch()