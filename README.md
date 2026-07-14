---
title: Smart Document Extraction
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Smart Document Extraction Pipeline (SDEP)

A production-ready document extraction system for receipts and invoices with modular architecture, powered by YOLO layout detection, Tesseract OCR, and LayoutLMv3 key information extraction.

## Project Overview

This repository contains the full end-to-end Smart Document Extraction workflow, including:
- document image samples
- trained model checkpoints
- training notebooks for each development phase
- the inference pipeline and export utilities
- a Gradio-based web app for testing

## Features ✨

- **Multi-format Extraction**: Receipts, invoices, and structured documents
- **Layout-aware OCR**: YOLO26 detects document regions → Tesseract extracts text
- **Smart Entity Recognition**: LayoutLMv3 fine-tuned on CORD v2 (F1: 0.971)
- **Annotated Visualization**: Bounding boxes and confidence scores overlaid on documents
- **Multiple Export Formats**: JSON, CSV, Excel (.xlsx), Word (.docx), PDF
- **CPU-only Inference**: Runs on free hardware without GPU
- **Structured Output**: Pydantic-validated JSON schema for vendor, line items, totals, payment info

## Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/yourusername/sdep-space.git
cd sdep-space

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (Command Prompt):
.\.venv\Scripts\activate.bat

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: First run will auto-download pre-trained models (~500MB) from Hugging Face Hub.

### 3. Run the Application

#### Option A: Gradio Web UI (Recommended)
```bash
python app.py
```
Opens at `http://localhost:7860` — drag & drop document images for extraction.

#### Option B: Local Debugging
```bash
python app_local.py
```
Runs inference on sample images in `examples/` folder with debug output.

#### Option C: Python API
```python
from src.pipeline import PathAPipeline

# Initialize pipeline
pipe = PathAPipeline(
    yolo_checkpoint="path/to/yolo_model.pt",
    kie_checkpoint="daudshah/layoutlmv3-cord-receipts",
    yolo_device="cpu",
    kie_device="cpu"
)

# Run on an image
result = pipe.run("path/to/document.jpg")
print(result.json())  # Pydantic model → JSON
```

## Architecture

### Data Flow (Path A: Modular Pipeline)

```
Input Image
    ↓
[YOLO26 Layout Detector] → Region bounding boxes + labels
    ↓
[Tesseract OCR] → Extract text from each region
    ↓
[LayoutLMv3 KIE] → Token-level entity classification
    ↓
[Pydantic Structurer] → Validated JSON schema
    ↓
Output: JSON + Overlay Image + Export Files
```

### Model Performance

| Component | Metric | Score |
|-----------|--------|-------|
| LayoutLMv3 (KIE) | F1 (Entity) | 0.971 |
| YOLO26 (Layout) | mAP50 | 0.735 |
| OCR | Tesseract | CPU-optimized |

### Models

- **KIE Model**: [layoutlmv3-cord-receipts](https://huggingface.co/daudshah/layoutlmv3-cord-receipts) - fine-tuned on CORD dataset
- **Layout Model**: [yolo26-doclaynet-layout](https://huggingface.co/daudshah/yolo26-doclaynet-layout) - trained on DocLayNet

## Project Structure

```
Smart-Document-Extraction/
├── app.py                                # Gradio web UI
├── app_local.py                          # Local debugging and sample run script
├── requirements.txt                      # Python dependencies
├── packages.txt                          # System-level dependencies
├── output_formatter.py                   # Export utilities for JSON/CSV/Excel/PDF
├── example_1.csv                         # Sample CSV output
├── example_1.pdf                         # Sample PDF output
├── example_2.xlsx                       # Sample Excel output
├── examples/                             # Sample document images
│   ├── example_1.jpg
│   ├── example_2.jpg
│   └── example_3.jpg
├── models/                               # Pretrained model checkpoints
│   ├── layoutlmv3_cord/best/
│   └── yolo26_doclaynet/
├── phase by phase model traning/        # Jupyter notebooks for each training phase
│   ├── phase1-data-explorationebd24f8e4e.ipynb
│   ├── phase2-yolo-layout-finetuning.ipynb
│   ├── phase3-paddleocr-integration.ipynb
│   ├── phase4-layoutlmv3-kie.ipynb
│   ├── phase5-pipeline-integration.ipynb
│   └── phase6-donut-baseline.ipynb
├── src/                                  # Core project modules
│   ├── pipeline.py
│   ├── layout_detector.py
│   ├── ocr_engine.py
│   ├── kie_model.py
│   ├── json_structurer.py
│   ├── schema.py
│   └── visualiser.py
└── README.md                             # Project documentation
```

## Configuration

### Environment Variables

Create `.env` file (optional):
```env
YOLO_CONF=0.25              # YOLO confidence threshold
KIE_DEVICE=cpu              # or 'cuda' for GPU
YOLO_DEVICE=cpu             # or 'cuda' for GPU
HF_TOKEN=your_hf_token      # For private model access
```

### Output Schema

The pipeline extracts and validates:

```python
{
  "vendor": {
    "name": "Store Name",
    "address": "...",
    "phone": "..."
  },
  "items": [
    {
      "description": "Product Name",
      "quantity": 1.0,
      "unit_price": 9.99,
      "amount": 9.99
    }
  ],
  "totals": {
    "subtotal": 9.99,
    "tax": 0.80,
    "total": 10.79
  },
  "payment": {
    "method": "Card",
    "reference": "..."
  },
  "metadata": {
    "confidence": 0.95,
    "timestamp": "2024-01-15T..."
  }
}
```

## Dependencies

### Python Packages
- **torch** ≥2.0.0 – PyTorch framework
- **transformers** ≥4.40.0 – HuggingFace transformers (LayoutLMv3)
- **ultralytics** ≥8.3.0 – YOLO framework
- **pytesseract** ≥0.3.10 – OCR wrapper
- **pydantic** ≥2.0.0 – Data validation
- **gradio** ≥4.0.0 – Web UI
- **opencv-python-headless** ≥4.8.0 – Image processing
- **pandas** ≥2.0.0 – Data manipulation
- **openpyxl** ≥3.1.0 – Excel export

### System Dependencies
Install from `packages.txt` (if using Linux/WSL):
```bash
cat packages.txt | xargs apt-get install -y
```

## Usage Examples

### 1. Web UI (Gradio)
```bash
python app.py
# Navigate to http://localhost:7860
# 1. Upload image
# 2. View structured JSON output
# 3. Download in preferred format
```

### 2. Batch Processing
```python
from pathlib import Path
from src.pipeline import PathAPipeline
from output_formatter import export_all

pipe = PathAPipeline()
results = []

for img_path in Path("documents").glob("*.jpg"):
    result = pipe.run(img_path)
    results.append(result.dict())
    # Export to all formats
    export_all(result, output_dir=f"output/{img_path.stem}")
```

### 3. API Integration
```python
import json
from src.pipeline import PathAPipeline

pipe = PathAPipeline()

# Process from file
result = pipe.run("receipt.jpg")
print(json.dumps(result.dict(), indent=2))

# Process from PIL Image
from PIL import Image
img = Image.open("invoice.png")
result = pipe.run(img)
```

## Troubleshooting

### Issue: Models fail to download
**Solution**: Pre-download models manually
```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('daudshah/layoutlmv3-cord-receipts', 'config.json')
hf_hub_download('daudshah/yolo26-doclaynet-layout', 'yolo26_doclaynet_best.pt')
"
```

### Issue: Tesseract not found
**Solution**: Install Tesseract-OCR
- **Windows**: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

### Issue: Out of memory
**Solution**: Reduce batch size or enable GPU if available
```python
pipe = PathAPipeline(yolo_device="cuda", kie_device="cuda")
```

### Issue: Gradio port already in use
**Solution**: Specify custom port
```bash
python app.py --server_port 7861
```

## Development

### Project Setup for Contributors
```bash
git clone <repo-url>
cd sdep-space
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
python app_local.py  # Test local run
```

### Add Custom Extraction Logic
Edit `src/json_structurer.py` to add field mappings:
```python
class ReceiptExtractor:
    def extract(self, tokens, bboxes, labels):
        # Custom logic here
        pass
```

### Debug Mode
```python
from src.pipeline import PathAPipeline
pipe = PathAPipeline(verbose=True)
result = pipe.run("image.jpg")
# Prints detailed extraction steps
```

## Performance Metrics

| Metric | Value | Hardware |
|--------|-------|----------|
| Avg Processing Time | ~5-10s | CPU (4 cores) |
| Memory Usage | ~2GB peak | Inference time |
| Model Size | ~500MB | Compressed |
| Accuracy (F1) | 0.971 | LayoutLMv3 |

## License

This project is open source under the MIT License. See `LICENSE` file for details.

## Citation

If you use this project, please cite:

```bibtex
@software{sdep2024,
  title = {Smart Document Extraction Pipeline},
  author = {Daud Shah},
  year = {2024},
  url = {https://github.com/yourusername/sdep-space}
}
```

## Support & Contributing

- **Issues**: Report bugs on [GitHub Issues](https://github.com/yourusername/sdep-space/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/yourusername/sdep-space/discussions)
- **Pull Requests**: Contributions welcome! Please follow [CONTRIBUTING.md](CONTRIBUTING.md)

## Roadmap

- [ ] Batch processing API endpoint
- [ ] Multi-language OCR support
- [ ] Custom field templates
- [ ] Real-time confidence visualization
- [ ] Mobile app integration
- [ ] REST API server

## Contact

Author: Daud Shah  
Email: [sdaud4214@gmail.com](mailto:sdaud4214@gmail.com)  
LinkedIn: [Your Profile](https://linkedin.com/in/yourusername)

---

**Last Updated**: 2024-07-13  
**Status**: Active Development ✨
