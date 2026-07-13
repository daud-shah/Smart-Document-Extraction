"""pipeline.py -- Path A end-to-end orchestration."""
import time
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).parent))

from layout_detector import LayoutDetector
from ocr_engine import OCREngine
from kie_model import KIEModel
from json_structurer import structure
from visualiser import draw_pipeline_output


class PathAPipeline:
    def __init__(self, yolo_checkpoint, kie_checkpoint,
                 yolo_device="cpu", kie_device="cpu", yolo_conf=0.25):
        print("Loading LayoutDetector...")
        self.detector = LayoutDetector(yolo_checkpoint, device=yolo_device, conf=yolo_conf)
        print("Loading OCREngine...")
        self.ocr = OCREngine(lang="en", use_gpu=False)
        print("Loading KIEModel on " + kie_device + "...")
        self.kie = KIEModel(kie_checkpoint, device=kie_device)
        print("Pipeline ready")

    def run(self, image_path, save_overlay=None, save_json=None):
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")

        regions, yolo_ms = self.detector.detect(image_path)
        if not regions:
            regions = self.detector.fallback_regions(image)

        t_ocr = time.time()
        fake_result = _FakeYOLOResult(regions)
        ocr_output = self.ocr.run_pipeline(image, fake_result, self.detector.classes)
        ocr_ms = (time.time() - t_ocr) * 1000

        all_words, all_bboxes = [], []
        img_w, img_h = image.size
        for region_data in ocr_output.values():
            for word in region_data.get("words", []):
                all_words.append(word["text"])
                bx1, by1, bx2, by2 = word["xyxy_bbox"]
                all_bboxes.append([
                    max(0, min(1000, int(bx1 / img_w * 1000))),
                    max(0, min(1000, int(by1 / img_h * 1000))),
                    max(0, min(1000, int(bx2 / img_w * 1000))),
                    max(0, min(1000, int(by2 / img_h * 1000))),
                ])

        t_kie = time.time()
        token_preds = self.kie.predict(image, all_words, all_bboxes) if all_words else []
        kie_ms = (time.time() - t_kie) * 1000

        field_confs = defaultdict(list)
        for item in token_preds:
            label = item.get("label", "O")
            if label != "O":
                field = label.replace("B-", "").replace("I-", "")
                field_confs[field].append(item["confidence"])
        field_confidences = dict(
            (f, round(sum(v) / len(v), 4)) for f, v in field_confs.items()
        )

        extraction = structure(
            token_predictions=token_preds,
            field_confidences=field_confidences,
            yolo_ms=yolo_ms, ocr_ms=ocr_ms, kie_ms=kie_ms,
        )

        timings = {
            "yolo_ms": round(yolo_ms, 1),
            "ocr_ms": round(ocr_ms, 1),
            "kie_ms": round(kie_ms, 1),
            "total_ms": round(yolo_ms + ocr_ms + kie_ms, 1),
        }

        if save_overlay:
            draw_pipeline_output(
                image_path=image_path, regions=regions,
                pipeline_output=ocr_output, extraction=extraction,
                save_path=save_overlay,
            )

        if save_json:
            Path(save_json).write_text(
                json.dumps(extraction.to_dict(), indent=2), encoding="utf-8"
            )

        return {
            "extraction": extraction,
            "regions": regions,
            "ocr_output": ocr_output,
            "timings": timings,
        }


class _FakeBox:
    def __init__(self, regions):
        import torch
        boxes = [r["bbox"] for r in regions]
        cls_map = {"Text": 0, "Table": 1, "Section-header": 2, "Title": 3, "Picture": 4}
        cls_ids = [cls_map.get(r["class_name"], 0) for r in regions]
        confs = [r["confidence"] for r in regions]
        self.xyxy = torch.tensor(boxes, dtype=torch.float32)
        self.cls = torch.tensor(cls_ids, dtype=torch.float32)
        self.conf = torch.tensor(confs, dtype=torch.float32)
        self._n = len(regions)

    def __len__(self):
        return self._n


class _FakeYOLOResult:
    def __init__(self, regions):
        self.boxes = _FakeBox(regions) if regions else None