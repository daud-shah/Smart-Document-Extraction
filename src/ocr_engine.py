"""
ocr_engine.py -- Phase 3
Pytesseract wrapper for the Smart Document Extraction Pipeline.
No dependency conflicts. Works on CPU. HF Space compatible.
Requires: apt-get install tesseract-ocr
"""
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output


def _run_tesseract(img_pil, psm=6):
    data = pytesseract.image_to_data(
        img_pil,
        output_type = Output.DICT,
        config      = f"--psm {psm} --oem 3",
    )
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not text or conf < 0:
            continue
        x1 = data["left"][i]
        y1 = data["top"][i]
        x2 = x1 + data["width"][i]
        y2 = y1 + data["height"][i]
        words.append({
            "text":       text,
            "confidence": round(conf / 100.0, 4),
            "quad_bbox":  [[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
            "xyxy_bbox":  [x1, y1, x2, y2],
        })
    return words


class OCREngine:
    def __init__(self, lang="en", use_gpu=False):
        # lang and use_gpu kept for API compatibility -- tesseract is always CPU
        self._lang = lang

    def run_on_image(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return _run_tesseract(image, psm=11)

    def run_on_region(self, full_image, region_xyxy):
        x1, y1, x2, y2 = [int(v) for v in region_xyxy]
        crop  = full_image.crop((x1, y1, x2, y2))
        words = _run_tesseract(crop, psm=6)
        # Shift bboxes from crop coords to full image coords
        for w in words:
            w["quad_bbox"] = [[p[0]+x1, p[1]+y1] for p in w["quad_bbox"]]
            bx1,by1,bx2,by2 = w["xyxy_bbox"]
            w["xyxy_bbox"] = [bx1+x1, by1+y1, bx2+x1, by2+y1]
        return words

    def run_pipeline(self, full_image, yolo_result, yolo_classes):
        output = {}
        if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            return output
        boxes   = yolo_result.boxes.xyxy.cpu().numpy()
        cls_ids = yolo_result.boxes.cls.cpu().numpy().astype(int)
        confs   = yolo_result.boxes.conf.cpu().numpy()
        for idx, (box, cls_id, conf) in enumerate(zip(boxes, cls_ids, confs)):
            cls_name = yolo_classes[cls_id] if cls_id < len(yolo_classes) else str(cls_id)
            words    = self.run_on_region(full_image, box)
            output[idx] = {
                "region_class": cls_name,
                "region_bbox":  [float(v) for v in box],
                "confidence":   round(float(conf), 4),
                "words":        words,
            }
        return output