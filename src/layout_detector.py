"""layout_detector.py -- YOLO26 layout detection wrapper."""
import time
from PIL import Image
from ultralytics import YOLO

YOLO_CLASSES = ["Text", "Table", "Section-header", "Title", "Picture"]


class LayoutDetector:
    def __init__(self, checkpoint_path, device="cpu", conf=0.25, iou=0.45):
        self.model = YOLO(str(checkpoint_path))
        self.device = device
        self.conf = conf
        self.iou = iou
        self.classes = YOLO_CLASSES

    def detect(self, image_path):
        t0 = time.time()
        result = self.model.predict(
            source=str(image_path), imgsz=640, conf=self.conf,
            iou=self.iou, device=self.device, verbose=False,
        )[0]
        latency_ms = (time.time() - t0) * 1000

        regions = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for box, cls_id, conf in zip(boxes, cls_ids, confs):
                cls_name = self.classes[cls_id] if cls_id < len(self.classes) else str(cls_id)
                regions.append({
                    "class_name": cls_name,
                    "bbox": [float(v) for v in box],
                    "confidence": round(float(conf), 4),
                })

        regions.sort(key=lambda r: r["bbox"][1])
        return regions, latency_ms

    def fallback_regions(self, image):
        w, h = image.size
        return [{"class_name": "Text", "bbox": [0.0, 0.0, float(w), float(h)], "confidence": 0.5}]