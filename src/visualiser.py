"""visualiser.py -- OpenCV annotated overlay for Path A output."""
import cv2
import numpy as np
from PIL import Image

REGION_COLORS = {
    "Text": (219, 152, 52), "Table": (113, 204, 46),
    "Section-header": (60, 76, 231), "Title": (15, 196, 241),
    "Picture": (182, 89, 155),
}
DEFAULT_COLOR = (128, 128, 128)


def draw_pipeline_output(image_path, regions, pipeline_output, extraction, save_path=None):
    img = cv2.imread(str(image_path))
    if img is None:
        pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    overlay = img.copy()

    for region in regions:
        x1, y1, x2, y2 = [int(v) for v in region["bbox"]]
        color = REGION_COLORS.get(region["class_name"], DEFAULT_COLOR)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = region["class_name"] + " " + str(round(region["confidence"], 2))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    for idx, region_data in pipeline_output.items():
        color = REGION_COLORS.get(region_data.get("region_class", "Text"), DEFAULT_COLOR)
        for word in region_data.get("words", []):
            bx1, by1, bx2, by2 = [int(v) for v in word["xyxy_bbox"]]
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, 1)

    result = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)

    if save_path:
        cv2.imwrite(str(save_path), result)

    return result