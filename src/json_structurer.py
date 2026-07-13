"""json_structurer.py -- groups KIE tokens into DocumentExtraction schema."""
import re
from collections import defaultdict
from typing import List, Dict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from schema import DocumentExtraction, VendorInfo, DocumentMeta, LineItem, Totals, Payment, ExtractionMeta


def _to_float(text):
    cleaned = re.sub(r"[^\d.,]", "", text.strip())
    cleaned = cleaned.replace(",", ".")
    parts = cleaned.split(".")
    if len(parts) > 2:
        cleaned = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return round(float(cleaned), 2)
    except (ValueError, TypeError):
        return 0.0


def _join_tokens(token_list):
    text = ""
    for tok in token_list:
        tok = tok.replace("\u2581", " ").strip()
        if not tok:
            continue
        text = (text + " " + tok) if text else tok
    return text.strip()


def structure(token_predictions, field_confidences, yolo_ms=0, ocr_ms=0, kie_ms=0):
    field_tokens = defaultdict(list)
    for item in token_predictions:
        label = item.get("label", "O")
        token = item.get("token", "")
        if label == "O" or not token:
            continue
        field = label.replace("B-", "").replace("I-", "")
        field_tokens[field].append(token)

    def get(field):
        return _join_tokens(field_tokens.get(field, []))

    line_items = []
    name_text = get("ITEM_NAME")
    price_text = get("ITEM_PRICE")
    qty_text = get("ITEM_QTY")
    if name_text or price_text:
        line_items.append(LineItem(
            description=name_text,
            quantity=_to_float(qty_text) if qty_text else 1.0,
            unit_price=_to_float(price_text),
            total=_to_float(price_text),
        ))

    totals = Totals(
        subtotal=_to_float(get("SUBTOTAL")),
        tax=_to_float(get("TAX")),
        discount=_to_float(get("DISCOUNT")),
        total=_to_float(get("TOTAL")),
    )

    payment_text = get("PAYMENT")
    payment = Payment(method="", amount_paid=_to_float(payment_text), change=0.0)

    total_ms = int(yolo_ms + ocr_ms + kie_ms)
    meta = ExtractionMeta(
        model_path="path_a", processing_time_ms=total_ms,
        ocr_engine="tesseract", layout_model="yolo26_doclaynet", kie_model="layoutlmv3_cord",
    )

    doc = DocumentExtraction(
        vendor=VendorInfo(), document=DocumentMeta(), line_items=line_items,
        totals=totals, payment=payment, extraction_meta=meta,
    )
    doc.flag_low_confidence(field_confidences)
    return doc