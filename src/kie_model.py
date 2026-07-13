"""kie_model.py -- Phase 4
LayoutLMv3 KIE inference wrapper.
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

LABEL_LIST = [
    "O",
    "B-ITEM_NAME",  "I-ITEM_NAME",
    "B-ITEM_QTY",   "I-ITEM_QTY",
    "B-ITEM_PRICE", "I-ITEM_PRICE",
    "B-SUBTOTAL",   "I-SUBTOTAL",
    "B-TAX",        "I-TAX",
    "B-DISCOUNT",   "I-DISCOUNT",
    "B-TOTAL",      "I-TOTAL",
    "B-PAYMENT",    "I-PAYMENT",
]
id2label = {i: l for i, l in enumerate(LABEL_LIST)}
label2id = {l: i for i, l in enumerate(LABEL_LIST)}


class KIEModel:
    def __init__(self, checkpoint_dir, device="cpu"):
        self.device    = device
        self.processor = LayoutLMv3Processor.from_pretrained(checkpoint_dir, apply_ocr=False)
        self.model     = LayoutLMv3ForTokenClassification.from_pretrained(
            checkpoint_dir, num_labels=len(LABEL_LIST),
            id2label=id2label, label2id=label2id,
        ).to(device).eval()

    def predict(self, image, words, bboxes):
        if not words:
            return []
        encoding = self.processor(
            image, words, boxes=bboxes,
            max_length=512, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
        pred_ids = outputs.logits.argmax(-1).squeeze().cpu().numpy()
        logits   = outputs.logits.squeeze().cpu().numpy()
        tokens   = self.processor.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"].squeeze()
        )

        results = []
        for token, pred_id, logit in zip(tokens, pred_ids, logits):
            if token in ["<s>", "</s>", "<pad>"]:
                continue
            label = id2label.get(int(pred_id), "O")
            conf  = float(np.exp(logit[int(pred_id)]) / np.exp(logit).sum())
            results.append({
                "token": token.replace("▁", " ").strip(),
                "label": label,
                "confidence": round(conf, 4),
            })
        return results