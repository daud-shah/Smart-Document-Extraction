
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import re, json


class VendorInfo(BaseModel):
    name:    str = Field(default="")
    address: str = Field(default="")
    phone:   str = Field(default="")
    tax_id:  str = Field(default="")


class DocumentMeta(BaseModel):
    type:     str = Field(default="receipt",  description="receipt | invoice")
    number:   str = Field(default="")
    date:     str = Field(default="",         description="YYYY-MM-DD")
    currency: str = Field(default="USD",      description="USD | EUR | GBP | MYR | IDR")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        v = v.lower().strip()
        return v if v in {"receipt", "invoice"} else "receipt"

    @field_validator("date")
    @classmethod
    def normalise_date(cls, v: str) -> str:
        if not v:
            return ""
        v = v.strip()
        if re.match(r"\d{4}-\d{2}-\d{2}", v):
            return v
        m = re.match(r"(\d{1,2})[/\\-](\d{1,2})[/\\-](\d{2,4})", v)
        if m:
            d, mo, y = m.groups()
            y = ("20" + y) if len(y) == 2 else y
            return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"
        return v


class LineItem(BaseModel):
    description: str   = Field(default="")
    quantity:    float = Field(default=1.0, ge=0)
    unit_price:  float = Field(default=0.0, ge=0)
    total:       float = Field(default=0.0, ge=0)


class Totals(BaseModel):
    subtotal: float = Field(default=0.0, ge=0)
    tax:      float = Field(default=0.0, ge=0)
    discount: float = Field(default=0.0, ge=0)
    total:    float = Field(default=0.0, ge=0)


class Payment(BaseModel):
    method:      str   = Field(default="",  description="cash | card | ewallet")
    amount_paid: float = Field(default=0.0, ge=0)
    change:      float = Field(default=0.0, ge=0)


class ExtractionMeta(BaseModel):
    model_path:            str       = Field(default="path_a",    description="path_a | path_b")
    overall_confidence:    float     = Field(default=0.0,         ge=0.0, le=1.0)
    low_confidence_fields: List[str] = Field(default_factory=list)
    processing_time_ms:    int       = Field(default=0,           ge=0)
    ocr_engine:            str       = Field(default="paddleocr", description="paddleocr | trocr")
    layout_model:          str       = Field(default="yolo26_doclaynet")
    kie_model:             str       = Field(default="layoutlmv3_cord")


class DocumentExtraction(BaseModel):
    vendor:          VendorInfo     = Field(default_factory=VendorInfo)
    document:        DocumentMeta   = Field(default_factory=DocumentMeta)
    line_items:      List[LineItem] = Field(default_factory=list)
    totals:          Totals         = Field(default_factory=Totals)
    payment:         Payment        = Field(default_factory=Payment)
    extraction_meta: ExtractionMeta = Field(default_factory=ExtractionMeta)

    def to_dict(self) -> dict:
        return self.model_dump()

    def flag_low_confidence(self, field_confidences: dict, threshold: float = 0.85) -> None:
        self.extraction_meta.low_confidence_fields = [
            f for f, score in field_confidences.items() if score < threshold
        ]
        if field_confidences:
            self.extraction_meta.overall_confidence = round(
                sum(field_confidences.values()) / len(field_confidences), 4
            )

    @classmethod
    def empty(cls) -> "DocumentExtraction":
        return cls()
