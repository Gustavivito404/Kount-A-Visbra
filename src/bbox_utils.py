from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw


def detect_bbox_white_background(
    img: Image.Image,
    white_threshold: int = 245,
    min_foreground_pixels: int = 50,
) -> Optional[list[int]]:
    arr = np.array(img.convert("RGB"))
    background_mask = np.all(arr >= white_threshold, axis=2)
    foreground_mask = ~background_mask

    ys, xs = np.where(foreground_mask)

    if len(xs) < min_foreground_pixels or len(ys) < min_foreground_pixels:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def build_preview_image(
    img: Image.Image,
    bbox_xyxy: list[int] | list[float] | None,
    max_size: tuple[int, int] = (700, 420),
) -> Image.Image:
    preview = img.convert("RGB").copy()

    if bbox_xyxy is not None:
        draw = ImageDraw.Draw(preview)
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)

    preview.thumbnail(max_size)
    return preview


def save_bbox_metadata(
    json_path: Path,
    image_filename: str,
    image_size: tuple[int, int],
    bbox_xyxy: list[float] | list[int],
    class_id: int = 0,
    class_name: str = "objeto",
) -> None:
    w, h = image_size
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]

    bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
    yolo = [
        ((x1 + x2) / 2.0) / w,
        ((y1 + y2) / 2.0) / h,
        (x2 - x1) / w,
        (y2 - y1) / h,
    ]

    payload = {
        "image_filename": image_filename,
        "image_width": w,
        "image_height": h,
        "class_id": class_id,
        "class_name": class_name,
        "bbox_format": "XYXY",
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_xywh": bbox_xywh,
        "bbox_yolo_cxcywh_normalized": yolo,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_bbox_metadata(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de bbox: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))
