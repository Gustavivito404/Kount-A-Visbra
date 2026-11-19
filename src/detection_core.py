from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
import json
import os
import random

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import batched_nms


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT_CLASS_NAME = "__root__"
DEFAULT_DETECTOR_MODEL_NAME = "detector_best.pth"


@dataclass
class DetectionRecord:
    image_path: Path
    label_path: Path
    class_name: str
    source_group: str


class DetectionDataset(Dataset):
    """
    Dataset para detección.

    En la v11 los JSON actuales de AugmentData contienen normalmente
    una bbox por imagen. El lector también queda preparado para un futuro
    formato con una lista 'objects'.
    """

    def __init__(
        self,
        records: list[DetectionRecord],
        class_names: list[str],
    ) -> None:
        self.records = records
        self.class_names = class_names
        self.class_to_detector_id = {
            name: idx + 1 for idx, name in enumerate(class_names)
        }  # 0 queda reservado para background
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]

        with Image.open(record.image_path) as img:
            image = img.convert("RGB")
            image_tensor = self.to_tensor(image)

        data = json.loads(record.label_path.read_text(encoding="utf-8"))

        boxes: list[list[float]] = []
        labels: list[int] = []

        if isinstance(data.get("objects"), list):
            for obj in data["objects"]:
                bbox = obj.get("bbox_xyxy")
                class_name = str(obj.get("class_name", record.class_name))
                if not bbox or class_name not in self.class_to_detector_id:
                    continue
                boxes.append([float(v) for v in bbox])
                labels.append(self.class_to_detector_id[class_name])
        else:
            bbox = data.get("bbox_xyxy")
            class_name = str(data.get("class_name", record.class_name))
            if bbox and class_name in self.class_to_detector_id:
                boxes.append([float(v) for v in bbox])
                labels.append(self.class_to_detector_id[class_name])

        if not boxes:
            raise ValueError(
                f"El label no contiene bounding boxes válidas: {record.label_path}"
            )

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
        }

        return image_tensor, target


def detection_collate_fn(batch):
    return tuple(zip(*batch))


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def _base_source_stem(stem: str) -> str:
    if "_aug_" in stem:
        return stem.split("_aug_", 1)[0]
    return stem


def _read_label_class_name(label_path: Path) -> str:
    data = json.loads(label_path.read_text(encoding="utf-8"))

    if isinstance(data.get("objects"), list) and data["objects"]:
        return str(data["objects"][0].get("class_name", ROOT_CLASS_NAME))

    return str(data.get("class_name", ROOT_CLASS_NAME))


def discover_detection_dataset(
    dataset_output_dir: Path,
) -> tuple[list[DetectionRecord], list[str]]:
    """
    Usa exactamente la salida actual de AugmentData:

      output/original_images
      output/original_labels
      output/augmented_images
      output/augmented_labels
    """
    dataset_output_dir = Path(dataset_output_dir)

    pairs = [
        (
            dataset_output_dir / "original_images",
            dataset_output_dir / "original_labels",
        ),
        (
            dataset_output_dir / "augmented_images",
            dataset_output_dir / "augmented_labels",
        ),
    ]

    discovered: list[tuple[Path, Path, str, str]] = []
    class_names: set[str] = set()

    for images_root, labels_root in pairs:
        if not images_root.exists() or not labels_root.exists():
            continue

        for image_path in sorted(images_root.rglob("*")):
            if not _is_image(image_path):
                continue

            rel = image_path.relative_to(images_root)
            label_path = (labels_root / rel).with_suffix(".json")

            if not label_path.exists():
                continue

            class_name = _read_label_class_name(label_path)
            source_group = (
                f"{class_name}::{_base_source_stem(image_path.stem)}"
            )

            discovered.append(
                (image_path, label_path, class_name, source_group)
            )
            class_names.add(class_name)

    if not discovered:
        raise ValueError(
            "No se encontraron pares imagen/label en el output. "
            "Primero genera el dataset aumentado en la pestaña 1."
        )

    classes = sorted(class_names)

    if len(classes) < 1:
        raise ValueError("No se encontró ninguna clase válida.")

    records = [
        DetectionRecord(
            image_path=image_path,
            label_path=label_path,
            class_name=class_name,
            source_group=source_group,
        )
        for image_path, label_path, class_name, source_group in discovered
    ]

    return records, classes


def grouped_detection_split(
    records: list[DetectionRecord],
    class_names: list[str],
    val_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[int], list[int], list[str]]:
    """
    Mantiene una imagen original y sus augmentaciones en el mismo conjunto.
    """
    rng = random.Random(seed)
    warnings: list[str] = []
    train_indices: list[int] = []
    val_indices: list[int] = []

    for class_name in class_names:
        group_to_indices: dict[str, list[int]] = {}

        for idx, record in enumerate(records):
            if record.class_name != class_name:
                continue
            group_to_indices.setdefault(record.source_group, []).append(idx)

        groups = list(group_to_indices.keys())
        rng.shuffle(groups)

        if len(groups) < 2:
            warnings.append(
                f"La clase '{class_name}' solo tiene un grupo/original. "
                "Se dejará completa en entrenamiento."
            )
            for group in groups:
                train_indices.extend(group_to_indices[group])
            continue

        n_val = max(1, int(round(len(groups) * val_fraction)))
        n_val = min(n_val, len(groups) - 1)
        val_groups = set(groups[:n_val])

        for group, indices in group_to_indices.items():
            if group in val_groups:
                val_indices.extend(indices)
            else:
                train_indices.extend(indices)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    if not val_indices:
        warnings.append(
            "No fue posible crear validación agrupada. "
            "Se realizará un split simple por imagen."
        )
        indices = list(range(len(records)))
        rng.shuffle(indices)
        if len(indices) < 2:
            raise ValueError(
                "Se necesitan al menos 2 imágenes para separar entrenamiento "
                "y validación."
            )
        n_val = max(1, int(round(len(indices) * val_fraction)))
        n_val = min(n_val, len(indices) - 1)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

    return train_indices, val_indices, warnings


def build_detector(
    num_object_classes: int,
    use_pretrained: bool = True,
):
    """
    Faster R-CNN + MobileNetV3 320 FPN para prototipos relativamente ligeros.

    Si use_pretrained=True, torchvision puede descargar los pesos COCO la
    primera vez que se ejecute en una computadora.
    """
    architecture = "fasterrcnn_mobilenet_v3_large_320_fpn"

    try:
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        if use_pretrained:
            weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        else:
            model = fasterrcnn_mobilenet_v3_large_320_fpn(
                weights=None,
                weights_backbone=None,
            )
    except (ImportError, AttributeError):
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn,
        )

        architecture = "fasterrcnn_mobilenet_v3_large_fpn"

        if use_pretrained:
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        else:
            model = fasterrcnn_mobilenet_v3_large_fpn(
                weights=None,
                weights_backbone=None,
            )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_object_classes + 1,  # + background
    )

    return model, architecture


def _box_iou_xyxy(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
        0.0, float(box_a[3] - box_a[1])
    )
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
        0.0, float(box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def evaluate_detector_simple(
    model,
    loader: DataLoader,
    device: torch.device,
    confidence_threshold: float = 0.40,
    iou_threshold: float = 0.50,
) -> dict:
    """
    Métrica simple para la v11.

    Considera una GT como encontrada si existe una predicción:
    - de la misma clase
    - score >= threshold
    - IoU >= threshold
    """
    model.eval()

    gt_total = 0
    gt_detected = 0
    prediction_total = 0
    matched_predictions = 0

    with torch.no_grad():
        for images, targets in loader:
            images_device = [img.to(device) for img in images]
            outputs = model(images_device)

            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"].detach().cpu()
                pred_labels = output["labels"].detach().cpu()
                pred_scores = output["scores"].detach().cpu()

                keep = pred_scores >= confidence_threshold
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]

                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()

                prediction_total += len(pred_boxes)

                used_predictions: set[int] = set()

                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    gt_total += 1
                    best_index = None
                    best_iou = 0.0

                    for pred_index, (pred_box, pred_label) in enumerate(
                        zip(pred_boxes, pred_labels)
                    ):
                        if pred_index in used_predictions:
                            continue
                        if int(pred_label.item()) != int(gt_label.item()):
                            continue

                        iou = _box_iou_xyxy(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_index = pred_index

                    if best_index is not None and best_iou >= iou_threshold:
                        gt_detected += 1
                        matched_predictions += 1
                        used_predictions.add(best_index)

    recall = 100.0 * gt_detected / max(gt_total, 1)
    precision = 100.0 * matched_predictions / max(prediction_total, 1)

    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "gt_total": gt_total,
        "prediction_total": prediction_total,
    }


def train_detector(
    dataset_output_dir: Path,
    model_path: Path,
    epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 0.0005,
    val_fraction: float = 0.20,
    seed: int = 42,
    use_pretrained: bool = True,
    validation_confidence: float = 0.40,
    validation_iou: float = 0.50,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    if epochs < 1:
        raise ValueError("epochs debe ser >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size debe ser >= 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser > 0.")
    if not (0.05 <= val_fraction <= 0.5):
        raise ValueError("val_fraction debe estar entre 0.05 y 0.50.")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    records, class_names = discover_detection_dataset(Path(dataset_output_dir))
    train_indices, val_indices, warnings = grouped_detection_split(
        records,
        class_names,
        val_fraction=val_fraction,
        seed=seed,
    )

    dataset = DetectionDataset(records, class_names)
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=detection_collate_fn,
    )

    model, architecture = build_detector(
        num_object_classes=len(class_names),
        use_pretrained=use_pretrained,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.0001,
    )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_epoch = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        batches = 0

        for images, targets in train_loader:
            images_device = [img.to(device) for img in images]
            targets_device = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]

            loss_dict = model(images_device, targets_device)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            optimizer.step()

            running_loss += float(losses.item())
            batches += 1

        train_loss = running_loss / max(batches, 1)

        metrics = evaluate_detector_simple(
            model,
            val_loader,
            device,
            confidence_threshold=validation_confidence,
            iou_threshold=validation_iou,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        }
        history.append(row)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_epoch = epoch

            checkpoint = {
                "format_version": 1,
                "model_type": "FasterRCNN",
                "architecture": architecture,
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "validation_confidence": validation_confidence,
                "validation_iou": validation_iou,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            torch.save(checkpoint, model_path)

        if progress_callback is not None:
            progress_callback({
                "epoch": epoch,
                "epochs": epochs,
                "train_loss": train_loss,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "device": str(device),
            })

    return {
        "model_path": str(model_path),
        "class_names": class_names,
        "num_classes": len(class_names),
        "total_images": len(records),
        "train_images": len(train_indices),
        "val_images": len(val_indices),
        "device": str(device),
        "architecture": architecture,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "warnings": warnings,
        "history": history,
    }


def _load_checkpoint(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_detector_for_inference(
    model_path: Path,
    device: torch.device | None = None,
):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el detector: {model_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = _load_checkpoint(model_path, map_location=device)
    class_names = list(checkpoint["class_names"])

    # Para cargar un checkpoint propio no necesitamos volver a descargar COCO.
    model, architecture = build_detector(
        num_object_classes=len(class_names),
        use_pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint, class_names, device, architecture


def _predict_tensor(
    image_tensor: torch.Tensor,
    model,
    class_names: list[str],
    device: torch.device,
    confidence_threshold: float,
    nms_iou_threshold: float = 0.8,
) -> list[dict]:
    """
    Ejecuta inferencia, filtra por confianza y aplica un NMS adicional
    por clase para reducir dobles conteos.
    """
    if not (0.0 < confidence_threshold < 1.0):
        raise ValueError("confidence_threshold debe estar entre 0 y 1.")

    if not (0.0 < nms_iou_threshold < 1.0):
        raise ValueError("nms_iou_threshold debe estar entre 0 y 1.")

    with torch.no_grad():
        output = model([image_tensor.to(device)])[0]

    boxes = output["boxes"].detach().cpu()
    labels = output["labels"].detach().cpu()
    scores = output["scores"].detach().cpu()

    keep_confidence = scores >= confidence_threshold
    boxes = boxes[keep_confidence]
    labels = labels[keep_confidence]
    scores = scores[keep_confidence]

    # NMS por clase: cajas de clases distintas no se eliminan entre si.
    if len(boxes) > 0:
        keep_nms = batched_nms(
            boxes,
            scores,
            labels,
            iou_threshold=nms_iou_threshold,
        )
        boxes = boxes[keep_nms]
        labels = labels[keep_nms]
        scores = scores[keep_nms]

    detections: list[dict] = []

    for box, label, score in zip(boxes, labels, scores):
        score_value = float(score.item())
        detector_id = int(label.item())

        if detector_id <= 0 or detector_id > len(class_names):
            continue

        class_name = class_names[detector_id - 1]

        detections.append({
            "class_id": detector_id - 1,
            "class_name": class_name,
            "score": score_value,
            "percentage": score_value * 100.0,
            "bbox_xyxy": [float(v) for v in box.tolist()],
        })

    return detections

def predict_pil_image(
    image: Image.Image,
    model_path: Path,
    confidence_threshold: float = 0.50,
    nms_iou_threshold: float = 0.35,
) -> dict:
    model, checkpoint, class_names, device, architecture = (
        load_detector_for_inference(model_path)
    )

    rgb = image.convert("RGB")
    tensor = transforms.ToTensor()(rgb)

    detections = _predict_tensor(
        tensor,
        model,
        class_names,
        device,
        confidence_threshold,
        nms_iou_threshold,
    )

    counts = {name: 0 for name in class_names}

    for detection in detections:
        counts[detection["class_name"]] += 1

    counts = {name: value for name, value in counts.items() if value > 0}

    return {
        "detections": detections,
        "counts": counts,
        "total": len(detections),
        "class_names": class_names,
        "device": str(device),
        "architecture": architecture,
        "best_f1": checkpoint.get("best_f1"),
        "best_epoch": checkpoint.get("best_epoch"),
        "nms_iou_threshold": nms_iou_threshold,
    }


def predict_image_file(
    image_path: Path,
    model_path: Path,
    confidence_threshold: float = 0.50,
    nms_iou_threshold: float = 0.35,
) -> dict:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    with Image.open(image_path) as img:
        return predict_pil_image(
            img,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )


def draw_detections(
    image: Image.Image,
    detections: list[dict],
    line_width: int = 3,
) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        label = (
            f"{detection['class_name']} "
            f"{detection['percentage']:.1f}%"
        )

        draw.rectangle(
            [int(x1), int(y1), int(x2), int(y2)],
            outline="red",
            width=line_width,
        )

        text_x = int(x1)
        text_y = max(0, int(y1) - 14)

        draw.rectangle(
            [text_x, text_y, text_x + max(90, len(label) * 7), text_y + 14],
            fill="white",
        )
        draw.text(
            (text_x + 2, text_y),
            label,
            fill="black",
        )

    return result


def capture_webcam_image(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    warmup_frames: int = 8,
) -> Image.Image:
    """
    Captura un solo frame de una webcam.

    OpenCV se usa únicamente para acceso a la cámara.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Para usar webcam instala OpenCV: pip install opencv-python"
        ) from exc

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"No se pudo abrir la webcam con índice {camera_index}."
        )

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        frame = None

        for _ in range(max(1, warmup_frames)):
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate

        if frame is None:
            raise RuntimeError("La webcam no devolvió una imagen válida.")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    finally:
        cap.release()


def load_detector_metadata(model_path: Path) -> dict:
    model_path = Path(model_path)
    checkpoint = _load_checkpoint(model_path, map_location="cpu")

    return {
        "model_type": checkpoint.get("model_type"),
        "architecture": checkpoint.get("architecture"),
        "class_names": list(checkpoint.get("class_names", [])),
        "best_f1": checkpoint.get("best_f1"),
        "best_epoch": checkpoint.get("best_epoch"),
        "created_at": checkpoint.get("created_at"),
        "validation_confidence": checkpoint.get("validation_confidence"),
        "validation_iou": checkpoint.get("validation_iou"),
    }
