from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
import random

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT_CLASS_NAME = "__root__"
DEFAULT_MODEL_NAME = "classifier_best.pth"


@dataclass
class ImageRecord:
    path: Path
    class_name: str
    class_id: int
    source_group: str


class ClassificationDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        image_size: int = 128,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.transform = build_inference_transform(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as img:
            image = img.convert("RGB")
            tensor = self.transform(image)
        return tensor, record.class_id


class ObjectClassifierCNN(nn.Module):
    """
    CNN de clasificación para imágenes completas.
    No hace detección ni conteo individual.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_inference_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def _class_name_from_relative_path(relative_path: Path) -> str:
    if len(relative_path.parts) <= 1:
        return ROOT_CLASS_NAME
    return relative_path.parts[0]


def _source_group_from_stem(stem: str) -> str:
    """
    Une original y sus augmentaciones en el mismo grupo:
    pieza_01.jpg
    pieza_01_aug_001.jpg
    pieza_01_aug_002.jpg
    -> pieza_01
    """
    marker = "_aug_"
    if marker in stem:
        return stem.split(marker, 1)[0]
    return stem


def discover_dataset(dataset_output_dir: Path) -> tuple[list[ImageRecord], list[str]]:
    """
    Lee el mismo output producido por augment_core.py:
      output/original_images
      output/augmented_images
    """
    dataset_output_dir = Path(dataset_output_dir)
    roots = [
        dataset_output_dir / "original_images",
        dataset_output_dir / "augmented_images",
    ]

    discovered: list[tuple[Path, str, str]] = []
    class_names: set[str] = set()

    for root in roots:
        if not root.exists():
            continue

        for path in sorted(root.rglob("*")):
            if not _is_image(path):
                continue
            rel = path.relative_to(root)
            class_name = _class_name_from_relative_path(rel)
            source_group = f"{class_name}::{_source_group_from_stem(path.stem)}"
            discovered.append((path, class_name, source_group))
            class_names.add(class_name)

    classes = sorted(class_names)
    if not discovered:
        raise ValueError(
            "No se encontraron imágenes en output/original_images ni "
            "output/augmented_images. Primero genera el dataset aumentado."
        )

    if len(classes) < 2:
        raise ValueError(
            "Se necesitan al menos 2 clases distintas para entrenar el clasificador. "
            "Organiza el input por carpetas de clase, por ejemplo: "
            "input/tornillos y input/tuercas."
        )

    class_to_id = {name: idx for idx, name in enumerate(classes)}
    records = [
        ImageRecord(
            path=path,
            class_name=class_name,
            class_id=class_to_id[class_name],
            source_group=source_group,
        )
        for path, class_name, source_group in discovered
    ]

    return records, classes


def grouped_train_val_split(
    records: list[ImageRecord],
    class_names: list[str],
    val_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[int], list[int], list[str]]:
    """
    Intenta impedir que una imagen original y sus augmentaciones terminen
    repartidas entre entrenamiento y validación.

    Si una clase solo tiene un grupo/original, esa clase queda completa en
    entrenamiento y se reporta una advertencia.
    """
    rng = random.Random(seed)
    warnings: list[str] = []
    train_indices: list[int] = []
    val_indices: list[int] = []

    for class_id, class_name in enumerate(class_names):
        group_to_indices: dict[str, list[int]] = {}

        for idx, record in enumerate(records):
            if record.class_id != class_id:
                continue
            group_to_indices.setdefault(record.source_group, []).append(idx)

        groups = list(group_to_indices.keys())
        rng.shuffle(groups)

        if len(groups) < 2:
            warnings.append(
                f"La clase '{class_name}' solo tiene un original/grupo fuente; "
                "se usará completa para entrenamiento y no aportará validación."
            )
            for group in groups:
                train_indices.extend(group_to_indices[group])
            continue

        n_val_groups = max(1, int(round(len(groups) * val_fraction)))
        n_val_groups = min(n_val_groups, len(groups) - 1)

        val_groups = set(groups[:n_val_groups])

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
            "Se hará un split simple por imagen; la exactitud puede verse optimista "
            "si hay augmentaciones muy parecidas."
        )
        all_indices = list(range(len(records)))
        rng.shuffle(all_indices)
        n_val = max(1, int(round(len(all_indices) * val_fraction)))
        n_val = min(n_val, len(all_indices) - 1)
        val_indices = all_indices[:n_val]
        train_indices = all_indices[n_val:]

    return train_indices, val_indices, warnings


def _load_torch_checkpoint(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def train_classifier(
    dataset_output_dir: Path,
    model_path: Path,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    image_size: int = 128,
    val_fraction: float = 0.20,
    seed: int = 42,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    if epochs < 1:
        raise ValueError("epochs debe ser >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size debe ser >= 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser > 0.")
    if image_size < 32:
        raise ValueError("image_size debe ser >= 32.")
    if not (0.05 <= val_fraction <= 0.5):
        raise ValueError("val_fraction debe estar entre 0.05 y 0.50.")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    records, class_names = discover_dataset(Path(dataset_output_dir))
    train_indices, val_indices, warnings = grouped_train_val_split(
        records,
        class_names,
        val_fraction=val_fraction,
        seed=seed,
    )

    dataset = ClassificationDataset(records, image_size=image_size)
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
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = ObjectClassifierCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_accuracy = -1.0
    best_epoch = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        train_accuracy = 100.0 * train_correct / max(train_total, 1)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * labels.size(0)
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_accuracy = 100.0 * val_correct / max(val_total, 1)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(row)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch

            checkpoint = {
                "format_version": 1,
                "model_type": "ObjectClassifierCNN",
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": image_size,
                "normalization": {
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5],
                },
                "best_val_accuracy": best_val_accuracy,
                "best_epoch": best_epoch,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            torch.save(checkpoint, model_path)

        if progress_callback is not None:
            progress_callback({
                "epoch": epoch,
                "epochs": epochs,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "best_val_accuracy": best_val_accuracy,
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
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "warnings": warnings,
        "history": history,
    }


def load_model_for_inference(
    model_path: Path,
    device: torch.device | None = None,
):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = _load_torch_checkpoint(model_path, map_location=device)
    class_names = list(checkpoint["class_names"])
    image_size = int(checkpoint.get("image_size", 128))

    model = ObjectClassifierCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, checkpoint, device, image_size, class_names


def predict_image(
    image_path: Path,
    model_path: Path,
) -> dict:
    model, checkpoint, device, image_size, class_names = load_model_for_inference(model_path)
    transform = build_inference_transform(image_size)

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    with Image.open(image_path) as img:
        image = img.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()

    values = [
        {
            "class_id": idx,
            "class_name": class_name,
            "probability": float(probabilities[idx].item()),
            "percentage": float(probabilities[idx].item() * 100.0),
        }
        for idx, class_name in enumerate(class_names)
    ]
    values.sort(key=lambda item: item["probability"], reverse=True)

    top = values[0]

    return {
        "image_path": str(image_path),
        "model_path": str(model_path),
        "device": str(device),
        "predicted_class": top["class_name"],
        "confidence": top["probability"],
        "confidence_percentage": top["percentage"],
        "probabilities": values,
        "best_val_accuracy": checkpoint.get("best_val_accuracy"),
        "best_epoch": checkpoint.get("best_epoch"),
        "image_size": image_size,
    }


def load_checkpoint_metadata(model_path: Path) -> dict:
    model_path = Path(model_path)
    checkpoint = _load_torch_checkpoint(model_path, map_location="cpu")
    return {
        "class_names": list(checkpoint.get("class_names", [])),
        "image_size": checkpoint.get("image_size"),
        "best_val_accuracy": checkpoint.get("best_val_accuracy"),
        "best_epoch": checkpoint.get("best_epoch"),
        "created_at": checkpoint.get("created_at"),
        "model_type": checkpoint.get("model_type"),
    }
