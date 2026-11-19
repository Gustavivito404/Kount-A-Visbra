from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import BoundingBoxFormat
from tqdm import tqdm

from bbox_utils import detect_bbox_white_background, save_bbox_metadata


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MANIFEST_NAME = ".aug_manifest.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def find_images(input_dir: Path) -> list[Path]:
    return sorted([p for p in input_dir.rglob("*") if is_image_file(p)])


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def relative_class_folder(input_dir: Path, image_path: Path) -> Path:
    return image_path.parent.relative_to(input_dir)


def save_manifest(
    manifest_path: Path,
    generated_files: list[str],
    bbox_json_files: list[str],
    copied_original_files: list[str],
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "generated_files": sorted(set(generated_files)),
        "bbox_json_files": sorted(set(bbox_json_files)),
        "copied_original_files": sorted(set(copied_original_files)),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"No se encontró el manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_detection_transform(enable_hflip: bool = True) -> T.Compose:
    ops: list = []

    if enable_hflip:
        ops.append(T.RandomHorizontalFlip(p=0.5))

    ops.extend([
        T.RandomRotation(degrees=15),
        T.RandomAffine(
            degrees=0,
            translate=(0.06, 0.06),
            scale=(0.95, 1.08),
            shear=(-6, 6)
        ),
        T.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02
        ),
        T.ClampBoundingBoxes(),
        T.SanitizeBoundingBoxes(min_size=2),
    ])
    return T.Compose(ops)


def generate_augmented_dataset_with_bboxes(
    input_dir: Path,
    output_dir: Path,
    copies_per_image: int,
    seed: int = 42,
    white_threshold: int = 245,
    min_foreground_pixels: int = 50,
    enable_hflip: bool = True,
) -> dict:
    if not input_dir.exists():
        raise FileNotFoundError(f"La carpeta de entrada no existe: {input_dir}")

    images = find_images(input_dir)
    if not images:
        raise ValueError(f"No se encontraron imágenes válidas en: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    original_images_dir = output_dir / "original_images"
    original_labels_dir = output_dir / "original_labels"
    augmented_images_dir = output_dir / "augmented_images"
    augmented_labels_dir = output_dir / "augmented_labels"

    for d in [original_images_dir, original_labels_dir, augmented_images_dir, augmented_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    transform = build_detection_transform(enable_hflip=enable_hflip)
    set_seed(seed)

    generated_files: list[str] = []
    bbox_json_files: list[str] = []
    copied_original_files: list[str] = []
    skipped: list[str] = []

    for image_path in tqdm(images, desc="Procesando dataset"):
        rel_folder = relative_class_folder(input_dir, image_path)
        (original_images_dir / rel_folder).mkdir(parents=True, exist_ok=True)
        (original_labels_dir / rel_folder).mkdir(parents=True, exist_ok=True)
        (augmented_images_dir / rel_folder).mkdir(parents=True, exist_ok=True)
        (augmented_labels_dir / rel_folder).mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as img:
            img = ensure_rgb(img)
            w, h = img.size

            bbox = detect_bbox_white_background(
                img,
                white_threshold=white_threshold,
                min_foreground_pixels=min_foreground_pixels
            )
            if bbox is None:
                skipped.append(str(image_path))
                continue

            stem = image_path.stem
            suffix = image_path.suffix.lower()

            original_copy_path = original_images_dir / rel_folder / f"{stem}{suffix}"
            img.save(original_copy_path, quality=95)
            copied_original_files.append(str(original_copy_path.resolve()))

            orig_bbox_json = original_labels_dir / rel_folder / f"{stem}.json"
            save_bbox_metadata(
                json_path=orig_bbox_json,
                image_filename=original_copy_path.name,
                image_size=(w, h),
                bbox_xyxy=bbox,
                class_id=0,
                class_name="objeto"
            )
            bbox_json_files.append(str(orig_bbox_json.resolve()))

            boxes = tv_tensors.BoundingBoxes(
                [bbox],
                format=BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            )
            labels = torch.tensor([0], dtype=torch.int64)

            for i in range(1, copies_per_image + 1):
                target = {
                    "boxes": boxes,
                    "labels": labels,
                }

                aug_img, aug_target = transform(img, target)
                aug_boxes = aug_target["boxes"]
                aug_labels = aug_target["labels"]

                if len(aug_boxes) == 0 or len(aug_labels) == 0:
                    continue

                aug_bbox = [float(x) for x in aug_boxes[0].tolist()]

                out_name = f"{stem}_aug_{i:03d}{suffix}"
                out_img_path = augmented_images_dir / rel_folder / out_name
                aug_img.save(out_img_path, quality=95)
                generated_files.append(str(out_img_path.resolve()))

                out_bbox_json = augmented_labels_dir / rel_folder / f"{stem}_aug_{i:03d}.json"
                save_bbox_metadata(
                    json_path=out_bbox_json,
                    image_filename=out_img_path.name,
                    image_size=aug_img.size,
                    bbox_xyxy=aug_bbox,
                    class_id=int(aug_labels[0].item()),
                    class_name="objeto"
                )
                bbox_json_files.append(str(out_bbox_json.resolve()))

    manifest_path = output_dir / MANIFEST_NAME
    save_manifest(
        manifest_path=manifest_path,
        generated_files=generated_files,
        bbox_json_files=bbox_json_files,
        copied_original_files=copied_original_files,
    )

    return {
        "original_images_found": len(images),
        "copied_original_images": len(set(copied_original_files)),
        "augmented_images_created": len(generated_files),
        "label_files_created": len(set(bbox_json_files)),
        "skipped_images": skipped,
        "manifest_path": str(manifest_path),
    }


def clean_generated_dataset(output_dir: Path) -> dict:
    manifest_path = output_dir / MANIFEST_NAME
    data = load_manifest(manifest_path)

    generated_files: Iterable[str] = data.get("generated_files", [])
    bbox_json_files: Iterable[str] = data.get("bbox_json_files", [])
    copied_original_files: Iterable[str] = data.get("copied_original_files", [])

    all_files = sorted(set(
        list(generated_files) +
        list(bbox_json_files) +
        list(copied_original_files)
    ))

    deleted_images = 0
    deleted_labels = 0
    deleted_copied_originals = 0

    for file_str in all_files:
        file_path = Path(file_str)
        if not file_path.exists():
            continue

        suffix = file_path.suffix.lower()
        file_path.unlink()

        if suffix == ".json":
            deleted_labels += 1
        elif "_aug_" in file_path.stem:
            deleted_images += 1
        else:
            deleted_copied_originals += 1

    for path in sorted(output_dir.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass

    if manifest_path.exists():
        manifest_path.unlink()

    return {
        "deleted_augmented_images": deleted_images,
        "deleted_labels": deleted_labels,
        "deleted_copied_originals": deleted_copied_originals,
    }
