from __future__ import annotations
import json
import re
import unicodedata
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CLASS_NAMES = [
    'tornillo', 'tuerca', 'rondana',
    'tornillo_oxidado', 'tuerca_oxidada', 'rondana_oxidada',
]
ANNOTATION_FOLDER = '.kounta_annotations'


def normalized_class(value: str) -> str | None:
    value = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower()
    value = re.sub(r'[^a-z0-9]+', '_', value).strip('_')
    aliases = {
        'tornillos': 'tornillo', 'tuercas': 'tuerca', 'rondanas': 'rondana',
        'arandela': 'rondana', 'arandelas': 'rondana',
        'tornillos_oxidados': 'tornillo_oxidado', 'tuercas_oxidadas': 'tuerca_oxidada',
        'rondanas_oxidadas': 'rondana_oxidada', 'arandelas_oxidadas': 'rondana_oxidada',
        'tornillo_oxidada': 'tornillo_oxidado', 'tuerca_oxidado': 'tuerca_oxidada',
        'rondana_oxidado': 'rondana_oxidada',
    }
    value = aliases.get(value, value)
    return value if value in CLASS_NAMES else None


def infer_folder_class(image_path: Path, input_dir: Path) -> str:
    rel = image_path.relative_to(input_dir)
    if len(rel.parts) >= 2:
        found = normalized_class(rel.parts[0])
        if found:
            return found
    return CLASS_NAMES[0]


def find_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(
        p for p in input_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        and ANNOTATION_FOLDER not in p.parts
    )


def annotation_path(input_dir: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(input_dir)
    return input_dir / ANNOTATION_FOLDER / rel.parent / f'{rel.name}.json'


def _validate_objects(objects: list[dict], image_size: tuple[int, int]) -> list[dict]:
    w, h = image_size
    cleaned = []
    for item in objects:
        cls = normalized_class(item.get('class_name', ''))
        bbox = item.get('bbox_xyxy')
        if cls is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            x1, x2 = sorted((max(0.0, min(x1, w)), max(0.0, min(x2, w))))
            y1, y2 = sorted((max(0.0, min(y1, h)), max(0.0, min(y2, h))))
        except (TypeError, ValueError):
            continue
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cleaned.append({'class_name': cls, 'bbox_xyxy': [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]})
    return cleaned


def read_annotations(input_dir: Path, image_path: Path, output_dir: Path | None = None) -> list[dict]:
    path = annotation_path(input_dir, image_path)
    # Compatibilidad con JSON originales de versiones anteriores, de una BBox.
    legacy = None
    if not path.is_file() and output_dir is not None:
        rel = image_path.relative_to(input_dir)
        legacy = output_dir / 'original_labels' / rel.with_suffix('.json')
    actual = path if path.is_file() else legacy
    if actual is None or not actual.is_file():
        return []
    try:
        data = json.loads(actual.read_text(encoding='utf-8'))
        raw = data.get('objects')
        if not isinstance(raw, list):
            raw = [{'class_name': data.get('class_name', infer_folder_class(image_path, input_dir)),
                    'bbox_xyxy': data.get('bbox_xyxy')}]
        with Image.open(image_path) as img:
            size = img.size
        # Archivos antiguos podrían utilizar alias o nombres de carpetas en plural.
        return _validate_objects(raw, size)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return []


def has_saved_annotation(input_dir: Path, image_path: Path) -> bool:
    return annotation_path(input_dir, image_path).is_file()


def save_annotations(input_dir: Path, image_path: Path, objects: list[dict]) -> Path:
    path = annotation_path(input_dir, image_path)
    with Image.open(image_path) as im:
        size = im.size
    valid = _validate_objects(objects, size)
    data = {'format_version': 2, 'image': str(image_path.relative_to(input_dir).as_posix()),
            'image_size': list(size), 'objects': valid}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)
    return path


def detect_grayscale_boxes(image: Image.Image, threshold: int = 225,
                           min_area: int = 150, padding: int = 2) -> list[list[float]]:
    """Propuesta automática para objetos oscuros sobre fondo claro; requiere revisión visual."""
    import cv2
    import numpy as np
    arr = np.asarray(image.convert('L'))
    _, mask = cv2.threshold(arr, int(threshold), 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = arr.shape
    results = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        results.append([float(max(0, x-padding)), float(max(0, y-padding)),
                        float(min(w, x+bw+padding)), float(min(h, y+bh+padding))])
    return sorted(results, key=lambda b: (b[1], b[0]))


def draw_overlay(image: Image.Image, objects: list[dict], selected: int = -1) -> Image.Image:
    from PIL import ImageDraw
    result = image.convert('RGB').copy()
    draw = ImageDraw.Draw(result)
    colors = ['#DB3B39', '#2776C7', '#30A567', '#B445AB', '#E08A24', '#6E63C6']
    for i, obj in enumerate(objects):
        cls = normalized_class(obj.get('class_name'))
        if cls is None:
            continue
        box = obj['bbox_xyxy']
        color = '#F2CF41' if i == selected else colors[CLASS_NAMES.index(cls)]
        draw.rectangle(box, outline=color, width=4 if i == selected else 3)
        x, y = int(box[0]), max(0, int(box[1]) - 17)
        label = f'{i + 1}. {cls}'
        draw.rectangle((x, y, x + len(label)*7 + 6, y + 16), fill=color)
        draw.text((x + 3, y), label, fill='black' if i == selected else 'white')
    return result


def load_label_objects(label_file: Path) -> list[dict]:
    try:
        raw = json.loads(label_file.read_text(encoding='utf-8'))
        return list(raw.get('objects', []))
    except (OSError, ValueError, KeyError):
        return []


# Limpieza SEGURA y limitada a artefactos regenerables del output.
# No toca input, anotaciones originales, modelos .pt ni reportes de entrenamiento.
def _clear_generated_items(output_dir: Path, directories: tuple[str, ...],
                           files: tuple[str, ...]) -> dict:
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f'No existe la carpeta output: {output_dir}')

    # Validar todas las rutas antes de borrar algo. No seguir enlaces simbólicos.
    targets = [(output_dir / name, True) for name in directories]
    targets += [(output_dir / name, False) for name in files]
    for target, should_be_dir in targets:
        if target.is_symlink():
            raise ValueError(f'Por seguridad no se borra un enlace simbólico: {target}')
        if target.exists() and target.is_dir() != should_be_dir:
            raise ValueError(f'El elemento no tiene el tipo esperado: {target}')

    removed = []
    image_count = 0
    for target, should_be_dir in targets:
        if not target.exists():
            continue
        if should_be_dir:
            image_count += sum(1 for f in target.rglob('*') if f.is_file())
            shutil.rmtree(target)
        else:
            target.unlink()
        removed.append(target.name)
    return {'removed': removed, 'files_in_directories': image_count,
            'output': str(output_dir)}


def clear_augmentation_output(output_dir: Path) -> dict:
    """Borra solo copias aumentadas, JSON aumentados y su reporte."""
    return _clear_generated_items(
        output_dir,
        directories=('augmented_images', 'augmented_labels'),
        files=('augmentation_report.json',),
    )


def clear_yolo_export_output(output_dir: Path) -> dict:
    """Borra el dataset YOLO exportado y el split persistido.

    ADVERTENCIA: borrar split_assignments.json permite generar otra división
    train/val/test en la próxima exportación.
    """
    return _clear_generated_items(
        output_dir,
        directories=('yolo_dataset',),
        files=('split_assignments.json',),
    )
