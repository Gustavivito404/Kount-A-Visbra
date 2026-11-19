from __future__ import annotations
import json
import random
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2 as T
from torchvision import tv_tensors

from .project_core import CLASS_NAMES, find_images, read_annotations


def build_transform(enable_flip=True, brightness=0.30, contrast=0.20,
                    saturation=0.08, hue=0.01):
    ops = []
    if enable_flip:
        ops.append(T.RandomHorizontalFlip(0.5))
    ops += [
        T.RandomRotation(15),
        T.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.08), shear=(-6, 6)),
        T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        T.ClampBoundingBoxes(),
        T.SanitizeBoundingBoxes(min_size=2),
    ]
    return T.Compose(ops)


def noise_effects(image: Image.Image, rng: random.Random, np_rng: np.random.Generator,
                  salt_pepper_prob: float, salt_pepper_amount: float,
                  gaussian_prob: float, gaussian_sigma: float) -> Image.Image:
    apply_sp = salt_pepper_amount > 0 and rng.random() < salt_pepper_prob
    apply_gauss = gaussian_sigma > 0 and rng.random() < gaussian_prob
    if not apply_sp and not apply_gauss:
        return image
    arr = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
    # El ruido es fotométrico: nunca modifica las BBox.
    if apply_sp:
        pixels = arr.reshape(-1, 3)
        n = min(pixels.shape[0], int(pixels.shape[0] * salt_pepper_amount))
        if n:
            inds = np_rng.choice(pixels.shape[0], n, replace=False)
            pixels[inds[:n//2]] = 0
            pixels[inds[n//2:]] = 255
    if apply_gauss:
        arr = np.clip(arr.astype(np.float32) + np_rng.normal(0, gaussian_sigma, arr.shape), 0, 255).astype('uint8')
    return Image.fromarray(arr)


def generate_augmentations(input_dir: Path, output_dir: Path, copies: int = 10,
                           seed: int = 42, enable_flip: bool = True,
                           brightness: float = .30, contrast: float = .20,
                           saturation: float = .08, hue: float = .01,
                           salt_pepper_prob: float = .30, salt_pepper_amount: float = .003,
                           gaussian_prob: float = .15, gaussian_sigma: float = 4.0,
                           progress: Callable[[dict], None] | None = None) -> dict:
    if copies < 1 or copies > 200:
        raise ValueError('Copias por imagen debe estar entre 1 y 200.')
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if (input_dir == output_dir or input_dir.is_relative_to(output_dir)
            or output_dir.is_relative_to(input_dir)):
        raise ValueError('Output no puede ser la carpeta input ni su carpeta contenedora.')
    if not all(0 <= x <= 1 for x in (brightness, contrast, saturation, salt_pepper_prob, gaussian_prob)):
        raise ValueError('Los ajustes de imagen deben estar entre 0 y 1.')
    if not (0 <= hue <= .5 and 0 <= salt_pepper_amount <= .1 and 0 <= gaussian_sigma <= 100):
        raise ValueError('Valores de ruido/hue fuera de rango.')

    originals = find_images(input_dir)
    records = []
    for path in originals:
        objects = read_annotations(input_dir, path, output_dir)
        if objects:
            records.append((path, objects))
    if not records:
        raise ValueError('No hay fotos con BBox guardadas. Etiqueta primero en la pestaña BBox.')

    augmented_images = output_dir / 'augmented_images'
    augmented_labels = output_dir / 'augmented_labels'
    # Solo se reemplazan las salidas de augmentation, NUNCA los originales ni sus anotaciones.
    for p in (augmented_images, augmented_labels):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)

    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    transform = build_transform(enable_flip, brightness, contrast, saturation, hue)
    created = 0
    skipped = 0
    for k, (path, objects) in enumerate(records, 1):
        relative = path.relative_to(input_dir)
        image_parent = augmented_images / relative.parent
        label_parent = augmented_labels / relative.parent
        image_parent.mkdir(parents=True, exist_ok=True)
        label_parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as img:
            img = img.convert('RGB')
            width, height = img.size
            boxes = tv_tensors.BoundingBoxes(
                [obj['bbox_xyxy'] for obj in objects], format='XYXY', canvas_size=(height, width))
            labels = torch.tensor([CLASS_NAMES.index(obj['class_name']) for obj in objects], dtype=torch.int64)
            for i in range(1, copies+1):
                aug_img, target = transform(img, {'boxes': boxes, 'labels': labels})
                if len(target['boxes']) == 0:
                    skipped += 1
                    continue
                # Se altera la imagen después de calcular las BBox transformadas.
                aug_img = noise_effects(aug_img, rng, np_rng, salt_pepper_prob,
                                        salt_pepper_amount, gaussian_prob, gaussian_sigma)
                stem = f'{relative.stem}_aug_{i:03d}'
                img_file = image_parent / f'{stem}.jpg'
                label_file = label_parent / f'{stem}.json'
                aug_img.save(img_file, quality=95)
                aug_objects = [
                    {'class_name': CLASS_NAMES[int(cls)], 'bbox_xyxy': [round(float(v), 2) for v in box.tolist()]}
                    for box, cls in zip(target['boxes'], target['labels'])
                ]
                label_file.write_text(json.dumps({
                    'format_version': 2, 'source_image': relative.as_posix(),
                    'image_size': list(aug_img.size), 'objects': aug_objects,
                }, ensure_ascii=False, indent=2), encoding='utf-8')
                created += 1
        if progress:
            progress({'done': k, 'total': len(records), 'created': created})
    summary = {'labeled_originals': len(records), 'unlabeled_originals': len(originals) - len(records),
               'augmented_images': created, 'skipped': skipped, 'seed': seed,
               'settings': {'copies': copies, 'flip': enable_flip, 'brightness': brightness,
                            'contrast': contrast, 'saturation': saturation, 'hue': hue,
                            'salt_pepper_prob': salt_pepper_prob,
                            'salt_pepper_amount': salt_pepper_amount,
                            'gaussian_prob': gaussian_prob, 'gaussian_sigma': gaussian_sigma}}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'augmentation_report.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary
