from __future__ import annotations
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
import yaml

from .project_core import CLASS_NAMES, find_images, read_annotations, load_label_objects, normalized_class


def _yolo_lines(objects: list[dict], size: tuple[int, int]) -> list[str]:
    w, h = size
    if w < 1 or h < 1:
        return []
    result = []
    for obj in objects:
        cls = normalized_class(obj.get('class_name', ''))
        box = obj.get('bbox_xyxy', [])
        if cls not in CLASS_NAMES or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2-x1 < 2 or y2-y1 < 2:
            continue
        result.append(f'{CLASS_NAMES.index(cls)} {(x1+x2)/(2*w):.7f} {(y1+y2)/(2*h):.7f} {(x2-x1)/w:.7f} {(y2-y1)/h:.7f}')
    return result


def _pick_splits(records: list[tuple[Path, list[dict]]], input_dir: Path,
                 output_dir: Path, seed: int, val_frac: float, test_frac: float) -> dict[str, str]:
    """Persiste asignaciones de originales; los aumentos siempre heredan el split."""
    manifest_file = output_dir / 'split_assignments.json'
    previous = {}
    if manifest_file.exists():
        old = json.loads(manifest_file.read_text(encoding='utf-8'))
        previous = old.get('assignments', {})
        if old.get('seed') != seed or old.get('val_frac') != val_frac or old.get('test_frac') != test_frac:
            # Al cambiar parámetros se solicita hacer un split nuevo explícitamente.
            raise ValueError('Ya existe un split con otra seed/fracción. Borra o mueve split_assignments.json '
                             'si quieres rehacer entrenamiento/validación/prueba desde cero.')
    assignments = {}
    per_class: dict[str, list[str]] = {}
    for image, objects in records:
        rel = image.relative_to(input_dir).as_posix()
        class_name = objects[0]['class_name']
        per_class.setdefault(class_name, []).append(rel)
        if previous.get(rel) in ('train', 'val', 'test'):
            assignments[rel] = previous[rel]
    rng = random.Random(seed)
    for cls, paths in sorted(per_class.items()):
        paths = sorted(paths)
        existing = [p for p in paths if p in assignments]
        fresh = [p for p in paths if p not in assignments]
        rng.shuffle(fresh)
        n = len(paths)
        # No sacrificar todo un grupo pequeño: val >= 1 con >= 2, test >= 1 con >= 7.
        target_val = max(1, round(n*val_frac)) if n >= 2 else 0
        target_test = max(1, round(n*test_frac)) if n >= 7 else 0
        if target_val + target_test >= n:
            target_test = max(0, n-target_val-1)
        assigned_val = sum(assignments[p]=='val' for p in existing)
        assigned_test = sum(assignments[p]=='test' for p in existing)
        for p in fresh:
            if assigned_val < target_val:
                split = 'val'
                assigned_val += 1
            elif assigned_test < target_test:
                split = 'test'
                assigned_test += 1
            else:
                split = 'train'
            assignments[p] = split
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps({'seed': seed, 'val_frac': val_frac,
                                         'test_frac': test_frac, 'assignments': assignments},
                                        ensure_ascii=False, indent=2), encoding='utf-8')
    return assignments


def export_yolo_dataset(input_dir: Path, output_dir: Path, seed: int = 42,
                        val_frac: float = 0.20, test_frac: float = 0.10,
                        use_augmented: bool = True,
                        progress: Callable[[str], None] | None = None) -> dict:
    """Exporta fotos originales + aumentos de TRAIN; val/test solo fotos reales."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if (input_dir == output_dir or input_dir.is_relative_to(output_dir)
            or output_dir.is_relative_to(input_dir)):
        raise ValueError('Output no puede contener la carpeta input.')
    if val_frac < .05 or test_frac < 0 or val_frac+test_frac >= .50:
        raise ValueError('Usa validación entre 0.05 y 0.49; val+test < 0.50.')
    records = [(p, read_annotations(input_dir, p, output_dir)) for p in find_images(input_dir)]
    records = [(p, o) for p, o in records if o]
    if not records:
        raise ValueError('No hay imágenes originales etiquetadas.')
    assignments = _pick_splits(records, input_dir, output_dir, seed, val_frac, test_frac)
    root = output_dir / 'yolo_dataset'
    if root.exists():
        shutil.rmtree(root)
    for split in ('train', 'val', 'test'):
        (root / 'images' / split).mkdir(parents=True)
        (root / 'labels' / split).mkdir(parents=True)

    counters = {'train': 0, 'val': 0, 'test': 0, 'augmented_train': 0, 'skipped_augmented': 0}

    def copy_pair(image: Path, objects: list[dict], split: str, relative: Path, suffix: str):
        # __orig y __aug evitan conflictos entre archivos con el mismo nombre de imagen.
        target_stem = f'{relative.stem}{suffix}'
        target_img = root / 'images' / split / relative.parent / (target_stem + image.suffix.lower())
        target_txt = root / 'labels' / split / relative.parent / (target_stem + '.txt')
        target_img.parent.mkdir(parents=True, exist_ok=True)
        target_txt.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image) as img:
            size = img.size
        lines = _yolo_lines(objects, size)
        if not lines:
            return False
        shutil.copy2(image, target_img)
        target_txt.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return True

    for img, objects in records:
        rel = img.relative_to(input_dir)
        split = assignments[rel.as_posix()]
        if copy_pair(img, objects, split, rel, '__orig'):
            counters[split] += 1

    if use_augmented:
        aug_img_root = output_dir / 'augmented_images'
        aug_label_root = output_dir / 'augmented_labels'
        if aug_img_root.exists():
            for img in find_images(aug_img_root):
                rel = img.relative_to(aug_img_root)
                label = (aug_label_root / rel).with_suffix('.json')
                if not label.is_file():
                    continue
                try:
                    meta = json.loads(label.read_text(encoding='utf-8'))
                    source_rel = meta.get('source_image', '')
                    if assignments.get(source_rel) != 'train':
                        counters['skipped_augmented'] += 1
                        continue
                    if copy_pair(img, list(meta.get('objects', [])), 'train', rel, '__aug'):
                        counters['train'] += 1
                        counters['augmented_train'] += 1
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
        else:
            if progress:
                progress('No existen aumentadas: entrenamiento con originales únicamente.')

    if not counters['train'] or not counters['val']:
        raise ValueError('Faltan imágenes para train/val. Agrega originales etiquetadas por clase.')

    yaml_file = root / 'dataset.yaml'
    config = {'path': str(root), 'train': 'images/train', 'val': 'images/val',
              'test': 'images/test', 'nc': len(CLASS_NAMES),
              'names': dict(enumerate(CLASS_NAMES))}
    yaml_file.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    report = {'dataset_yaml': str(yaml_file), 'labeled_originals': len(records),
              'counts': counters, 'class_names': CLASS_NAMES, 'assignments': len(assignments),
              'note': 'Las augmentaciones solo se exportan a train; val/test son fotos originales.'}
    (root / 'export_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if progress:
        progress(f'Exportación terminada: train={counters["train"]}, val={counters["val"]}, test={counters["test"]}')
    return report


def _require_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError as exc:
        raise ImportError('Falta ultralytics. Instala dependencias con: pip install -r requirements.txt') from exc


def train_yolo(input_dir: Path, output_dir: Path, *, base_model: str = 'yolo11n.pt',
               epochs: int = 35, batch: int = 4, imgsz: int = 640,
               lr0: float = .001, patience: int = 8, seed: int = 42,
               val_frac: float = .20, test_frac: float = .10,
               use_augmented: bool = True, mosaic: float = .25,
               progress: Callable[[dict], None] | None = None) -> dict:
    if epochs < 1 or batch < 1 or imgsz < 128 or lr0 <= 0 or patience < 0 or not (0 <= mosaic <= 1):
        raise ValueError('Valores de entrenamiento inválidos.')
    if base_model.lower().endswith('.pth'):
        raise ValueError('Un checkpoint Faster R-CNN .pth no es compatible con YOLO. '
                         'Usa un modelo YOLO preentrenado .pt o best.pt anterior.')
    def note(message: str):
        if progress:
            progress({'stage': 'message', 'message': message})
    note('Exportando dataset YOLO con separación segura por originales...')
    export = export_yolo_dataset(input_dir, output_dir, seed=seed, val_frac=val_frac,
                                 test_frac=test_frac, use_augmented=use_augmented, progress=note)
    YOLO = _require_yolo()
    model = YOLO(base_model)
    device = '0' if torch.cuda.is_available() else 'cpu'
    note(f'Modelo={base_model}; dispositivo={device}; imágenes={export["counts"]}')
    run_name = 'kounta_' + datetime.now().strftime('%Y%m%d_%H%M%S')

    def epoch_callback(trainer):
        epoch = int(trainer.epoch) + 1
        metrics = {str(k): float(v) for k, v in getattr(trainer, 'metrics', {}).items()
                   if isinstance(v, (int, float))}
        if progress:
            progress({'stage': 'epoch', 'epoch': epoch, 'epochs': epochs,
                      'metrics': metrics, 'device': device})

    def train_epoch_start(trainer):
        trainer._kounta_batch_count = 0
        if progress:
            progress({'stage': 'batch', 'epoch': int(trainer.epoch)+1,
                      'epochs': epochs, 'batch': 0,
                      'batches': len(trainer.train_loader)})

    def train_batch_end(trainer):
        count = getattr(trainer, '_kounta_batch_count', 0) + 1
        trainer._kounta_batch_count = count
        total_batches = max(1, len(trainer.train_loader))
        interval = max(1, total_batches // 20)
        if progress and (count == 1 or count % interval == 0 or count == total_batches):
            progress({'stage': 'batch', 'epoch': int(trainer.epoch)+1,
                      'epochs': epochs, 'batch': count, 'batches': total_batches})

    model.add_callback('on_train_epoch_start', train_epoch_start)
    model.add_callback('on_train_batch_end', train_batch_end)
    model.add_callback('on_fit_epoch_end', epoch_callback)
    results = model.train(
        data=export['dataset_yaml'], epochs=epochs, batch=batch, imgsz=imgsz,
        lr0=lr0, optimizer='AdamW', patience=patience, seed=seed,
        device=device, workers=0, plots=True, cache=False,
        project=str(output_dir / 'models' / 'runs'), name=run_name,
        exist_ok=False, verbose=True,
        # Aumentación interna ligera para no borrar señales cromáticas de oxidación.
        mosaic=mosaic, mixup=0.0, hsv_h=0.01, hsv_s=0.08, hsv_v=0.15,
        degrees=5.0, translate=0.04, scale=0.15, shear=0.0,
        fliplr=0.5, flipud=0.0,
    )
    trainer = model.trainer
    best = Path(trainer.best)
    last = Path(trainer.last)
    if not best.is_file():
        raise FileNotFoundError(f'No se generó best.pt: {best}')
    model_dir = output_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    chosen = model_dir / f'{run_name}_best.pt'
    shutil.copy2(best, chosen)
    # Datos reproducibles de arquitectura y métricas útiles para la tesis.
    trained_net = model.model
    parameter_count = sum(p.numel() for p in trained_net.parameters())
    trainable_parameters = sum(p.numel() for p in trained_net.parameters() if p.requires_grad)
    metrics = getattr(trainer, 'metrics', {}) or {}
    last_metrics = {str(k): float(v) for k, v in metrics.items()
                    if isinstance(v, (int, float))}
    report = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'architecture': type(trained_net).__name__ + ' (Ultralytics YOLO)',
        'parameter_count': parameter_count, 'trainable_parameters': trainable_parameters,
        'num_classes': len(CLASS_NAMES), 'class_names': CLASS_NAMES,
        'last_metrics': last_metrics,
        'model_input': base_model, 'best_model': str(chosen), 'best_run': str(best),
        'last_run': str(last), 'run_folder': str(best.parent.parent),
        'device': device, 'epochs': epochs, 'batch': batch, 'imgsz': imgsz,
        'lr0': lr0, 'patience': patience, 'seed': seed, 'val_frac': val_frac,
        'test_frac': test_frac, 'use_augmented': use_augmented, 'mosaic': mosaic,
        'internal_augmentation': {'mixup': 0.0, 'hsv_h': 0.01, 'hsv_s': 0.08,
                                  'hsv_v': 0.15, 'degrees': 5.0,
                                  'translate': 0.04, 'scale': 0.15,
                                  'fliplr': 0.5, 'flipud': 0.0},
        'export': export,
    }
    (model_dir / f'{run_name}_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def predict_image(image: Image.Image, model_path: str, conf: float = .5,
                  iou: float = .45, max_det: int = 300) -> dict:
    if not (0 < conf < 1 and 0 < iou < 1 and max_det >= 1):
        raise ValueError('Confianza/IoU deben estar entre 0 y 1 y max_det >= 1.')
    if not Path(model_path).is_file():
        raise FileNotFoundError(f'No se encuentra el modelo YOLO: {model_path}')
    YOLO = _require_yolo()
    model = YOLO(str(model_path))
    results = model.predict(source=image.convert('RGB'), conf=conf, iou=iou,
                            max_det=max_det, verbose=False, device='0' if torch.cuda.is_available() else 'cpu')
    result = results[0]
    boxes = result.boxes
    detections = []
    if boxes is not None:
        for box in boxes:
            index = int(box.cls.item())
            name = str(result.names[index])
            detections.append({'class_name': normalized_class(name) or name,
                               'class_id': index,
                               'score': float(box.conf.item()),
                               'bbox_xyxy': [round(float(v), 2) for v in box.xyxy[0].cpu().tolist()]})
    counts = {}
    for detection in detections:
        counts[detection['class_name']] = counts.get(detection['class_name'], 0) + 1
    return {'detections': detections, 'counts': counts, 'total': len(detections)}


def prelabel_images_batch(images: list[Path], input_dir: Path, output_dir: Path,
                          model_path: str, confidence: float, allowed_classes: set[str],
                          keep_existing: bool = True, progress=None) -> dict:
    """Run inference with ONE loaded YOLO model and persist proposed BBox for a profile.

    Existing annotations are skipped by default; if replacement is enabled, a
    pre-existing annotation is only replaced after valid detections are found.
    All labels remain tentative and must be reviewed in the manual editor.
    """
    from .project_core import has_saved_annotation, save_annotations

    if not 0 < confidence < 1:
        raise ValueError('Confianza inválida.')
    if not Path(model_path).is_file():
        raise FileNotFoundError(f'No se encuentra el modelo: {model_path}')
    if not allowed_classes:
        raise ValueError('El perfil no tiene clases configuradas.')
    YOLO = _require_yolo()
    model = YOLO(str(model_path))
    model_names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    unknown = sorted({str(name) for name in model_names if normalized_class(str(name)) not in allowed_classes})
    if unknown:
        raise ValueError('Las clases del modelo no corresponden al perfil seleccionado: '+', '.join(unknown))

    counts = {'updated': 0, 'skipped': 0, 'empty': 0, 'total': len(images)}
    for index, path in enumerate(images, 1):
        path = Path(path)
        if keep_existing and (has_saved_annotation(input_dir, path)
                              or read_annotations(input_dir, path, output_dir)):
            counts['skipped'] += 1
        else:
            with Image.open(path) as source_image:
                image = source_image.convert('RGB')
            result = model.predict(source=image, conf=confidence, iou=.45, max_det=300,
                                   verbose=False, device='0' if torch.cuda.is_available() else 'cpu')[0]
            candidates = []
            if result.boxes is not None:
                for box in result.boxes:
                    name = normalized_class(str(result.names[int(box.cls.item())]))
                    if name not in allowed_classes:
                        continue
                    candidates.append({'class_name': name,
                                       'bbox_xyxy': [float(x) for x in box.xyxy[0].cpu().tolist()]})
            if candidates:
                save_annotations(input_dir, path, candidates)
                counts['updated'] += 1
            else:
                counts['empty'] += 1
        if progress is not None:
            progress(index, len(images), counts['updated'])
    return counts
