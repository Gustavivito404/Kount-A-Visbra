"""Prueba local de BBox, augmentation y exportación sin descargar modelos YOLO.

Ejecuta: python -m src.smoke_test
"""
from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image, ImageDraw
from .project_core import save_annotations, read_annotations, detect_grayscale_boxes
from .augment_core import generate_augmentations
from .yolo_core import export_yolo_dataset


def main():
    with TemporaryDirectory(prefix='kounta_smoke_') as tmp:
        inp = Path(tmp)/'input'
        out = Path(tmp)/'output'
        for cls in ['tornillo', 'tuerca', 'rondana']:
            (inp/cls).mkdir(parents=True)
            for i in range(8):
                im = Image.new('RGB', (150, 120), 'white')
                ImageDraw.Draw(im).rectangle((20, 30, 65, 90), fill='gray')
                path = inp/cls/f'{i:03d}.png'
                im.save(path)
                save_annotations(inp, path, [{'class_name': cls, 'bbox_xyxy': [20, 30, 65, 90]}])
        mixed = inp/'mezclas'/'dos_piezas.png'
        mixed.parent.mkdir(parents=True)
        im = Image.new('RGB', (180, 120), 'white')
        draw = ImageDraw.Draw(im)
        draw.rectangle((10, 15, 60, 50), fill='gray')
        draw.rectangle((110, 60, 170, 100), fill='gray')
        im.save(mixed)
        save_annotations(inp, mixed, [
            {'class_name': 'tornillo', 'bbox_xyxy': [10, 15, 60, 50]},
            {'class_name': 'tuerca', 'bbox_xyxy': [110, 60, 170, 100]},
        ])
        assert len(read_annotations(inp, mixed, out)) == 2
        assert len(detect_grayscale_boxes(im, threshold=225, min_area=20)) == 2
        summary = generate_augmentations(inp, out, copies=2, seed=42)
        assert summary['augmented_images'] == 50
        label = out/'augmented_labels'/'mezclas'/'dos_piezas_aug_001.json'
        assert len(json.loads(label.read_text())['objects']) == 2
        export = export_yolo_dataset(inp, out, seed=42, val_frac=.2, test_frac=.1)
        assert export['counts']['train'] > 0 and export['counts']['val'] > 0
        assignments = json.loads((out/'split_assignments.json').read_text())['assignments']
        for label in (out/'yolo_dataset'/'labels'/'train').rglob('*__aug.txt'):
            rel = label.relative_to(out/'yolo_dataset'/'labels'/'train')
            original_json = out/'augmented_labels'/rel.with_name(rel.name.replace('__aug.txt', '.json'))
            assert original_json.is_file()
            source_rel = json.loads(original_json.read_text())['source_image']
            assert assignments[source_rel] == 'train'
        print('OK | Etiquetado múltiple, aumento, exportación YOLO y split sin fuga')


if __name__ == '__main__':
    main()
