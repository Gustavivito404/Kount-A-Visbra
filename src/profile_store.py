"""Persistent, per-product profiles. All writes originate from the engineer GUI."""
from __future__ import annotations
import json
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path

DEFAULT_CLASSES = ['tornillo', 'tuerca', 'rondana', 'tornillo_oxidado', 'tuerca_oxidada', 'rondana_oxidada']

def slug(text: str) -> str:
    value = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('ascii').lower()
    return re.sub(r'[^a-z0-9]+', '_', value).strip('_')

class ProfileManager:
    def __init__(self, root: Path):
        self.root = Path(root).resolve() / 'profiles'
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.list_profiles():
            self.create('Tornillería', DEFAULT_CLASSES, aruco_mode='optional')

    def set_root(self, profiles_directory: Path) -> None:
        """Switch to an existing profiles folder without copying or deleting any data.

        The built-in profiles/ directory is ALWAYS created on fresh installations by
        __init__. An external empty folder is allowed and is not populated implicitly.
        """
        directory = Path(profiles_directory).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError('Selecciona una carpeta de perfiles existente.')
        if directory.is_symlink():
            raise ValueError('No se permiten enlaces simbólicos para la carpeta de perfiles.')
        self.root = directory

    def list_profiles(self) -> list[dict]:
        records = []
        for folder in sorted(self.root.iterdir()):
            if folder.is_dir() and not folder.is_symlink() and (folder / 'config.json').is_file():
                try:
                    records.append(self.load(folder.name))
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    continue
        return records

    def profile_dir(self, profile_id: str) -> Path:
        if not profile_id or slug(profile_id) != profile_id:
            raise ValueError('Identificador de perfil inválido.')
        target = self.root / profile_id
        if target.is_symlink():
            raise ValueError('No se permiten perfiles mediante enlaces simbólicos.')
        return target

    def load(self, profile_id: str) -> dict:
        root = self.profile_dir(profile_id)
        cfg = json.loads((root / 'config.json').read_text(encoding='utf-8'))
        if cfg.get('id') != profile_id:
            raise ValueError('Identificador de perfil inconsistente.')
        if not cfg.get('classes') or len(set(cfg['classes'])) != len(cfg['classes']):
            raise ValueError('El perfil no tiene clases válidas.')
        return cfg

    def save(self, cfg: dict) -> None:
        folder = self.profile_dir(cfg['id'])
        if not folder.is_dir():
            raise ValueError('Perfil inexistente.')
        path = folder / 'config.json'
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(path)

    def create(self, name: str, classes: list[str], aruco_mode: str = 'none') -> dict:
        profile_id = slug(name)
        if not profile_id or len(profile_id) > 65:
            raise ValueError('Escribe un nombre de perfil válido (hasta 65 caracteres).')
        if aruco_mode not in ('none', 'optional', 'required'):
            raise ValueError('Configuración ArUco inválida.')
        names = [slug(c) for c in classes]
        if not names or any(not x or len(x) > 60 or x == 'mezclas' for x in names) or len(names) != len(set(names)):
            raise ValueError('Indica clases no vacías y sin duplicados.')
        folder = self.profile_dir(profile_id)
        if folder.exists():
            raise ValueError('Ya existe un perfil con ese nombre.')
        folder.mkdir(parents=True)
        try:
            for cls in names:
                (folder / 'input' / cls).mkdir(parents=True, exist_ok=True)
            (folder / 'input' / 'mezclas').mkdir(parents=True, exist_ok=True)
            for path in ('annotations', 'output/augmented_images', 'output/augmented_labels',
                         'output/models', 'results'):
                (folder / path).mkdir(parents=True, exist_ok=True)
            cfg = {
                'id': profile_id, 'name': name.strip(), 'description': '', 'classes': names,
                'aruco_mode': aruco_mode, 'aruco': {
                    'reference_width_mm': None, 'reference_height_mm': None,
                    'marker_size_mm': 22.9, 'threshold': 120, 'min_area_px': 500,
                    'max_area_px': 150000, 'max_rmse_mm': 3.0,
                },
                'status': 'development', 'approved_model': None,
                'created_at': datetime.now().isoformat(timespec='seconds'),
            }
            self.save(cfg)
            return cfg
        except Exception:
            shutil.rmtree(folder)
            raise

    def add_class(self, profile_id: str, name: str) -> str:
        cfg = self.load(profile_id)
        cls = slug(name)
        if not cls or len(cls) > 60 or cls == 'mezclas' or cls in cfg['classes']:
            raise ValueError('Clase vacía, duplicada o reservada.')
        (self.profile_dir(profile_id) / 'input' / cls).mkdir(parents=True, exist_ok=True)
        cfg['classes'].append(cls)
        # The old approved model is now out of date; operator can't use it.
        cfg['status'], cfg['approved_model'] = 'development', None
        self.save(cfg)
        return cls

    def remove_class(self, profile_id: str, name: str) -> None:
        cfg = self.load(profile_id)
        if name not in cfg['classes'] or len(cfg['classes']) < 2:
            raise ValueError('La clase no existe o es la última clase del perfil.')
        folder = self.profile_dir(profile_id)
        cls_dir = folder / 'input' / name
        if cls_dir.exists() and any(cls_dir.iterdir()):
            raise ValueError('La clase tiene archivos; primero debes revisar sus imágenes y etiquetas.')
        if any((folder / 'input' / '.kounta_annotations').rglob('*.json')):
            raise ValueError('Existen anotaciones: eliminar clases podría invalidar etiquetas. No se permite.')
        if any((folder / 'output' / 'models').rglob('*.pt')):
            raise ValueError('Existen modelos entrenados: no se permite eliminar clases sin migración.')
        cfg['classes'].remove(name)
        cfg['status'], cfg['approved_model'] = 'development', None
        self.save(cfg)
        if cls_dir.is_dir():
            cls_dir.rmdir()

    def import_images(self, profile_id: str, paths: list[Path], category: str) -> int:
        cfg = self.load(profile_id)
        if category not in cfg['classes'] + ['mezclas']:
            raise ValueError('Selecciona una carpeta de destino válida.')
        target = self.profile_dir(profile_id) / 'input' / category
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        count = 0
        for original in paths:
            original = Path(original)
            if not original.is_file() or original.suffix.lower() not in extensions:
                continue
            destination = target / original.name
            if destination.exists():
                stem, suffix = original.stem, original.suffix
                k = 2
                while destination.exists():
                    destination = target / f'{stem}_{k}{suffix}'
                    k += 1
            shutil.copy2(original, destination)
            count += 1
        return count

    def list_models(self, profile_id: str) -> list[Path]:
        folder = self.profile_dir(profile_id) / 'output' / 'models'
        return sorted((p for p in folder.rglob('*.pt') if p.is_file()), reverse=True)

    def approve(self, profile_id: str, model: Path) -> None:
        cfg = self.load(profile_id)
        root = self.profile_dir(profile_id).resolve()
        path = Path(model).resolve()
        if not path.is_file() or not path.is_relative_to(root / 'output' / 'models'):
            raise ValueError('El modelo autorizado debe estar en output/models del perfil.')
        # Verify YOLO class mapping before activating production.
        from ultralytics import YOLO
        net = YOLO(str(path))
        model_classes = [str(net.names[i]) for i in sorted(net.names)]
        if model_classes != cfg['classes']:
            raise ValueError(f'Clases del modelo incompatibles. Modelo: {model_classes}; perfil: {cfg["classes"]}.')
        cfg['approved_model'] = str(path.relative_to(root).as_posix())
        cfg['status'] = 'approved'
        self.save(cfg)

    def revoke(self, profile_id: str) -> None:
        cfg = self.load(profile_id)
        cfg['approved_model'] = None
        cfg['status'] = 'validation' if self.list_models(profile_id) else 'development'
        self.save(cfg)

    def approved_profiles(self) -> list[dict]:
        available = []
        for cfg in self.list_profiles():
            path = cfg.get('approved_model')
            if cfg.get('status') == 'approved' and path and (self.profile_dir(cfg['id']) / path).is_file():
                available.append(cfg)
        return available
