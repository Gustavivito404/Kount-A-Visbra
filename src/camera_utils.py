from __future__ import annotations
import os
from PIL import Image


def scan_cameras(max_index: int = 7) -> list[dict]:
    """Busca dispositivos utilizables: lista para selección, sin pedir índices manuales."""
    import cv2
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
    found = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, backend)
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            found.append({'index': index, 'label': f'Cámara {len(found)+1} · {width}×{height} (dispositivo {index})'})
        finally:
            cap.release()
    return found


def capture_camera(index: int, width: int = 1280, height: int = 720,
                   autofocus: bool = True, manual_focus: int = 50) -> Image.Image:
    import cv2
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    try:
        if not cap.isOpened():
            raise RuntimeError(f'No se pudo abrir la cámara seleccionada (dispositivo {index}).')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
        if not autofocus:
            cap.set(cv2.CAP_PROP_FOCUS, max(0, min(255, int(manual_focus))))
        last = None
        for _ in range(5):
            ok, frame = cap.read()
            if ok and frame is not None:
                last = frame
        if last is None:
            raise RuntimeError('La cámara no entregó ningún frame.')
        return Image.fromarray(cv2.cvtColor(last, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
