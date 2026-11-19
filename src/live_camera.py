"""Single continuously-open webcam reader; GUI/YOLO read most recent frame.
The camera reader never uses Tkinter. Use stop() before a new camera scan.
"""
from __future__ import annotations
import os
import threading
import time
from PIL import Image


class LiveCamera:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._frame = None
        self._counter = 0
        self.index = None
        self.error = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def start(self, index: int, width=1280, height=720, autofocus=True, focus=50):
        if self.running and self.index == int(index):
            return
        self.stop()
        self.index = int(index)
        self._frame = None
        self._counter = 0
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(int(index), int(width), int(height), bool(autofocus), int(focus), self._stop), daemon=True)
        self._thread.start()

    def _run(self, index, width, height, autofocus, focus, stop):
        import cv2
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        try:
            if not cap.isOpened():
                self.error = f'No se pudo abrir la cámara {index}.'
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
            if not autofocus:
                cap.set(cv2.CAP_PROP_FOCUS, max(0, min(255, focus)))
            failures = 0
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures > 25:
                        self.error = 'La cámara dejó de enviar imágenes.'
                        break
                    stop.wait(.06)
                    continue
                failures = 0
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                with self._lock:
                    self._frame = img
                    self._counter += 1
        except Exception as exc:
            self.error = f'Error de captura: {exc}'
        finally:
            cap.release()

    def latest(self):
        with self._lock:
            return self._counter, self._frame.copy() if self._frame is not None else None

    def wait_frame(self, timeout=8.0, cancel=None):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cancel is not None and cancel.is_set():
                raise InterruptedError('Secuencia cancelada.')
            _, img = self.latest()
            if img is not None:
                return img
            if self.error:
                raise RuntimeError(self.error)
            time.sleep(.06)
        raise TimeoutError('La cámara no entregó una imagen a tiempo.')

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
        self._thread = None
        self.index = None
        with self._lock:
            self._frame = None
            self._counter = 0
