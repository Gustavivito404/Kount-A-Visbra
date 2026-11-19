"""Tk-side non-blocking preview, continuous YOLO and ArUco visualization."""
from __future__ import annotations
import threading
import time
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image

from . import aruco_core
from .project_core import normalized_class
from .yolo_core import predict_image


class LiveUIMixin:
    def _get_inspection_model(self):
        if not self.active_profile: return ''
        if self._engineering_view():
            return self.test_model.get().strip()
        approved = self.active_profile.get('approved_model')
        return str(self.profiles.profile_dir(self.active_profile['id']) / approved) if approved else ''

    def _inspect_image(self, image, model):
        cfg = self.active_profile
        conf, iou, max_det = self._sequence_count_config if self._sequence_running else (
            float(self.test_conf.get()), float(self.test_iou.get()), int(self.test_maxdet.get()))
        calibration = aruco_core.analyze(image, cfg['aruco']) if cfg['aruco_mode'] != 'none' else None
        if calibration is not None and cfg['aruco_mode'] == 'required' and not calibration['valid']:
            return {'detections': [], 'counts': {}, 'total': 0, 'preview': calibration['preview'],
                    'calibration': {'detected': calibration['detected'], 'valid': False,
                                    'reason': calibration['reason'], 'rmse_mm': calibration['rmse_mm']},
                    'blocked': True, 'contour_count': None}
        result = predict_image(image, model, conf=conf, iou=iou, max_det=max_det)
        if calibration is not None:
            result = aruco_core.combine(image, result, calibration,
                                        compare=((self._sequence_test_mode if self._sequence_running else self.test_mode.get()) == 'Comparativo'))
        return result

    def _selected_camera_index(self):
        choice = self.camera_combo.current()
        if 0 <= choice < len(self.available_cameras):
            return self.available_cameras[choice]['index']
        raise ValueError('Busca cámaras y selecciona un dispositivo para continuar.')

    def _camera_options(self):
        return (int(self.camera_width.get()), int(self.camera_height.get()),
                bool(self.camera_autofocus.get()), int(self.camera_focus.get()))

    def _start_live_preview(self, kind='test', index=None, infer=True):
        if self._sequence_running:
            messagebox.showwarning('Cámara', 'Hay una secuencia en ejecución.'); return
        try:
            if index is None:
                if kind == 'aruco':
                    pos = self.aruco_camera_choice.current()
                    if pos < 0 or pos >= len(self.available_cameras):
                        raise ValueError('Busca cámaras en Prueba y conteo y selecciona una.')
                    index = self.available_cameras[pos]['index']
                else:
                    index = self._selected_camera_index()
            options = self._camera_options()
        except Exception as exc:
            messagebox.showerror('Cámara en vivo', str(exc)); return
        self._stop_live_inference()
        self._reported_cam_error=None
        self._camera_stream.start(index, *options)
        self._camera_context = kind
        self._last_camera_frame_id = -1
        self._last_aruco_at = 0
        self.test_detections = []
        self.test_display_image = None
        if kind == 'test':
            self.test_summary.set('Cámara en vivo. Iniciando visualización…')
            if infer:
                self._start_live_inference()
        elif kind == 'aruco' and hasattr(self, 'aruco_live_status'):
            self.aruco_live_status.set('Buscando marcadores ArUco en cámara…')

    def _user_stop_live_preview(self):
        if self._sequence_running:
            messagebox.showwarning('Cámara', 'Detén la secuencia antes de cerrar la cámara.')
            return
        self._stop_live_preview()

    def _stop_live_preview(self):
        self._stop_live_inference()
        self._camera_stream.stop()
        self._camera_context = None
        if hasattr(self, 'aruco_live_status'):
            self.aruco_live_status.set('Cámara detenida.')

    def _start_live_inference(self, model_path=None):
        self._stop_live_inference()
        model_path = (model_path or self._get_inspection_model()).strip()
        if not model_path or not Path(model_path).is_file():
            self.test_summary.set('Cámara en vivo sin YOLO: selecciona un modelo válido.')
            return
        try:
            conf, iou, max_det = float(self.test_conf.get()), float(self.test_iou.get()), int(self.test_maxdet.get())
            if not (0 < conf < 1 and 0 < iou < 1 and max_det >= 1):
                raise ValueError('Confianza/IoU inválidos.')
        except ValueError as exc:
            messagebox.showerror('YOLO', str(exc)); return
        self._live_generation += 1
        generation = self._live_generation
        stop = self._live_infer_stop = threading.Event()
        def process():
            try:
                import torch
                from ultralytics import YOLO
                predictor = YOLO(model_path)  # Cargar una sola vez por sesión de video
                prev_id = -1
                last_calibration = None
                last_calibration_at = 0.0
                aruco_mode = self.active_profile['aruco_mode'] if self.active_profile else 'none'
                aruco_settings = dict(self.active_profile['aruco']) if self.active_profile else {}
                while not stop.is_set():
                    frame_id, image = self._camera_stream.latest()
                    if image is None or frame_id == prev_id:
                        stop.wait(.06)
                        continue
                    prev_id = frame_id
                    result = predictor.predict(source=image, conf=conf, iou=iou, max_det=max_det,
                                               verbose=False, device='0' if torch.cuda.is_available() else 'cpu')[0]
                    detections = []
                    for box in (result.boxes or []):
                        ci = int(box.cls.item())
                        name = str(result.names[ci])
                        detections.append({'class_name': normalized_class(name) or name,
                                           'class_id': ci, 'score': float(box.conf.item()),
                                           'bbox_xyxy': [float(x) for x in box.xyxy[0].cpu().tolist()]})
                    note=''
                    if aruco_mode != 'none':
                        if last_calibration is None or time.monotonic()-last_calibration_at > .9:
                            last_calibration = aruco_core.analyze(image, aruco_settings)
                            last_calibration_at = time.monotonic()
                        if last_calibration['valid']:
                            import cv2
                            import numpy as np
                            poly = last_calibration['polygon'].astype(np.float32)
                            detections = [d for d in detections if cv2.pointPolygonTest(
                                poly, ((d['bbox_xyxy'][0]+d['bbox_xyxy'][2])/2,
                                       (d['bbox_xyxy'][1]+d['bbox_xyxy'][3])/2), False) >= 0]
                            note='ArUco válido · ROI aplicada'
                        elif aruco_mode == 'required':
                            detections = []
                            note='ArUco obligatorio sin calibración: inspección bloqueada'
                        else:
                            note='ArUco no válido · conteo de toda la imagen'
                    counts = {}
                    for d in detections:
                        counts[d['class_name']] = counts.get(d['class_name'], 0)+1
                    self._emit('live_count', (generation, detections, counts, note))
                    stop.wait(.12)  # Mantener UI reactiva; saltar frames intermedios
            except Exception as exc:
                if not stop.is_set():
                    self._emit('live_error', (generation, str(exc)))
        self._live_infer_thread = threading.Thread(target=process, daemon=True)
        self._live_infer_thread.start()

    def _stop_live_inference(self):
        self._live_generation += 1
        self._live_infer_stop.set()

    def _poll_live(self):
        try:
            if self._camera_context is not None:
                seq, image = self._camera_stream.latest()
                if self._camera_stream.error and self._camera_stream.error != self._reported_cam_error:
                    self._reported_cam_error = self._camera_stream.error
                    self.status_var.set(self._camera_stream.error)
                if image is not None and seq != self._last_camera_frame_id:
                    self._last_camera_frame_id = seq
                    if self._camera_context == 'test':
                        self.test_image = image
                        self.test_display_image = None
                        self._rerender_test()
                    elif self._camera_context == 'aruco':
                        if time.monotonic() - self._last_aruco_at >= .7 and not self._aruco_processing:
                            self._last_aruco_at = time.monotonic()
                            self._aruco_processing = True
                            settings = dict(self.active_profile['aruco']) if self.active_profile else {}
                            # Vista previa responde a campos actuales, incluso antes de Guardar.
                            for key, var in getattr(self,'aruco_vars',{}).items():
                                try:
                                    text=var.get().strip()
                                    settings[key]=float(text) if text else None
                                except ValueError:
                                    pass
                            def analyze():
                                try: self._emit('aruco_live_result', aruco_core.analyze(image, settings))
                                except Exception as exc: self._emit('aruco_live_error', str(exc))
                            threading.Thread(target=analyze, daemon=True).start()
        finally:
            try: self.after(150, self._poll_live)
            except tk.TclError: pass

    def _save_distribution(self):
        if not self.active_profile:
            messagebox.showinfo('Distribución', 'Selecciona un perfil.'); return
        image = self.test_image
        if image is None:
            messagebox.showinfo('Distribución', 'Abre una imagen o activa la cámara primero.'); return
        dest = self.profiles.profile_dir(self.active_profile['id']) / 'input' / 'mezclas'
        dest.mkdir(parents=True, exist_ok=True)
        filename = 'distribucion_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.png'
        try:
            output = dest / filename
            image.convert('RGB').save(output, format='PNG')
            self.status_var.set('Distribución sin etiquetas guardada: ' + str(output))
            messagebox.showinfo('Distribución guardada', 'Fotografía original guardada en mezclas.\nRevísala y etiquétala antes del entrenamiento.')
            if self._engineering_view():
                self._refresh_images()
        except Exception as exc:
            messagebox.showerror('Guardar distribución', str(exc))
