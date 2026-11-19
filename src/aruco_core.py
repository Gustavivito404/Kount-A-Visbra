"""Four-ArUco calibration + independent threshold-based contour counting.

Adapted from 4_aruco.py: retains IDs 0/1/2/3, center-to-center homography,
marker-size RMSE, threshold, min/max area and oriented mm measurements.
No camera capture, GUI event loop or mutable process globals.
"""
from __future__ import annotations
from collections import Counter

import cv2
import numpy as np
from PIL import Image

EXPECTED_IDS = (0, 1, 2, 3)

def analyze(image: Image.Image, settings: dict) -> dict:
    rgb = np.asarray(image.convert('RGB'))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    aruco = getattr(cv2, 'aruco', None)
    if aruco is None or not hasattr(aruco, 'ArucoDetector'):
        raise RuntimeError('ArUco no disponible. Instala opencv-contrib-python (no mezclar con opencv-python).')
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50), aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    markers = {}
    if ids is not None:
        for pts, value in zip(corners, ids.flatten()):
            if int(value) in EXPECTED_IDS:
                markers[int(value)] = pts.reshape(4, 2).astype(np.float32)
    diagnostic = {'markers': sorted(markers), 'detected': len(markers), 'valid': False,
                  'reason': 'Se requieren los cuatro ArUco (0, 1, 2, 3).',
                  'rmse_mm': None, 'polygon': None, 'homography': None,
                  'contours': [], 'contour_count': None}
    for idx, pts in markers.items():
        cv2.polylines(bgr, [pts.astype(np.int32)], True, (30, 205, 80), 2)
        at = np.round(pts.mean(axis=0)).astype(int)
        cv2.putText(bgr, f'ID {idx}', tuple(at), cv2.FONT_HERSHEY_SIMPLEX, .7, (30, 205, 80), 2)
    diagnostic['preview'] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if len(markers) != 4:
        return diagnostic
    try:
        width = float(settings.get('reference_width_mm') or 0)
        height = float(settings.get('reference_height_mm') or 0)
        if width <= 0 or height <= 0:
            diagnostic['reason'] = 'Configura las distancias centro-a-centro del perfil.'
            return diagnostic
        pixel_centers = np.asarray([markers[i].mean(axis=0) for i in EXPECTED_IDS], dtype=np.float32)
        physical = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        poly = pixel_centers.astype(np.int32)
        if cv2.contourArea(pixel_centers) < 150 or not cv2.isContourConvex(poly):
            diagnostic['reason'] = 'Los centros ArUco no forman una región convexa válida.'
            return diagnostic
        homography = cv2.getPerspectiveTransform(pixel_centers, physical)
        if not np.isfinite(homography).all():
            diagnostic['reason'] = 'Transformación geométrica inválida.'
            return diagnostic
        marker_size = float(settings.get('marker_size_mm', 22.9))
        errors = []
        for pts in markers.values():
            mapped = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), homography).reshape(4, 2)
            errors.extend([np.linalg.norm(mapped[(i+1) % 4] - mapped[i]) - marker_size for i in range(4)])
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        diagnostic['rmse_mm'] = rmse
        max_error = float(settings.get('max_rmse_mm', 3.0))
        if rmse > max_error:
            diagnostic['reason'] = f'Error de referencia {rmse:.2f} mm > tolerancia {max_error:.2f} mm.'
            return diagnostic
        diagnostic.update(valid=True, reason='Calibración válida.', homography=homography,
                          polygon=pixel_centers)
        cv2.polylines(bgr, [poly], True, (235, 205, 40), 3)
        gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        _, mask = cv2.threshold(gray, int(settings.get('threshold', 120)), 255, cv2.THRESH_BINARY_INV)
        for pts in markers.values():
            x,y,w,h = cv2.boundingRect(pts.astype(np.int32))
            cv2.rectangle(mask, (max(0,x-20), max(0,y-20)),
                          (min(mask.shape[1],x+w+20), min(mask.shape[0],y+h+20)), 0, -1)
        region = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(region, [poly], 255)
        mask = cv2.bitwise_and(mask, region)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        measured = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not float(settings.get('min_area_px', 500)) <= area <= float(settings.get('max_area_px', 150000)):
                continue
            rect = cv2.minAreaRect(contour)
            corners = cv2.boxPoints(rect).astype(np.float32)
            pts = cv2.perspectiveTransform(corners.reshape(-1,1,2), homography).reshape(4,2)
            a = (np.linalg.norm(pts[1]-pts[0])+np.linalg.norm(pts[3]-pts[2])) / 2
            b = (np.linalg.norm(pts[2]-pts[1])+np.linalg.norm(pts[0]-pts[3])) / 2
            if max(a,b) > 200 or min(a,b) > 150:
                continue
            measured.append({'bbox_xyxy': list(map(int, (cv2.boundingRect(contour)[0], cv2.boundingRect(contour)[1],
                         cv2.boundingRect(contour)[0]+cv2.boundingRect(contour)[2], cv2.boundingRect(contour)[1]+cv2.boundingRect(contour)[3]))),
                             'box': corners, 'center': rect[0], 'length_mm': round(float(max(a,b)),2),
                             'width_mm': round(float(min(a,b)),2), 'area_px': float(area)})
        diagnostic['contours'] = measured
        diagnostic['contour_count'] = len(measured)
        diagnostic['preview'] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return diagnostic
    except (cv2.error, ValueError, TypeError, OverflowError) as exc:
        diagnostic['reason'] = f'Error de calibración: {exc}'
        return diagnostic


def combine(image: Image.Image, yolo: dict, calibration: dict, compare: bool = False) -> dict:
    """Filter objects by center inside polygon; link contours by rectangle IoU.
    If missing calibration YOLO remains full-frame; dimensions unavailable.
    """
    canvas = np.asarray(calibration['preview'].convert('RGB')).copy()
    poly = calibration.get('polygon') if calibration['valid'] else None
    contours = calibration['contours'] if calibration['valid'] else []
    matches_used = set()
    filtered = []
    for det in yolo['detections']:
        b = det['bbox_xyxy']
        center = ((b[0]+b[2])/2, (b[1]+b[3])/2)
        if poly is not None and cv2.pointPolygonTest(poly.astype(np.float32), center, False) < 0:
            continue
        record = dict(det)
        record['length_mm'] = None
        record['width_mm'] = None
        best_idx, best_iou = None, 0
        for i, contour in enumerate(contours):
            if i in matches_used:
                continue
            q = contour['bbox_xyxy']
            xx1, yy1, xx2, yy2 = max(b[0],q[0]), max(b[1],q[1]), min(b[2],q[2]), min(b[3],q[3])
            overlap = max(0,xx2-xx1)*max(0,yy2-yy1)
            union = (b[2]-b[0])*(b[3]-b[1])+(q[2]-q[0])*(q[3]-q[1])-overlap
            score = overlap/union if union>0 else 0
            if score > best_iou:
                best_idx, best_iou = i, score
        if best_idx is not None and best_iou >= .15:
            matches_used.add(best_idx)
            contour = contours[best_idx]
            record['length_mm'] = contour['length_mm']
            record['width_mm'] = contour['width_mm']
        filtered.append(record)
    for i, det in enumerate(filtered):
        x1,y1,x2,y2 = (int(x) for x in det['bbox_xyxy'])
        cv2.rectangle(canvas, (x1,y1),(x2,y2),(49,153,238),2)
        title = f'{i+1}. {det["class_name"]}'
        if det['length_mm'] is not None:
            title += f' {det["length_mm"]:.1f}x{det["width_mm"]:.1f} mm'
        cv2.putText(canvas, title, (x1,max(16,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,110,240),2)
    counts = dict(Counter(d['class_name'] for d in filtered))
    return {'detections': filtered, 'counts': counts, 'total':len(filtered),
            'calibration': {'detected': calibration['detected'], 'valid':calibration['valid'],
                            'reason':calibration['reason'], 'rmse_mm':calibration['rmse_mm']},
            'contour_count': calibration['contour_count'] if compare else None,
            'preview': Image.fromarray(canvas)}
