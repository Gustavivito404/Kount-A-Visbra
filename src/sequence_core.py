"""Validated, per-profile sequence persistence. No hardware side effects."""
from __future__ import annotations
import json
from pathlib import Path

KIND_LABELS = {'motor': 'Movimiento de motor', 'led': 'Iluminación LED', 'camera': 'Cámara'}


def validate_step(step: dict) -> dict:
    if not isinstance(step, dict):
        raise ValueError('La instrucción debe ser un objeto.')
    kind = step.get('type')
    if kind == 'motor':
        motor, direction, pwm = int(step.get('motor', 0)), int(step.get('direction', 0)), int(step.get('pwm', -1))
        duration = float(step.get('duration', 0))
        if motor not in (1, 2) or direction not in (1, 2) or not 0 <= pwm <= 100 or not 0.1 <= duration <= 120:
            raise ValueError('Motor: elige M1/M2, sentido, PWM 0–100 y duración 0.1–120 s.')
        return {'type': kind, 'motor': motor, 'direction': direction, 'pwm': pwm, 'duration': round(duration, 3)}
    if kind == 'led':
        action = step.get('action')
        if action not in ('on', 'off'):
            raise ValueError('LED: selecciona encender o apagar.')
        rgb = step.get('rgb', [255, 255, 255])
        if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
            raise ValueError('LED: RGB necesita tres canales.')
        rgb = [int(c) for c in rgb]
        if any(c < 0 or c > 255 for c in rgb):
            raise ValueError('LED: RGB debe estar entre 0 y 255.')
        return {'type': kind, 'action': action, 'rgb': rgb}
    if kind == 'camera':
        mode = step.get('mode')
        if mode not in ('auto', 'manual'):
            raise ValueError('Cámara: selecciona automático o manual.')
        index = int(step.get('camera', -1))
        if not 0 <= index <= 32:
            raise ValueError('Cámara: selecciona un dispositivo válido.')
        settle = float(step.get('settle', 1.0))
        if not 0 <= settle <= 30:
            raise ValueError('Espera de estabilización entre 0 y 30 s.')
        return {'type': kind, 'mode': mode, 'camera': index, 'camera_name': str(step.get('camera_name', ''))[:100], 'settle': round(settle, 2)}
    raise ValueError('Tipo de instrucción desconocido.')


def describe(step: dict) -> str:
    if step['type'] == 'motor':
        return f"M{step['motor']} · {step['pwm']}% · {'Horario' if step['direction']==1 else 'Antihorario'} · {step['duration']:g} s"
    if step['type'] == 'led':
        return ('Encender · RGB ' + ','.join(str(c) for c in step['rgb'])) if step['action']=='on' else 'Apagar'
    return f"{'Automático' if step['mode']=='auto' else 'Manual en vivo'} · {step.get('camera_name') or 'Cámara '+str(step['camera'])} · espera {step['settle']:g} s"


class SequenceStore:
    def __init__(self, profile_dir: Path):
        self.path = Path(profile_dir) / 'work_sequence.json'

    def load(self) -> list[dict]:
        if not self.path.is_file():
            return []
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('format_version') != 1 or not isinstance(data.get('steps'), list):
            raise ValueError('Formato de secuencia inválido.')
        if len(data['steps']) > 500:
            raise ValueError('Secuencia excesivamente larga (máximo 500 pasos).')
        return [validate_step(item) for item in data['steps']]

    def save(self, steps: list[dict]) -> Path:
        if len(steps) > 500:
            raise ValueError('Se permiten hasta 500 pasos.')
        cleaned = [validate_step(item) for item in steps]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix('.json.tmp')
        temp.write_text(json.dumps({'format_version': 1, 'steps': cleaned}, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(self.path)
        return self.path
