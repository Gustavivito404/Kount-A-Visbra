"""Wi-Fi TCP client for the original Kount-A ESP32 firmware (no firmware change).

Firmware commands: M_RUN:d1:d2:pwm1:pwm2:duration_ms, MOTOR:0, RGB:r:g:b.
OK means *accepted*, NOT motor stopped. Operations must be serialised by caller.
"""
from __future__ import annotations
import socket


class ESP32Link:
    def __init__(self, host='192.168.4.1', port=8080, timeout=4.0):
        self.host = str(host).strip()
        self.port = int(port)
        self.timeout = float(timeout)

    def send(self, command: str) -> str:
        if not self.host or '\n' in command or '\r' in command:
            raise ValueError('Dirección o comando inválido.')
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as conn:
                conn.settimeout(self.timeout)
                conn.sendall((command + '\n').encode('utf-8'))
                response = conn.recv(256).decode('utf-8', errors='replace').strip()
        except (OSError, TimeoutError) as exc:
            raise ConnectionError(f'No se pudo comunicar con ESP32 ({self.host}:{self.port}): {exc}') from exc
        if response != 'OK':
            raise RuntimeError(f'ESP32 rechazó «{command}»: {response or "sin respuesta"}')
        return response

    def motor(self, motor: int, direction: int, pwm: int, duration: float) -> str:
        if motor not in (1, 2) or direction not in (1, 2):
            raise ValueError('Motor/dirección inválidos.')
        if not 0 <= int(pwm) <= 100 or not 0.1 <= float(duration) <= 120:
            raise ValueError('PWM debe estar entre 0 y 100; duración entre 0.1 y 120 segundos.')
        fields = (direction, 0, int(pwm), 0) if motor == 1 else (0, direction, 0, int(pwm))
        return self.send(f'M_RUN:{fields[0]}:{fields[1]}:{fields[2]}:{fields[3]}:{round(float(duration)*1000)}')

    def rgb(self, red: int, green: int, blue: int) -> str:
        channels = (int(red), int(green), int(blue))
        if any(not 0 <= c <= 255 for c in channels):
            raise ValueError('RGB debe estar entre 0 y 255.')
        return self.send('RGB:' + ':'.join(str(c) for c in channels))

    def stop(self) -> str:
        return self.send('MOTOR:0')
