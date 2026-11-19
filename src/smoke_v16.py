"""Self-contained v16 tests: profile sequences, RGB/motor TCP protocol.
Does not require a real camera, YOLO model, or ESP32.
Usage: python -m src.smoke_v16
"""
from __future__ import annotations
import socket
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from .sequence_core import SequenceStore, validate_step, describe
from .esp32_link import ESP32Link


def main():
    with TemporaryDirectory(prefix='kounta_v16_') as folder:
        store=SequenceStore(Path(folder))
        steps=[{'type':'motor','motor':1,'direction':2,'pwm':50,'duration':1.2},
               {'type':'led','action':'on','rgb':[255,180,70]},
               {'type':'camera','mode':'manual','camera':1,'camera_name':'Cámara USB','settle':1.5},
               {'type':'led','action':'off','rgb':[0,0,0]}]
        store.save(steps)
        assert store.load()==[validate_step(s) for s in steps]
        assert 'Motor' not in describe(store.load()[0]) or describe(store.load()[0]).startswith('M1')
        for invalid in [{'type':'motor','motor':1,'direction':1,'pwm':101,'duration':1},
                        {'type':'motor','motor':1,'direction':1,'pwm':30,'duration':0},
                        {'type':'camera','mode':'invalid','camera':0},
                        {'type':'led','action':'on','rgb':[256,0,0]}]:
            try:
                validate_step(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError(f'El paso inválido fue aceptado: {invalid}')
        server=socket.socket()
        server.bind(('127.0.0.1',0))
        server.listen(4)
        port=server.getsockname()[1]
        commands=[]
        def handle():
            for _ in range(3):
                conn,_=server.accept()
                with conn:
                    commands.append(conn.recv(256).decode().strip())
                    conn.sendall(b'OK\n')
            server.close()
        th=threading.Thread(target=handle,daemon=True)
        th.start()
        link=ESP32Link('127.0.0.1',port)
        link.rgb(255,180,70)
        link.motor(2,1,65,2)
        link.stop()
        th.join(timeout=5)
        assert commands==['RGB:255:180:70','M_RUN:0:1:0:65:2000','MOTOR:0'],commands
    print('OK | v16: secuencias por perfil, validación de campos y protocolo TCP simulado')


if __name__=='__main__':
    main()
