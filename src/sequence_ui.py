"""Compact visual sequence editor and supervised/automatic executor for Kount-A.
All Tk operations run on the GUI thread; execution uses the existing ESP32 protocol.
"""
from __future__ import annotations
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser

from .esp32_link import ESP32Link
from .sequence_core import SequenceStore, validate_step, describe, KIND_LABELS


class SequenceUIMixin:
    def _esp_init_ui_state(self):
        self.esp_connection_text = tk.StringVar(value='ESP32: sin comprobar')
        self._esp_lamps = []
        self._esp_connected = False
        self._esp_confirmed_endpoint = None
        self._esp_probe_revision = 0
        self._esp_probe_pending = False
        self._esp_monitoring = False
        self.esp32_ip.trace_add('write', self._esp_endpoint_changed)
        self.esp32_port.trace_add('write', self._esp_endpoint_changed)

    def _esp_endpoint_changed(self, *_):
        self._esp_probe_revision += 1
        self._esp_monitoring = False
        self._esp_probe_pending = False
        self._esp_update_indicator(False, 'ESP32: sin comprobar')

    def _esp_update_indicator(self, connected, message):
        self._esp_connected = bool(connected)
        self._esp_confirmed_endpoint = self._esp_endpoint() if connected else None
        self.esp_connection_text.set(message)
        color = '#24B85A' if connected else '#D94343'
        active = []
        for lamp in self._esp_lamps:
            try:
                if lamp.winfo_exists():
                    lamp.configure(bg=color)
                    active.append(lamp)
            except tk.TclError:
                pass
        self._esp_lamps = active

    def _esp_status_widget(self, parent):
        frame = ttk.Frame(parent)
        lamp = tk.Frame(frame, width=22, height=16,
                        bg='#24B85A' if self._esp_connected else '#D94343',
                        highlightthickness=1, highlightbackground='#555555')
        lamp.pack(side='left', padx=(0, 7))
        lamp.pack_propagate(False)
        ttk.Label(frame, textvariable=self.esp_connection_text,
                  wraplength=200).pack(side='left', fill='x', expand=True)
        self._esp_lamps.append(lamp)
        return frame

    def _esp_endpoint(self):
        try:
            address = self.esp32_ip.get().strip()
            port = int(self.esp32_port.get())
            if not address or not 0 < port < 65536:
                return None
            return (address, port)
        except ValueError:
            return None

    def _esp_connect(self):
        self._esp_monitoring = True
        self._esp_probe(manual=True)

    def _esp_probe(self, manual=False):
        endpoint = self._esp_endpoint()
        if endpoint is None:
            self._esp_monitoring = False
            self._esp_update_indicator(False, 'ESP32: IP o puerto inválido')
            return
        if self._esp_probe_pending:
            return
        self._esp_probe_pending = True
        revision = self._esp_probe_revision
        if manual:
            self._esp_update_indicator(False, 'ESP32: comprobando…')
        def worker():
            try:
                ESP32Link(*endpoint, timeout=2.5).probe()
                self._emit('esp_probe_result', (revision, endpoint, True, ''))
            except Exception as exc:
                self._emit('esp_probe_result', (revision, endpoint, False, str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _esp_receive_probe_result(self, data):
        revision, endpoint, ok, error = data
        if revision != self._esp_probe_revision or endpoint != self._esp_endpoint():
            return
        self._esp_probe_pending = False
        self._esp_update_indicator(ok, 'ESP32: conectada' if ok else 'ESP32: sin conexión')
        if error:
            self.status_var.set(error)
        if self._esp_monitoring:
            # Los comandos del firmware usan sockets breves: comprobar periódicamente.
            self.after(4500, self._esp_periodic_probe)

    def _esp_periodic_probe(self):
        if self._esp_monitoring and self.winfo_exists():
            self._esp_probe()

    def _sequence_store(self):
        return SequenceStore(self.profiles.profile_dir(self.active_profile['id']))

    def _load_sequence(self):
        self.sequence_steps = []
        self.sequence_dirty = False
        if self.active_profile:
            try:
                self.sequence_steps = self._sequence_store().load()
            except (ValueError, OSError, TypeError) as exc:
                self.status_var.set('Secuencia: ' + str(exc))
        self._refresh_sequence_tree()

    def _refresh_sequence_tree(self):
        tree = getattr(self, 'seq_tree', None)
        if tree is None or not tree.winfo_exists():
            return
        selected = tree.selection()
        chosen = int(selected[0]) if selected else -1
        tree.delete(*tree.get_children())
        for i, step in enumerate(self.sequence_steps):
            tree.insert('', 'end', iid=str(i), values=(i+1, KIND_LABELS[step['type']], describe(step)))
        if 0 <= chosen < len(self.sequence_steps):
            tree.selection_set(str(chosen))
        self.sequence_status.set(f'{len(self.sequence_steps)} instrucciones' + (' · sin guardar' if self.sequence_dirty else ' · guardada'))

    def _build_sequence_tab(self, tab):
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)
        sidebar = ttk.LabelFrame(tab, text='Herramientas', padding=9)
        sidebar.grid(row=0, column=0, sticky='ns', padx=(0, 5))
        tools = [('+ Agregar', self._seq_add), ('Editar', self._seq_edit),
                 ('↑ Subir', lambda: self._seq_move(-1)), ('↓ Bajar', lambda: self._seq_move(1)),
                 ('Eliminar', self._seq_delete), ('Guardar secuencia', self._seq_save)]
        for label, callback in tools:
            ttk.Button(sidebar, text=label, command=callback,
                       style='Primary.TButton' if label == 'Guardar secuencia' else 'TButton').pack(fill='x', pady=4)
        ttk.Separator(sidebar).pack(fill='x', pady=12)
        ttk.Label(sidebar, text='Perfil activo:', wraplength=140).pack(anchor='w')
        self.sequence_profile_label = ttk.Label(sidebar, text=self.active_profile['name'] if self.active_profile else 'Sin perfil', wraplength=140)
        self.sequence_profile_label.pack(anchor='w', pady=(3, 9))
        ttk.Label(sidebar, text='Conexión ESP32').pack(anchor='w', pady=(3, 2))
        ttk.Label(sidebar, text='IP:').pack(anchor='w')
        ttk.Entry(sidebar, textvariable=self.esp32_ip, width=17).pack(fill='x', pady=(0, 3))
        ttk.Label(sidebar, text='Puerto:').pack(anchor='w')
        ttk.Entry(sidebar, textvariable=self.esp32_port, width=10).pack(fill='x', pady=(0, 5))
        self._esp_status_widget(sidebar).pack(fill='x', pady=(2, 5))
        ttk.Button(sidebar, text='Conectar ESP32', command=self._esp_connect).pack(fill='x', pady=4)
        ttk.Button(sidebar, text='Probar secuencia', command=self._start_sequence).pack(fill='x', pady=4)
        ttk.Button(sidebar, text='Detener ejecución', command=self._stop_sequence).pack(fill='x', pady=4)
        main = ttk.LabelFrame(tab, text='Instrucciones en orden de ejecución', padding=8)
        main.grid(row=0, column=1, sticky='nsew')
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        self.sequence_status = tk.StringVar(value='Secuencia sin cargar')
        ttk.Label(main, textvariable=self.sequence_status).grid(row=0, column=0, sticky='w', pady=(0, 6))
        self.seq_tree = ttk.Treeview(main, columns=('num', 'kind', 'info'), show='headings', selectmode='browse')
        for col, heading, size in [('num', 'N.º', 60), ('kind', 'Instrucción', 165), ('info', 'Parámetros', 430)]:
            self.seq_tree.heading(col, text=heading)
            self.seq_tree.column(col, width=size, minwidth=50, stretch=col == 'info', anchor='w')
        self.seq_tree.grid(row=1, column=0, sticky='nsew')
        sc = ttk.Scrollbar(main, orient='vertical', command=self.seq_tree.yview)
        sc.grid(row=1, column=1, sticky='ns')
        self.seq_tree.configure(yscrollcommand=sc.set)
        self.seq_tree.bind('<Double-1>', lambda _e: self._seq_edit())
        ttk.Label(main, text='Selecciona una fila para editarla, subirla o bajarla. Desplázate con la rueda del mouse.',
                  wraplength=660).grid(row=2, column=0, sticky='w', pady=7)
        self._refresh_sequence_tree()

    def _seq_selected(self):
        selected = self.seq_tree.selection()
        return int(selected[0]) if selected else -1

    def _seq_add(self):
        if not self.active_profile or self._sequence_busy():
            return
        step = self._seq_step_dialog()
        if step is not None:
            idx = self._seq_selected()
            insert_at = min(idx + 1, len(self.sequence_steps)) if idx >= 0 else len(self.sequence_steps)
            self.sequence_steps.insert(insert_at, step)
            self.sequence_dirty = True
            self._refresh_sequence_tree()
            self.seq_tree.selection_set(str(insert_at))
            self.seq_tree.see(str(insert_at))

    def _seq_edit(self):
        if self._sequence_busy(): return
        idx = self._seq_selected()
        if idx < 0:
            messagebox.showinfo('Secuencia', 'Selecciona primero una instrucción.'); return
        edited = self._seq_step_dialog(self.sequence_steps[idx])
        if edited is not None:
            self.sequence_steps[idx] = edited
            self.sequence_dirty = True
            self._refresh_sequence_tree()

    def _seq_move(self, delta):
        if self._sequence_busy(): return
        idx = self._seq_selected()
        dest = idx + delta
        if idx < 0 or not 0 <= dest < len(self.sequence_steps): return
        self.sequence_steps[idx], self.sequence_steps[dest] = self.sequence_steps[dest], self.sequence_steps[idx]
        self.sequence_dirty = True
        self._refresh_sequence_tree()
        self.seq_tree.selection_set(str(dest))
        self.seq_tree.see(str(dest))

    def _seq_delete(self):
        if self._sequence_busy(): return
        idx = self._seq_selected()
        if idx < 0: return
        if not messagebox.askyesno('Eliminar instrucción', f'¿Eliminar paso {idx+1}?'):
            return
        self.sequence_steps.pop(idx)
        self.sequence_dirty = True
        self._refresh_sequence_tree()

    def _seq_save(self):
        if not self.active_profile or self._sequence_busy(): return False
        try:
            path = self._sequence_store().save(self.sequence_steps)
            self.sequence_dirty = False
            self._refresh_sequence_tree()
            self.status_var.set('Secuencia guardada en perfil: ' + str(path))
            return True
        except Exception as exc:
            messagebox.showerror('Guardar secuencia', str(exc))
            return False

    def _resolve_sequence_dirty(self):
        if not self._engineering_view() or not getattr(self, 'sequence_dirty', False): return True
        answer = messagebox.askyesnocancel('Secuencia sin guardar', 'La secuencia tiene cambios. ¿Guardarlos antes de salir?')
        if answer is None: return False
        return self._seq_save() if answer else True

    def _sequence_busy(self):
        if getattr(self, '_sequence_running', False):
            messagebox.showwarning('Secuencia activa', 'Detén la ejecución antes de modificar las instrucciones.')
            return True
        return False

    def _seq_step_dialog(self, existing=None):
        data = dict(existing or {'type': 'motor', 'motor': 1, 'direction': 1, 'pwm': 45, 'duration': 1.5})
        dialog = tk.Toplevel(self)
        dialog.title('Editar instrucción' if existing else 'Nueva instrucción')
        dialog.transient(self)
        dialog.configure(bg=self._palette()['bg'])
        dialog.resizable(False, False)
        body = ttk.Frame(dialog, padding=20)
        body.pack(fill='both', expand=True)
        ttk.Label(body, text='Tipo de instrucción').pack(anchor='w', pady=(0, 4))
        type_var = tk.StringVar(value=KIND_LABELS.get(data['type'], KIND_LABELS['motor']))
        kind_picker = ttk.Combobox(body, textvariable=type_var, values=list(KIND_LABELS.values()), state='readonly', width=43)
        kind_picker.pack(fill='x')
        fields = ttk.Frame(body)
        fields.pack(fill='both', expand=True, pady=12)
        vars = {}
        current_camera = data.get('camera', 0)
        choices = {c['label']: c['index'] for c in self.available_cameras}
        if not choices:
            choices = {f'Cámara {i} (dispositivo {i})': i for i in range(4)}
        if int(current_camera) not in choices.values():
            choices[f"{data.get('camera_name') or 'Cámara'} (dispositivo {current_camera})"] = int(current_camera)

        def add_row(label, variable, values=None):
            row = ttk.Frame(fields)
            row.pack(fill='x', pady=5)
            ttk.Label(row, text=label, width=18).pack(side='left')
            if values is None:
                widget = ttk.Entry(row, textvariable=variable, width=25)
            else:
                widget = ttk.Combobox(row, textvariable=variable, values=values, width=27, state='readonly')
            widget.pack(side='left', fill='x', expand=True)
            return widget

        def repaint(*_):
            for child in fields.winfo_children(): child.destroy()
            kind = next((k for k, v in KIND_LABELS.items() if v == type_var.get()), 'motor')
            if kind == 'motor':
                vars['motor'] = tk.StringVar(value=str(data.get('motor', 1)))
                vars['direction'] = tk.StringVar(value='Horario' if data.get('direction', 1) == 1 else 'Antihorario')
                vars['pwm'] = tk.StringVar(value=str(data.get('pwm', 45)))
                vars['duration'] = tk.StringVar(value=str(data.get('duration', 1.5)))
                add_row('Motor', vars['motor'], ['1', '2'])
                add_row('Dirección', vars['direction'], ['Horario', 'Antihorario'])
                add_row('PWM (0–100 %)', vars['pwm'])
                add_row('Duración (s)', vars['duration'])
            elif kind == 'led':
                vars['action'] = tk.StringVar(value='Encender' if data.get('action', 'on') == 'on' else 'Apagar')
                rgb = data.get('rgb', [255, 255, 255])
                vars['r'], vars['g'], vars['b'] = (tk.StringVar(value=str(x)) for x in rgb)
                add_row('Acción', vars['action'], ['Encender', 'Apagar'])
                for channel in ('r', 'g', 'b'):
                    add_row('Canal ' + channel.upper(), vars[channel])
                def pick_color():
                    current = tuple(max(0, min(255, int(vars[c].get()))) for c in ('r', 'g', 'b'))
                    picked = colorchooser.askcolor(color='#%02x%02x%02x' % current, parent=dialog)
                    if picked[0]:
                        for c, value in zip(('r', 'g', 'b'), picked[0]): vars[c].set(str(int(value)))
                ttk.Button(fields, text='Elegir color RGB…', command=pick_color).pack(fill='x', pady=7)
            else:
                vars['mode'] = tk.StringVar(value='Manual — tiempo real' if data.get('mode', 'manual') == 'manual' else 'Automático — captura única')
                camera_label = next((label for label, index in choices.items() if index == int(current_camera)), next(iter(choices)))
                vars['camera'] = tk.StringVar(value=camera_label)
                vars['settle'] = tk.StringVar(value=str(data.get('settle', 1.0)))
                add_row('Modo', vars['mode'], ['Automático — captura única', 'Manual — tiempo real'])
                add_row('Dispositivo', vars['camera'], list(choices))
                add_row('Estabilización (s)', vars['settle'])
                ttk.Label(fields, text='Busca cámaras en Prueba y conteo para mostrar sus nombres.', wraplength=340).pack(anchor='w', pady=6)
        kind_picker.bind('<<ComboboxSelected>>', repaint)
        repaint()
        result = {'step': None}
        def close(save=False):
            if save:
                try:
                    kind = next(k for k, value in KIND_LABELS.items() if value == type_var.get())
                    if kind == 'motor':
                        step = {'type': kind, 'motor': int(vars['motor'].get()),
                                'direction': 1 if vars['direction'].get() == 'Horario' else 2,
                                'pwm': int(vars['pwm'].get()), 'duration': float(vars['duration'].get())}
                    elif kind == 'led':
                        step = {'type': kind, 'action': 'on' if vars['action'].get() == 'Encender' else 'off',
                                'rgb': [int(vars[c].get()) for c in ('r', 'g', 'b')]}
                    else:
                        selected = vars['camera'].get()
                        step = {'type': kind, 'mode': 'manual' if vars['mode'].get().startswith('Manual') else 'auto',
                                'camera': choices[selected], 'camera_name': selected.split(' · ')[0],
                                'settle': float(vars['settle'].get())}
                    result['step'] = validate_step(step)
                except (ValueError, KeyError, TypeError) as exc:
                    messagebox.showerror('Instrucción inválida', str(exc), parent=dialog)
                    return
            dialog.destroy()
        buttons = ttk.Frame(body)
        buttons.pack(fill='x')
        ttk.Button(buttons, text='Cancelar', command=lambda: close(False)).pack(side='right', padx=5)
        ttk.Button(buttons, text='Guardar instrucción', style='Primary.TButton', command=lambda: close(True)).pack(side='right', padx=5)
        dialog.protocol('WM_DELETE_WINDOW', lambda: close(False))
        dialog.bind('<Escape>', lambda _e: close(False))
        dialog.update_idletasks()
        dialog.geometry(f'+{max(0,self.winfo_rootx()+(self.winfo_width()-dialog.winfo_width())//2)}+{max(0,self.winfo_rooty()+(self.winfo_height()-dialog.winfo_height())//2)}')
        dialog.grab_set()
        kind_picker.focus_set()
        self.wait_window(dialog)
        return result['step']

    def _start_sequence(self):
        if self._sequence_busy(): return
        if not self.active_profile:
            messagebox.showwarning('Secuencia', 'Selecciona un perfil.'); return
        if self._engineering_view():
            if self.sequence_dirty and not self._seq_save(): return
            steps = list(self.sequence_steps)
        else:
            try: steps = self._sequence_store().load()
            except Exception as exc:
                messagebox.showerror('Secuencia', str(exc)); return
        if not steps:
            messagebox.showwarning('Secuencia', 'Este perfil aún no tiene una secuencia guardada.'); return
        if self.busy:
            messagebox.showwarning('Proceso activo', 'Finaliza las otras operaciones antes de iniciar una secuencia.'); return
        if any(s['type'] in ('motor', 'led') for s in steps):
            if not self._esp_connected or self._esp_confirmed_endpoint != self._esp_endpoint():
                messagebox.showwarning('ESP32 sin conexión',
                    'Antes de ejecutar motores o iluminación, pulsa «Conectar ESP32» '
                    'y comprueba que el indicador esté verde.')
                return
        try:
            address = self.esp32_ip.get().strip()
            port = int(self.esp32_port.get())
            if not 0 < port < 65536 or not address:
                raise ValueError('Dirección IP o puerto incorrectos.')
            model = self._get_inspection_model()
            if any(s['type'] == 'camera' for s in steps) and not model:
                raise ValueError('Selecciona un modelo YOLO o autoriza uno para producción.')
            # Check local path before moving motors.
            if any(s['type'] == 'camera' for s in steps):
                from pathlib import Path
                if not Path(model).is_file():
                    raise FileNotFoundError(f'No existe modelo YOLO: {model}')
        except Exception as exc:
            messagebox.showerror('Secuencia', str(exc)); return
        try:
            self._sequence_camera_options = self._camera_options()
            self._sequence_count_config = (float(self.test_conf.get()),float(self.test_iou.get()),int(self.test_maxdet.get()))
            if not (0 < self._sequence_count_config[0] < 1 and 0 < self._sequence_count_config[1] < 1 and self._sequence_count_config[2] >= 1):
                raise ValueError('Confianza, IoU o máximo de detecciones inválidos.')
            self._sequence_test_mode = self.test_mode.get()
        except Exception as exc:
            messagebox.showerror('Secuencia',str(exc)); return
        self._stop_live_inference()
        self._sequence_hardware_used = any(s['type'] in ('motor','led') for s in steps)
        self._sequence_stop_event = threading.Event()
        self._sequence_manual_event = threading.Event()
        self._sequence_running = True
        self._sequence_auto_model = model
        if self._engineering_view() and hasattr(self, 'inspection_tab'):
            self.nb.select(self.inspection_tab)
        self.sequence_run_state.set(f'Iniciando secuencia de {len(steps)} instrucciones…')
        def runner():
            link = ESP32Link(address, port)
            try:
                for i, step in enumerate(steps, 1):
                    if self._sequence_stop_event.is_set():
                        raise InterruptedError('Secuencia detenida.')
                    self._emit('sequence_progress', f'Paso {i}/{len(steps)}: {KIND_LABELS[step["type"]]} — {describe(step)}')
                    if step['type'] == 'motor':
                        # OK = accepted, never means motor finished. Stop + margin later.
                        link.motor(step['motor'], step['direction'], step['pwm'], step['duration'])
                        if self._sequence_stop_event.wait(step['duration'] + 1.3):
                            raise InterruptedError('Secuencia detenida.')
                        link.stop()
                    elif step['type'] == 'led':
                        link.rgb(*(step['rgb'] if step['action'] == 'on' else (0, 0, 0)))
                    else:
                        self._sequence_run_camera(step, model)
                self._emit('sequence_complete', 'Secuencia completada.')
            except InterruptedError as exc:
                self._emit('sequence_complete', str(exc))
            except Exception as exc:
                self._emit('sequence_error', str(exc))
            finally:
                # Best effort: no software stop replaces a hardware emergency stop.
                if self._sequence_hardware_used:
                    try: link.stop()
                    except Exception as exc: self._emit('sequence_safety_error', str(exc))
                    try: link.rgb(0, 0, 0)
                    except Exception: pass
                self._emit('sequence_finished', None)
        threading.Thread(target=runner, daemon=True).start()

    def _sequence_run_camera(self, step, model):
        device_index=step['camera']
        # El nombre ayuda a resolver cambios en el orden de los índices USB.
        known_name=str(step.get('camera_name','')).strip()
        if known_name:
            matches=[c for c in self.available_cameras if c.get('name')==known_name]
            if matches:
                device_index=next((c['index'] for c in matches if c['index']==device_index), matches[0]['index'])
        self._camera_stream.start(device_index, *self._sequence_camera_options)
        self._camera_context = 'test'
        self._camera_stream.wait_frame(cancel=self._sequence_stop_event)
        if self._sequence_stop_event.wait(step['settle']):
            raise InterruptedError('Secuencia detenida.')
        if step['mode'] == 'manual':
            self._sequence_manual_event.clear()
            self._emit('sequence_manual', {'camera': device_index, 'model': model})
            while not self._sequence_manual_event.wait(.1):
                if self._sequence_stop_event.is_set():
                    raise InterruptedError('Secuencia detenida.')
            if self._sequence_stop_event.is_set():
                raise InterruptedError('Secuencia detenida.')
            self._emit('sequence_manual_end', None)
        frame = self._camera_stream.wait_frame(cancel=self._sequence_stop_event)
        result = self._inspect_image(frame, model)
        self._emit('sequence_count', (frame, result))
        # En ambas modalidades se libera la cámara al terminar esta instrucción.
        self._camera_stream.stop()
        self._camera_context = None
        if result.get('blocked'):
            raise RuntimeError('Calibración ArUco obligatoria inválida. Inspección bloqueada; secuencia detenida.')

    def _sequence_continue(self):
        if not self._sequence_running: return
        self._stop_live_inference()
        self.sequence_continue_btn.grid_remove()
        self._sequence_manual_event.set()

    def _stop_sequence(self):
        if not getattr(self, '_sequence_running', False): return
        self._sequence_stop_event.set()
        self._sequence_manual_event.set()
        self._stop_live_inference()
        self.sequence_run_state.set('Deteniendo secuencia…')
        # Send emergency stop independently of worker wait. Best effort on TCP.
        if self._sequence_hardware_used:
            address, port = self.esp32_ip.get().strip(), int(self.esp32_port.get())
            def emergency():
                try: ESP32Link(address, port, 2).stop()
                except Exception as exc: self._emit('sequence_safety_error', str(exc))
            threading.Thread(target=emergency, daemon=True).start()
