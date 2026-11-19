from __future__ import annotations
import json
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

from .project_core import (
    CLASS_NAMES, find_images, read_annotations, save_annotations,
    infer_folder_class, detect_grayscale_boxes, load_label_objects,
    clear_augmentation_output, clear_yolo_export_output,
)
from .augment_core import generate_augmentations
from .yolo_core import train_yolo, predict_image, export_yolo_dataset
from .camera_utils import scan_cameras, capture_camera


class KountApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Kount-A | Dataset, BBox, Augmentación y YOLO v14')
        self.geometry('1500x890')
        self.minsize(1120, 720)
        self.message_queue = queue.Queue()
        self.project_root = Path(__file__).resolve().parent.parent
        self.session_path = self.project_root / 'ui_session_config.json'
        self.input_var = tk.StringVar(value=str(self.project_root / 'input'))
        self.output_var = tk.StringVar(value=str(self.project_root / 'output'))
        self.status_var = tk.StringVar(value='Selecciona las carpetas input y output.')
        self.current_image: Path | None = None
        self.current_pil: Image.Image | None = None
        self.current_objects: list[dict] = []
        self.selected_bbox = -1
        self.dirty = False
        self.image_files: list[Path] = []
        self.editor_view = {}
        self.preview_refs = {}
        self.aug_orig_files: list[Path] = []
        self.aug_generated_files: list[Path] = []
        self.aug_current_original = None
        self.aug_current_aug = None
        self.test_image = None
        self.test_detections = []
        self.available_cameras: list[dict] = []
        self.busy = set()
        self._mouse_start = None
        self._rubber_id = None
        self._configure_vars()
        self._build_ui()
        self._load_session()
        self.after(100, self._poll_queue)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    def _configure_vars(self):
        self.bbox_class = tk.StringVar(value=CLASS_NAMES[0])
        self.draw_mode = tk.StringVar(value='Nueva')
        self.gray_threshold = tk.StringVar(value='225')
        self.gray_min_area = tk.StringVar(value='150')
        self.auto_keep = tk.BooleanVar(value=False)
        self.prelabel_model = tk.StringVar()
        self.prelabel_conf = tk.StringVar(value='0.40')
        self.aug_copies = tk.StringVar(value='10')
        self.aug_seed = tk.StringVar(value='42')
        self.aug_flip = tk.BooleanVar(value=True)
        self.aug_brightness = tk.StringVar(value='0.30')
        self.aug_contrast = tk.StringVar(value='0.20')
        self.aug_saturation = tk.StringVar(value='0.08')
        self.aug_hue = tk.StringVar(value='0.01')
        self.aug_snp_prob = tk.StringVar(value='0.30')
        self.aug_snp_amount = tk.StringVar(value='0.003')
        self.aug_gaussian_prob = tk.StringVar(value='0.15')
        self.aug_gaussian_sigma = tk.StringVar(value='4.0')
        self.train_base = tk.StringVar(value='yolo11n.pt')
        self.train_epochs = tk.StringVar(value='35')
        self.train_batch = tk.StringVar(value='4')
        self.train_imgsz = tk.StringVar(value='640')
        self.train_lr = tk.StringVar(value='0.001')
        self.train_patience = tk.StringVar(value='8')
        self.train_seed = tk.StringVar(value='42')
        self.train_val = tk.StringVar(value='0.20')
        self.train_test = tk.StringVar(value='0.10')
        self.train_aug = tk.BooleanVar(value=True)
        self.train_mosaic = tk.StringVar(value='0.25')
        self.test_model = tk.StringVar()
        self.test_source = tk.StringVar(value='Sin imagen seleccionada.')
        self.test_conf = tk.StringVar(value='0.50')
        self.test_iou = tk.StringVar(value='0.45')
        self.test_maxdet = tk.StringVar(value='300')
        self.camera_choice = tk.StringVar(value='Presiona Buscar cámaras')
        self.camera_width = tk.StringVar(value='1280')
        self.camera_height = tk.StringVar(value='720')

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill='both', expand=True)
        top = ttk.Frame(root)
        top.pack(fill='x', pady=(0, 8))
        ttk.Label(top, text='Input de fotografías').grid(row=0, column=0, sticky='w')
        ttk.Entry(top, textvariable=self.input_var).grid(row=0, column=1, sticky='ew', padx=5)
        ttk.Button(top, text='Examinar', command=self._choose_input).grid(row=0, column=2)
        ttk.Label(top, text='Output de resultados').grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(top, textvariable=self.output_var).grid(row=1, column=1, sticky='ew', padx=5)
        ttk.Button(top, text='Examinar', command=self._choose_output).grid(row=1, column=2)
        ttk.Button(top, text='Recargar imágenes', command=self._refresh_images).grid(row=0, column=3, rowspan=2, padx=12)
        top.columnconfigure(1, weight=1)
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill='both', expand=True)
        tabs = [ttk.Frame(self.nb, padding=6) for _ in range(4)]
        for tab, name in zip(tabs, ('1 · BBox y etiquetado', '2 · Augmentación y revisión',
                                    '3 · Entrenamiento YOLO', '4 · Prueba y conteo')):
            self.nb.add(tab, text=name)
        self._build_bbox_tab(tabs[0])
        self._build_augment_tab(tabs[1])
        self._build_train_tab(tabs[2])
        self._build_test_tab(tabs[3])
        ttk.Label(root, textvariable=self.status_var, anchor='w').pack(fill='x', pady=(6, 0))

    def _build_bbox_tab(self, tab):
        tab.columnconfigure(0, weight=2)
        tab.columnconfigure(1, weight=7)
        tab.columnconfigure(2, weight=3)
        tab.rowconfigure(0, weight=1)
        left = ttk.LabelFrame(tab, text='Imágenes originales', padding=6)
        left.grid(row=0, column=0, sticky='nsew', padx=3)
        self.orig_list = tk.Listbox(left, exportselection=False, width=30)
        self.orig_list.pack(fill='both', expand=True, side='left')
        s = ttk.Scrollbar(left, orient='vertical', command=self.orig_list.yview)
        s.pack(fill='y', side='left')
        self.orig_list.configure(yscrollcommand=s.set)
        self.orig_list.bind('<<ListboxSelect>>', self._on_image_select)

        center = ttk.LabelFrame(tab, text='Arrastra con el mouse para dibujar la BBox', padding=6)
        center.grid(row=0, column=1, sticky='nsew', padx=3)
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)
        self.editor_canvas = tk.Canvas(center, bg='#282C32', highlightthickness=0, cursor='crosshair')
        self.editor_canvas.grid(row=0, column=0, sticky='nsew')
        self.editor_canvas.bind('<Configure>', lambda _e: self._render_editor())
        self.editor_canvas.bind('<ButtonPress-1>', self._bbox_mouse_down)
        self.editor_canvas.bind('<B1-Motion>', self._bbox_mouse_drag)
        self.editor_canvas.bind('<ButtonRelease-1>', self._bbox_mouse_up)
        self.editor_info = tk.StringVar(value='Selecciona una foto para etiquetarla.')
        ttk.Label(center, textvariable=self.editor_info, wraplength=650).grid(row=1, column=0, sticky='w', pady=6)

        right = ttk.Frame(tab)
        right.grid(row=0, column=2, sticky='nsew', padx=3)
        right.columnconfigure(0, weight=1)
        pane = ttk.LabelFrame(right, text='Edición manual', padding=8)
        pane.grid(row=0, column=0, sticky='ew')
        ttk.Label(pane, text='Clase de la BBox').pack(anchor='w')
        ttk.Combobox(pane, textvariable=self.bbox_class, values=CLASS_NAMES, state='readonly').pack(fill='x', pady=3)
        ttk.Label(pane, text='Modo al arrastrar').pack(anchor='w', pady=(7, 0))
        ttk.Combobox(pane, textvariable=self.draw_mode, values=['Nueva', 'Reemplazar seleccionada'],
                     state='readonly').pack(fill='x', pady=3)
        btnrow = ttk.Frame(pane)
        btnrow.pack(fill='x', pady=4)
        ttk.Button(btnrow, text='Aplicar clase', command=self._change_selected_class).pack(side='left')
        ttk.Button(btnrow, text='Eliminar BBox', command=self._delete_selected).pack(side='left', padx=5)
        ttk.Button(pane, text='Guardar BBox de esta imagen', command=self._save_current).pack(fill='x', pady=5)

        auto = ttk.LabelFrame(right, text='Detección auxiliar: escala de grises', padding=8)
        auto.grid(row=1, column=0, sticky='ew', pady=(8, 0))
        row = ttk.Frame(auto); row.pack(fill='x')
        ttk.Label(row, text='Umbral').pack(side='left')
        ttk.Entry(row, width=6, textvariable=self.gray_threshold).pack(side='left', padx=4)
        ttk.Label(row, text='Área mínima').pack(side='left')
        ttk.Entry(row, width=7, textvariable=self.gray_min_area).pack(side='left', padx=4)
        ttk.Checkbutton(auto, text='Conservar cajas existentes', variable=self.auto_keep).pack(anchor='w', pady=3)
        ttk.Button(auto, text='Proponer BBox en esta imagen', command=self._auto_gray_current).pack(fill='x', pady=3)
        ttk.Button(auto, text='Auto gris en fotos sin etiqueta (lote)', command=self._auto_gray_batch).pack(fill='x', pady=3)

        pre = ttk.LabelFrame(right, text='Pre-etiquetar con modelo YOLO anterior', padding=8)
        pre.grid(row=2, column=0, sticky='ew', pady=(8, 0))
        ttk.Entry(pre, textvariable=self.prelabel_model).pack(fill='x')
        ttk.Button(pre, text='Seleccionar best.pt', command=lambda: self._choose_model(self.prelabel_model)).pack(fill='x', pady=3)
        line = ttk.Frame(pre); line.pack(fill='x')
        ttk.Label(line, text='Confianza').pack(side='left')
        ttk.Entry(line, textvariable=self.prelabel_conf, width=8).pack(side='left', padx=6)
        ttk.Button(pre, text='Pre-etiquetar foto seleccionada', command=self._start_prelabel).pack(fill='x', pady=5)

        boxes = ttk.LabelFrame(right, text='Objetos de la fotografía', padding=6)
        boxes.grid(row=3, column=0, sticky='nsew', pady=(8, 0))
        right.rowconfigure(3, weight=1)
        self.box_tree = ttk.Treeview(boxes, columns=('clase', 'xyxy'), show='headings', height=7)
        self.box_tree.heading('clase', text='Clase')
        self.box_tree.heading('xyxy', text='BBox XYXY')
        self.box_tree.column('clase', width=130)
        self.box_tree.column('xyxy', width=175)
        self.box_tree.pack(fill='both', expand=True)
        self.box_tree.bind('<<TreeviewSelect>>', self._on_box_select)

    def _build_augment_tab(self, tab):
        tab.columnconfigure(0, weight=2)
        tab.columnconfigure(1, weight=5)
        tab.columnconfigure(2, weight=5)
        tab.rowconfigure(0, weight=1)
        settings = ttk.LabelFrame(tab, text='Configuración de augmentation', padding=9)
        settings.grid(row=0, column=0, sticky='nsew', padx=3)
        entries = [
            ('Copias por fotografía', self.aug_copies), ('Seed', self.aug_seed),
            ('Brillo (0–1)', self.aug_brightness), ('Contraste (0–1)', self.aug_contrast),
            ('Saturación (0–1)', self.aug_saturation), ('Hue (0–0.5)', self.aug_hue),
            ('Prob. sal/pimienta', self.aug_snp_prob), ('Cantidad sal/pimienta', self.aug_snp_amount),
            ('Prob. ruido gaussiano', self.aug_gaussian_prob), ('Sigma gaussiana', self.aug_gaussian_sigma),
        ]
        for i, (label, var) in enumerate(entries):
            ttk.Label(settings, text=label).grid(row=i, column=0, sticky='w', pady=4)
            ttk.Entry(settings, textvariable=var, width=9).grid(row=i, column=1, sticky='ew', pady=4, padx=4)
        settings.columnconfigure(1, weight=1)
        ttk.Checkbutton(settings, text='Flip horizontal', variable=self.aug_flip).grid(row=len(entries), column=0, columnspan=2, sticky='w', pady=5)
        ttk.Button(settings, text='Generar imágenes aumentadas', command=self._start_augment).grid(row=len(entries)+1, column=0, columnspan=2, sticky='ew', pady=6)
        self.aug_progress = ttk.Progressbar(settings, maximum=100)
        self.aug_progress.grid(row=len(entries)+2, column=0, columnspan=2, sticky='ew', pady=4)
        self.aug_summary = tk.StringVar(value='Las BBox se transforman junto con las fotografías.')
        ttk.Label(settings, textvariable=self.aug_summary, wraplength=260).grid(row=len(entries)+3, column=0, columnspan=2, sticky='w', pady=8)
        ttk.Button(settings, text='Recargar galerías', command=self._refresh_augment_lists).grid(row=len(entries)+4, column=0, columnspan=2, sticky='ew')
        ttk.Button(settings, text='Borrar imágenes aumentadas', command=self._start_clear_augmented).grid(
            row=len(entries)+5, column=0, columnspan=2, sticky='ew', pady=(8, 0))
        self.aug_orig_list, self.aug_orig_canvas = self._gallery_panel(tab, 1, 'Original y sus BBox', self._aug_orig_select)
        self.aug_gen_list, self.aug_gen_canvas = self._gallery_panel(tab, 2, 'Aumentada y sus BBox', self._aug_generated_select)

    def _gallery_panel(self, parent, column, title, callback):
        frame = ttk.LabelFrame(parent, text=title, padding=6)
        frame.grid(row=0, column=column, sticky='nsew', padx=3)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=2)
        frame.rowconfigure(1, weight=5)
        lst = tk.Listbox(frame, height=8, exportselection=False)
        lst.grid(row=0, column=0, sticky='nsew', pady=(0, 4))
        lst.bind('<<ListboxSelect>>', callback)
        canvas = tk.Canvas(frame, bg='#282C32', highlightthickness=0)
        canvas.grid(row=1, column=0, sticky='nsew')
        canvas.bind('<Configure>', lambda _e: self._rerender_gallery(canvas))
        return lst, canvas

    def _build_train_tab(self, tab):
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=6)
        tab.rowconfigure(0, weight=1)
        options = ttk.LabelFrame(tab, text='Entrenamiento / Fine-tuning de YOLO', padding=12)
        options.grid(row=0, column=0, sticky='nsew', padx=4)
        ttk.Label(options, text='Modelo inicial: yolo11n.pt o best.pt de YOLO anterior', wraplength=330).grid(row=0, column=0, columnspan=2, sticky='w')
        ttk.Entry(options, textvariable=self.train_base).grid(row=1, column=0, sticky='ew', pady=5)
        ttk.Button(options, text='Buscar .pt', command=lambda: self._choose_model(self.train_base)).grid(row=1, column=1)
        params = [
            ('Épocas', self.train_epochs), ('Batch', self.train_batch),
            ('Resolución imgsz', self.train_imgsz), ('Learning rate', self.train_lr),
            ('Early stopping (patience)', self.train_patience), ('Seed', self.train_seed),
            ('Validación', self.train_val), ('Prueba (test)', self.train_test),
            ('Mosaic YOLO (0–1)', self.train_mosaic),
        ]
        for i, (label, var) in enumerate(params, 2):
            ttk.Label(options, text=label).grid(row=i, column=0, sticky='w', pady=5)
            ttk.Entry(options, textvariable=var, width=12).grid(row=i, column=1, sticky='w', padx=6)
        ttk.Checkbutton(options, text='Usar aumentadas SOLO en train', variable=self.train_aug).grid(row=11, column=0, columnspan=2, sticky='w', pady=7)
        ttk.Button(options, text='Exportar dataset YOLO', command=self._start_export).grid(row=12, column=0, columnspan=2, sticky='ew', pady=5)
        ttk.Button(options, text='Entrenar modelo', command=self._start_training).grid(row=13, column=0, columnspan=2, sticky='ew', pady=5)
        ttk.Label(options, text='Los .pth de Faster R-CNN no son pesos YOLO.\nNo se modifica tu entrenamiento antiguo.', wraplength=310).grid(row=14, column=0, columnspan=2, sticky='w', pady=6)
        ttk.Button(options, text='Borrar dataset YOLO exportado', command=self._start_clear_yolo_export).grid(
            row=15, column=0, columnspan=2, sticky='ew', pady=(12, 4))
        ttk.Label(options, text='Borra yolo_dataset y split_assignments.json; conserva modelos y sus métricas.',
                  wraplength=310).grid(row=16, column=0, columnspan=2, sticky='w', pady=3)
        options.columnconfigure(0, weight=1)

        right = ttk.LabelFrame(tab, text='Avance, métricas y archivos del modelo', padding=10)
        right.grid(row=0, column=1, sticky='nsew', padx=4)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)
        self.train_progress = ttk.Progressbar(right, maximum=100)
        self.train_progress.grid(row=0, column=0, sticky='ew', pady=8)
        self.train_state = tk.StringVar(value='Listo para exportar o entrenar.')
        ttk.Label(right, textvariable=self.train_state, wraplength=780).grid(row=1, column=0, sticky='w', pady=4)
        self.train_log = tk.Text(right, wrap='word', state='disabled')
        self.train_log.grid(row=2, column=0, sticky='nsew', pady=5)
        ttk.Button(right, text='Usar el último best.pt en Prueba y pre-etiquetado',
                   command=self._copy_latest_model).grid(row=3, column=0, sticky='w', pady=6)
        self.last_trained = None

    def _build_test_tab(self, tab):
        tab.columnconfigure(0, weight=7)
        tab.columnconfigure(1, weight=3)
        tab.rowconfigure(1, weight=1)
        toolbar = ttk.LabelFrame(tab, text='Modelo, imágenes y cámaras', padding=9)
        toolbar.grid(row=0, column=0, columnspan=2, sticky='ew', pady=5)
        ttk.Label(toolbar, text='Modelo YOLO').grid(row=0, column=0, sticky='w')
        ttk.Entry(toolbar, textvariable=self.test_model, width=45).grid(row=0, column=1, columnspan=3, sticky='ew', padx=4)
        ttk.Button(toolbar, text='Buscar best.pt', command=lambda: self._choose_model(self.test_model)).grid(row=0, column=4)
        ttk.Button(toolbar, text='Abrir fotografía', command=self._test_open).grid(row=1, column=0, pady=5)
        ttk.Button(toolbar, text='Contar objetos', command=self._start_test_count).grid(row=1, column=1, pady=5)
        ttk.Label(toolbar, text='Conf.').grid(row=1, column=2)
        ttk.Entry(toolbar, textvariable=self.test_conf, width=7).grid(row=1, column=3, sticky='w')
        ttk.Label(toolbar, text='IoU NMS').grid(row=1, column=4)
        ttk.Entry(toolbar, textvariable=self.test_iou, width=7).grid(row=1, column=5, sticky='w')
        ttk.Label(toolbar, text='Máx.').grid(row=1, column=6)
        ttk.Entry(toolbar, textvariable=self.test_maxdet, width=6).grid(row=1, column=7, sticky='w')
        ttk.Separator(toolbar, orient='horizontal').grid(row=2, column=0, columnspan=8, sticky='ew', pady=7)
        ttk.Button(toolbar, text='Buscar cámaras conectadas', command=self._start_camera_scan).grid(row=3, column=0)
        self.camera_combo = ttk.Combobox(toolbar, textvariable=self.camera_choice, state='readonly', width=40)
        self.camera_combo.grid(row=3, column=1, columnspan=3, sticky='ew', padx=4)
        ttk.Label(toolbar, text='W').grid(row=3, column=4)
        ttk.Entry(toolbar, textvariable=self.camera_width, width=7).grid(row=3, column=5)
        ttk.Label(toolbar, text='H').grid(row=3, column=6)
        ttk.Entry(toolbar, textvariable=self.camera_height, width=7).grid(row=3, column=7)
        ttk.Button(toolbar, text='Capturar foto', command=lambda: self._start_capture(False)).grid(row=4, column=0, pady=5)
        ttk.Button(toolbar, text='Capturar y contar', command=lambda: self._start_capture(True)).grid(row=4, column=1, pady=5)
        ttk.Label(toolbar, textvariable=self.test_source, wraplength=900).grid(row=4, column=2, columnspan=6, sticky='w')
        toolbar.columnconfigure(1, weight=1)
        show = ttk.LabelFrame(tab, text='Imagen y detecciones', padding=6)
        show.grid(row=1, column=0, sticky='nsew', padx=3)
        show.rowconfigure(0, weight=1)
        show.columnconfigure(0, weight=1)
        self.test_canvas = tk.Canvas(show, bg='#282C32', highlightthickness=0)
        self.test_canvas.grid(row=0, column=0, sticky='nsew')
        self.test_canvas.bind('<Configure>', lambda _e: self._rerender_test())
        result = ttk.LabelFrame(tab, text='Conteo por clase y detalles', padding=10)
        result.grid(row=1, column=1, sticky='nsew', padx=3)
        result.rowconfigure(1, weight=1)
        result.columnconfigure(0, weight=1)
        self.test_summary = tk.StringVar(value='Sin conteo.')
        ttk.Label(result, textvariable=self.test_summary, font=('Segoe UI', 12, 'bold'),
                  wraplength=360).grid(row=0, column=0, sticky='w', pady=5)
        self.test_details = tk.Text(result, wrap='word', state='disabled', width=36)
        self.test_details.grid(row=1, column=0, sticky='nsew')

    def _get_paths(self):
        raw_in, raw_out = self.input_var.get().strip(), self.output_var.get().strip()
        if not raw_in or not raw_out:
            raise ValueError('Selecciona las carpetas input y output.')
        input_dir, output_dir = Path(raw_in).resolve(), Path(raw_out).resolve()
        if not input_dir.is_dir():
            raise FileNotFoundError(f'Input no existe: {input_dir}')
        if (input_dir == output_dir or input_dir.is_relative_to(output_dir)
            or output_dir.is_relative_to(input_dir)):
            raise ValueError('Output no puede contener el input.')
        output_dir.mkdir(parents=True, exist_ok=True)
        return input_dir, output_dir

    def _choose_input(self):
        if not self._resolve_dirty():
            return
        path = filedialog.askdirectory(title='Carpeta de imágenes originales')
        if path:
            self.input_var.set(path)
            self._refresh_images()

    def _choose_output(self):
        if not self._resolve_dirty():
            return
        path = filedialog.askdirectory(title='Carpeta output')
        if path:
            self.output_var.set(path)
            self._refresh_images()

    def _refresh_images(self):
        try:
            input_dir, _ = self._get_paths()
            self.image_files = find_images(input_dir)
            self.orig_list.delete(0, 'end')
            for p in self.image_files:
                self.orig_list.insert('end', str(p.relative_to(input_dir)))
            self._refresh_augment_lists()
            self.status_var.set(f'{len(self.image_files)} fotografías originales encontradas.')
        except Exception as exc:
            self.status_var.set(str(exc))

    def _resolve_dirty(self):
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel('Cambios sin guardar', 'La BBox de esta foto tiene cambios. ¿Guardar antes de continuar?')
        if answer is None:
            return False
        return self._save_current() if answer else True

    def _on_image_select(self, _event=None):
        selected = self.orig_list.curselection()
        if not selected:
            return
        path = self.image_files[selected[0]]
        if self.current_image == path:
            return
        if not self._resolve_dirty():
            return
        try:
            input_dir, output_dir = self._get_paths()
            with Image.open(path) as im:
                image = im.convert('RGB').copy()
            self.current_image = path
            self.current_pil = image
            self.current_objects = read_annotations(input_dir, path, output_dir)
            self.bbox_class.set(infer_folder_class(path, input_dir))
            self.dirty = False
            self.selected_bbox = -1
            self._update_box_tree()
            self._render_editor()
        except Exception as exc:
            messagebox.showerror('Imagen', str(exc))

    def _save_current(self):
        if self.current_image is None:
            return True
        try:
            input_dir, _ = self._get_paths()
            saved = save_annotations(input_dir, self.current_image, self.current_objects)
            self.dirty = False
            self.status_var.set(f'Anotaciones guardadas: {saved}')
            return True
        except Exception as exc:
            messagebox.showerror('Guardar BBox', str(exc))
            return False

    def _render_image(self, canvas: tk.Canvas, image: Image.Image | None,
                      objects: list[dict] | None, key: str, selected: int = -1):
        canvas.delete('all')
        if image is None:
            canvas.create_text(max(40, canvas.winfo_width() // 2), max(40, canvas.winfo_height() // 2),
                               text='Sin imagen', fill='white')
            return None
        width, height = image.size
        cw, ch = max(20, canvas.winfo_width()), max(20, canvas.winfo_height())
        scale = min((cw-16)/width, (ch-16)/height)
        nw, nh = max(1, round(width*scale)), max(1, round(height*scale))
        left, top = (cw-nw)//2, (ch-nh)//2
        im = image.resize((nw, nh), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(im)
        self.preview_refs[key] = photo
        canvas.create_image(left, top, image=photo, anchor='nw')
        colors = ['#EC4B47', '#449CFF', '#36D48C', '#F166CF', '#F4A844', '#9D89FF']
        for i, obj in enumerate(objects or []):
            try:
                x1, y1, x2, y2 = obj['bbox_xyxy']
                color = '#FFD53F' if i == selected else colors[CLASS_NAMES.index(obj['class_name'])]
                sx1, sy1 = left+x1*scale, top+y1*scale
                sx2, sy2 = left+x2*scale, top+y2*scale
                canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=3 if i==selected else 2)
                txt = f'{i+1}. {obj["class_name"]}'
                label_y = max(top, sy1-19)
                rid = canvas.create_rectangle(sx1, label_y, sx1+len(txt)*7+8, label_y+18,
                                              fill=color, outline=color)
                canvas.create_text(sx1+4, label_y+9, text=txt, fill='#151515', anchor='w')
            except (KeyError, ValueError, IndexError):
                continue
        return {'scale': scale, 'x0': left, 'y0': top, 'w': width, 'h': height}

    def _render_editor(self):
        self.editor_view = self._render_image(self.editor_canvas, self.current_pil,
                                              self.current_objects, 'editor', self.selected_bbox) or {}
        if self.current_image and self.current_pil:
            self.editor_info.set(f'{self.current_image.name} · {self.current_pil.width}×{self.current_pil.height} '
                                 f'· {len(self.current_objects)} objeto(s)'
                                 + (' · CAMBIOS SIN GUARDAR' if self.dirty else ' · guardado/sin cambios'))

    def _screen_to_image(self, x: float, y: float):
        v = self.editor_view
        if not v:
            return None
        ox, oy, scale = v['x0'], v['y0'], v['scale']
        if x < ox or y < oy or x > ox+v['w']*scale or y > oy+v['h']*scale:
            return None
        return (max(0, min(v['w'], (x-ox)/scale)), max(0, min(v['h'], (y-oy)/scale)))

    def _bbox_mouse_down(self, event):
        point = self._screen_to_image(event.x, event.y)
        if point is None:
            return
        self._mouse_start = point
        self._rubber_id = self.editor_canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                                outline='#FFE46B', dash=(4, 3), width=2)

    def _bbox_mouse_drag(self, event):
        if self._mouse_start is None or self._rubber_id is None:
            return
        v = self.editor_view
        start = self._mouse_start
        end = self._screen_to_image(event.x, event.y)
        if end is None:
            end = (max(0, min(v['w'], (event.x-v['x0'])/v['scale'])),
                   max(0, min(v['h'], (event.y-v['y0'])/v['scale'])))
        x1, y1 = v['x0']+start[0]*v['scale'], v['y0']+start[1]*v['scale']
        x2, y2 = v['x0']+end[0]*v['scale'], v['y0']+end[1]*v['scale']
        self.editor_canvas.coords(self._rubber_id, x1, y1, x2, y2)

    def _bbox_mouse_up(self, event):
        if self._mouse_start is None:
            return
        start = self._mouse_start
        self._mouse_start = None
        if self._rubber_id is not None:
            self.editor_canvas.delete(self._rubber_id)
            self._rubber_id = None
        end = self._screen_to_image(event.x, event.y)
        if end is None:
            v = self.editor_view
            end = (max(0, min(v['w'], (event.x-v['x0'])/v['scale'])),
                   max(0, min(v['h'], (event.y-v['y0'])/v['scale'])))
        x1, x2 = sorted((start[0], end[0]))
        y1, y2 = sorted((start[1], end[1]))
        if x2-x1 < 3 or y2-y1 < 3:
            # Clic simple sobre una caja permite seleccionarla.
            self.selected_bbox = -1
            for i in range(len(self.current_objects)-1, -1, -1):
                a, b, c, d = self.current_objects[i]['bbox_xyxy']
                if a <= end[0] <= c and b <= end[1] <= d:
                    self.selected_bbox = i
                    break
            self._update_box_tree()
            self._render_editor()
            return
        obj = {'class_name': self.bbox_class.get(),
               'bbox_xyxy': [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]}
        if self.draw_mode.get() == 'Reemplazar seleccionada' and 0 <= self.selected_bbox < len(self.current_objects):
            self.current_objects[self.selected_bbox] = obj
        else:
            self.current_objects.append(obj)
            self.selected_bbox = len(self.current_objects)-1
        self.dirty = True
        self._update_box_tree()
        self._render_editor()

    def _update_box_tree(self):
        if not hasattr(self, 'box_tree'):
            return
        for item in self.box_tree.get_children():
            self.box_tree.delete(item)
        for i, obj in enumerate(self.current_objects):
            self.box_tree.insert('', 'end', iid=str(i), values=(obj['class_name'],
                                  ', '.join(str(round(float(v))) for v in obj['bbox_xyxy'])))
        if 0 <= self.selected_bbox < len(self.current_objects):
            self.box_tree.selection_set(str(self.selected_bbox))

    def _on_box_select(self, _event=None):
        ids = self.box_tree.selection()
        if not ids:
            return
        self.selected_bbox = int(ids[0])
        self.bbox_class.set(self.current_objects[self.selected_bbox]['class_name'])
        self._render_editor()

    def _change_selected_class(self):
        if 0 <= self.selected_bbox < len(self.current_objects):
            self.current_objects[self.selected_bbox]['class_name'] = self.bbox_class.get()
            self.dirty = True
            self._update_box_tree()
            self._render_editor()

    def _delete_selected(self):
        if 0 <= self.selected_bbox < len(self.current_objects):
            self.current_objects.pop(self.selected_bbox)
            self.selected_bbox = -1
            self.dirty = True
            self._update_box_tree()
            self._render_editor()

    def _auto_gray_current(self):
        if self.current_pil is None:
            return
        try:
            threshold, area = int(self.gray_threshold.get()), int(self.gray_min_area.get())
            if not (0 <= threshold <= 255 and area >= 1):
                raise ValueError('Umbral 0–255; área >= 1')
            boxes = detect_grayscale_boxes(self.current_pil, threshold, area)
            if not self.auto_keep.get() and self.current_objects:
                if not messagebox.askyesno('Sustituir BBox', '¿Sustituir todas las BBox de esta foto por las propuestas en gris?'):
                    return
                self.current_objects = []
            self.current_objects += [{'class_name': self.bbox_class.get(), 'bbox_xyxy': b} for b in boxes]
            self.dirty = True
            self.selected_bbox = -1
            self._update_box_tree()
            self._render_editor()
            self.status_var.set(f'Propuestas {len(boxes)} BBox. Revísalas y guarda manualmente.')
        except Exception as exc:
            messagebox.showerror('Umbral gris', str(exc))

    def _auto_gray_batch(self):
        if not messagebox.askyesno('Autodetección por lote',
                   'Solo se propondrán BBox para fotos sin anotaciones anteriores. '
                   'Debes revisar visualmente las etiquetas generadas. ¿Continuar?'):
            return
        if not self._resolve_dirty():
            return
        try:
            inp, out = self._get_paths()
            threshold, area = int(self.gray_threshold.get()), int(self.gray_min_area.get())
            files = list(self.image_files)
        except Exception as exc:
            messagebox.showerror('Autodetección', str(exc)); return
        def worker():
            try:
                updated, skipped = 0, 0
                for p in files:
                    if read_annotations(inp, p, out):
                        skipped += 1
                        continue
                    with Image.open(p) as im:
                        boxes = detect_grayscale_boxes(im, threshold, area)
                    if boxes:
                        cls = infer_folder_class(p, inp)
                        save_annotations(inp, p, [{'class_name': cls, 'bbox_xyxy': b} for b in boxes])
                        updated += 1
                self._emit('auto_batch_done', f'BBox automáticas propuestas para {updated} fotografías; {skipped} ya tenían anotaciones.')
            except Exception as exc:
                self._emit('error', str(exc))
        self._start_worker('auto_batch', worker)

    def _start_prelabel(self):
        if self.current_image is None:
            messagebox.showwarning('Pre-etiquetado', 'Selecciona primero una foto.'); return
        model_path = self.prelabel_model.get().strip()
        if not model_path or model_path.endswith('.pth'):
            messagebox.showerror('Pre-etiquetado', 'Selecciona un best.pt entrenado con YOLO.'); return
        if self.current_objects and not messagebox.askyesno('Reemplazar anotaciones', 'El modelo reemplazará las BBox de esta imagen. ¿Continuar?'):
            return
        image = self.current_pil.copy()
        confidence = float(self.prelabel_conf.get())
        current_path = self.current_image
        def worker():
            try:
                result = predict_image(image, model_path, conf=confidence)
                objects = [{'class_name': d['class_name'], 'bbox_xyxy': d['bbox_xyxy']}
                           for d in result['detections'] if d['class_name'] in CLASS_NAMES]
                self._emit('prelabel_done', {'path': current_path, 'objects': objects})
            except Exception as exc:
                self._emit('error', str(exc))
        self._start_worker('prelabel', worker)

    def _refresh_augment_lists(self):
        self.aug_orig_files = []
        self.aug_generated_files = []
        self.aug_orig_list.delete(0, 'end')
        self.aug_gen_list.delete(0, 'end')
        try:
            inp, out = self._get_paths()
        except Exception:
            return
        self.aug_orig_files = find_images(inp)
        for p in self.aug_orig_files:
            self.aug_orig_list.insert('end', p.relative_to(inp).as_posix())
        augdir = out / 'augmented_images'
        if augdir.exists():
            self.aug_generated_files = find_images(augdir)
            for p in self.aug_generated_files:
                self.aug_gen_list.insert('end', p.relative_to(augdir).as_posix())

    def _aug_orig_select(self, _event=None):
        indices = self.aug_orig_list.curselection()
        if not indices: return
        try:
            inp, out = self._get_paths()
            path = self.aug_orig_files[indices[0]]
            with Image.open(path) as im:
                img = im.convert('RGB').copy()
            self.aug_current_original = (img, read_annotations(inp, path, out))
            self._rerender_gallery(self.aug_orig_canvas)
        except Exception as exc:
            self.status_var.set(str(exc))

    def _aug_generated_select(self, _event=None):
        indices = self.aug_gen_list.curselection()
        if not indices: return
        try:
            _, out = self._get_paths()
            path = self.aug_generated_files[indices[0]]
            rel = path.relative_to(out / 'augmented_images')
            labels_path = (out / 'augmented_labels' / rel).with_suffix('.json')
            with Image.open(path) as im:
                img = im.convert('RGB').copy()
            self.aug_current_aug = (img, load_label_objects(labels_path))
            self._rerender_gallery(self.aug_gen_canvas)
        except Exception as exc:
            self.status_var.set(str(exc))

    def _rerender_gallery(self, canvas):
        selected = self.aug_current_original if canvas is self.aug_orig_canvas else self.aug_current_aug
        if selected:
            self._render_image(canvas, selected[0], selected[1],
                               'aug_original' if canvas is self.aug_orig_canvas else 'aug_generated')

    def _start_augment(self):
        if not self._resolve_dirty(): return
        try:
            inp, out = self._get_paths()
            kwargs = dict(
                copies=int(self.aug_copies.get()), seed=int(self.aug_seed.get()),
                enable_flip=bool(self.aug_flip.get()), brightness=float(self.aug_brightness.get()),
                contrast=float(self.aug_contrast.get()), saturation=float(self.aug_saturation.get()),
                hue=float(self.aug_hue.get()), salt_pepper_prob=float(self.aug_snp_prob.get()),
                salt_pepper_amount=float(self.aug_snp_amount.get()),
                gaussian_prob=float(self.aug_gaussian_prob.get()),
                gaussian_sigma=float(self.aug_gaussian_sigma.get()),
            )
        except Exception as exc:
            messagebox.showerror('Augmentation', str(exc)); return
        if not messagebox.askyesno('Regenerar aumentadas', 'Se reemplazarán únicamente las imágenes aumentadas y sus etiquetas del output. ¿Continuar?'):
            return
        self.aug_progress.configure(value=0)
        def worker():
            try:
                report = generate_augmentations(inp, out, **kwargs,
                          progress=lambda state: self._emit('aug_progress', state))
                self._emit('aug_done', report)
            except Exception as exc:
                self._emit('error', f'Augmentation: {exc}')
        self._start_worker('augment', worker)

    def _can_cleanup_output(self) -> bool:
        # No borrar archivos que están siendo utilizados por otros procesos.
        if self.busy:
            messagebox.showwarning(
                'Proceso en curso',
                'Espera a que terminen la aumentación, exportación, entrenamiento o cualquier otra tarea antes de borrar archivos.')
            return False
        return True

    def _start_clear_augmented(self):
        if not self._can_cleanup_output():
            return
        try:
            _, out = self._get_paths()
        except Exception as exc:
            messagebox.showerror('Borrar aumentación', str(exc))
            return
        if not messagebox.askyesno(
            'Confirmar borrado de aumentación',
            f'Se borrarán SOLO estos elementos de:\n{out}\n\n'
            '• augmented_images/\n• augmented_labels/\n• augmentation_report.json\n\n'
            'Las fotos originales, BBox manuales, el dataset YOLO exportado y los modelos se conservarán.\n\n'
            '¿Eliminar permanentemente estos archivos?',
            icon='warning',
        ):
            return

        def worker():
            try:
                report = clear_augmentation_output(out)
                self._emit('clear_aug_done', report)
            except Exception as exc:
                self._emit('error', f'Borrar aumentación: {exc}')
        self._start_worker('clear_augmented', worker)

    def _start_clear_yolo_export(self):
        if not self._can_cleanup_output():
            return
        try:
            _, out = self._get_paths()
        except Exception as exc:
            messagebox.showerror('Borrar dataset YOLO', str(exc))
            return
        if not messagebox.askyesno(
            'Confirmar borrado del dataset YOLO',
            f'Se borrarán SOLO estos elementos de:\n{out}\n\n'
            '• yolo_dataset/ (copias train, val y test; labels; dataset.yaml)\n'
            '• split_assignments.json (asignación train/val/test)\n\n'
            'IMPORTANTE: al borrar el split, una futura exportación puede repartir las imágenes de otra forma. '
            'Para comparar modelos en la tesis, conserva una copia del JSON si necesitas repetir exactamente la partición.\n\n'
            'Los modelos .pt y reportes de entrenamiento NO se borrarán.\n\n'
            '¿Eliminar permanentemente estos archivos?',
            icon='warning',
        ):
            return

        def worker():
            try:
                report = clear_yolo_export_output(out)
                self._emit('clear_yolo_done', report)
            except Exception as exc:
                self._emit('error', f'Borrar dataset YOLO: {exc}')
        self._start_worker('clear_yolo', worker)

    def _train_parameters(self):
        inp, out = self._get_paths()
        params = dict(seed=int(self.train_seed.get()), val_frac=float(self.train_val.get()),
                      test_frac=float(self.train_test.get()),
                      use_augmented=bool(self.train_aug.get()))
        return inp, out, params

    def _start_export(self):
        if not self._resolve_dirty():
            return
        try:
            inp, out, params = self._train_parameters()
        except Exception as exc:
            messagebox.showerror('Dataset YOLO', str(exc)); return
        def worker():
            try:
                report = export_yolo_dataset(inp, out, **params,
                                progress=lambda m: self._emit('train_msg', m))
                self._emit('export_done', report)
            except Exception as exc:
                self._emit('error', f'Exportación YOLO: {exc}')
        self._start_worker('training', worker)

    def _start_training(self):
        if not self._resolve_dirty():
            return
        try:
            inp, out, params = self._train_parameters()
            params.update(base_model=self.train_base.get().strip(),
                          epochs=int(self.train_epochs.get()), batch=int(self.train_batch.get()),
                          imgsz=int(self.train_imgsz.get()), lr0=float(self.train_lr.get()),
                          patience=int(self.train_patience.get()),
                          mosaic=float(self.train_mosaic.get()))
        except Exception as exc:
            messagebox.showerror('Entrenamiento', str(exc)); return
        self.train_progress.configure(value=0)
        self.train_state.set('Preparando datos y modelo...')
        def worker():
            try:
                report = train_yolo(inp, out, **params,
                                    progress=lambda data: self._emit('train_progress', data))
                self._emit('train_done', report)
            except Exception as exc:
                self._emit('error', f'Entrenamiento YOLO: {exc}')
        self._start_worker('training', worker)

    def _append_train_log(self, text):
        self.train_log.configure(state='normal')
        self.train_log.insert('end', str(text).rstrip() + '\n')
        self.train_log.see('end')
        self.train_log.configure(state='disabled')

    def _choose_model(self, var):
        path = filedialog.askopenfilename(title='Seleccionar modelo YOLO', filetypes=[('Modelo YOLO', '*.pt'), ('Todos', '*.*')])
        if path:
            var.set(path)

    def _copy_latest_model(self):
        if not self.last_trained:
            messagebox.showinfo('Modelo', 'Todavía no has entrenado un modelo en esta sesión.')
            return
        self.test_model.set(self.last_trained)
        self.prelabel_model.set(self.last_trained)
        self.status_var.set('Modelo seleccionado en Prueba y BBox: ' + self.last_trained)

    def _test_open(self):
        path = filedialog.askopenfilename(title='Imagen de prueba', filetypes=[('Imágenes', '*.png *.jpg *.jpeg *.bmp *.webp'), ('Todas', '*.*')])
        if not path:
            return
        try:
            with Image.open(path) as im:
                self.test_image = im.convert('RGB').copy()
            self.test_detections = []
            self.test_source.set(path)
            self.test_summary.set('Imagen cargada. Pulsa Contar objetos.')
            self._rerender_test()
        except Exception as exc:
            messagebox.showerror('Imagen', str(exc))

    def _rerender_test(self):
        boxes = [{'class_name': d['class_name'], 'bbox_xyxy': d['bbox_xyxy']}
                 for d in self.test_detections if d['class_name'] in CLASS_NAMES]
        self._render_image(self.test_canvas, self.test_image, boxes, 'test')

    def _start_test_count(self):
        if self.test_image is None:
            messagebox.showinfo('Conteo', 'Selecciona una foto o captura una imagen de cámara.')
            return
        try:
            model = self.test_model.get().strip()
            image = self.test_image.copy()
            conf = float(self.test_conf.get())
            iou = float(self.test_iou.get())
            maxdet = int(self.test_maxdet.get())
        except Exception as exc:
            messagebox.showerror('Parámetros', str(exc)); return
        self.test_summary.set('Detectando objetos...')
        def worker():
            try:
                result = predict_image(image, model, conf=conf, iou=iou, max_det=maxdet)
                self._emit('test_done', result)
            except Exception as exc:
                self._emit('error', f'Conteo: {exc}')
        self._start_worker('test', worker)

    def _start_camera_scan(self):
        self.camera_choice.set('Buscando cámaras...')
        def worker():
            try:
                found = scan_cameras(7)
                self._emit('camera_found', found)
            except Exception as exc:
                self._emit('error', f'Buscar cámaras: {exc}')
        self._start_worker('camera', worker)

    def _start_capture(self, count_after):
        selection = self.camera_combo.current()
        if selection < 0 or selection >= len(self.available_cameras):
            messagebox.showinfo('Cámaras', 'Primero pulsa Buscar cámaras y selecciona una.')
            return
        try:
            index = self.available_cameras[selection]['index']
            width, height = int(self.camera_width.get()), int(self.camera_height.get())
        except Exception as exc:
            messagebox.showerror('Cámaras', str(exc)); return
        def worker():
            try:
                img = capture_camera(index, width, height)
                self._emit('capture_done', (img, index, count_after))
            except Exception as exc:
                self._emit('error', f'Cámara: {exc}')
        self._start_worker('capture', worker)

    def _start_worker(self, key: str, fn):
        if key in self.busy:
            messagebox.showinfo('Proceso activo', 'Ya está ejecutándose este proceso.')
            return
        self.busy.add(key)
        def run():
            try:
                fn()
            finally:
                self._emit('worker_finished', key)
        threading.Thread(target=run, daemon=True).start()

    def _emit(self, event, value):
        self.message_queue.put((event, value))

    def _poll_queue(self):
        try:
            for _ in range(80):
                try:
                    event, data = self.message_queue.get_nowait()
                except queue.Empty:
                    break
                if event == 'worker_finished':
                    self.busy.discard(data)
                elif event == 'error':
                    self.status_var.set(data)
                    self._append_train_log('ERROR: ' + data)
                    messagebox.showerror('Kount-A', data)
                elif event == 'aug_progress':
                    self.aug_progress.configure(value=100 * data['done'] / max(1, data['total']))
                    self.aug_summary.set(f'{data["done"]}/{data["total"]} originales · {data["created"]} aumentadas')
                elif event == 'aug_done':
                    self.aug_progress.configure(value=100)
                    self.aug_summary.set(f'Listo: {data["augmented_images"]} aumentadas de {data["labeled_originals"]} originales etiquetados. Sin etiqueta: {data["unlabeled_originals"]}')
                    self._refresh_augment_lists()
                elif event == 'clear_aug_done':
                    self.aug_current_aug = None
                    self.aug_gen_canvas.delete('all')
                    self.aug_progress.configure(value=0)
                    self._refresh_augment_lists()
                    names = ', '.join(data['removed']) if data['removed'] else 'ninguno (ya estaba limpio)'
                    self.aug_summary.set(f'Aumentación eliminada: {names}.')
                    self.status_var.set('Limpieza de aumentación terminada. Originales y modelos conservados.')
                    messagebox.showinfo('Borrado completado',
                                        f'Elementos eliminados: {names}\nArchivos dentro de carpetas: {data["files_in_directories"]}')
                elif event == 'clear_yolo_done':
                    names = ', '.join(data['removed']) if data['removed'] else 'ninguno (ya estaba limpio)'
                    self._append_train_log(f'Limpieza dataset YOLO: {names}. Modelos conservados.')
                    self.train_state.set(f'Dataset YOLO eliminado: {names}.')
                    self.status_var.set('Limpieza YOLO terminada; modelos y métricas conservados.')
                    messagebox.showinfo('Borrado completado',
                                        f'Elementos eliminados: {names}\nArchivos dentro de carpetas: {data["files_in_directories"]}')
                elif event == 'auto_batch_done':
                    self.status_var.set(data)
                    messagebox.showinfo('BBox automáticas', data)
                    # Recargar la BBox de la foto activa después del lote.
                    if self.current_image and not self.dirty:
                        inp, out = self._get_paths()
                        self.current_objects = read_annotations(inp, self.current_image, out)
                        self._update_box_tree()
                        self._render_editor()
                elif event == 'prelabel_done':
                    if self.current_image == data['path']:
                        self.current_objects = data['objects']
                        self.dirty = True
                        self.selected_bbox = -1
                        self._update_box_tree()
                        self._render_editor()
                        self.status_var.set(f'{len(self.current_objects)} BBox pre-etiquetadas. Revísalas y guárdalas.')
                elif event == 'train_msg':
                    self._append_train_log(data)
                elif event == 'train_progress':
                    if data.get('stage') == 'message':
                        self._append_train_log(data['message'])
                        self.train_state.set(data['message'])
                    elif data.get('stage') == 'batch':
                        epoch, total = data['epoch'], data['epochs']
                        batch, batches = data['batch'], max(1, data['batches'])
                        self.train_progress.configure(value=100*((epoch-1)+batch/batches)/max(total, 1))
                        self.train_state.set(f'Época {epoch}/{total} · Batch {batch}/{batches} (entrenando...)')
                    elif data.get('stage') == 'epoch':
                        epoch, total = data['epoch'], data['epochs']
                        self.train_progress.configure(value=100*epoch/max(total,1))
                        metrics = ' | '.join(f'{k}={v:.4f}' for k,v in data['metrics'].items())
                        line = f'Época {epoch}/{total} | {metrics or "métricas disponibles al completar validación"}'
                        self._append_train_log(line)
                        self.train_state.set(line[:200])
                elif event == 'export_done':
                    line = f'Exportación YOLO completa: {data["counts"]} | {data["dataset_yaml"]}'
                    self._append_train_log(line)
                    self.train_state.set(line)
                elif event == 'train_done':
                    self.last_trained = data['best_model']
                    self.train_progress.configure(value=100)
                    self.train_state.set(f'Modelo listo: {data["best_model"]}')
                    self._append_train_log('Entrenamiento completo. best.pt = ' + data['best_model'])
                    self._copy_latest_model()
                elif event == 'test_done':
                    self.test_detections = data['detections']
                    self._rerender_test()
                    self.test_summary.set(f'Total: {data["total"]}')
                    lines = ['CONTEO POR CLASE:']
                    lines += [f'{cls}: {count}' for cls, count in sorted(data['counts'].items())]
                    lines += ['', 'DETECCIONES:']
                    lines += [f'{i+1}. {d["class_name"]}: {d["score"]*100:.1f}%\n   BBox {d["bbox_xyxy"]}'
                              for i, d in enumerate(data['detections'])]
                    if not data['detections']:
                        lines.append('No hubo objetos sobre la confianza mínima.')
                    self._set_text(self.test_details, '\n'.join(lines))
                elif event == 'camera_found':
                    self.available_cameras = data
                    labels = [c['label'] for c in data]
                    self.camera_combo.configure(values=labels)
                    self.camera_choice.set(labels[0] if labels else 'No se encontraron cámaras disponibles')
                    if data:
                        self.camera_combo.current(0)
                    self.status_var.set(f'{len(data)} cámara(s) encontradas.')
                elif event == 'capture_done':
                    image, index, count_after = data
                    self.test_image = image
                    self.test_detections = []
                    self.test_source.set(f'Cámara seleccionada · dispositivo {index} · {image.width}×{image.height}')
                    self.test_summary.set('Frame capturado.')
                    self._rerender_test()
                    if count_after:
                        self._start_test_count()
        finally:
            self.after(100, self._poll_queue)

    @staticmethod
    def _set_text(widget: tk.Text, value: str):
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', value)
        widget.configure(state='disabled')

    def _load_session(self):
        if not self.session_path.is_file():
            self._refresh_images()
            return
        try:
            obj = json.loads(self.session_path.read_text(encoding='utf-8'))
            self.input_var.set(obj.get('input', ''))
            self.output_var.set(obj.get('output', ''))
            for key, value in obj.get('settings', {}).items():
                var = getattr(self, key, None)
                if isinstance(var, (tk.StringVar, tk.BooleanVar)):
                    var.set(value)
            self._refresh_images()
        except Exception as exc:
            self.status_var.set(f'No se pudo restaurar la sesión: {exc}')

    def _save_session(self):
        names = [
            'aug_copies', 'aug_seed', 'aug_flip', 'aug_brightness', 'aug_contrast',
            'aug_saturation', 'aug_hue', 'aug_snp_prob', 'aug_snp_amount',
            'aug_gaussian_prob', 'aug_gaussian_sigma',
            'train_base', 'train_epochs', 'train_batch', 'train_imgsz',
            'train_lr', 'train_patience', 'train_seed', 'train_val', 'train_test',
            'train_aug', 'train_mosaic', 'test_model', 'prelabel_model', 'gray_threshold',
            'gray_min_area', 'test_conf', 'test_iou', 'test_maxdet',
        ]
        payload = {'input': self.input_var.get(), 'output': self.output_var.get(),
                   'settings': {name: getattr(self, name).get() for name in names}}
        self.session_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def _on_close(self):
        if self.busy and not messagebox.askyesno(
            'Procesos en marcha',
            'Hay procesos ejecutándose. Al cerrar se pueden interrumpir y perder resultados. '
            '¿Deseas cerrar igualmente?'
        ):
            return
        if not self._resolve_dirty():
            return
        try:
            self._save_session()
        except OSError:
            pass
        self.destroy()


def main():
    app = KountApp()
    app.mainloop()


if __name__ == '__main__':
    main()
