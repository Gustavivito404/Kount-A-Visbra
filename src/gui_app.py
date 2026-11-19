from __future__ import annotations

import json
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

try:
    # Ejecución recomendada desde la raíz:
    # python -m src.gui_app
    from .augment_core import (
        CLASS_CONFIG_FILENAME,
        ROOT_CLASS_NAME,
        clean_generated_dataset,
        ensure_rgb,
        find_images,
        generate_augmented_dataset_with_bboxes,
        get_class_name,
        get_class_names,
        load_class_config_json,
        save_class_config_json,
    )
    from .bbox_utils import (
        build_preview_image,
        detect_bbox_white_background,
        detect_bbox_configurable,
        load_bbox_metadata,
    )
    from .detection_core import (
        DEFAULT_DETECTOR_MODEL_NAME,
        capture_webcam_image,
        draw_detections,
        load_detector_metadata,
        predict_image_file,
        predict_pil_image,
        train_detector,
    )
except ImportError:
    # También permite:
    # python src/gui_app.py
    from augment_core import (
        CLASS_CONFIG_FILENAME,
        ROOT_CLASS_NAME,
        clean_generated_dataset,
        ensure_rgb,
        find_images,
        generate_augmented_dataset_with_bboxes,
        get_class_name,
        get_class_names,
        load_class_config_json,
        save_class_config_json,
    )
    from bbox_utils import (
        build_preview_image,
        detect_bbox_white_background,
        detect_bbox_configurable,
        load_bbox_metadata,
    )
    from detection_core import (
        DEFAULT_DETECTOR_MODEL_NAME,
        capture_webcam_image,
        draw_detections,
        load_detector_metadata,
        predict_image_file,
        predict_pil_image,
        train_detector,
    )


NOTES_TEXT = """\
Notas de uso

1. La obtención de bounding boxes tiene dos modos:
   - white: usa un umbral para separar un fondo blanco.
   - rgb_range: selecciona el objeto con rangos independientes R, G y B.

2. Los parámetros globales sirven como respaldo general:
   - Seed
   - Modo de BBox
   - Rangos RGB
   - Umbral de blanco
   - Mínimo de pixeles del objeto

3. Si configuras umbrales por clase, esos valores tienen prioridad sobre los globales.
   La clase se toma del primer nivel de carpeta dentro de input.
   Ejemplo:
   input/
   ├── tornillos/
   └── tuercas/

4. La vista superior central muestra la bbox detectada en la imagen original.
   La vista inferior central muestra la bbox ya guardada de una imagen augmentada.

5. La lista inferior derecha no genera una imagen random.
   Solo lee lo que ya está guardado en:
   - output/augmented_images
   - output/augmented_labels

6. El borrado elimina:
   - imágenes augmentadas
   - labels JSON
   - originales copiados al output

7. La configuración persistente del dataset se guarda en:
   augment_thresholds_config.json
   dentro de la carpeta input.

8. La última ruta de input y output usada por la interfaz se guarda en:
   ui_session_config.json
   junto al proyecto de la app.
"""

SESSION_CONFIG_FILENAME = "ui_session_config.json"


class NotesWindow(tk.Toplevel):
    def __init__(self, parent: "AugmentApp") -> None:
        super().__init__(parent)
        self.title("Notas")
        self.geometry("760x520")
        self.minsize(680, 440)

        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="Notas", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))

        text = tk.Text(container, wrap="word")
        text.pack(fill="both", expand=True)
        text.insert("1.0", NOTES_TEXT)
        text.configure(state="disabled")


class AugmentApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Augment Dataset + Detector / Conteo v12")
        self.geometry("1420x830")
        self.minsize(1280, 760)

        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.copies_var = tk.StringVar(value="5")
        self.seed_var = tk.StringVar(value="42")
        self.threshold_var = tk.StringVar(value="245")
        self.min_fg_var = tk.StringVar(value="50")
        self.flip_var = tk.BooleanVar(value=True)

        # Extracción de bounding box: modo clásico o rango RGB
        self.detection_mode_var = tk.StringVar(value="white")
        self.r_min_var = tk.StringVar(value="0")
        self.r_max_var = tk.StringVar(value="240")
        self.g_min_var = tk.StringVar(value="0")
        self.g_max_var = tk.StringVar(value="240")
        self.b_min_var = tk.StringVar(value="0")
        self.b_max_var = tk.StringVar(value="240")

        self.class_params: dict[str, dict[str, int]] = {}
        self.current_scope_var = tk.StringVar(value="General / default")
        self.scope_options: list[str] = ["General / default"]
        self.previous_scope_display = "General / default"

        self.original_image_paths: list[Path] = []
        self.augmented_image_paths: list[Path] = []

        self.original_preview_photo = None
        self.augmented_preview_photo = None

        self._building_scope_values = False

        # Detector / conteo (v11)
        self.det_epochs_var = tk.StringVar(value="10")
        self.det_batch_var = tk.StringVar(value="2")
        self.det_lr_var = tk.StringVar(value="0.0005")
        self.det_val_fraction_var = tk.StringVar(value="0.20")
        self.det_seed_var = tk.StringVar(value="42")
        self.det_pretrained_var = tk.BooleanVar(value=True)
        self.det_confidence_var = tk.StringVar(value="0.50")
        self.det_model_path_var = tk.StringVar()
        self.det_image_path_var = tk.StringVar()
        self.det_camera_index_var = tk.StringVar(value="0")
        self.det_camera_width_var = tk.StringVar(value="1280")
        self.det_camera_height_var = tk.StringVar(value="720")
        self.det_status_var = tk.StringVar(value="Listo para entrenar detector.")
        self.det_result_var = tk.StringVar(value="Sin conteo.")
        self.det_model_info_var = tk.StringVar(value="Sin detector cargado.")
        self.det_preview_photo = None
        self.det_current_pil_image = None
        self.det_training = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(80, self.initialize_from_session)

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Augment Dataset + Detector / Conteo v12",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", pady=(0, 10))

        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill="both", expand=True)

        augment_tab = ttk.Frame(self.notebook, padding=4)
        training_tab = ttk.Frame(self.notebook, padding=4)
        test_tab = ttk.Frame(self.notebook, padding=4)

        self.notebook.add(augment_tab, text="1. Dataset / Augmentación")
        self.notebook.add(training_tab, text="2. Entrenamiento detector")
        self.notebook.add(test_tab, text="3. Prueba de conteo")

        main = ttk.Panedwindow(augment_tab, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, padding=8)
        center = ttk.Frame(main, padding=8)
        right = ttk.Frame(main, padding=8)

        main.add(left, weight=3)
        main.add(center, weight=6)
        main.add(right, weight=4)

        self._build_left_panel(left)
        self._build_center_panel(center)
        self._build_right_panel(right)

        self._build_detector_training_tab(training_tab)
        self._build_detector_test_tab(test_tab)


    def _build_left_panel(self, parent: ttk.Frame) -> None:
        form = ttk.LabelFrame(parent, text="Configuración general", padding=10)
        form.pack(fill="x")

        ttk.Label(form, text="Carpeta input").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.input_dir_var, width=40).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(form, text="Examinar", command=self.select_input_dir).grid(row=0, column=2)

        ttk.Label(form, text="Carpeta output").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.output_dir_var, width=40).grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(form, text="Examinar", command=self.select_output_dir).grid(row=1, column=2)

        ttk.Label(form, text="Copias por imagen").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.copies_var, width=12).grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(form, text="Seed").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.seed_var, width=12).grid(row=3, column=1, sticky="w", padx=6)

        ttk.Separator(form, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="ew", pady=7)

        ttk.Label(form, text="Modo de BBox").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Combobox(
            form,
            textvariable=self.detection_mode_var,
            state="readonly",
            values=["white", "rgb_range"],
            width=16
        ).grid(row=5, column=1, sticky="w", padx=6)

        ttk.Label(
            form,
            text="white: fondo blanco | rgb_range: color del objeto",
            justify="left"
        ).grid(row=6, column=1, columnspan=2, sticky="w", padx=6)

        ttk.Label(form, text="R min / max").grid(row=7, column=0, sticky="w", pady=3)
        r_frame = ttk.Frame(form)
        r_frame.grid(row=7, column=1, sticky="w", padx=6)
        ttk.Entry(r_frame, textvariable=self.r_min_var, width=6).pack(side="left")
        ttk.Label(r_frame, text="  a  ").pack(side="left")
        ttk.Entry(r_frame, textvariable=self.r_max_var, width=6).pack(side="left")

        ttk.Label(form, text="G min / max").grid(row=8, column=0, sticky="w", pady=3)
        g_frame = ttk.Frame(form)
        g_frame.grid(row=8, column=1, sticky="w", padx=6)
        ttk.Entry(g_frame, textvariable=self.g_min_var, width=6).pack(side="left")
        ttk.Label(g_frame, text="  a  ").pack(side="left")
        ttk.Entry(g_frame, textvariable=self.g_max_var, width=6).pack(side="left")

        ttk.Label(form, text="B min / max").grid(row=9, column=0, sticky="w", pady=3)
        b_frame = ttk.Frame(form)
        b_frame.grid(row=9, column=1, sticky="w", padx=6)
        ttk.Entry(b_frame, textvariable=self.b_min_var, width=6).pack(side="left")
        ttk.Label(b_frame, text="  a  ").pack(side="left")
        ttk.Entry(b_frame, textvariable=self.b_max_var, width=6).pack(side="left")

        ttk.Separator(form, orient="horizontal").grid(row=10, column=0, columnspan=3, sticky="ew", pady=7)

        ttk.Label(form, text="Editar parámetros de").grid(row=11, column=0, sticky="w", pady=4)
        self.scope_combo = ttk.Combobox(
            form,
            textvariable=self.current_scope_var,
            state="readonly",
            values=self.scope_options,
            width=26
        )
        self.scope_combo.grid(row=11, column=1, sticky="w", padx=6)
        self.scope_combo.bind("<<ComboboxSelected>>", self.on_scope_changed)
        ttk.Button(form, text="Recargar clases", command=self.reload_classes_and_sync_json).grid(row=11, column=2)

        ttk.Label(form, text="Umbral blanco").grid(row=12, column=0, sticky="w", pady=4)
        self.scope_threshold_entry = ttk.Entry(form, width=12)
        self.scope_threshold_entry.grid(row=12, column=1, sticky="w", padx=6)
        self.scope_threshold_entry.bind("<FocusOut>", self.on_scope_entries_focus_out)

        ttk.Label(form, text="Mín. pixeles objeto").grid(row=13, column=0, sticky="w", pady=4)
        self.scope_min_fg_entry = ttk.Entry(form, width=12)
        self.scope_min_fg_entry.grid(row=13, column=1, sticky="w", padx=6)
        self.scope_min_fg_entry.bind("<FocusOut>", self.on_scope_entries_focus_out)

        ttk.Checkbutton(form, text="Permitir flip horizontal", variable=self.flip_var).grid(
            row=14, column=1, sticky="w", padx=6, pady=5
        )

        form.columnconfigure(1, weight=1)

        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=(10, 0))
        ttk.Button(btns, text="Generar augment dataset", command=self.start_generate).pack(side="left")
        ttk.Button(btns, text="Borrar proceso generado", command=self.start_clean).pack(side="left", padx=8)

        tools = ttk.Frame(parent)
        tools.pack(fill="x", pady=(10, 0))
        ttk.Button(tools, text="Ver notas", command=self.open_notes_window).pack(side="left")

        self.class_config_info_var = tk.StringVar(value="Sin configuración por clase aplicada.")
        ttk.Label(parent, textvariable=self.class_config_info_var, justify="left").pack(anchor="w", pady=(10, 0))

        self.session_info_var = tk.StringVar(
            value="La sesión recordará la última carpeta input y output usadas."
        )
        ttk.Label(parent, textvariable=self.session_info_var, justify="left").pack(anchor="w", pady=(8, 0))


    def _build_center_panel(self, parent: ttk.Frame) -> None:
        top = ttk.LabelFrame(parent, text="Original seleccionado con bbox detectada", padding=10)
        top.pack(fill="both", expand=True)

        self.original_preview_label = ttk.Label(
            top,
            text="Selecciona una imagen original para ver su bbox detectada.",
            anchor="center",
            justify="center"
        )
        self.original_preview_label.pack(fill="both", expand=True)

        self.original_preview_info_var = tk.StringVar(value="Sin original seleccionado.")
        ttk.Label(top, textvariable=self.original_preview_info_var, justify="left").pack(anchor="w", pady=(10, 0))

        mid_controls = ttk.Frame(parent)
        mid_controls.pack(fill="x", pady=8)

        ttk.Button(
            mid_controls,
            text="Actualizar previsualización original",
            command=self.update_original_preview
        ).pack(side="left")

        ttk.Button(
            mid_controls,
            text="Limpiar previsualizaciones",
            command=self.clear_all_previews
        ).pack(side="left", padx=8)

        bottom = ttk.LabelFrame(parent, text="Imagen augmentada guardada con bbox leída del label", padding=10)
        bottom.pack(fill="both", expand=True)

        self.augmented_preview_label = ttk.Label(
            bottom,
            text="Selecciona una imagen augmentada para ver la bbox ya generada.",
            anchor="center",
            justify="center"
        )
        self.augmented_preview_label.pack(fill="both", expand=True)

        self.augmented_preview_info_var = tk.StringVar(value="Sin augmentada seleccionada.")
        ttk.Label(bottom, textvariable=self.augmented_preview_info_var, justify="left").pack(anchor="w", pady=(10, 0))

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        top = ttk.LabelFrame(parent, text="Originales en input", padding=10)
        top.pack(fill="both", expand=True)

        top_btns = ttk.Frame(top)
        top_btns.pack(fill="x", pady=(0, 8))
        ttk.Button(top_btns, text="Cargar originales", command=self.refresh_original_list).pack(side="left")

        self.original_listbox = tk.Listbox(top, exportselection=False)
        self.original_listbox.pack(fill="both", expand=True, side="left")
        self.original_listbox.bind("<<ListboxSelect>>", self.on_original_select)

        top_scrollbar = ttk.Scrollbar(top, orient="vertical", command=self.original_listbox.yview)
        top_scrollbar.pack(fill="y", side="right")
        self.original_listbox.configure(yscrollcommand=top_scrollbar.set)

        bottom = ttk.LabelFrame(parent, text="Augmentadas guardadas en output", padding=10)
        bottom.pack(fill="both", expand=True, pady=(10, 0))

        bottom_btns = ttk.Frame(bottom)
        bottom_btns.pack(fill="x", pady=(0, 8))
        ttk.Button(bottom_btns, text="Cargar augmentadas", command=self.refresh_augmented_list).pack(side="left")

        self.augmented_listbox = tk.Listbox(bottom, exportselection=False)
        self.augmented_listbox.pack(fill="both", expand=True, side="left")
        self.augmented_listbox.bind("<<ListboxSelect>>", self.on_augmented_select)

        bottom_scrollbar = ttk.Scrollbar(bottom, orient="vertical", command=self.augmented_listbox.yview)
        bottom_scrollbar.pack(fill="y", side="right")
        self.augmented_listbox.configure(yscrollcommand=bottom_scrollbar.set)

    def _build_detector_training_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        train_frame = ttk.LabelFrame(parent, text="Entrenamiento del detector", padding=12)
        train_frame.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        train_frame.columnconfigure(1, weight=1)
        train_frame.columnconfigure(3, weight=1)

        ttk.Label(
            train_frame,
            text=(
                "Esta pestaña se dedica únicamente al entrenamiento. El detector usa "
                "original_images/original_labels y augmented_images/augmented_labels."
            ),
            justify="left",
            wraplength=1200
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

        ttk.Label(train_frame, text="Dataset (output)").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(train_frame, textvariable=self.output_dir_var, state="readonly").grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=6
        )

        ttk.Label(train_frame, text="Modelo .pth").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(train_frame, textvariable=self.det_model_path_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", padx=6
        )
        ttk.Button(
            train_frame,
            text="Elegir / cargar modelo",
            command=self.select_detector_model_path
        ).grid(row=2, column=3, sticky="e")

        ttk.Label(train_frame, text="Épocas").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(train_frame, textvariable=self.det_epochs_var, width=10).grid(row=3, column=1, sticky="w", padx=6)
        ttk.Label(train_frame, text="Batch size").grid(row=3, column=2, sticky="w", pady=4)
        ttk.Entry(train_frame, textvariable=self.det_batch_var, width=10).grid(row=3, column=3, sticky="w", padx=6)

        ttk.Label(train_frame, text="Learning rate").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(train_frame, textvariable=self.det_lr_var, width=10).grid(row=4, column=1, sticky="w", padx=6)
        ttk.Label(train_frame, text="Validación").grid(row=4, column=2, sticky="w", pady=4)
        ttk.Entry(train_frame, textvariable=self.det_val_fraction_var, width=10).grid(row=4, column=3, sticky="w", padx=6)

        ttk.Label(train_frame, text="Seed").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(train_frame, textvariable=self.det_seed_var, width=10).grid(row=5, column=1, sticky="w", padx=6)
        ttk.Checkbutton(
            train_frame,
            text="Usar pesos COCO preentrenados",
            variable=self.det_pretrained_var
        ).grid(row=5, column=2, columnspan=2, sticky="w", padx=6)

        controls = ttk.Frame(train_frame)
        controls.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(10, 4))

        self.det_train_button = ttk.Button(
            controls,
            text="Entrenar detector",
            command=self.start_detector_training
        )
        self.det_train_button.pack(side="left")
        ttk.Button(
            controls,
            text="Ruta recomendada",
            command=self.sync_detector_model_path
        ).pack(side="left", padx=8)
        ttk.Button(
            controls,
            text="Información del modelo",
            command=self.refresh_detector_model_info
        ).pack(side="left")

        self.det_progress = ttk.Progressbar(
            train_frame, orient="horizontal", mode="determinate", maximum=100
        )
        self.det_progress.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(8, 4))

        ttk.Label(train_frame, textvariable=self.det_status_var).grid(
            row=8, column=0, columnspan=4, sticky="w", pady=(4, 0)
        )
        ttk.Label(train_frame, textvariable=self.det_model_info_var, justify="left").grid(
            row=9, column=0, columnspan=4, sticky="w", pady=(4, 0)
        )

        log_frame = ttk.LabelFrame(parent, text="Avance por épocas / información", padding=12)
        log_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.det_log_text = tk.Text(log_frame, wrap="word")
        self.det_log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.det_log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.det_log_text.configure(yscrollcommand=log_scroll.set, state="disabled")

        ttk.Label(
            log_frame,
            text="El mejor checkpoint se guarda automáticamente según el F1 de validación.",
            justify="left"
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        self.after(180, self.sync_detector_model_path)


    def _build_detector_test_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=7)
        parent.columnconfigure(1, weight=3)
        parent.rowconfigure(1, weight=1)

        top = ttk.LabelFrame(parent, text="Fuente de imagen", padding=10)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=6)

        ttk.Button(top, text="Seleccionar imagen", command=self.select_detector_test_image).pack(side="left")
        ttk.Button(top, text="Contar objetos", command=self.start_detector_prediction).pack(side="left", padx=8)
        ttk.Button(top, text="Limpiar", command=self.clear_detector_test).pack(side="left", padx=(0, 16))

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(top, text="Webcam índice").pack(side="left")
        ttk.Entry(top, textvariable=self.det_camera_index_var, width=4).pack(side="left", padx=4)
        ttk.Label(top, text="W").pack(side="left")
        ttk.Entry(top, textvariable=self.det_camera_width_var, width=6).pack(side="left", padx=4)
        ttk.Label(top, text="H").pack(side="left")
        ttk.Entry(top, textvariable=self.det_camera_height_var, width=6).pack(side="left", padx=4)
        ttk.Button(top, text="Capturar frame", command=self.start_webcam_capture).pack(side="left", padx=6)
        ttk.Button(top, text="Capturar y contar", command=self.start_webcam_capture_and_count).pack(side="left")

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(top, text="Confianza mín.").pack(side="left")
        ttk.Entry(top, textvariable=self.det_confidence_var, width=7).pack(side="left", padx=5)

        image_frame = ttk.LabelFrame(parent, text="Imagen y bounding boxes", padding=10)
        image_frame.grid(row=1, column=0, sticky="nsew", padx=(6, 3), pady=6)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(1, weight=1)

        ttk.Label(
            image_frame,
            textvariable=self.det_image_path_var,
            justify="left",
            wraplength=900
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.det_preview_label = ttk.Label(
            image_frame,
            text=(
                "Selecciona una imagen para realizar una prueba de conteo.\n"
                "Aquí se mostrará la imagen anotada en un área grande."
            ),
            anchor="center",
            justify="center"
        )
        self.det_preview_label.grid(row=1, column=0, sticky="nsew")

        result_frame = ttk.LabelFrame(parent, text="Resultado", padding=12)
        result_frame.grid(row=1, column=1, sticky="nsew", padx=(3, 6), pady=6)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(3, weight=1)

        ttk.Label(
            result_frame,
            textvariable=self.det_result_var,
            font=("Segoe UI", 13, "bold"),
            justify="left",
            wraplength=360
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.det_count_tree = ttk.Treeview(
            result_frame,
            columns=("class", "count"),
            show="headings",
            height=7
        )
        self.det_count_tree.heading("class", text="Clase")
        self.det_count_tree.heading("count", text="Cantidad")
        self.det_count_tree.column("class", width=210, anchor="w")
        self.det_count_tree.column("count", width=80, anchor="center")
        self.det_count_tree.grid(row=1, column=0, sticky="ew")

        ttk.Label(result_frame, text="Detalle de detecciones", font=("Segoe UI", 10, "bold")).grid(
            row=2, column=0, sticky="w", pady=(12, 4)
        )

        self.det_result_text = tk.Text(result_frame, wrap="word", width=38)
        self.det_result_text.grid(row=3, column=0, sticky="nsew")
        result_scroll = ttk.Scrollbar(result_frame, orient="vertical", command=self.det_result_text.yview)
        result_scroll.grid(row=3, column=1, sticky="ns")
        self.det_result_text.configure(yscrollcommand=result_scroll.set, state="disabled")

        ttk.Label(
            result_frame,
            text="La webcam es opcional; puedes trabajar solo con imágenes guardadas.",
            justify="left",
            wraplength=360
        ).grid(row=4, column=0, sticky="w", pady=(10, 0))


    def set_detector_result_text(self, text: str) -> None:
        if not hasattr(self, "det_result_text"):
            return
        self.det_result_text.configure(state="normal")
        self.det_result_text.delete("1.0", "end")
        self.det_result_text.insert("1.0", text)
        self.det_result_text.configure(state="disabled")


    def append_detector_log(self, text: str) -> None:
        self.det_log_text.configure(state="normal")
        self.det_log_text.insert("end", text.rstrip() + "\n")
        self.det_log_text.see("end")
        self.det_log_text.configure(state="disabled")

    def sync_detector_model_path(self) -> None:
        output_dir = self._get_output_dir()

        if output_dir is None:
            if not self.det_model_path_var.get().strip():
                self.det_model_path_var.set("")
            return

        recommended = output_dir / "models" / DEFAULT_DETECTOR_MODEL_NAME
        current = self.det_model_path_var.get().strip()

        if not current:
            self.det_model_path_var.set(str(recommended))
            return

        current_path = Path(current)

        if (
            current_path.name == DEFAULT_DETECTOR_MODEL_NAME
            and current_path.parent.name == "models"
        ):
            self.det_model_path_var.set(str(recommended))

    def select_detector_model_path(self) -> None:
        self.sync_detector_model_path()
        initial = self.det_model_path_var.get().strip()
        initial_dir = (
            str(Path(initial).parent)
            if initial
            else str(Path.cwd())
        )

        path = filedialog.asksaveasfilename(
            title="Guardar detector",
            defaultextension=".pth",
            filetypes=[
                ("PyTorch detector", "*.pth"),
                ("Todos los archivos", "*.*")
            ],
            initialdir=initial_dir,
            initialfile=(
                Path(initial).name
                if initial
                else DEFAULT_DETECTOR_MODEL_NAME
            ),
        )

        if path:
            self.det_model_path_var.set(path)

    def start_detector_training(self) -> None:
        if self.det_training:
            messagebox.showinfo(
                "Entrenamiento",
                "Ya hay un entrenamiento del detector en curso."
            )
            return

        threading.Thread(
            target=self.detector_training_action,
            daemon=True
        ).start()

    def detector_training_action(self) -> None:
        try:
            output_dir = self._get_output_dir()

            if output_dir is None or not output_dir.exists():
                raise FileNotFoundError(
                    "Selecciona primero el output del dataset "
                    "en la pestaña 1."
                )

            self.sync_detector_model_path()

            model_path_raw = self.det_model_path_var.get().strip()

            if not model_path_raw:
                raise ValueError(
                    "Define dónde guardar el detector .pth."
                )

            epochs = int(self.det_epochs_var.get().strip())
            batch_size = int(self.det_batch_var.get().strip())
            learning_rate = float(self.det_lr_var.get().strip())
            val_fraction = float(
                self.det_val_fraction_var.get().strip()
            )
            seed = int(self.det_seed_var.get().strip())
            use_pretrained = bool(self.det_pretrained_var.get())

            self.det_training = True

            self.after(
                0,
                lambda: self.det_train_button.configure(
                    state="disabled"
                )
            )
            self.after(
                0,
                lambda: self.det_progress.configure(value=0)
            )
            self.after(
                0,
                lambda: self.det_status_var.set(
                    "Preparando detector..."
                )
            )
            self.after(
                0,
                lambda: self.append_detector_log(
                    "\n=== Nuevo entrenamiento de detección ===\n"
                    f"Dataset: {output_dir}\n"
                    f"Modelo: {model_path_raw}\n"
                    f"Pesos COCO preentrenados: {use_pretrained}"
                )
            )

            def progress(data: dict) -> None:
                epoch = int(data["epoch"])
                total = int(data["epochs"])
                percent = 100.0 * epoch / max(total, 1)

                line = (
                    f"Época {epoch}/{total} | "
                    f"loss={data['train_loss']:.4f} | "
                    f"precision={data['precision']:.2f}% | "
                    f"recall={data['recall']:.2f}% | "
                    f"F1={data['f1']:.2f}%"
                )

                self.after(
                    0,
                    lambda p=percent:
                    self.det_progress.configure(value=p)
                )
                self.after(
                    0,
                    lambda l=line:
                    self.append_detector_log(l)
                )
                self.after(
                    0,
                    lambda e=epoch, t=total, d=data:
                    self.det_status_var.set(
                        f"Entrenando {e}/{t} | "
                        f"{d['device']} | "
                        f"mejor F1={d['best_f1']:.2f}%"
                    )
                )

            summary = train_detector(
                dataset_output_dir=output_dir,
                model_path=Path(model_path_raw),
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                val_fraction=val_fraction,
                seed=seed,
                use_pretrained=use_pretrained,
                progress_callback=progress,
            )

            def finish() -> None:
                self.det_progress.configure(value=100)
                self.det_status_var.set(
                    "Entrenamiento terminado. "
                    f"Mejor F1={summary['best_f1']:.2f}% "
                    f"(época {summary['best_epoch']})."
                )

                self.append_detector_log(
                    f"Arquitectura: {summary['architecture']}\n"
                    f"Clases: {', '.join(summary['class_names'])}\n"
                    f"Imágenes: {summary['total_images']} "
                    f"(train={summary['train_images']}, "
                    f"val={summary['val_images']})\n"
                    f"Dispositivo: {summary['device']}\n"
                    f"Mejor detector: {summary['model_path']}"
                )

                for warning in summary.get("warnings", []):
                    self.append_detector_log(
                        "AVISO: " + warning
                    )

                self.refresh_detector_model_info()

                messagebox.showinfo(
                    "Entrenamiento completado",
                    (
                        f"Detector guardado en:\n"
                        f"{summary['model_path']}\n\n"
                        f"Clases: {summary['num_classes']}\n"
                        f"Imágenes: {summary['total_images']}\n"
                        f"Mejor F1: {summary['best_f1']:.2f}%"
                    )
                )

            self.after(0, finish)

        except Exception as e:
            self.after(
                0,
                lambda: self.det_status_var.set(
                    "Error de entrenamiento."
                )
            )
            self.after(
                0,
                lambda err=str(e):
                self.append_detector_log("ERROR: " + err)
            )
            self.after(
                0,
                lambda err=str(e):
                messagebox.showerror(
                    "Error de entrenamiento del detector",
                    err
                )
            )
        finally:
            self.det_training = False
            self.after(
                0,
                lambda: self.det_train_button.configure(
                    state="normal"
                )
            )

    def refresh_detector_model_info(self) -> None:
        try:
            raw = self.det_model_path_var.get().strip()

            if not raw:
                self.det_model_info_var.set(
                    "Sin ruta de detector."
                )
                return

            path = Path(raw)

            if not path.exists():
                self.det_model_info_var.set(
                    "El detector aún no existe en la ruta indicada."
                )
                return

            info = load_detector_metadata(path)
            classes = ", ".join(info.get("class_names", []))
            best = info.get("best_f1")
            best_text = (
                "N/D"
                if best is None
                else f"{float(best):.2f}%"
            )

            self.det_model_info_var.set(
                f"Arquitectura: {info.get('architecture')} | "
                f"mejor F1={best_text}\n"
                f"Clases: {classes}"
            )

        except Exception as e:
            self.det_model_info_var.set(
                f"No se pudo leer el detector: {e}"
            )

    def clear_detector_test(self) -> None:
        self.det_current_pil_image = None
        self.det_image_path_var.set("")
        self.det_preview_photo = None
        self.det_preview_label.configure(
            image="",
            text=(
                "Selecciona una imagen para realizar una prueba de conteo.\n"
                "Aquí se mostrará la imagen anotada en un área grande."
            )
        )
        self.det_result_var.set("Sin conteo.")
        self.set_detector_result_text("")

        for item in self.det_count_tree.get_children():
            self.det_count_tree.delete(item)


    def select_detector_test_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Selecciona una imagen para contar",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Todos los archivos", "*.*")
            ]
        )

        if not path:
            return

        try:
            with Image.open(path) as img:
                self.det_current_pil_image = img.convert("RGB").copy()

            self.det_image_path_var.set(path)
            self.show_detector_preview(
                self.det_current_pil_image
            )
            self.det_result_var.set(
                "Imagen cargada. Presiona 'Contar objetos'."
            )

        except Exception as e:
            messagebox.showerror(
                "Error de imagen",
                str(e)
            )

    def show_detector_preview(self, image: Image.Image) -> None:
        preview = image.convert("RGB").copy()

        # El área de prueba ahora ocupa la mayor parte de la pestaña.
        width = self.det_preview_label.winfo_width()
        height = self.det_preview_label.winfo_height()
        max_width = width - 20 if width > 200 else 950
        max_height = height - 20 if height > 200 else 600

        preview.thumbnail((max_width, max_height))
        self.det_preview_photo = ImageTk.PhotoImage(preview)
        self.det_preview_label.configure(image=self.det_preview_photo, text="")


    def start_detector_prediction(self) -> None:
        threading.Thread(
            target=self.detector_prediction_action,
            daemon=True
        ).start()

    def detector_prediction_action(self) -> None:
        try:
            model_raw = self.det_model_path_var.get().strip()

            if not model_raw:
                raise ValueError(
                    "Selecciona o entrena primero un detector .pth."
                )

            if self.det_current_pil_image is None:
                raise ValueError(
                    "Selecciona una imagen o captura un frame "
                    "de la webcam."
                )

            confidence = float(
                self.det_confidence_var.get().strip()
            )

            if not (0.0 < confidence < 1.0):
                raise ValueError(
                    "La confianza mínima debe estar entre 0 y 1."
                )

            image_copy = self.det_current_pil_image.copy()

            self.after(
                0,
                lambda: self.det_result_var.set(
                    "Detectando y contando..."
                )
            )

            result = predict_pil_image(
                image_copy,
                model_path=Path(model_raw),
                confidence_threshold=confidence,
            )

            annotated = draw_detections(
                image_copy,
                result["detections"]
            )

            def finish() -> None:
                self.show_detector_preview(annotated)

                if len(result["counts"]) == 1:
                    only_class, only_count = next(iter(result["counts"].items()))
                    self.det_result_var.set(
                        f"{only_class}: {only_count} detectadas "
                        f"| total={result['total']} "
                        f"| dispositivo={result['device']}"
                    )
                else:
                    self.det_result_var.set(
                        f"Total detectado: {result['total']} "
                        f"| dispositivo: {result['device']}"
                    )

                for item in self.det_count_tree.get_children():
                    self.det_count_tree.delete(item)

                for class_name, count in sorted(
                    result["counts"].items()
                ):
                    self.det_count_tree.insert(
                        "",
                        "end",
                        values=(class_name, count)
                    )

                self.append_detector_log(
                    f"\nInferencia | total={result['total']} | "
                    f"threshold={confidence:.2f}"
                )

                detail_lines = [
                    f"Total detectado: {result['total']}",
                    f"Confianza mínima: {confidence:.2f}",
                    "",
                ]

                for index, detection in enumerate(result["detections"], start=1):
                    bbox_text = [round(v, 1) for v in detection["bbox_xyxy"]]
                    detail_lines.append(
                        f"{index}. {detection['class_name']}\n"
                        f"   Confianza: {detection['percentage']:.2f}%\n"
                        f"   BBox: {bbox_text}"
                    )
                    self.append_detector_log(
                        f"- {detection['class_name']} | "
                        f"{detection['percentage']:.2f}% | bbox={bbox_text}"
                    )

                if not result["detections"]:
                    detail_lines.append(
                        "No hubo detecciones por encima del umbral de confianza."
                    )

                self.set_detector_result_text("\n\n".join(detail_lines))

            self.after(0, finish)

        except Exception as e:
            self.after(
                0,
                lambda err=str(e):
                messagebox.showerror(
                    "Error de detección",
                    err
                )
            )

    def start_webcam_capture(self) -> None:
        threading.Thread(
            target=self.webcam_capture_action,
            args=(False,),
            daemon=True
        ).start()

    def start_webcam_capture_and_count(self) -> None:
        threading.Thread(
            target=self.webcam_capture_action,
            args=(True,),
            daemon=True
        ).start()

    def webcam_capture_action(self, count_after: bool = False) -> None:
        try:
            camera_index = int(
                self.det_camera_index_var.get().strip()
            )
            width = int(
                self.det_camera_width_var.get().strip()
            )
            height = int(
                self.det_camera_height_var.get().strip()
            )

            self.after(
                0,
                lambda: self.det_result_var.set(
                    "Capturando frame de webcam..."
                )
            )

            image = capture_webcam_image(
                camera_index=camera_index,
                width=width,
                height=height,
            )

            def show_capture() -> None:
                self.det_current_pil_image = image.copy()
                self.det_image_path_var.set(
                    f"Webcam {camera_index} | "
                    f"{image.size[0]}x{image.size[1]}"
                )
                self.show_detector_preview(image)
                self.det_result_var.set(
                    "Frame capturado correctamente."
                )

            self.after(0, show_capture)

            if count_after:
                # Espera a que Tk registre visualmente la captura y
                # luego lanza la inferencia en un hilo nuevo.
                self.after(
                    100,
                    self.start_detector_prediction
                )

        except Exception as e:
            self.after(
                0,
                lambda err=str(e):
                messagebox.showerror(
                    "Error de webcam",
                    err
                )
            )

    def open_notes_window(self) -> None:
        NotesWindow(self)

    def get_project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def get_session_json_path(self) -> Path:
        return self.get_project_root() / SESSION_CONFIG_FILENAME

    def load_session_json(self) -> dict:
        path = self.get_session_json_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_session_json(self) -> Path:
        path = self.get_session_json_path()
        payload = {
            "last_input_dir": self.input_dir_var.get().strip(),
            "last_output_dir": self.output_dir_var.get().strip(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def initialize_from_session(self) -> None:
        data = self.load_session_json()
        last_input = str(data.get("last_input_dir", "")).strip()
        last_output = str(data.get("last_output_dir", "")).strip()

        if last_input:
            self.input_dir_var.set(last_input)
        if last_output:
            self.output_dir_var.set(last_output)

        if last_input and Path(last_input).exists():
            self.load_persisted_config_from_input(ask_about_missing=True)
            self.refresh_original_list()

        if last_output and Path(last_output).exists():
            self.refresh_augmented_list()
            self.sync_detector_model_path()

        if last_input or last_output:
            self.session_info_var.set(
                "Sesión cargada: se restauraron las últimas rutas usadas."
            )

    def on_close(self) -> None:
        try:
            self.autosave_current_scope_values(silent=True)
            self.save_session_json()
        except Exception:
            pass
        self.destroy()

    def _get_detection_params(self):
        threshold = int(self.threshold_var.get().strip())
        min_fg = int(self.min_fg_var.get().strip())
        enable_hflip = bool(self.flip_var.get())
        detection_mode = self.detection_mode_var.get().strip()

        r_min, r_max = int(self.r_min_var.get()), int(self.r_max_var.get())
        g_min, g_max = int(self.g_min_var.get()), int(self.g_max_var.get())
        b_min, b_max = int(self.b_min_var.get()), int(self.b_max_var.get())

        if not (0 <= threshold <= 255):
            raise ValueError("El umbral de blanco global debe estar entre 0 y 255.")
        if min_fg < 1:
            raise ValueError("El mínimo global de pixeles debe ser >= 1.")
        if detection_mode not in {"white", "rgb_range"}:
            raise ValueError("Modo de BBox inválido.")

        values = [r_min, r_max, g_min, g_max, b_min, b_max]
        if any(v < 0 or v > 255 for v in values):
            raise ValueError("Los valores RGB deben estar entre 0 y 255.")
        if r_min > r_max or g_min > g_max or b_min > b_max:
            raise ValueError("Cada canal debe cumplir min <= max.")

        return (
            threshold,
            min_fg,
            enable_hflip,
            detection_mode,
            (r_min, g_min, b_min),
            (r_max, g_max, b_max),
        )


    def _validate_scope_values(self, white_threshold: str, min_fg: str) -> tuple[int, int]:
        w = int(white_threshold.strip())
        m = int(min_fg.strip())

        if not (0 <= w <= 255):
            raise ValueError("El umbral de blanco debe estar entre 0 y 255.")
        if m < 1:
            raise ValueError("El mínimo de pixeles del objeto debe ser >= 1.")
        return w, m

    def _update_class_config_label(self) -> None:
        if not self.class_params:
            self.class_config_info_var.set("Sin configuración por clase aplicada.")
            return

        lines = ["Configuración por clase activa:"]
        for class_name, data in sorted(self.class_params.items()):
            shown_name = "(root)" if class_name == ROOT_CLASS_NAME else class_name
            lines.append(
                f"- {shown_name}: threshold={data['white_threshold']}, min_fg={data['min_foreground_pixels']}"
            )
        self.class_config_info_var.set("\n".join(lines))

    def _get_input_dir(self) -> Path | None:
        raw = self.input_dir_var.get().strip()
        if not raw:
            return None
        return Path(raw)

    def _get_output_dir(self) -> Path | None:
        raw = self.output_dir_var.get().strip()
        if not raw:
            return None
        return Path(raw)

    def write_config_json(self) -> None:
        input_dir = self._get_input_dir()
        if input_dir is None or not input_dir.exists():
            return

        threshold, min_fg, _, detection_mode, rgb_min, rgb_max = self._get_detection_params()
        save_class_config_json(
            input_dir=input_dir,
            default_white_threshold=threshold,
            default_min_foreground_pixels=min_fg,
            class_params=self.class_params,
            detection_mode=detection_mode,
            rgb_min=rgb_min,
            rgb_max=rgb_max,
        )


    def load_persisted_config_from_input(self, ask_about_missing: bool = True) -> None:
        input_dir = self._get_input_dir()
        if input_dir is None or not input_dir.exists():
            return

        data = load_class_config_json(input_dir)
        default_data = data.get("default", {})
        classes_data = data.get("classes", {})

        if default_data:
            self.threshold_var.set(str(default_data.get("white_threshold", self.threshold_var.get())))
            self.min_fg_var.set(str(default_data.get("min_foreground_pixels", self.min_fg_var.get())))
            self.detection_mode_var.set(str(default_data.get("detection_mode", self.detection_mode_var.get())))

            rgb_min = default_data.get("rgb_min", [0, 0, 0])
            rgb_max = default_data.get("rgb_max", [240, 240, 240])
            if len(rgb_min) == 3:
                self.r_min_var.set(str(rgb_min[0]))
                self.g_min_var.set(str(rgb_min[1]))
                self.b_min_var.set(str(rgb_min[2]))
            if len(rgb_max) == 3:
                self.r_max_var.set(str(rgb_max[0]))
                self.g_max_var.set(str(rgb_max[1]))
                self.b_max_var.set(str(rgb_max[2]))

        loaded_class_params = {}
        for class_name, values in classes_data.items():
            loaded_class_params[class_name] = {
                "white_threshold": int(values.get("white_threshold", 245)),
                "min_foreground_pixels": int(values.get("min_foreground_pixels", 50)),
            }

        detected_classes = set(get_class_names(input_dir))
        persisted_classes = set(loaded_class_params.keys())
        missing_classes = sorted(persisted_classes - detected_classes)

        self.class_params = loaded_class_params

        if ask_about_missing and missing_classes:
            shown_missing = [("(root)" if c == ROOT_CLASS_NAME else c) for c in missing_classes]
            question = (
                f"Se encontraron clases guardadas en {CLASS_CONFIG_FILENAME} que ya no están en input:\n\n"
                + "\n".join(f"- {name}" for name in shown_missing)
                + "\n\n¿Quieres eliminarlas del JSON?\n"
                  "Si eliges 'No', se conservarán por si luego vuelves a incorporar esas clases."
            )
            remove = messagebox.askyesno("Clases faltantes detectadas", question)
            if remove:
                for cls in missing_classes:
                    self.class_params.pop(cls, None)
                self.write_config_json()

        self.rebuild_scope_selector()
        self.previous_scope_display = self.current_scope_var.get().strip()
        self.load_values_into_scope_fields(self.previous_scope_display)
        self._update_class_config_label()

    def rebuild_scope_selector(self) -> None:
        input_dir = self._get_input_dir()
        classes = get_class_names(input_dir) if input_dir and input_dir.exists() else []

        options = ["General / default"]
        for cls in classes:
            shown = "(root)" if cls == ROOT_CLASS_NAME else cls
            options.append(shown)

        previous = self.current_scope_var.get().strip() or "General / default"
        self.scope_options = options
        self.scope_combo["values"] = self.scope_options

        if previous not in self.scope_options:
            previous = "General / default"

        self.current_scope_var.set(previous)

    def scope_display_to_class_name(self, display_name: str) -> str | None:
        display_name = display_name.strip()
        if display_name == "General / default":
            return None
        if display_name == "(root)":
            return ROOT_CLASS_NAME
        return display_name

    def save_scope_display_values(self, scope_display: str, white_text: str, min_text: str, silent: bool = False) -> bool:
        try:
            white_threshold, min_fg = self._validate_scope_values(white_text, min_text)
            class_name = self.scope_display_to_class_name(scope_display)

            if class_name is None:
                self.threshold_var.set(str(white_threshold))
                self.min_fg_var.set(str(min_fg))
            else:
                self.class_params[class_name] = {
                    "white_threshold": white_threshold,
                    "min_foreground_pixels": min_fg,
                }

            self.write_config_json()
            self._update_class_config_label()
            return True
        except Exception as e:
            if not silent:
                messagebox.showerror("Error", str(e))
            return False

    def autosave_current_scope_values(self, silent: bool = False) -> bool:
        if self._building_scope_values:
            return True
        return self.save_scope_display_values(
            self.current_scope_var.get().strip(),
            self.scope_threshold_entry.get(),
            self.scope_min_fg_entry.get(),
            silent=silent
        )

    def load_values_into_scope_fields(self, scope_display: str | None = None) -> None:
        if scope_display is None:
            scope_display = self.current_scope_var.get().strip()

        class_name = self.scope_display_to_class_name(scope_display)

        global_threshold = self.threshold_var.get().strip()
        global_min_fg = self.min_fg_var.get().strip()

        if class_name is None:
            threshold_value = global_threshold
            min_fg_value = global_min_fg
        else:
            data = self.class_params.get(class_name, {})
            threshold_value = str(data.get("white_threshold", global_threshold))
            min_fg_value = str(data.get("min_foreground_pixels", global_min_fg))

        self._building_scope_values = True
        try:
            self.scope_threshold_entry.delete(0, "end")
            self.scope_threshold_entry.insert(0, threshold_value)

            self.scope_min_fg_entry.delete(0, "end")
            self.scope_min_fg_entry.insert(0, min_fg_value)
        finally:
            self._building_scope_values = False

    def on_scope_entries_focus_out(self, event=None) -> None:
        self.autosave_current_scope_values(silent=True)

    def on_scope_changed(self, event=None) -> None:
        new_scope = self.current_scope_var.get().strip()
        old_scope = self.previous_scope_display

        if not self.save_scope_display_values(
            old_scope,
            self.scope_threshold_entry.get(),
            self.scope_min_fg_entry.get(),
            silent=True
        ):
            self.current_scope_var.set(old_scope)
            return

        self.previous_scope_display = new_scope
        self.load_values_into_scope_fields(new_scope)

    def reload_classes_and_sync_json(self) -> None:
        try:
            self.autosave_current_scope_values(silent=True)
            self.load_persisted_config_from_input(ask_about_missing=True)
            self.refresh_original_list()
            self.save_session_json()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def select_input_dir(self) -> None:
        folder = filedialog.askdirectory(title="Selecciona la carpeta input")
        if folder:
            self.input_dir_var.set(folder)
            self.load_persisted_config_from_input(ask_about_missing=True)
            self.refresh_original_list()
            self.write_config_json()
            self.save_session_json()
            self.session_info_var.set(
                f"Sesión guardada. Último input: {folder}"
            )

    def select_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="Selecciona la carpeta output")
        if folder:
            self.output_dir_var.set(folder)
            self.refresh_augmented_list()
            self.sync_detector_model_path()
            self.save_session_json()
            self.session_info_var.set(
                f"Sesión guardada. Último output: {folder}"
            )

    def refresh_original_list(self) -> None:
        try:
            input_dir = self._get_input_dir()
            if input_dir is None or not input_dir.exists():
                raise FileNotFoundError("La carpeta input no existe.")

            self.original_image_paths = find_images(input_dir)
            self.original_listbox.delete(0, "end")

            for p in self.original_image_paths:
                self.original_listbox.insert("end", str(p.relative_to(input_dir)))

            self.rebuild_scope_selector()
            self.previous_scope_display = self.current_scope_var.get().strip()
            self.load_values_into_scope_fields(self.previous_scope_display)
            self._update_class_config_label()

            if not self.original_image_paths:
                self.clear_original_preview()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def refresh_augmented_list(self) -> None:
        try:
            output_dir = self._get_output_dir()
            if output_dir is None or not output_dir.exists():
                raise FileNotFoundError("La carpeta output no existe.")

            aug_dir = output_dir / "augmented_images"
            if not aug_dir.exists():
                self.augmented_image_paths = []
                self.augmented_listbox.delete(0, "end")
                self.clear_augmented_preview()
                return

            self.augmented_image_paths = find_images(aug_dir)
            self.augmented_listbox.delete(0, "end")

            for p in self.augmented_image_paths:
                self.augmented_listbox.insert("end", str(p.relative_to(aug_dir)))

            if not self.augmented_image_paths:
                self.clear_augmented_preview()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def get_selected_original_path(self) -> Path | None:
        selection = self.original_listbox.curselection()
        if not selection:
            return None
        return self.original_image_paths[selection[0]]

    def get_selected_augmented_path(self) -> Path | None:
        selection = self.augmented_listbox.curselection()
        if not selection:
            return None
        return self.augmented_image_paths[selection[0]]

    def on_original_select(self, event=None) -> None:
        self.update_original_preview()

    def on_augmented_select(self, event=None) -> None:
        self.update_augmented_preview()

    def update_original_preview(self) -> None:
        try:
            self.autosave_current_scope_values(silent=True)

            image_path = self.get_selected_original_path()
            if image_path is None:
                self.clear_original_preview()
                return

            input_dir = self._get_input_dir()
            threshold, min_fg, _, detection_mode, rgb_min, rgb_max = self._get_detection_params()

            class_name = get_class_name(input_dir, image_path)
            class_params = self.class_params.get(class_name, {})
            used_threshold = int(class_params.get("white_threshold", threshold))
            used_min_fg = int(class_params.get("min_foreground_pixels", min_fg))

            with Image.open(image_path) as img:
                img = ensure_rgb(img)
                bbox = detect_bbox_configurable(
                    img,
                    detection_mode=detection_mode,
                    white_threshold=used_threshold,
                    min_foreground_pixels=used_min_fg,
                    rgb_min=rgb_min,
                    rgb_max=rgb_max,
                )
                preview = build_preview_image(img, bbox, max_size=(820, 290))

            self.original_preview_photo = ImageTk.PhotoImage(preview)
            self.original_preview_label.configure(image=self.original_preview_photo, text="")

            shown_class = "(root)" if class_name == ROOT_CLASS_NAME else class_name
            mode_text = (
                f"modo={detection_mode} | RGB={rgb_min}..{rgb_max}"
                if detection_mode == "rgb_range"
                else f"modo=white | threshold={used_threshold}"
            )

            if bbox is None:
                self.original_preview_info_var.set(
                    f"Imagen: {image_path.name}\n"
                    f"Clase: {shown_class}\n"
                    f"No se detectó bbox. {mode_text} | min_fg={used_min_fg}"
                )
            else:
                x1, y1, x2, y2 = bbox
                self.original_preview_info_var.set(
                    f"Imagen: {image_path.name}\n"
                    f"Clase: {shown_class}\n"
                    f"BBox XYXY: [{x1}, {y1}, {x2}, {y2}]\n"
                    f"{mode_text} | min_fg={used_min_fg}"
                )
        except Exception as e:
            self.clear_original_preview()
            messagebox.showerror("Error de original", str(e))


    def update_augmented_preview(self) -> None:
        try:
            image_path = self.get_selected_augmented_path()
            if image_path is None:
                self.clear_augmented_preview()
                return

            output_dir = self._get_output_dir()
            aug_dir = output_dir / "augmented_images"
            labels_dir = output_dir / "augmented_labels"

            rel_path = image_path.relative_to(aug_dir)
            json_path = (labels_dir / rel_path).with_suffix(".json")

            data = load_bbox_metadata(json_path)
            bbox = data.get("bbox_xyxy")

            with Image.open(image_path) as img:
                img = ensure_rgb(img)
                preview = build_preview_image(img, bbox, max_size=(820, 290))

            self.augmented_preview_photo = ImageTk.PhotoImage(preview)
            self.augmented_preview_label.configure(image=self.augmented_preview_photo, text="")

            x1, y1, x2, y2 = [round(float(v), 2) for v in bbox]
            class_name = data.get("class_name", "desconocida")
            shown_class = "(root)" if class_name == ROOT_CLASS_NAME else class_name
            white_threshold = data.get("white_threshold")
            min_fg = data.get("min_foreground_pixels")

            self.augmented_preview_info_var.set(
                f"Imagen augmentada: {image_path.name}\n"
                f"Clase: {shown_class} | class_id: {data.get('class_id')}\n"
                f"BBox guardada XYXY: [{x1}, {y1}, {x2}, {y2}]\n"
                f"threshold guardado={white_threshold} | min_fg guardado={min_fg}"
            )
        except Exception as e:
            self.clear_augmented_preview()
            messagebox.showerror("Error de augmentada", str(e))

    def clear_original_preview(self) -> None:
        self.original_preview_photo = None
        self.original_preview_label.configure(image="", text="Vista original limpia.")
        self.original_preview_info_var.set("Sin original seleccionado.")

    def clear_augmented_preview(self) -> None:
        self.augmented_preview_photo = None
        self.augmented_preview_label.configure(image="", text="Vista augmentada limpia.")
        self.augmented_preview_info_var.set("Sin augmentada seleccionada.")

    def clear_all_previews(self) -> None:
        self.clear_original_preview()
        self.clear_augmented_preview()

    def start_generate(self) -> None:
        threading.Thread(target=self.generate_action, daemon=True).start()

    def start_clean(self) -> None:
        threading.Thread(target=self.clean_action, daemon=True).start()

    def generate_action(self) -> None:
        try:
            if not self.autosave_current_scope_values(silent=True):
                raise ValueError("No se pudieron guardar los valores actuales.")

            input_dir = self._get_input_dir()
            output_dir = self._get_output_dir()
            copies = int(self.copies_var.get().strip())
            seed = int(self.seed_var.get().strip())
            threshold, min_fg, enable_hflip, detection_mode, rgb_min, rgb_max = self._get_detection_params()

            if input_dir is None or not input_dir.exists():
                raise FileNotFoundError("La carpeta input no existe.")
            if output_dir is None:
                raise FileNotFoundError("Debes seleccionar una carpeta output.")
            if copies < 1:
                raise ValueError("El número de copias debe ser >= 1.")

            self.write_config_json()
            self.save_session_json()

            summary = generate_augmented_dataset_with_bboxes(
                input_dir=input_dir,
                output_dir=output_dir,
                copies_per_image=copies,
                seed=seed,
                white_threshold=threshold,
                min_foreground_pixels=min_fg,
                enable_hflip=enable_hflip,
                class_params=self.class_params,
                detection_mode=detection_mode,
                rgb_min=rgb_min,
                rgb_max=rgb_max,
            )

            self.after(0, self.refresh_augmented_list)

            skipped = summary.get("skipped_images", [])
            class_to_id = summary.get("class_to_id", {})
            class_lines = ["Mapa de clases:"]
            for class_name, class_id in class_to_id.items():
                shown_name = "(root)" if class_name == ROOT_CLASS_NAME else class_name
                class_lines.append(f"- {shown_name}: {class_id}")

            skipped_msg = ""
            if skipped:
                skipped_msg = "\n\nImágenes omitidas:\n" + "\n".join(skipped[:10])
                if len(skipped) > 10:
                    skipped_msg += f"\n... y {len(skipped) - 10} más."

            messagebox.showinfo(
                "Éxito",
                (
                    "Proceso completado.\n\n"
                    f"Modo BBox: {detection_mode}\n"
                    f"RGB min/max: {rgb_min} / {rgb_max}\n\n"
                    f"Originales encontradas: {summary['original_images_found']}\n"
                    f"Originales copiadas: {summary['copied_original_images']}\n"
                    f"Augmentadas creadas: {summary['augmented_images_created']}\n"
                    f"Labels creados: {summary['label_files_created']}\n\n"
                    + "\n".join(class_lines)
                    + skipped_msg
                )
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))


    def clean_action(self) -> None:
        try:
            output_dir = self._get_output_dir()
            if output_dir is None or not output_dir.exists():
                raise FileNotFoundError("La carpeta output no existe.")

            self.save_session_json()

            summary = clean_generated_dataset(output_dir=output_dir)

            self.after(0, self.refresh_augmented_list)

            messagebox.showinfo(
                "Limpieza completada",
                (
                    f"Imágenes augmentadas eliminadas: {summary['deleted_augmented_images']}\n"
                    f"Labels eliminados: {summary['deleted_labels']}\n"
                    f"Originales copiados eliminados: {summary['deleted_copied_originals']}"
                )
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    app = AugmentApp()
    app.mainloop()
