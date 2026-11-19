from __future__ import annotations

import json
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

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
    load_bbox_metadata,
)


NOTES_TEXT = """\
Notas de uso

1. El programa trabaja suponiendo que el objeto está sobre fondo blanco o casi blanco.

2. Los parámetros globales sirven como respaldo general:
   - Seed
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
        self.title("Augment Dataset + Bounding Boxes")
        self.geometry("1420x830")
        self.minsize(1280, 760)

        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.copies_var = tk.StringVar(value="5")
        self.seed_var = tk.StringVar(value="42")
        self.threshold_var = tk.StringVar(value="245")
        self.min_fg_var = tk.StringVar(value="50")
        self.flip_var = tk.BooleanVar(value=True)

        self.class_params: dict[str, dict[str, int]] = {}
        self.current_scope_var = tk.StringVar(value="General / default")
        self.scope_options: list[str] = ["General / default"]
        self.previous_scope_display = "General / default"

        self.original_image_paths: list[Path] = []
        self.augmented_image_paths: list[Path] = []

        self.original_preview_photo = None
        self.augmented_preview_photo = None

        self._building_scope_values = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(80, self.initialize_from_session)

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Flujo de augmentación con inspección de originales y augmentadas guardadas",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", pady=(0, 10))

        main = ttk.Panedwindow(container, orient="horizontal")
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

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        form = ttk.LabelFrame(parent, text="Configuración general", padding=10)
        form.pack(fill="x")

        ttk.Label(form, text="Carpeta input").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.input_dir_var, width=42).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(form, text="Examinar", command=self.select_input_dir).grid(row=0, column=2)

        ttk.Label(form, text="Carpeta output").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.output_dir_var, width=42).grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(form, text="Examinar", command=self.select_output_dir).grid(row=1, column=2)

        ttk.Label(form, text="Copias por imagen").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.copies_var, width=12).grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(form, text="Seed").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.seed_var, width=12).grid(row=3, column=1, sticky="w", padx=6)

        ttk.Separator(form, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="ew", pady=8)

        ttk.Label(form, text="Editar umbrales de").grid(row=5, column=0, sticky="w", pady=5)
        self.scope_combo = ttk.Combobox(
            form,
            textvariable=self.current_scope_var,
            state="readonly",
            values=self.scope_options,
            width=28
        )
        self.scope_combo.grid(row=5, column=1, sticky="w", padx=6)
        self.scope_combo.bind("<<ComboboxSelected>>", self.on_scope_changed)

        ttk.Button(form, text="Recargar clases", command=self.reload_classes_and_sync_json).grid(row=5, column=2)

        ttk.Label(form, text="Umbral de blanco").grid(row=6, column=0, sticky="w", pady=5)
        self.scope_threshold_entry = ttk.Entry(form, width=12)
        self.scope_threshold_entry.grid(row=6, column=1, sticky="w", padx=6)
        self.scope_threshold_entry.bind("<FocusOut>", self.on_scope_entries_focus_out)

        ttk.Label(form, text="Mín. pixeles objeto").grid(row=7, column=0, sticky="w", pady=5)
        self.scope_min_fg_entry = ttk.Entry(form, width=12)
        self.scope_min_fg_entry.grid(row=7, column=1, sticky="w", padx=6)
        self.scope_min_fg_entry.bind("<FocusOut>", self.on_scope_entries_focus_out)

        ttk.Checkbutton(form, text="Permitir flip horizontal", variable=self.flip_var).grid(
            row=8, column=1, sticky="w", padx=6, pady=6
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

    def _get_detection_params(self) -> tuple[int, int, bool]:
        threshold = int(self.threshold_var.get().strip())
        min_fg = int(self.min_fg_var.get().strip())
        enable_hflip = bool(self.flip_var.get())

        if not (0 <= threshold <= 255):
            raise ValueError("El umbral de blanco global debe estar entre 0 y 255.")
        if min_fg < 1:
            raise ValueError("El mínimo global de pixeles debe ser >= 1.")

        return threshold, min_fg, enable_hflip

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

        default_threshold, default_min_fg, _ = self._get_detection_params()
        save_class_config_json(
            input_dir=input_dir,
            default_white_threshold=default_threshold,
            default_min_foreground_pixels=default_min_fg,
            class_params=self.class_params,
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
            global_threshold, global_min_fg, _ = self._get_detection_params()

            class_name = get_class_name(input_dir, image_path)
            class_params = self.class_params.get(class_name, {})
            used_threshold = int(class_params.get("white_threshold", global_threshold))
            used_min_fg = int(class_params.get("min_foreground_pixels", global_min_fg))

            with Image.open(image_path) as img:
                img = ensure_rgb(img)
                bbox = detect_bbox_white_background(
                    img,
                    white_threshold=used_threshold,
                    min_foreground_pixels=used_min_fg
                )
                preview = build_preview_image(img, bbox, max_size=(820, 290))

            self.original_preview_photo = ImageTk.PhotoImage(preview)
            self.original_preview_label.configure(image=self.original_preview_photo, text="")

            shown_class = "(root)" if class_name == ROOT_CLASS_NAME else class_name

            if bbox is None:
                self.original_preview_info_var.set(
                    f"Imagen: {image_path.name}\n"
                    f"Clase: {shown_class}\n"
                    f"No se detectó bbox con threshold={used_threshold} y min_fg={used_min_fg}."
                )
            else:
                x1, y1, x2, y2 = bbox
                self.original_preview_info_var.set(
                    f"Imagen: {image_path.name}\n"
                    f"Clase: {shown_class}\n"
                    f"BBox original XYXY: [{x1}, {y1}, {x2}, {y2}]\n"
                    f"threshold usado={used_threshold} | min_fg usado={used_min_fg}"
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
                raise ValueError("No se pudieron guardar los valores actuales de umbral.")

            input_dir = self._get_input_dir()
            output_dir = self._get_output_dir()
            copies = int(self.copies_var.get().strip())
            seed = int(self.seed_var.get().strip())
            threshold, min_fg, enable_hflip = self._get_detection_params()

            if input_dir is None or not input_dir.exists():
                raise FileNotFoundError("La carpeta input no existe.")
            if output_dir is None:
                raise FileNotFoundError("Debes seleccionar una carpeta output.")
            if copies < 1:
                raise ValueError("El número de copias debe ser >= 1.")

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
                    f"Originales encontradas: {summary['original_images_found']}\n"
                    f"Originales copiadas: {summary['copied_original_images']}\n"
                    f"Augmentadas creadas: {summary['augmented_images_created']}\n"
                    f"Labels creados: {summary['label_files_created']}\n"
                    f"Manifest: {summary['manifest_path']}\n\n"
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
