from __future__ import annotations

import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from augment_core import (
    ROOT_CLASS_NAME,
    clean_generated_dataset,
    ensure_rgb,
    find_images,
    generate_augmented_dataset_with_bboxes,
    get_class_name,
    get_class_names,
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

3. Si asignas umbrales por clase, esos valores tienen prioridad sobre los globales.
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

7. Si una imagen no detecta objeto, revisa:
   - umbral de blanco
   - mínimo de pixeles
   - calidad del fondo
"""


DEFAULT_SCOPE_KEY = "__default_scope__"


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
        self.geometry("1400x820")
        self.minsize(1260, 760)

        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.copies_var = tk.StringVar(value="5")
        self.seed_var = tk.StringVar(value="42")
        self.flip_var = tk.BooleanVar(value=True)

        # Valores por defecto/globales
        self.default_threshold_var = tk.StringVar(value="245")
        self.default_min_fg_var = tk.StringVar(value="50")

        # Editor visible en UI (cambia según el scope seleccionado)
        self.threshold_editor_var = tk.StringVar(value="245")
        self.min_fg_editor_var = tk.StringVar(value="50")
        self.threshold_scope_var = tk.StringVar(value="General / default")

        self.class_params: dict[str, dict[str, int]] = {}
        self.scope_display_to_key: dict[str, str] = {"General / default": DEFAULT_SCOPE_KEY}
        self.current_scope_key: str = DEFAULT_SCOPE_KEY

        self.original_image_paths: list[Path] = []
        self.augmented_image_paths: list[Path] = []

        self.original_preview_photo = None
        self.augmented_preview_photo = None

        self._build_ui()
        self._refresh_scope_options()

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

        ttk.Separator(form, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 10))

        ttk.Label(form, text="Editar umbrales de").grid(row=5, column=0, sticky="w", pady=5)
        self.scope_combo = ttk.Combobox(
            form,
            textvariable=self.threshold_scope_var,
            state="readonly",
            width=26,
        )
        self.scope_combo.grid(row=5, column=1, sticky="w", padx=6)
        self.scope_combo.bind("<<ComboboxSelected>>", self.on_scope_changed)

        ttk.Button(form, text="Recargar clases", command=self.reload_scope_options).grid(row=5, column=2, sticky="w")

        ttk.Label(form, text="Umbral de blanco").grid(row=6, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.threshold_editor_var, width=12).grid(row=6, column=1, sticky="w", padx=6)

        ttk.Label(form, text="Mín. pixeles objeto").grid(row=7, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.min_fg_editor_var, width=12).grid(row=7, column=1, sticky="w", padx=6)

        ttk.Button(form, text="Guardar valores de esta selección", command=self.save_current_scope_values).grid(
            row=8, column=1, sticky="w", padx=6, pady=(6, 4)
        )

        ttk.Checkbutton(form, text="Permitir flip horizontal", variable=self.flip_var).grid(
            row=9, column=1, sticky="w", padx=6, pady=6
        )

        form.columnconfigure(1, weight=1)

        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=(10, 0))
        ttk.Button(btns, text="Generar augment dataset", command=self.start_generate).pack(side="left")
        ttk.Button(btns, text="Borrar proceso generado", command=self.start_clean).pack(side="left", padx=8)

        tools = ttk.Frame(parent)
        tools.pack(fill="x", pady=(10, 0))
        ttk.Button(tools, text="Ver notas", command=self.open_notes_window).pack(side="left")

        self.class_config_info_var = tk.StringVar(value="Sin configuración específica por clase.")
        ttk.Label(parent, textvariable=self.class_config_info_var, justify="left").pack(anchor="w", pady=(10, 0))

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

    def _display_name_for_class(self, class_name: str) -> str:
        return "General / default" if class_name == DEFAULT_SCOPE_KEY else ("(root)" if class_name == ROOT_CLASS_NAME else class_name)

    def _refresh_scope_options(self) -> None:
        input_dir = Path(self.input_dir_var.get().strip())
        class_names = get_class_names(input_dir) if input_dir.exists() else []

        display_values = ["General / default"]
        self.scope_display_to_key = {"General / default": DEFAULT_SCOPE_KEY}

        for class_name in class_names:
            display = self._display_name_for_class(class_name)
            display_values.append(display)
            self.scope_display_to_key[display] = class_name

        self.scope_combo["values"] = display_values

        # Mantener selección si todavía existe
        current_display = self.threshold_scope_var.get().strip() or "General / default"
        if current_display not in display_values:
            current_display = "General / default"
            self.current_scope_key = DEFAULT_SCOPE_KEY

        self.threshold_scope_var.set(current_display)
        self.current_scope_key = self.scope_display_to_key[current_display]
        self._load_scope_values_into_editor(self.current_scope_key)
        self._update_class_config_label()

    def reload_scope_options(self) -> None:
        try:
            self.save_current_scope_values(show_message=False)
        except Exception:
            pass
        self._refresh_scope_options()

    def _load_scope_values_into_editor(self, scope_key: str) -> None:
        if scope_key == DEFAULT_SCOPE_KEY:
            self.threshold_editor_var.set(self.default_threshold_var.get())
            self.min_fg_editor_var.set(self.default_min_fg_var.get())
            return

        data = self.class_params.get(scope_key, {})
        self.threshold_editor_var.set(str(data.get("white_threshold", self.default_threshold_var.get())))
        self.min_fg_editor_var.set(str(data.get("min_foreground_pixels", self.default_min_fg_var.get())))

    def _validate_editor_values(self) -> tuple[int, int]:
        white_threshold = int(self.threshold_editor_var.get().strip())
        min_fg = int(self.min_fg_editor_var.get().strip())

        if not (0 <= white_threshold <= 255):
            raise ValueError("El umbral de blanco debe estar entre 0 y 255.")
        if min_fg < 1:
            raise ValueError("El mínimo de pixeles del objeto debe ser >= 1.")

        return white_threshold, min_fg

    def save_current_scope_values(self, show_message: bool = True) -> None:
        white_threshold, min_fg = self._validate_editor_values()
        scope_key = self.current_scope_key

        if scope_key == DEFAULT_SCOPE_KEY:
            self.default_threshold_var.set(str(white_threshold))
            self.default_min_fg_var.set(str(min_fg))
        else:
            self.class_params[scope_key] = {
                "white_threshold": white_threshold,
                "min_foreground_pixels": min_fg,
            }

        self._update_class_config_label()

        if show_message:
            shown_name = self._display_name_for_class(scope_key)
            messagebox.showinfo("Guardado", f"Se guardaron los valores de '{shown_name}'.")

    def on_scope_changed(self, event=None) -> None:
        selected_display = self.threshold_scope_var.get().strip()
        new_scope_key = self.scope_display_to_key.get(selected_display, DEFAULT_SCOPE_KEY)

        # Intentar guardar lo actual antes de cambiar
        try:
            self.save_current_scope_values(show_message=False)
        except Exception as e:
            # Restaurar selección previa si falla la validación
            prev_display = self._display_name_for_class(self.current_scope_key)
            self.threshold_scope_var.set(prev_display)
            messagebox.showerror("Error", f"No se pudo cambiar de selección porque los valores actuales no son válidos.\n\n{e}")
            return

        self.current_scope_key = new_scope_key
        self._load_scope_values_into_editor(new_scope_key)

    def _get_default_detection_params(self) -> tuple[int, int, bool]:
        # Asegura que el editor actual quede persistido antes de leer defaults
        self.save_current_scope_values(show_message=False)

        threshold = int(self.default_threshold_var.get().strip())
        min_fg = int(self.default_min_fg_var.get().strip())
        enable_hflip = bool(self.flip_var.get())

        if not (0 <= threshold <= 255):
            raise ValueError("El umbral de blanco global debe estar entre 0 y 255.")
        if min_fg < 1:
            raise ValueError("El mínimo global de pixeles debe ser >= 1.")

        return threshold, min_fg, enable_hflip

    def _update_class_config_label(self) -> None:
        if not self.class_params:
            self.class_config_info_var.set(
                "Default activo: "
                f"threshold={self.default_threshold_var.get()}, "
                f"min_fg={self.default_min_fg_var.get()}\n"
                "Sin configuración específica por clase."
            )
            return

        lines = [
            "Default activo: "
            f"threshold={self.default_threshold_var.get()}, "
            f"min_fg={self.default_min_fg_var.get()}",
            "Configuración específica por clase:"
        ]
        for class_name, data in sorted(self.class_params.items()):
            shown_name = self._display_name_for_class(class_name)
            lines.append(
                f"- {shown_name}: threshold={data['white_threshold']}, min_fg={data['min_foreground_pixels']}"
            )
        self.class_config_info_var.set("\n".join(lines))

    def select_input_dir(self) -> None:
        folder = filedialog.askdirectory(title="Selecciona la carpeta input")
        if folder:
            self.input_dir_var.set(folder)
            self.refresh_original_list()
            self._refresh_scope_options()

    def select_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="Selecciona la carpeta output")
        if folder:
            self.output_dir_var.set(folder)
            self.refresh_augmented_list()

    def refresh_original_list(self) -> None:
        try:
            input_dir = Path(self.input_dir_var.get().strip())
            if not input_dir.exists():
                raise FileNotFoundError("La carpeta input no existe.")

            self.original_image_paths = find_images(input_dir)
            self.original_listbox.delete(0, "end")

            for p in self.original_image_paths:
                self.original_listbox.insert("end", str(p.relative_to(input_dir)))

            self._refresh_scope_options()

            if not self.original_image_paths:
                self.clear_original_preview()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def refresh_augmented_list(self) -> None:
        try:
            output_dir = Path(self.output_dir_var.get().strip())
            aug_dir = output_dir / "augmented_images"

            if not output_dir.exists():
                raise FileNotFoundError("La carpeta output no existe.")
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
            image_path = self.get_selected_original_path()
            if image_path is None:
                self.clear_original_preview()
                return

            input_dir = Path(self.input_dir_var.get().strip())
            global_threshold, global_min_fg, _ = self._get_default_detection_params()

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

            shown_class = self._display_name_for_class(class_name)

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

            output_dir = Path(self.output_dir_var.get().strip())
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
            shown_class = self._display_name_for_class(class_name)
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
            input_dir = Path(self.input_dir_var.get().strip())
            output_dir = Path(self.output_dir_var.get().strip())
            copies = int(self.copies_var.get().strip())
            seed = int(self.seed_var.get().strip())
            threshold, min_fg, enable_hflip = self._get_default_detection_params()

            if not input_dir.exists():
                raise FileNotFoundError("La carpeta input no existe.")
            if copies < 1:
                raise ValueError("El número de copias debe ser >= 1.")

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
                shown_name = self._display_name_for_class(class_name)
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
            output_dir = Path(self.output_dir_var.get().strip())
            if not output_dir.exists():
                raise FileNotFoundError("La carpeta output no existe.")

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
