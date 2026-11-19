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

7. Si una imagen no detecta objeto, revisa:
   - umbral de blanco
   - mínimo de pixeles
   - calidad del fondo
"""


class ThresholdConfigWindow(tk.Toplevel):
    def __init__(self, parent: "AugmentApp") -> None:
        super().__init__(parent)
        self.parent = parent
        self.title("Configurar umbrales por clase")
        self.geometry("760x520")
        self.minsize(700, 460)

        self.entries: dict[str, dict[str, tk.Entry]] = {}

        self._build_ui()
        self.load_classes()

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Umbrales por clase",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(
            container,
            text=(
                "Aquí puedes dar valores específicos por clase.\n"
                "Si una clase usa valores distintos, el detector los aplicará durante la preview y la generación."
            ),
            justify="left"
        ).pack(anchor="w", pady=(0, 10))

        headers = ttk.Frame(container)
        headers.pack(fill="x")
        ttk.Label(headers, text="Clase", width=28).grid(row=0, column=0, sticky="w")
        ttk.Label(headers, text="Umbral de blanco", width=18).grid(row=0, column=1, sticky="w")
        ttk.Label(headers, text="Mín. pixeles objeto", width=18).grid(row=0, column=2, sticky="w")

        self.canvas = tk.Canvas(container, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, side="left")

        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(fill="y", side="right")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.inner = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Button(btns, text="Recargar clases", command=self.load_classes).pack(side="left")
        ttk.Button(btns, text="Usar valores globales", command=self.fill_with_global_values).pack(side="left", padx=8)
        ttk.Button(btns, text="Aplicar", command=self.apply_changes).pack(side="right")

    def _on_inner_configure(self, event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event=None) -> None:
        self.canvas.itemconfig(self.canvas_window, width=self.canvas.winfo_width())

    def load_classes(self) -> None:
        for child in self.inner.winfo_children():
            child.destroy()
        self.entries.clear()

        input_dir = Path(self.parent.input_dir_var.get().strip())
        classes = get_class_names(input_dir) if input_dir.exists() else []

        if not classes:
            ttk.Label(
                self.inner,
                text="No se detectaron clases todavía. Primero selecciona una carpeta input válida.",
                justify="left"
            ).pack(anchor="w", pady=8)
            return

        global_threshold = self.parent.threshold_var.get().strip()
        global_min_fg = self.parent.min_fg_var.get().strip()

        for row, class_name in enumerate(classes):
            shown_name = "(root)" if class_name == ROOT_CLASS_NAME else class_name

            ttk.Label(self.inner, text=shown_name, width=28).grid(row=row, column=0, sticky="w", pady=4)

            th_entry = ttk.Entry(self.inner, width=16)
            minfg_entry = ttk.Entry(self.inner, width=16)

            existing = self.parent.class_params.get(class_name, {})
            th_entry.insert(0, str(existing.get("white_threshold", global_threshold)))
            minfg_entry.insert(0, str(existing.get("min_foreground_pixels", global_min_fg)))

            th_entry.grid(row=row, column=1, sticky="w", padx=4, pady=4)
            minfg_entry.grid(row=row, column=2, sticky="w", padx=4, pady=4)

            self.entries[class_name] = {
                "white_threshold": th_entry,
                "min_foreground_pixels": minfg_entry,
            }

    def fill_with_global_values(self) -> None:
        global_threshold = self.parent.threshold_var.get().strip()
        global_min_fg = self.parent.min_fg_var.get().strip()

        for data in self.entries.values():
            data["white_threshold"].delete(0, "end")
            data["white_threshold"].insert(0, global_threshold)

            data["min_foreground_pixels"].delete(0, "end")
            data["min_foreground_pixels"].insert(0, global_min_fg)

    def apply_changes(self) -> None:
        new_params: dict[str, dict[str, int]] = {}

        try:
            for class_name, widgets in self.entries.items():
                white_threshold = int(widgets["white_threshold"].get().strip())
                min_fg = int(widgets["min_foreground_pixels"].get().strip())

                if not (0 <= white_threshold <= 255):
                    raise ValueError(f"El umbral de blanco de la clase '{class_name}' debe estar entre 0 y 255.")
                if min_fg < 1:
                    raise ValueError(f"El mínimo de pixeles de la clase '{class_name}' debe ser >= 1.")

                new_params[class_name] = {
                    "white_threshold": white_threshold,
                    "min_foreground_pixels": min_fg,
                }

            self.parent.class_params = new_params
            messagebox.showinfo("Éxito", "Configuración por clase aplicada.")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))


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
        self.threshold_var = tk.StringVar(value="245")
        self.min_fg_var = tk.StringVar(value="50")
        self.flip_var = tk.BooleanVar(value=True)

        self.class_params: dict[str, dict[str, int]] = {}

        self.original_image_paths: list[Path] = []
        self.augmented_image_paths: list[Path] = []

        self.original_preview_photo = None
        self.augmented_preview_photo = None

        self._build_ui()

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

        ttk.Label(form, text="Umbral de blanco global").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.threshold_var, width=12).grid(row=4, column=1, sticky="w", padx=6)

        ttk.Label(form, text="Mín. pixeles global").grid(row=5, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.min_fg_var, width=12).grid(row=5, column=1, sticky="w", padx=6)

        ttk.Checkbutton(form, text="Permitir flip horizontal", variable=self.flip_var).grid(
            row=6, column=1, sticky="w", padx=6, pady=6
        )

        form.columnconfigure(1, weight=1)

        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=(10, 0))
        ttk.Button(btns, text="Generar augment dataset", command=self.start_generate).pack(side="left")
        ttk.Button(btns, text="Borrar proceso generado", command=self.start_clean).pack(side="left", padx=8)

        tools = ttk.Frame(parent)
        tools.pack(fill="x", pady=(10, 0))
        ttk.Button(tools, text="Configurar umbrales por clase", command=self.open_threshold_config_window).pack(side="left")
        ttk.Button(tools, text="Ver notas", command=self.open_notes_window).pack(side="left", padx=8)

        self.class_config_info_var = tk.StringVar(value="Sin configuración por clase aplicada.")
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

    def open_threshold_config_window(self) -> None:
        ThresholdConfigWindow(self)

    def _get_detection_params(self) -> tuple[int, int, bool]:
        threshold = int(self.threshold_var.get().strip())
        min_fg = int(self.min_fg_var.get().strip())
        enable_hflip = bool(self.flip_var.get())

        if not (0 <= threshold <= 255):
            raise ValueError("El umbral de blanco global debe estar entre 0 y 255.")
        if min_fg < 1:
            raise ValueError("El mínimo global de pixeles debe ser >= 1.")

        return threshold, min_fg, enable_hflip

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

    def select_input_dir(self) -> None:
        folder = filedialog.askdirectory(title="Selecciona la carpeta input")
        if folder:
            self.input_dir_var.set(folder)
            self.refresh_original_list()

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

            self._update_class_config_label()

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
            input_dir = Path(self.input_dir_var.get().strip())
            output_dir = Path(self.output_dir_var.get().strip())
            copies = int(self.copies_var.get().strip())
            seed = int(self.seed_var.get().strip())
            threshold, min_fg, enable_hflip = self._get_detection_params()

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
