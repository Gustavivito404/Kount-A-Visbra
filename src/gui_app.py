from __future__ import annotations
import json
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from PIL import Image, ImageTk

from .project_core import (
    CLASS_NAMES, find_images, read_annotations, save_annotations,
    infer_folder_class, detect_grayscale_boxes, load_label_objects,
    clear_augmentation_output, clear_yolo_export_output,
)
from .augment_core import generate_augmentations
from .yolo_core import train_yolo, predict_image, export_yolo_dataset
from .camera_utils import scan_cameras, capture_camera
from .profile_store import ProfileManager
from .auth_store import UserStore
from . import aruco_core
from .project_core import set_active_classes


class KountApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Kount-A v15 | Plataforma de inspección')
        # Fit the initial window inside smaller laptop screens while retaining the v15 layout.
        available_w = self.winfo_screenwidth()
        available_h = self.winfo_screenheight()
        self.geometry(f'{min(1500, max(930, available_w-60))}x{min(900, max(600, available_h-85))}')
        self.minsize(930, 600)
        self.project_root = Path(__file__).resolve().parent.parent
        self.profiles = ProfileManager(self.project_root)
        self.users = UserStore(self.project_root)
        self.profiles_location_file = self.project_root / 'config' / 'profiles_location.json'
        # The built-in profiles/ remains part of every new installation.
        try:
            saved_location = json.loads(self.profiles_location_file.read_text(encoding='utf-8')).get('directory')
            if saved_location and Path(saved_location).is_dir():
                self.profiles.set_root(Path(saved_location))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        self.profiles_root_var = tk.StringVar(value=str(self.profiles.root))
        self.view_mode = 'engineer'  # owner may switch to operator view without changing credentials
        self.theme_file = self.project_root / 'config' / 'appearance.json'
        self.theme_name = 'dark'
        try:
            self.theme_name = json.loads(self.theme_file.read_text(encoding='utf-8')).get('theme','dark')
        except (OSError, ValueError):
            pass
        self.role = None
        self.username = None
        self.active_profile = None
        self.message_queue = queue.Queue()
        self.busy = set()
        self.session_path = self.project_root / 'config' / 'ui_session_config.json'
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.status_var = tk.StringVar(value='Inicia sesión para continuar.')
        self.current_image = None
        self.current_pil = None
        self.current_objects = []
        self.selected_bbox = -1
        self.dirty = False
        self.image_files = []
        self.editor_view = {}
        self.preview_refs = {}
        self.aug_orig_files = []
        self.aug_generated_files = []
        self.aug_current_original = None
        self.aug_current_aug = None
        self.test_image = None
        self.test_detections = []
        self.test_display_image = None
        self.available_cameras = []
        self._mouse_start = None
        self._rubber_id = None
        self._configure_vars()
        self._apply_theme()
        self._build_login()
        self.after(100, self._poll_queue)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    def _engineering_view(self):
        return self.role in ('owner', 'engineer') and self.view_mode == 'engineer'

    def _visible_profiles(self):
        return (self.profiles.list_profiles() if self._engineering_view()
                else self.profiles.approved_profiles())

    def _palette(self):
        if self.theme_name == 'light':
            return dict(bg='#FFFFFF', panel='#F2F6FB', field='#FFFFFF', text='#101820',
                        subtle='#526276', blue='#1675D1', blue_hover='#075FAF', border='#CED9E5',
                        canvas='#E8EFF7', selected='#D5E9FC')
        return dict(bg='#181B20', panel='#252A31', field='#2D333C', text='#F1F6FC',
                    subtle='#B8C7D8', blue='#1675D1', blue_hover='#298CE8', border='#46505D',
                    canvas='#151A21', selected='#244D78')

    def _apply_theme(self):
        p = self._palette()
        self.configure(bg=p['bg'])
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('.', background=p['bg'], foreground=p['text'], font=('Segoe UI',10))
        style.configure('TFrame', background=p['bg'])
        style.configure('TLabel', background=p['bg'], foreground=p['text'])
        style.configure('TLabelFrame', background=p['bg'], foreground=p['text'], bordercolor=p['border'])
        style.configure('TLabelFrame.Label', background=p['bg'], foreground=p['blue'], font=('Segoe UI',10,'bold'))
        style.configure('TButton', background=p['panel'], foreground=p['text'], padding=(9,6), bordercolor=p['border'])
        style.map('TButton', background=[('active',p['blue_hover']),('pressed',p['blue'])],
                  foreground=[('active','#FFFFFF')])
        style.configure('Primary.TButton', background=p['blue'], foreground='#FFFFFF', padding=(12,7), bordercolor=p['blue'])
        style.map('Primary.TButton', background=[('active',p['blue_hover'])])
        style.configure('TNotebook', background=p['bg'], bordercolor=p['border'])
        style.configure('TNotebook.Tab', background=p['panel'], foreground=p['text'], padding=(13,9))
        style.map('TNotebook.Tab', background=[('selected',p['blue']),('active',p['blue_hover'])],
                  foreground=[('selected','#FFFFFF'),('active','#FFFFFF')])
        for name in ('TEntry','TCombobox','TSpinbox'):
            style.configure(name, fieldbackground=p['field'], foreground=p['text'],
                            background=p['panel'], bordercolor=p['border'])
        style.map('TCombobox', fieldbackground=[('readonly',p['field'])],
                  foreground=[('readonly',p['text'])])
        style.configure('Treeview', background=p['field'], fieldbackground=p['field'],
                        foreground=p['text'], rowheight=26)
        style.map('Treeview', background=[('selected',p['blue'])], foreground=[('selected','#FFFFFF')])
        style.configure('Treeview.Heading', background=p['panel'], foreground=p['text'])
        style.configure('TCheckbutton', background=p['bg'], foreground=p['text'])
        style.configure('TRadiobutton', background=p['bg'], foreground=p['text'])
        style.configure('TProgressbar', background=p['blue'], troughcolor=p['panel'])
        self._recolor_tk(self)
        try:
            self.theme_file.parent.mkdir(parents=True, exist_ok=True)
            self.theme_file.write_text(json.dumps({'theme':self.theme_name}),encoding='utf-8')
        except OSError:
            pass

    def _recolor_tk(self, parent):
        p = self._palette()
        for obj in parent.winfo_children():
            try:
                if isinstance(obj, tk.Listbox):
                    obj.configure(bg=p['field'], fg=p['text'], selectbackground=p['blue'],
                                  selectforeground='white', highlightbackground=p['border'])
                elif isinstance(obj, tk.Text):
                    obj.configure(bg=p['field'], fg=p['text'], insertbackground=p['text'],
                                  selectbackground=p['blue'], highlightbackground=p['border'])
                elif isinstance(obj, tk.Canvas):
                    obj.configure(bg=p['canvas'])
                elif isinstance(obj, (tk.Entry,)):
                    obj.configure(bg=p['field'], fg=p['text'], insertbackground=p['text'])
            except tk.TclError:
                pass
            self._recolor_tk(obj)

    def _toggle_theme(self):
        self.theme_name = 'light' if self.theme_name == 'dark' else 'dark'
        self._apply_theme()
        if hasattr(self,'theme_button') and self.theme_button.winfo_exists():
            self.theme_button.configure(text='☀ Claro / ☾ Oscuro')

    def _build_login(self):
        self.login_screen = ttk.Frame(self, padding=35)
        self.login_screen.pack(fill='both',expand=True)
        wrapper = ttk.Frame(self.login_screen, padding=30)
        wrapper.place(relx=.5,rely=.49,anchor='center')
        logo_file = self.project_root / 'assets' / 'kounta_logo.png'
        if logo_file.is_file():
            try:
                with Image.open(logo_file) as im:
                    photo = im.convert('RGBA').copy()
                photo.thumbnail((370,185))
                self.login_logo = ImageTk.PhotoImage(photo)
                ttk.Label(wrapper, image=self.login_logo).pack(pady=(0,17))
            except (OSError, ValueError):
                ttk.Label(wrapper,text='KOUNT-A',font=('Segoe UI',34,'bold')).pack(pady=16)
        else:
            ttk.Label(wrapper,text='KOUNT-A',font=('Segoe UI',34,'bold')).pack(pady=16)
        ttk.Label(wrapper,text='Plataforma de inspección y conteo',font=('Segoe UI',12)).pack(pady=(0,20))
        self.login_user = tk.StringVar()
        self.login_pass = tk.StringVar()
        for title, var, hidden in [('Usuario',self.login_user,False),('Contraseña',self.login_pass,True)]:
            ttk.Label(wrapper,text=title).pack(anchor='w',pady=(8,2))
            ent=ttk.Entry(wrapper,textvariable=var,width=37,show='●' if hidden else '')
            ent.pack(fill='x',ipady=5)
            if not hidden:
                ent.focus_set()
        if not self.users.has_users:
            ttk.Label(wrapper,text='Primera ejecución: crea la cuenta del dueño.',
                      foreground=self._palette()['blue']).pack(pady=12)
            label='Crear cuenta del dueño'
        else:
            if not self.users.has_owner:
                ttk.Label(wrapper,text='Actualización: inicia sesión con un ingeniero para activar al dueño.',
                          foreground=self._palette()['blue'],wraplength=360).pack(pady=12)
            label='Iniciar sesión'
        ttk.Button(wrapper,text=label,style='Primary.TButton',command=self._submit_login).pack(fill='x',pady=(20,10))
        self.bind('<Return>',lambda _e:self._submit_login())
        ttk.Button(wrapper,text='☀ Claro / ☾ Oscuro',command=self._toggle_theme).pack(pady=5)
        self._recolor_tk(self)

    def _submit_login(self):
        username,password=self.login_user.get(),self.login_pass.get()
        try:
            if not self.users.has_users:
                self.users.add_user(username,password,'owner')
                role='owner'
            else:
                role=self.users.login(username,password)
                if not role:
                    messagebox.showerror('Acceso','Usuario o contraseña incorrectos.')
                    return
                if not self.users.has_owner and role == 'engineer':
                    if messagebox.askyesno('Actualizar propietario',
                            'Esta instalación anterior no tiene dueño. ¿Quieres convertir '
                            'esta cuenta de ingeniero verificada en dueño?'):
                        self.users.promote_legacy_engineer(username, password)
                        role='owner'
                if not self.users.has_owner:
                    messagebox.showerror('Acceso','Primero es necesario establecer una cuenta del dueño.')
                    return
        except Exception as exc:
            messagebox.showerror('Usuario',str(exc)); return
        self.username=username.strip().lower()
        self.role=role
        self.view_mode='engineer' if role in ('owner','engineer') else 'operator'
        self.unbind('<Return>')
        self.login_screen.destroy()
        self._open_workspace()

    def _open_workspace(self):
        all_profiles=self._visible_profiles()
        if all_profiles:
            self._select_profile(all_profiles[0]['id'],rebuild=False)
        self._build_ui()
        if self._engineering_view():
            self._load_session()
        else:
            self._refresh_operator_profiles()
        self._recolor_tk(self)

    def _configure_vars(self):
        self.bbox_class = tk.StringVar(value=CLASS_NAMES[0])
        self.draw_mode = tk.StringVar(value='Nueva')
        self.gray_threshold = tk.StringVar(value='225')
        self.gray_min_area = tk.StringVar(value='150')
        self.auto_keep = tk.BooleanVar(value=False)
        self.prelabel_keep_existing = tk.BooleanVar(value=True)
        self.bbox_editing_mode = tk.StringVar(value='Edición manual')
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
        self.camera_autofocus = tk.BooleanVar(value=True)
        self.camera_focus = tk.StringVar(value='50')
        self.test_mode = tk.StringVar(value='Combinado')
        self.active_profile_var = tk.StringVar(value='Sin perfil')
        self.aruco_mode_var = tk.StringVar(value='optional')

    def _build_ui(self):
        self.workspace = ttk.Frame(self,padding=8)
        self.workspace.pack(fill='both',expand=True)
        bar=ttk.Frame(self.workspace)
        bar.pack(fill='x',pady=(0,6))
        ttk.Label(bar,text='KOUNT-A  |  v15',font=('Segoe UI',16,'bold'),
                  foreground=self._palette()['blue']).pack(side='left',padx=(0,10))
        role_label={'owner':'Dueño','engineer':'Ingeniero','operator':'Operador'}[self.role]
        ttk.Label(bar,text=f'{self.username} · {role_label}').pack(side='left',padx=(0,8))
        ttk.Label(bar,text='Perfil:').pack(side='left',padx=(4,3))
        self.profile_selector=ttk.Combobox(bar,textvariable=self.active_profile_var,state='readonly',width=22)
        self.profile_selector.pack(side='left',padx=(0,4))
        self.profile_selector.bind('<<ComboboxSelected>>', self._on_profile_combo)
        if self._engineering_view():
            ttk.Button(bar,text='Explorar perfiles…',command=self._choose_profiles_root).pack(side='left',padx=3)
        ttk.Button(bar,text='Cerrar sesión',command=self._logout).pack(side='right',padx=3)
        self.theme_button=ttk.Button(bar,text='☀ Claro / ☾ Oscuro',command=self._toggle_theme)
        self.theme_button.pack(side='right',padx=3)
        if self._engineering_view():
            ttk.Button(bar,text='Recargar imágenes',command=self._refresh_images).pack(side='right',padx=3)
        if self.role=='owner':
            ttk.Button(bar,text='Vista operador' if self._engineering_view() else 'Vista ingeniero',
                       command=self._toggle_owner_view).pack(side='right',padx=3)
        self.nb=ttk.Notebook(self.workspace)
        self.nb.pack(fill='both',expand=True)
        if self._engineering_view():
            names=('1 · Perfiles', '2 · BBox y etiquetado','3 · Augmentación y revisión',
                   '4 · Entrenamiento YOLO','5 · Prueba y conteo','6 · Calibración ArUco')
            if self.role=='owner':
                names+=('7 · Usuarios',)
            tabs=[ttk.Frame(self.nb,padding=6) for _ in names]
            for tab,name in zip(tabs,names): self.nb.add(tab,text=name)
            self._build_profiles_tab(tabs[0])
            self._build_bbox_tab(tabs[1])
            self._build_augment_tab(tabs[2])
            self._build_train_tab(tabs[3])
            self._build_test_tab(tabs[4])
            self._build_aruco_tab(tabs[5])
            if self.role=='owner':
                self._build_users_tab(tabs[6])
        else:
            tab=ttk.Frame(self.nb,padding=6)
            self.nb.add(tab,text='Producción · Captura y análisis')
            self._build_operator_tab(tab)
        ttk.Label(self.workspace,textvariable=self.status_var,anchor='w').pack(fill='x',pady=(6,0))
        self._refresh_profile_selector()

    def _toggle_owner_view(self):
        if self.role != 'owner':
            return
        if self.busy:
            messagebox.showwarning('Proceso activo','Espera a que terminen los procesos.');return
        if self._engineering_view():
            if not self._resolve_dirty():return
            self._save_session()
            self.view_mode='operator'
        else:
            self.view_mode='engineer'
        self.workspace.destroy()
        visible=self._visible_profiles()
        if not self.active_profile or self.active_profile['id'] not in {p['id'] for p in visible}:
            if visible:
                self._select_profile(visible[0]['id'],rebuild=False)
            else:
                self._clear_profile()
        self._build_ui()
        if self._engineering_view():
            self._load_session()
        else:
            self._refresh_operator_profiles()
        self._recolor_tk(self)

    def _choose_profiles_root(self):
        if not self._engineering_view():return
        if self.busy:
            messagebox.showwarning('Proceso activo','Espera a que termine el proceso actual.');return
        if not self._resolve_dirty():return
        selected=filedialog.askdirectory(title='Seleccionar carpeta principal de perfiles',
                                         initialdir=str(self.profiles.root),mustexist=True)
        if not selected:return
        new_root=Path(selected).resolve()
        if new_root==self.profiles.root:return
        try:
            self.profiles.set_root(new_root)
            self.profiles_location_file.parent.mkdir(parents=True,exist_ok=True)
            self.profiles_location_file.write_text(
                json.dumps({'directory':str(self.profiles.root)},ensure_ascii=False,indent=2),encoding='utf-8')
            self.profiles_root_var.set(str(self.profiles.root))
            profiles=self.profiles.list_profiles()
            if profiles:
                self._select_profile(profiles[0]['id'])
            else:
                self._clear_profile()
                self._refresh_profiles_tab()
                self._refresh_images()
            self._refresh_profile_selector()
            self.status_var.set('Carpeta de perfiles: '+str(self.profiles.root))
        except (OSError,ValueError) as exc:
            messagebox.showerror('Carpeta de perfiles',str(exc))

    def _clear_profile(self):
        self.active_profile=None
        self.active_profile_var.set('Sin perfiles')
        self.input_var.set('')
        self.output_var.set('')
        self.current_image=self.current_pil=None
        self.current_objects=[]
        self.image_files=[]
        self.test_image=self.test_display_image=None
        self.test_detections=[]
        self.test_model.set('')
        self.dirty=False
        if self._engineering_view():
            if hasattr(self,'orig_list') and self.orig_list.winfo_exists():
                self.orig_list.delete(0,'end')
                self._update_box_tree()
                self._render_editor()
            if hasattr(self,'classes_list') and self.classes_list.winfo_exists():
                self.classes_list.delete(0,'end')
                self.import_combo.configure(values=[])
                self.model_combo.configure(values=[])
                self.profile_info.set('No hay perfiles en la carpeta seleccionada.')
        if hasattr(self,'test_summary') and self.test_summary is not None:
            self.test_summary.set('No hay perfil seleccionado.')

    def _logout(self):
        if self.busy:
            messagebox.showwarning('Proceso activo','Espera a que terminen los procesos antes de cerrar sesión.')
            return
        if self._engineering_view() and not self._resolve_dirty():return
        if self._engineering_view():
            try:self._save_session()
            except OSError:pass
        self.workspace.destroy()
        self.username=self.role=None
        self.active_profile=None
        self._build_login()

    def _refresh_profile_selector(self):
        profiles=self._visible_profiles()
        self.profile_ids=[p['id'] for p in profiles]
        labels=[p['name'] for p in profiles]
        self.profile_selector.configure(values=labels)
        if self.active_profile and self.active_profile['id'] in self.profile_ids:
            self.profile_selector.current(self.profile_ids.index(self.active_profile['id']))
        else:
            self.profile_selector.set('Elige un perfil' if self._engineering_view() else 'Sin perfiles autorizados')

    def _on_profile_combo(self,_event=None):
        i=self.profile_selector.current()
        if i<0 or i>=len(self.profile_ids):return
        new=self.profile_ids[i]
        if self.active_profile and new==self.active_profile['id']:return
        if self.busy:
            messagebox.showwarning('Proceso activo','Espera a que finalice el proceso antes de cambiar de perfil.')
            self._refresh_profile_selector();return
        if self._engineering_view() and not self._resolve_dirty():
            self._refresh_profile_selector();return
        self._select_profile(new)

    def _select_profile(self, profile_id, rebuild=True):
        cfg=self.profiles.load(profile_id)
        if not self._engineering_view() and cfg not in self.profiles.approved_profiles():
            raise ValueError('Perfil no autorizado.')
        self.active_profile=cfg
        set_active_classes(cfg['classes'])
        folder=self.profiles.profile_dir(profile_id)
        self.input_var.set(str(folder/'input'))
        self.output_var.set(str(folder/'output'))
        self.current_image=self.current_pil=None
        self.current_objects=[]
        self.dirty=False
        self.test_image=None
        self.test_display_image=None
        self.test_detections=[]
        self.last_trained=None
        self.aruco_mode_var.set(cfg['aruco_mode'])
        self.test_mode.set('Combinado')
        if hasattr(self, 'test_mode_combo') and self.test_mode_combo.winfo_exists():
            self.test_mode_combo.configure(values=('Combinado','Comparativo') if cfg['aruco_mode']!='none' else ('Combinado',))
        approved=cfg.get('approved_model')
        self.test_model.set(str(folder/approved) if approved else '')
        if rebuild:
            if self._engineering_view():
                self.bbox_class.set(CLASS_NAMES[0])
                self.bbox_combo.configure(values=CLASS_NAMES)
                self.aug_current_original=self.aug_current_aug=None
                self._refresh_images()
                self._refresh_profiles_tab()
                self._load_aruco_config()
                self.test_summary.set('Sin conteo.')
                self._set_text(self.test_details,'')
                self._rerender_test()
                self._render_editor()
            else:
                self.test_summary.set('Listo para capturar.')
                self._set_text(self.test_details,'')
                self._rerender_test()
            self._refresh_profile_selector()
            if hasattr(self,'test_mode_combo'):
                self.test_mode_combo.configure(values=('Combinado',) if cfg['aruco_mode']=='none' else ('Combinado','Comparativo'))
        self.status_var.set(f'Perfil activo: {cfg["name"]} · {len(cfg["classes"])} clases')

    def _build_profiles_tab(self, tab):
        tab.columnconfigure(0,weight=2);tab.columnconfigure(1,weight=4);tab.rowconfigure(0,weight=1)
        left=ttk.LabelFrame(tab,text='Perfiles del sistema',padding=12)
        left.grid(row=0,column=0,sticky='nsew',padx=4)
        self.profiles_list=tk.Listbox(left,exportselection=False,height=15)
        self.profiles_list.pack(fill='both',expand=True,pady=4)
        self.profiles_list.bind('<<ListboxSelect>>',self._profiles_item_select)
        ttk.Button(left,text='Crear perfil',command=self._create_profile).pack(fill='x',pady=5)
        ttk.Button(left,text='Activar perfil seleccionado',command=self._activate_list_profile).pack(fill='x',pady=5)
        ttk.Button(left,text='Importar proyecto v14 existente',command=self._import_legacy_project).pack(fill='x',pady=5)
        right=ttk.Frame(tab);right.grid(row=0,column=1,sticky='nsew',padx=4)
        right.columnconfigure(0,weight=1)
        classes=ttk.LabelFrame(right,text='Clases del perfil activo',padding=10)
        classes.grid(row=0,column=0,sticky='ew',pady=5)
        self.profile_info=tk.StringVar(value='')
        ttk.Label(classes,textvariable=self.profile_info,wraplength=670).pack(anchor='w',pady=5)
        self.classes_list=tk.Listbox(classes,exportselection=False,height=7)
        self.classes_list.pack(fill='x',pady=5)
        row=ttk.Frame(classes);row.pack(fill='x',pady=4)
        ttk.Button(row,text='Agregar clase',command=self._add_class).pack(side='left',padx=3)
        ttk.Button(row,text='Eliminar clase vacía',command=self._remove_class).pack(side='left',padx=3)
        images=ttk.LabelFrame(right,text='Fotografías originales',padding=10)
        images.grid(row=1,column=0,sticky='ew',pady=5)
        ttk.Label(images,text='Importar imágenes desde Windows a una clase o a mezclas.').pack(anchor='w')
        self.import_class=tk.StringVar()
        self.import_combo=ttk.Combobox(images,textvariable=self.import_class,state='readonly',width=32)
        self.import_combo.pack(fill='x',pady=6)
        ttk.Button(images,text='Importar fotografías al perfil',command=self._import_images).pack(fill='x',pady=3)
        models=ttk.LabelFrame(right,text='Autorización de modelos para producción',padding=10)
        models.grid(row=2,column=0,sticky='ew',pady=5)
        self.model_picker=tk.StringVar()
        self.model_combo=ttk.Combobox(models,textvariable=self.model_picker,state='readonly')
        self.model_combo.pack(fill='x',pady=5)
        row=ttk.Frame(models);row.pack(fill='x')
        ttk.Button(row,text='Actualizar modelos',command=self._refresh_profiles_tab).pack(side='left',padx=3)
        ttk.Button(row,text='Autorizar modelo',command=self._approve_model,style='Primary.TButton').pack(side='left',padx=3)
        ttk.Button(row,text='Revocar autorización',command=self._revoke_model).pack(side='left',padx=3)
        if self.role=='owner':
            users=ttk.LabelFrame(right,text='Administración de usuarios',padding=10)
            users.grid(row=3,column=0,sticky='ew',pady=5)
            ttk.Label(users,text='Gestiona cuentas y contraseñas desde 7 · Usuarios.').pack(anchor='w')
        self._refresh_profiles_tab()

    def _refresh_profiles_tab(self):
        if not hasattr(self,'profiles_list'):return
        available=self.profiles.list_profiles()
        self.profiles_list.delete(0,'end')
        descriptions={'approved':'PRODUCCIÓN','validation':'VALIDACIÓN','development':'DESARROLLO'}
        for cfg in available:
            self.profiles_list.insert('end',f'{cfg["name"]}   ·   {descriptions.get(cfg["status"],cfg["status"])}')
        if self.active_profile:
            active=self.profiles.load(self.active_profile['id'])
            self.active_profile=active
            self.classes_list.delete(0,'end')
            for name in active['classes']:self.classes_list.insert('end',name)
            self.import_combo.configure(values=active['classes']+['mezclas'])
            self.import_class.set(active['classes'][0])
            self.profile_info.set(f'{active["name"]} · {len(active["classes"])} clases · ArUco: {active["aruco_mode"]} '
                                  f'· Estado: {descriptions.get(active["status"],active["status"])}')
            self._available_models=self.profiles.list_models(active['id'])
            self.model_combo.configure(values=[str(p.relative_to(self.profiles.profile_dir(active['id']))) for p in self._available_models])
            if self._available_models:self.model_combo.current(0)
            else:self.model_picker.set('No hay modelos entrenados')
        self._refresh_profile_selector()

    def _profiles_item_select(self,_event=None):
        # Selecting a list row is not activating it; use explicit button.
        pass

    def _activate_list_profile(self):
        selected=self.profiles_list.curselection()
        profiles=self.profiles.list_profiles()
        if not selected or selected[0]>=len(profiles):return
        if self.busy:
            messagebox.showwarning('Proceso activo','Finaliza el proceso actual antes de cambiar de perfil.');return
        if not self._resolve_dirty():return
        self._select_profile(profiles[selected[0]]['id'])

    def _import_legacy_project(self):
        if not self.active_profile or self.busy:return
        if self.active_profile['classes'] != ['tornillo','tuerca','rondana','tornillo_oxidado','tuerca_oxidada','rondana_oxidada']:
            messagebox.showerror('Migración v14','Selecciona un perfil con las seis clases de Tornillería v14.');return
        folder=filedialog.askdirectory(title='Elige la carpeta del proyecto v14 que contiene input y output')
        if not folder:return
        legacy=Path(folder).resolve()
        if not (legacy/'input').is_dir() or not (legacy/'output').is_dir():
            messagebox.showerror('Migración','La carpeta elegida debe contener input/ y output/.');return
        current=self.profiles.profile_dir(self.active_profile['id']).resolve()
        if legacy==current or legacy.is_relative_to(current) or current.is_relative_to(legacy):
            messagebox.showerror('Migración','El proyecto anterior y el perfil deben estar separados.');return
        if find_images(current/'input') or any((current/'input'/'.kounta_annotations').rglob('*.json')) or self.profiles.list_models(self.active_profile['id']):
            messagebox.showerror('Migración','Para evitar sobrescribir datos, importa en un perfil vacío sin imágenes, anotaciones o modelos.');return
        if not messagebox.askyesno('Importar v14','Se COPIARÁN input y output completos, incluyendo BBox, aumentaciones y modelos. No se eliminarán datos antiguos. Puede utilizar bastante espacio en disco. ¿Continuar?'):
            return
        def worker():
            try:
                import shutil
                for item in ('input','output'):
                    shutil.copytree(legacy/item,current/item,dirs_exist_ok=True)
                self._emit('legacy_done',str(legacy))
            except Exception as exc:
                self._emit('error',f'Importación v14: {exc}')
        self._start_worker('migration',worker)

    def _create_profile(self):
        if self.busy:
            messagebox.showwarning('Proceso activo','Espera al proceso actual.');return
        name=simpledialog.askstring('Nuevo perfil','Nombre del nuevo perfil:',parent=self)
        if not name:return
        classes=simpledialog.askstring('Clases','Clases iniciales separadas por coma (ej.: luneta_fija, luneta_movil):',parent=self)
        if not classes:return
        choice=simpledialog.askstring('ArUco','Modo ArUco: none / optional / required\n(none = sin ArUco; optional = opcional; required = obligatorio):',initialvalue='none',parent=self)
        if choice is None:return
        try:
            cfg=self.profiles.create(name,[x.strip() for x in classes.split(',')],choice.strip().lower())
            self._select_profile(cfg['id'])
            self._refresh_profiles_tab()
        except Exception as exc:messagebox.showerror('Crear perfil',str(exc))

    def _add_class(self):
        if not self.active_profile:return
        if self.busy:
            messagebox.showwarning('Proceso activo','Espera antes de cambiar clases.');return
        name=simpledialog.askstring('Nueva clase','Nombre de la clase:',parent=self)
        if not name:return
        try:
            self.profiles.add_class(self.active_profile['id'],name)
            self._select_profile(self.active_profile['id'])
        except Exception as exc:messagebox.showerror('Clase',str(exc))

    def _remove_class(self):
        if not self.active_profile:return
        selected=self.classes_list.curselection()
        if not selected:return
        name=self.classes_list.get(selected[0])
        if not messagebox.askyesno('Eliminar clase',f'¿Eliminar la clase {name}? Solo se permite si no hay datos dependientes.'):
            return
        try:
            self.profiles.remove_class(self.active_profile['id'],name)
            self._select_profile(self.active_profile['id'])
        except Exception as exc:messagebox.showerror('Eliminar clase',str(exc))

    def _import_images(self):
        if not self.active_profile:return
        if self.busy or not self._resolve_dirty():
            messagebox.showwarning('Proceso activo','Termina el proceso o guarda las BBox antes de importar.')
            return
        paths=filedialog.askopenfilenames(title='Importar fotografías',filetypes=[('Imágenes','*.jpg *.jpeg *.png *.bmp *.webp')])
        if not paths:return
        try:
            count=self.profiles.import_images(self.active_profile['id'],[Path(p) for p in paths],self.import_class.get())
            self._refresh_images()
            messagebox.showinfo('Importación',f'Se importaron {count} imágenes a {self.import_class.get()}.')
        except Exception as exc:messagebox.showerror('Importación',str(exc))

    def _approve_model(self):
        if not self.active_profile:return
        chosen=self.model_combo.current()
        if chosen<0 or chosen>=len(self._available_models):return
        if not messagebox.askyesno('Autorizar producción','Se aprobará esta versión del modelo para operadores. ¿Continuar?'):
            return
        if self.busy:
            messagebox.showwarning('Proceso activo','Termina el proceso actual.');return
        try:
            self.profiles.approve(self.active_profile['id'],self._available_models[chosen])
            self._refresh_profiles_tab()
            messagebox.showinfo('Autorizado','El perfil ya está disponible para el operador.')
        except Exception as exc:messagebox.showerror('Autorización',str(exc))

    def _revoke_model(self):
        if not self.active_profile:return
        if not messagebox.askyesno('Retirar autorización','El perfil dejará de estar disponible para operadores. ¿Continuar?'):
            return
        self.profiles.revoke(self.active_profile['id'])
        self._refresh_profiles_tab()

    def _build_users_tab(self, tab):
        if self.role!='owner':return
        tab.columnconfigure(0,weight=1)
        tab.rowconfigure(1,weight=1)
        ttk.Label(tab,text='Administración de usuarios · Solo dueño',
                  font=('Segoe UI',13,'bold')).grid(row=0,column=0,sticky='w',padx=9,pady=8)
        self.users_tree=ttk.Treeview(tab,columns=('username','role','enabled'),show='headings',height=11)
        for key,heading,width in [('username','Usuario',230),('role','Rol',170),('enabled','Acceso',130)]:
            self.users_tree.heading(key,text=heading)
            self.users_tree.column(key,width=width,anchor='w')
        self.users_tree.grid(row=1,column=0,sticky='nsew',padx=9,pady=5)
        row=ttk.Frame(tab)
        row.grid(row=2,column=0,sticky='w',padx=9,pady=10)
        ttk.Button(row,text='Crear usuario',style='Primary.TButton',command=self._add_user).pack(side='left',padx=(0,7))
        ttk.Button(row,text='Renombrar usuario',command=self._rename_user).pack(side='left',padx=7)
        ttk.Button(row,text='Cambiar rol',command=self._change_user_role).pack(side='left',padx=7)
        ttk.Button(row,text='Cambiar contraseña',command=self._reset_user_password).pack(side='left',padx=7)
        ttk.Button(row,text='Activar / desactivar',command=self._toggle_user_status).pack(side='left',padx=7)
        self._refresh_users_table()

    def _refresh_users_table(self):
        if self.role!='owner' or not hasattr(self,'users_tree') or not self.users_tree.winfo_exists():return
        for item in self.users_tree.get_children():self.users_tree.delete(item)
        for record in self.users.user_details():
            self.users_tree.insert('', 'end', iid=record['username'],
                values=(record['username'], {'owner':'Dueño','engineer':'Ingeniero','operator':'Operador'}[record['role']],
                        'Activo' if record['enabled'] else 'Desactivado'))

    def _selected_user(self):
        if self.role!='owner':return None
        selection=self.users_tree.selection()
        if not selection:
            messagebox.showwarning('Usuarios','Selecciona un usuario de la tabla.')
            return None
        return selection[0]

    def _add_user(self):
        if self.role!='owner':return
        username=simpledialog.askstring('Crear usuario','Usuario (mínimo 3 caracteres):',parent=self)
        if not username:return
        password=simpledialog.askstring('Crear usuario','Contraseña (mínimo 8 caracteres):',parent=self,show='*')
        if password is None:return
        role=simpledialog.askstring('Crear usuario','Rol: owner / engineer / operator',
                                    initialvalue='operator',parent=self)
        if role is None:return
        try:
            self.users.add_user(username,password,role.strip().lower())
            self._refresh_users_table()
            messagebox.showinfo('Usuario','Cuenta creada correctamente.')
        except Exception as exc:messagebox.showerror('Usuario',str(exc))

    def _rename_user(self):
        name=self._selected_user()
        if not name:return
        replacement=simpledialog.askstring('Renombrar usuario',f'Nuevo nombre para {name}:',
                                           initialvalue=name,parent=self)
        if not replacement or replacement.strip().lower()==name:return
        try:
            new=self.users.update_user(name,new_username=replacement)
            if name==self.username:
                self.username=new
                messagebox.showinfo('Usuario','Tu nombre de cuenta cambió. Se reflejará en el próximo inicio de sesión.')
            self._refresh_users_table()
        except Exception as exc:messagebox.showerror('Usuario',str(exc))

    def _change_user_role(self):
        name=self._selected_user()
        if not name:return
        role=simpledialog.askstring('Cambiar rol',f'Nuevo rol de {name}: owner / engineer / operator',
                                    parent=self)
        if role is None:return
        try:
            if name==self.username:
                raise ValueError('No cambies tu propio rol durante la sesión. Utiliza otra cuenta de dueño.')
            self.users.update_user(name,role=role.strip().lower())
            self._refresh_users_table()
        except Exception as exc:messagebox.showerror('Usuario',str(exc))

    def _reset_user_password(self):
        name=self._selected_user()
        if not name:return
        password=simpledialog.askstring('Nueva contraseña',f'Nueva contraseña para {name}:',
                                        parent=self,show='*')
        if password is None:return
        confirm=simpledialog.askstring('Confirmar contraseña','Repite la nueva contraseña:',
                                       parent=self,show='*')
        if confirm is None:return
        if password!=confirm:
            messagebox.showerror('Contraseña','Las contraseñas no coinciden.');return
        try:
            self.users.update_user(name,password=password)
            messagebox.showinfo('Contraseña','Contraseña actualizada.')
        except Exception as exc:messagebox.showerror('Contraseña',str(exc))

    def _toggle_user_status(self):
        name=self._selected_user()
        if not name:return
        record=next((r for r in self.users.user_details() if r['username']==name),None)
        if record is None:return
        if name==self.username:
            messagebox.showwarning('Usuarios','No desactives tu cuenta mientras tienes sesión iniciada.');return
        state=not record['enabled']
        if not messagebox.askyesno('Acceso',f'¿{"Activar" if state else "Desactivar"} a {name}?'):
            return
        try:
            self.users.update_user(name,enabled=state)
            self._refresh_users_table()
        except Exception as exc:messagebox.showerror('Usuario',str(exc))

    def _build_aruco_tab(self, tab):
        tab.columnconfigure(0,weight=1)
        group=ttk.LabelFrame(tab,text='Configuración por perfil — 4 marcadores ArUco',padding=15)
        group.grid(row=0,column=0,sticky='nw',padx=10,pady=10)
        fields=[('Distancia centro ID 0 → ID 1 (mm)','reference_width_mm'),
                ('Distancia centro ID 0 → ID 3 (mm)','reference_height_mm'),
                ('Lado negro del marcador (mm)','marker_size_mm'),
                ('Error RMSE máximo (mm)','max_rmse_mm'),
                ('Umbral de objetos (0–255)','threshold'),
                ('Área mínima (px²)','min_area_px'),('Área máxima (px²)','max_area_px')]
        self.aruco_vars={key:tk.StringVar() for _,key in fields}
        ttk.Label(group,text='Uso de ArUco en este perfil').grid(row=0,column=0,sticky='w',pady=7)
        ttk.Combobox(group,textvariable=self.aruco_mode_var,
                     values=('none','optional','required'),state='readonly',width=22).grid(row=0,column=1,padx=7,sticky='w')
        for i,(text,key) in enumerate(fields,1):
            ttk.Label(group,text=text).grid(row=i,column=0,sticky='w',pady=7)
            ttk.Entry(group,textvariable=self.aruco_vars[key],width=20).grid(row=i,column=1,sticky='w',padx=7)
        ttk.Button(group,text='Guardar calibración del perfil',command=self._save_aruco_config,
                   style='Primary.TButton').grid(row=len(fields)+1,column=0,columnspan=2,sticky='ew',pady=12)
        ttk.Label(group,text='0 → 1: ancho · 0 → 3: alto · ID 2: esquina inferior derecha\n'
                        'Con opcional: faltan marcadores → YOLO cuenta toda la foto, sin medidas.\n'
                        'Con required: una calibración inválida bloquea la inspección.',
                  wraplength=560).grid(row=len(fields)+2,column=0,columnspan=2,sticky='w')
        self._load_aruco_config()

    def _load_aruco_config(self):
        if not self.active_profile or not hasattr(self,'aruco_vars'):return
        cfg=self.profiles.load(self.active_profile['id'])
        self.aruco_mode_var.set(cfg['aruco_mode'])
        for key,var in self.aruco_vars.items():
            value=cfg['aruco'].get(key)
            var.set('' if value is None else str(value))

    def _save_aruco_config(self):
        if not self.active_profile:return
        if self.busy:
            messagebox.showwarning('Proceso activo','Espera a que finalice el proceso.');return
        try:
            cfg=self.profiles.load(self.active_profile['id'])
            mode=self.aruco_mode_var.get()
            if mode not in ('none','optional','required'):raise ValueError('Modo ArUco no válido.')
            values={}
            for key,var in self.aruco_vars.items():
                value=var.get().strip()
                values[key]=None if not value and key in ('reference_width_mm','reference_height_mm') else float(value)
            if values['reference_width_mm'] is not None and values['reference_width_mm']<=0:raise ValueError('El ancho debe ser positivo.')
            if values['reference_height_mm'] is not None and values['reference_height_mm']<=0:raise ValueError('El alto debe ser positivo.')
            if any(values[k]<=0 for k in ('marker_size_mm','max_rmse_mm','min_area_px','max_area_px')):
                raise ValueError('Las medidas, tolerancia y áreas deben ser positivas.')
            if not 0<=values['threshold']<=255:raise ValueError('El umbral debe estar entre 0 y 255.')
            if values['min_area_px']>=values['max_area_px']:raise ValueError('El área mínima debe ser menor que el área máxima.')
            cfg['aruco_mode']=mode
            cfg['aruco'].update(values)
            self.profiles.save(cfg)
            self.active_profile=cfg
            self._refresh_profiles_tab()
            messagebox.showinfo('Calibración','Parámetros guardados en el perfil.')
        except Exception as exc:messagebox.showerror('Calibración',str(exc))

    def _build_operator_tab(self,tab):
        tab.columnconfigure(0,weight=7);tab.columnconfigure(1,weight=3);tab.rowconfigure(1,weight=1)
        bar=ttk.LabelFrame(tab,text='Cámara y operación',padding=12)
        bar.grid(row=0,column=0,columnspan=2,sticky='ew',pady=5)
        ttk.Button(bar,text='Buscar cámaras',command=self._start_camera_scan).grid(row=0,column=0,padx=5,pady=5)
        self.camera_combo=ttk.Combobox(bar,textvariable=self.camera_choice,state='readonly',width=42)
        self.camera_combo.grid(row=0,column=1,columnspan=3,sticky='ew',padx=5)
        ttk.Label(bar,text='Ancho').grid(row=0,column=4,padx=3)
        ttk.Entry(bar,textvariable=self.camera_width,width=8).grid(row=0,column=5)
        ttk.Label(bar,text='Alto').grid(row=0,column=6,padx=3)
        ttk.Entry(bar,textvariable=self.camera_height,width=8).grid(row=0,column=7)
        ttk.Label(bar,text='Modo').grid(row=1,column=0,padx=4)
        self.test_mode_combo=ttk.Combobox(bar,textvariable=self.test_mode,values=('Combinado','Comparativo'),state='readonly',width=18)
        self.test_mode_combo.grid(row=1,column=1,sticky='w',padx=5)
        ttk.Button(bar,text='CAPTURAR Y ANALIZAR',style='Primary.TButton',
                   command=lambda:self._start_capture(True)).grid(row=1,column=2,columnspan=2,sticky='ew',padx=5,pady=9)
        ttk.Checkbutton(bar,text='Enfoque automático',variable=self.camera_autofocus).grid(row=2,column=0,columnspan=2,sticky='w',pady=5)
        ttk.Label(bar,text='Foco manual (0–255)').grid(row=2,column=2,sticky='w')
        ttk.Entry(bar,textvariable=self.camera_focus,width=8).grid(row=2,column=3,sticky='w')
        ttk.Label(bar,textvariable=self.test_source,wraplength=700).grid(row=3,column=0,columnspan=8,sticky='w',pady=5)
        bar.columnconfigure(3,weight=1)
        show=ttk.LabelFrame(tab,text='Fotografía y detecciones',padding=8)
        show.grid(row=1,column=0,sticky='nsew',padx=4)
        show.columnconfigure(0,weight=1);show.rowconfigure(0,weight=1)
        self.test_canvas=tk.Canvas(show,highlightthickness=0)
        self.test_canvas.grid(row=0,column=0,sticky='nsew')
        self.test_canvas.bind('<Configure>',lambda _e:self._rerender_test())
        result=ttk.LabelFrame(tab,text='Resultados del lote',padding=10)
        result.grid(row=1,column=1,sticky='nsew',padx=4)
        result.columnconfigure(0,weight=1);result.rowconfigure(1,weight=1)
        self.test_summary=tk.StringVar(value='Listo para comenzar.')
        ttk.Label(result,textvariable=self.test_summary,font=('Segoe UI',15,'bold'),
                  wraplength=320).grid(row=0,column=0,sticky='w',pady=7)
        self.test_details=tk.Text(result,state='disabled',wrap='word',width=37)
        self.test_details.grid(row=1,column=0,sticky='nsew')

    def _refresh_operator_profiles(self):
        self._refresh_profile_selector()
        if not self.active_profile:
            self.test_summary.set('No hay perfiles autorizados. Solicita su aprobación a ingeniería.')

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
        selector=ttk.LabelFrame(right,text='Modo de etiquetado',padding=7)
        selector.grid(row=0,column=0,sticky='ew')
        self.bbox_mode_combo=ttk.Combobox(selector,textvariable=self.bbox_editing_mode,
            values=('Edición manual','Detección auxiliar: escala de grises',
                    'Preetiquetado con modelo YOLO'),state='readonly')
        self.bbox_mode_combo.pack(fill='x',pady=2)
        self.bbox_mode_combo.bind('<<ComboboxSelected>>',self._show_bbox_mode)
        self.bbox_mode_panels={}
        pane=ttk.LabelFrame(right,text='Edición manual',padding=7)
        self.bbox_mode_panels['Edición manual']=pane
        ttk.Label(pane,text='Clase de la BBox').pack(anchor='w')
        self.bbox_combo=ttk.Combobox(pane,textvariable=self.bbox_class,values=CLASS_NAMES,state='readonly')
        self.bbox_combo.pack(fill='x',pady=2)
        ttk.Label(pane,text='Modo al arrastrar').pack(anchor='w',pady=(4,0))
        ttk.Combobox(pane,textvariable=self.draw_mode,values=['Nueva','Reemplazar seleccionada'],
                     state='readonly').pack(fill='x',pady=2)
        btnrow=ttk.Frame(pane);btnrow.pack(fill='x',pady=3)
        ttk.Button(btnrow,text='Aplicar clase',command=self._change_selected_class).pack(side='left')
        ttk.Button(btnrow,text='Eliminar BBox',command=self._delete_selected).pack(side='left',padx=4)
        ttk.Button(pane,text='Guardar BBox de esta imagen',command=self._save_current).pack(fill='x',pady=3)

        auto=ttk.LabelFrame(right,text='Detección auxiliar: escala de grises',padding=7)
        self.bbox_mode_panels['Detección auxiliar: escala de grises']=auto
        row=ttk.Frame(auto);row.pack(fill='x')
        ttk.Label(row,text='Umbral').pack(side='left')
        ttk.Entry(row,width=6,textvariable=self.gray_threshold).pack(side='left',padx=4)
        ttk.Label(row,text='Área mínima').pack(side='left')
        ttk.Entry(row,width=7,textvariable=self.gray_min_area).pack(side='left',padx=4)
        ttk.Checkbutton(auto,text='Conservar cajas existentes (imagen actual)',
                        variable=self.auto_keep).pack(anchor='w',pady=3)
        ttk.Button(auto,text='Proponer BBox en esta imagen',command=self._auto_gray_current).pack(fill='x',pady=2)
        ttk.Button(auto,text='Auto gris en fotos sin etiqueta (lote)',
                   command=self._auto_gray_batch).pack(fill='x',pady=2)

        pre=ttk.LabelFrame(right,text='Preetiquetado con modelo YOLO',padding=7)
        self.bbox_mode_panels['Preetiquetado con modelo YOLO']=pre
        ttk.Entry(pre,textvariable=self.prelabel_model).pack(fill='x')
        ttk.Button(pre,text='Seleccionar best.pt',
                   command=lambda:self._choose_model(self.prelabel_model)).pack(fill='x',pady=3)
        line=ttk.Frame(pre);line.pack(fill='x')
        ttk.Label(line,text='Confianza').pack(side='left')
        ttk.Entry(line,textvariable=self.prelabel_conf,width=8).pack(side='left',padx=6)
        ttk.Checkbutton(pre,text='Conservar imágenes ya etiquetadas (lote)',
                        variable=self.prelabel_keep_existing).pack(anchor='w',pady=3)
        ttk.Button(pre,text='Preetiquetar foto seleccionada',command=self._start_prelabel).pack(fill='x',pady=2)
        ttk.Button(pre,text='Preetiquetar todas las fotografías',style='Primary.TButton',
                   command=self._start_prelabel_batch).pack(fill='x',pady=3)

        boxes = ttk.LabelFrame(right, text='Objetos de la fotografía', padding=6)
        boxes.grid(row=2, column=0, sticky='nsew', pady=(5, 0))
        right.rowconfigure(2, weight=1)
        self.box_tree = ttk.Treeview(boxes, columns=('clase', 'xyxy'), show='headings', height=5)
        self.box_tree.heading('clase', text='Clase')
        self.box_tree.heading('xyxy', text='BBox XYXY')
        self.box_tree.column('clase', width=130)
        self.box_tree.column('xyxy', width=175)
        self.box_tree.pack(fill='both', expand=True)
        self.box_tree.bind('<<TreeviewSelect>>', self._on_box_select)
        self._show_bbox_mode()

    def _show_bbox_mode(self, _event=None):
        mode=self.bbox_editing_mode.get()
        if mode not in self.bbox_mode_panels:
            mode='Edición manual'
            self.bbox_editing_mode.set(mode)
        for panel in self.bbox_mode_panels.values():
            panel.grid_forget()
        self.bbox_mode_panels[mode].grid(row=1,column=0,sticky='ew',pady=(5,0))
        # The annotated-image table remains visible with every mode.

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
        ttk.Checkbutton(toolbar, text='Enfoque automático',variable=self.camera_autofocus).grid(row=6,column=0,columnspan=2,sticky='w',pady=4)
        ttk.Label(toolbar,text='Foco manual (0–255)').grid(row=6,column=2,columnspan=2)
        ttk.Entry(toolbar,textvariable=self.camera_focus,width=7).grid(row=6,column=4,sticky='w')
        ttk.Label(toolbar, text='Modo').grid(row=5, column=0, sticky='w', pady=4)
        self.test_mode_combo=ttk.Combobox(toolbar, textvariable=self.test_mode, values=('Combinado','Comparativo'), state='readonly', width=20)
        self.test_mode_combo.grid(row=5,column=1,sticky='w')
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
        messagebox.showinfo('Perfil','Las carpetas se administran desde la pestaña Perfiles.')
        return
        if not self._resolve_dirty():
            return
        path = filedialog.askdirectory(title='Carpeta de imágenes originales')
        if path:
            self.input_var.set(path)
            self._refresh_images()

    def _choose_output(self):
        messagebox.showinfo('Perfil','Las carpetas se administran desde la pestaña Perfiles.')
        return
        if not self._resolve_dirty():
            return
        path = filedialog.askdirectory(title='Carpeta output')
        if path:
            self.output_var.set(path)
            self._refresh_images()

    def _refresh_images(self):
        if not self._engineering_view():return
        if not self.active_profile:
            self.image_files=[]
            self.orig_list.delete(0,'end')
            if hasattr(self,'aug_orig_list'):
                self.aug_orig_list.delete(0,'end')
                self.aug_gen_list.delete(0,'end')
            return
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
                color = '#FFD53F' if i == selected else colors[CLASS_NAMES.index(obj['class_name']) % len(colors)]
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
        if self.busy:
            messagebox.showwarning('Proceso activo','Espera a que finalice el proceso anterior.');return
        if not self.active_profile:return
        if not messagebox.askyesno('Autodetección por lote',
                   'Solo se propondrán BBox para fotos sin anotaciones anteriores. '
                   'Debes revisar visualmente las etiquetas generadas. ¿Continuar?'):
            return
        if not self._resolve_dirty():
            return
        try:
            inp, out = self._get_paths()
            threshold, area = int(self.gray_threshold.get()), int(self.gray_min_area.get())
            if not (0<=threshold<=255 and area>=1):
                raise ValueError('Umbral 0–255; área >= 1')
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

    def _start_prelabel_batch(self):
        if self.busy:
            messagebox.showwarning('Proceso activo','Espera a que terminen los procesos anteriores.');return
        try:
            inp,out=self._get_paths()
            model_path=self.prelabel_model.get().strip()
            if not model_path or not Path(model_path).is_file():
                raise ValueError('Selecciona primero un modelo YOLO best.pt válido.')
            confidence=float(self.prelabel_conf.get())
            if not 0<confidence<1:
                raise ValueError('La confianza debe ser mayor que 0 y menor que 1.')
            files=list(self.image_files)
            if not files:raise ValueError('Este perfil todavía no contiene fotografías.')
            keep_existing=self.prelabel_keep_existing.get()
            allowed=set(CLASS_NAMES)
            if not messagebox.askyesno('Preetiquetado masivo',
                f'Se analizarán {len(files)} fotografías del perfil activo.\n'
                +('Se conservarán las anotaciones ya existentes.' if keep_existing else
                  'ATENCIÓN: se reemplazarán las anotaciones existentes si hay detecciones.')
                +'\nLas nuevas BBox deben revisarse manualmente. ¿Continuar?'):
                return
        except (OSError,ValueError) as exc:
            messagebox.showerror('Preetiquetado YOLO',str(exc));return
        if not self._resolve_dirty():return
        self.status_var.set(f'YOLO por lote: iniciando {len(files)} imágenes…')
        def worker():
            try:
                from .yolo_core import prelabel_images_batch
                result=prelabel_images_batch(files,inp,out,model_path,confidence,
                                             allowed,keep_existing,
                                             progress=lambda done,total,updated:self._emit(
                                               'prelabel_batch_progress',(done,total,updated)))
                self._emit('prelabel_batch_done',result)
            except Exception as exc:
                self._emit('error',f'Preetiquetado por lote: {exc}')
        self._start_worker('prelabel_batch',worker)

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
            self.test_display_image = None
            self.test_source.set(path)
            self.test_summary.set('Imagen cargada. Pulsa Contar objetos.')
            self._rerender_test()
        except Exception as exc:
            messagebox.showerror('Imagen', str(exc))

    def _rerender_test(self):
        if self.test_display_image is not None:
            self._render_image(self.test_canvas,self.test_display_image,[], 'test')
        else:
            boxes = [{'class_name': d['class_name'], 'bbox_xyxy': d['bbox_xyxy']}
                     for d in self.test_detections if d['class_name'] in CLASS_NAMES]
            self._render_image(self.test_canvas, self.test_image, boxes, 'test')

    def _start_test_count(self):
        if self.test_image is None:
            messagebox.showinfo('Conteo','Selecciona una foto o captura una imagen de cámara.');return
        if not self.active_profile:
            messagebox.showwarning('Perfil','Selecciona un perfil autorizado.');return
        try:
            cfg=dict(self.active_profile)
            if not self._engineering_view():
                approved=cfg.get('approved_model')
                if not approved:
                    raise ValueError('El perfil no dispone de un modelo autorizado.')
                model=str(self.profiles.profile_dir(cfg['id'])/approved)
            else:
                model=self.test_model.get().strip()
            image=self.test_image.copy()
            conf=float(self.test_conf.get());iou=float(self.test_iou.get());maxdet=int(self.test_maxdet.get())
            mode='Combinado' if cfg['aruco_mode']=='none' else self.test_mode.get()
        except Exception as exc:
            messagebox.showerror('Parámetros',str(exc));return
        self.test_summary.set('Analizando fotografía...')
        def worker():
            try:
                calibration = aruco_core.analyze(image,cfg['aruco']) if cfg['aruco_mode']!='none' else None
                if calibration is not None and cfg['aruco_mode']=='required' and not calibration['valid']:
                    result={'detections':[],'counts':{},'total':0,'preview':calibration['preview'],
                            'calibration':{'detected':calibration['detected'],'valid':False,
                                           'reason':calibration['reason'],'rmse_mm':calibration['rmse_mm']},
                            'blocked':True,'contour_count':None}
                else:
                    result=predict_image(image,model,conf=conf,iou=iou,max_det=maxdet)
                    if calibration is not None:
                        result=aruco_core.combine(image,result,calibration,compare=(mode=='Comparativo'))
                        if mode=='Comparativo' and not calibration['valid']:
                            result['comparison_unavailable']=True
                self._emit('test_done',result)
            except Exception as exc:
                self._emit('error',f'Identificación: {exc}')
        self._start_worker('test',worker)

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
            autofocus=self.camera_autofocus.get(); focus=int(self.camera_focus.get())
            if not 0<=focus<=255: raise ValueError('Foco manual entre 0 y 255.')
        except Exception as exc:
            messagebox.showerror('Cámaras', str(exc)); return
        def worker():
            try:
                img = capture_camera(index, width, height, autofocus=autofocus, manual_focus=focus)
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
                    if hasattr(self,'train_log') and self.train_log.winfo_exists():
                        self._append_train_log('ERROR: ' + data)
                    messagebox.showerror('Kount-A', data)
                elif event == 'legacy_done':
                    if self._engineering_view():
                        self._refresh_images()
                        self._refresh_profiles_tab()
                    messagebox.showinfo('Importación v14','Copia completada. Revisa tus imágenes, BBox y modelos antes de entrenar.')
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
                elif event == 'prelabel_batch_progress':
                    done,total,updated=data
                    self.status_var.set(f'YOLO por lote: {done}/{total} · {updated} imágenes etiquetadas')
                elif event == 'prelabel_batch_done':
                    self.status_var.set(f'YOLO por lote terminado: {data["updated"]} etiquetadas, '
                                        f'{data["skipped"]} conservadas, {data["empty"]} sin detecciones.')
                    if self.current_image and not self.dirty:
                        inp,out=self._get_paths()
                        self.current_objects=read_annotations(inp,self.current_image,out)
                        self._update_box_tree()
                        self._render_editor()
                    messagebox.showinfo('Preetiquetado YOLO',self.status_var.get())
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
                    if self.active_profile and self.active_profile['status']!='approved':
                        cfg=self.profiles.load(self.active_profile['id'])
                        cfg['status']='validation'
                        self.profiles.save(cfg)
                        self.active_profile=cfg
                    self.last_trained = data['best_model']
                    self.train_progress.configure(value=100)
                    self.train_state.set(f'Modelo listo: {data["best_model"]}')
                    self._append_train_log('Entrenamiento completo. best.pt = ' + data['best_model'])
                    self._copy_latest_model()
                    if self._engineering_view():self._refresh_profiles_tab()
                elif event == 'test_done':
                    self.test_detections = data['detections']
                    self.test_display_image = data.get('preview')
                    self._rerender_test()
                    self.test_summary.set('Inspección bloqueada' if data.get('blocked') else f'Total: {data["total"]}')
                    lines = ['CONTEO POR CLASE:']
                    if data.get('contour_count') is not None:
                        lines.extend(['',f'Conteo por contornos OpenCV: {data["contour_count"]}',
                                      f'Diferencia YOLO - OpenCV: {data["total"]-data["contour_count"]}'])
                    if 'calibration' in data:
                        cal=data['calibration']
                        lines.insert(0,f'ArUco: {cal["detected"]}/4 · '+('VÁLIDO' if cal['valid'] else 'SIN CALIBRACIÓN'))
                        lines.insert(1,cal['reason'])
                        if not cal['valid']:
                            lines.insert(2,'SIN MEDIDAS: cuando ArUco es opcional, YOLO cuenta la imagen completa.')
                    if data.get('comparison_unavailable'):
                        lines.append('Comparación deshabilitada: calibración ArUco no válida.')
                    if data.get('blocked'):
                        lines=['INSPECCIÓN BLOQUEADA: '+data['calibration']['reason']]

                    lines += [f'{cls}: {count}' for cls, count in sorted(data['counts'].items())]
                    lines += ['', 'DETECCIONES:']
                    lines += [f'{i+1}. {d["class_name"]}: {d["score"]*100:.1f}%\n   BBox {d["bbox_xyxy"]}' +
                              (f'\n   Largo {d["length_mm"]:.1f} mm · Ancho {d["width_mm"]:.1f} mm'
                               if d.get('length_mm') is not None else '')
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
                    self.test_display_image = None
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
            profile_id=obj.get('active_profile')
            if profile_id and profile_id in [p['id'] for p in self.profiles.list_profiles()]:
                self._select_profile(profile_id)
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
            'gray_min_area', 'bbox_editing_mode', 'prelabel_keep_existing', 'test_conf', 'test_iou', 'test_maxdet', 'test_mode',
            'camera_width','camera_height','camera_autofocus','camera_focus',
        ]
        payload = {'active_profile': self.active_profile['id'] if self.active_profile else None,
                   'settings': {name: getattr(self, name).get() for name in names if name not in ('test_model','prelabel_model')}}
        self.session_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def _on_close(self):
        if self.busy and not messagebox.askyesno(
            'Procesos en marcha',
            'Hay procesos ejecutándose. Al cerrar se pueden interrumpir y perder resultados. '
            '¿Deseas cerrar igualmente?'
        ):
            return
        if self._engineering_view() and not self._resolve_dirty():
            return
        try:
            if self._engineering_view(): self._save_session()
        except OSError:
            pass
        self.destroy()


def main():
    app = KountApp()
    app.mainloop()


if __name__ == '__main__':
    main()
