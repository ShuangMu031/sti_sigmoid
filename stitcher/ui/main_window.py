import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import time
from pathlib import Path
import multiprocessing as mp
from datetime import datetime
import queue

if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW("Eagle Eye Stitcher - 控制台")
    except Exception:
        pass

try:
    from stitcher.common.logger import get_logger, setup_logger
    from stitcher.io.image_io import cv_imread, cv_imwrite
    from stitcher.workers import run_stitching_worker
except ImportError:
    def get_logger(*args, **kwargs):
        import logging
        return logging.getLogger(__name__)

    def setup_logger(*args, **kwargs): pass

    logger = get_logger(__name__)

    def cv_imread(path): return cv2.imread(path)
    def cv_imwrite(path, img): return cv2.imwrite(path, img)
    def run_stitching_worker(*args): pass

logs_dir = Path(__file__).parent.parent.parent / 'logs'
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / f"img_stitcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
setup_logger('stitcher', 'INFO', str(log_file), True)
logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  Eagle Eye · 双区配色系统
#
#  左侧控制面板  ──  鹰羽白（温暖浅色）
#  右侧画布区域  ──  深夜蓝（中深色，不过暗）
#  金色分隔线    ──  鹰眼金，贯穿过渡
# ══════════════════════════════════════════════════════════════════════════════

# ── 左面板：鹰羽白 ───────────────────────────────────────────────
P_BG    = "#F5F1E8"   # 暖白底色（羽毛白）
P_MID   = "#EDE7D5"   # 参数区底色
P_DEEP  = "#D8CEB8"   # 深一档，边框/分隔
P_TEXT  = "#1E2540"   # 主文字：深海蓝
P_DIM   = "#7A7060"   # 次要文字：暖灰

# ── 右侧：深夜蓝（比上版本浅，不刺眼）────────────────────────────
R_BG     = "#1E2A3E"  # 主窗口背景
R_FRAME  = "#243044"  # LabelFrame 背景
R_CANVAS = "#151F30"  # 画布（最暗，凸显图像）

# ── 鹰眼金系列 ─────────────────────────────────────────────────
GOLD       = "#C8A84B"  # 标准金
GOLD_LIGHT = "#E8C866"  # 高亮金
GOLD_DARK  = "#7A6020"  # 暗金
GOLD_PALE  = "#D4BC78"  # 浅金（用于浅色面板上的点缀）

# ── 动作按钮 ───────────────────────────────────────────────────
STEEL      = "#3A6A9B"   # 钢铁蓝（选择）
STEEL_DARK = "#1E4A75"
AMBER      = "#B86018"   # 琥珀橙（清除）
AMBER_DARK = "#8A4810"
JADE       = "#2A7A5A"   # 苍玉绿（保存）
JADE_DARK  = "#1A5A3E"

# ── 右侧文字 ───────────────────────────────────────────────────
T_BRIGHT = "#EDE8DC"  # 亮文字
T_GOLD   = "#C8A84B"  # 金色文字
T_MED    = "#8A8EA8"  # 中灰文字
T_DIM    = "#4A4E60"  # 暗灰文字

SEP_COLOR = GOLD      # 分隔线颜色
# ══════════════════════════════════════════════════════════════════════════════


class StitchingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Eagle Eye Stitcher 🦅")

        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets', 'icons', 'cat_backup.ico')
            if os.path.exists(icon_path):
                try: self.root.iconbitmap(default=icon_path.replace('/', '\\'))
                except: pass
                try:
                    icon_image = Image.open(icon_path).resize((32, 32), Image.LANCZOS)
                    self.icon_photo = ImageTk.PhotoImage(icon_image)
                    self.root.wm_iconphoto(True, self.icon_photo)
                except: pass
        except Exception as e:
            logger.error(f"无法设置窗口图标: {e}")

        self.root.geometry("1100x750")
        self.root.minsize(900, 650)
        self.root.configure(bg=R_BG)

        self.image_paths = []
        self.result_image = None
        self.thumbnail_refs = []
        self.result_photo = None
        self.button_icons = self._load_button_icons()

        self.worker_process = None
        self.progress_queue = None
        self.result_queue = None
        self.is_processing = False
        self.temp_result_path = None
        self.last_stitching_time = 0.0

        self._anim_running = False
        self._anim_job_id = None
        self.animal_emojis = ["🦅", "⚡", "🦅", "✦"]

        self._create_widgets()
        self._layout_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        logger.info("GUI界面已初始化（鹰眼双区版）")

    def _load_button_icons(self):
        icons = {}
        icon_dir = Path(__file__).parent.parent.parent / 'assets' / 'icons'
        icon_files = {
            'select': 'select_images.png',
            'stitch': 'stitch.png',
            'clear':  'clear.png',
            'save':   'save.png'
        }
        for key, filename in icon_files.items():
            icon_path = icon_dir / filename
            if icon_path.exists():
                try:
                    image = Image.open(icon_path).resize((18, 18), Image.LANCZOS)
                    icons[key] = ImageTk.PhotoImage(image)
                except Exception as e:
                    logger.warning(f"加载按钮图标失败 {icon_path}: {e}")
        return icons

    def _create_widgets(self):
        self._create_menu()

        # ── 顶层框架 ──────────────────────────────────────────────────────────
        self.main_frame = tk.Frame(self.root, bg=R_BG)

        # ── 左面板（浅色鹰羽白）────────────────────────────────────────────────
        self.control_frame = tk.Frame(self.main_frame, bg=P_BG, padx=12, pady=12)

        # ── 金色分隔线（1px + 渐变感留白）──────────────────────────────────────
        self.sep_outer = tk.Frame(self.main_frame, bg=R_BG, width=18)
        self.sep_line  = tk.Frame(self.sep_outer, bg=SEP_COLOR, width=2)

        # ── 右侧容器（深夜蓝）──────────────────────────────────────────────────
        self.right_container = tk.Frame(self.main_frame, bg=R_BG)

        # ══ ttk 样式配置 ══════════════════════════════════════════════════════
        style = ttk.Style()
        style.theme_use('clam')

        FONT_MAIN  = ("Microsoft YaHei UI", 10)
        FONT_BOLD  = ("Microsoft YaHei UI", 10, "bold")
        FONT_SMALL = ("Microsoft YaHei UI", 9)

        # 右侧 LabelFrame（深色卡片）
        style.configure("TFrame",         background=R_BG)
        style.configure("TLabel",         background=R_FRAME, foreground=T_MED,  font=FONT_SMALL)
        style.configure("TLabelFrame",    background=R_FRAME, foreground=T_GOLD, font=FONT_BOLD,
                        bordercolor="#2E3D58", relief="flat", borderwidth=1)
        style.configure("TLabelFrame.Label", background=R_FRAME, foreground=T_GOLD, font=FONT_BOLD)

        # 左面板内部用的 ttk 滑块与进度条
        style.configure("TScale",
            background=P_MID, troughcolor=P_DEEP,
            sliderrelief="flat", sliderlength=18)
        style.configure("Custom.Horizontal.TProgressbar",
            troughcolor=P_DEEP, background=GOLD, thickness=7, borderwidth=0)
        style.configure("TCombobox",
            fieldbackground=P_BG, background=P_MID,
            foreground=P_TEXT, selectbackground=GOLD_DARK, selectforeground=P_BG,
            bordercolor=P_DEEP, arrowcolor=GOLD_DARK)
        style.map("TCombobox",
            fieldbackground=[('readonly', P_BG)],
            foreground=[('readonly', P_TEXT)])

        # ── 按钮 ──────────────────────────────────────────────────────────────
        # Select  → 钢铁蓝
        style.configure("Select.TButton", padding=10, relief="flat",
            background=STEEL, foreground=T_BRIGHT, font=FONT_BOLD, borderwidth=0)
        style.map("Select.TButton",
            background=[('active', "#4A80B8"), ('pressed', STEEL_DARK), ('disabled', "#283848")],
            foreground=[('disabled', "#4A5870")])

        # Stitch  → 鹰羽金（最醒目的主操作）
        style.configure("Stitch.TButton", padding=10, relief="flat",
            background=GOLD, foreground=P_TEXT, font=FONT_BOLD, borderwidth=0)
        style.map("Stitch.TButton",
            background=[('active', GOLD_LIGHT), ('pressed', GOLD_DARK), ('disabled', P_DEEP)],
            foreground=[('active', "#0D0800"), ('disabled', P_DIM)])

        # Clear   → 琥珀橙
        style.configure("Clear.TButton", padding=10, relief="flat",
            background=AMBER, foreground=T_BRIGHT, font=FONT_BOLD, borderwidth=0)
        style.map("Clear.TButton",
            background=[('active', "#CE7028"), ('pressed', AMBER_DARK), ('disabled', "#302010")],
            foreground=[('disabled', "#604030")])

        # Save    → 苍玉绿
        style.configure("Save.TButton", padding=10, relief="flat",
            background=JADE, foreground=T_BRIGHT, font=FONT_BOLD, borderwidth=0)
        style.map("Save.TButton",
            background=[('active', "#3A9A6A"), ('pressed', JADE_DARK), ('disabled', "#162418")],
            foreground=[('disabled', "#304038")])

        # ══ 品牌头部（左面板顶部）══════════════════════════════════════════════
        self.brand_frame = tk.Frame(self.control_frame, bg=P_BG)
        # 顶部金条
        self.brand_top_bar = tk.Frame(self.brand_frame, bg=GOLD, height=3)
        # 主标题行
        self.brand_title_row = tk.Frame(self.brand_frame, bg=P_BG)
        self.brand_icon  = tk.Label(self.brand_title_row, text="🦅",
            font=("Segoe UI Emoji", 20), bg=P_BG, fg=P_TEXT, anchor="w")
        self.brand_name  = tk.Label(self.brand_title_row, text="EAGLE EYE",
            font=("Microsoft YaHei UI", 14, "bold"), bg=P_BG, fg=P_TEXT, anchor="w")
        # 副标题
        self.brand_sub   = tk.Label(self.brand_frame, text="S T I T C H E R  ·  全景拼接引擎",
            font=("Microsoft YaHei UI", 7), bg=P_BG, fg=P_DIM, anchor="w")
        # 底部分隔
        self.brand_divider = tk.Frame(self.brand_frame, bg=P_DEEP, height=1)

        # ══ 四个主按钮 ══════════════════════════════════════════════════════════
        self.select_images_btn = ttk.Button(
            self.control_frame, text="选择图像", command=self.select_images,
            image=self.button_icons.get('select'), compound=tk.LEFT, style="Select.TButton")
        self.stitch_btn = ttk.Button(
            self.control_frame, text="开始拼接", command=self.start_stitching, state=tk.DISABLED,
            image=self.button_icons.get('stitch'), compound=tk.LEFT, style="Stitch.TButton")
        self.clear_btn = ttk.Button(
            self.control_frame, text="清除数据", command=self.clear_data,
            image=self.button_icons.get('clear'), compound=tk.LEFT, style="Clear.TButton")
        self.save_btn = ttk.Button(
            self.control_frame, text="保存结果", command=self.save_result, state=tk.DISABLED,
            image=self.button_icons.get('save'), compound=tk.LEFT, style="Save.TButton")

        # ══ 参数调节区域 ════════════════════════════════════════════════════════
        # 用纯 tk 控件保证颜色准确
        self.params_frame = tk.LabelFrame(
            self.control_frame, text="  ⚙  参数调节  ",
            bg=P_MID, fg=GOLD_DARK,
            font=("Microsoft YaHei UI", 9, "bold"),
            bd=1, relief="groove", padx=8, pady=8)

        self.seam_band_label = tk.Label(self.params_frame, text="融合带宽:",
            bg=P_MID, fg=P_DIM, font=("Microsoft YaHei UI", 9))
        self.seam_band_var = tk.IntVar(value=9)
        self.seam_band_scale = ttk.Scale(self.params_frame, from_=1, to=20,
            orient='horizontal', variable=self.seam_band_var, length=145)
        self.seam_band_value_label = tk.Label(self.params_frame, text="9",
            bg=GOLD_DARK, fg=GOLD_LIGHT,
            font=("Arial", 9, "bold"), width=3, relief="flat", padx=5)
        self.seam_band_var.trace_add('write', self._update_seam_band_value)

        self.feather_radius_label = tk.Label(self.params_frame, text="羽化半径:",
            bg=P_MID, fg=P_DIM, font=("Microsoft YaHei UI", 9))
        self.feather_radius_var = tk.IntVar(value=11)
        self.feather_radius_scale = ttk.Scale(self.params_frame, from_=1, to=30,
            orient='horizontal', variable=self.feather_radius_var, length=145)
        self.feather_radius_value_label = tk.Label(self.params_frame, text="11",
            bg=STEEL_DARK, fg="#A0C8E8",
            font=("Arial", 9, "bold"), width=3, relief="flat", padx=5)
        self.feather_radius_var.trace_add('write', self._update_feather_radius_value)

        self.gc_mode_label = tk.Label(self.params_frame, text="拼接模式:",
            bg=P_MID, fg=P_DIM, font=("Microsoft YaHei UI", 9))
        self.gc_mode_var = tk.StringVar(value="normal")
        self.gc_mode_combo = ttk.Combobox(
            self.params_frame,
            textvariable=self.gc_mode_var,
            values=["normal", "professional"],
            state="readonly", width=14,
            font=("Microsoft YaHei UI", 9))
        self.gc_mode_combo.bind("<<ComboboxSelected>>", self._on_gc_mode_changed)

        # 分隔线
        self.params_divider = tk.Frame(self.params_frame, bg=P_DEEP, height=1)

        self.stitching_time_label = tk.Label(self.params_frame, text="⏱  拼接时间: 0.00s",
            font=("Arial", 9), fg=GOLD_DARK, bg=P_MID)

        self.progress_frame_inner = tk.Frame(self.params_frame, bg=P_MID)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame_inner,
            variable=self.progress_var, maximum=100, mode='determinate',
            style="Custom.Horizontal.TProgressbar")
        self.progress_label = tk.Label(self.progress_frame_inner, text="就绪",
            fg=P_DIM, bg=P_MID, font=("Microsoft YaHei UI", 8))

        # ══ 状态栏 ══════════════════════════════════════════════════════════════
        self.status_var = tk.StringVar(value="🦅 待命")
        self.status_label = tk.Label(
            self.control_frame, textvariable=self.status_var,
            relief="flat", anchor=tk.W, padx=10, pady=6,
            bg=P_DEEP, fg=P_TEXT,
            font=("Microsoft YaHei UI", 9))

        # ══ 雄鹰动画区 ══════════════════════════════════════════════════════════
        self.animal_frame = tk.Frame(self.control_frame, bg=P_BG)
        # 金色顶线
        self.animal_top_line = tk.Frame(self.animal_frame, bg=GOLD_PALE, height=1)
        self.animal_label = tk.Label(
            self.animal_frame, text="🦅",
            font=("Segoe UI Emoji", 32), bg=P_BG, anchor="center")
        self.animal_hint_label = tk.Label(
            self.animal_frame, text='点击"开始拼接"雄鹰起飞！',
            font=("Microsoft YaHei UI", 8), bg=P_BG, fg=P_DIM)

        # ══ 右侧：图像区 + 结果区 ════════════════════════════════════════════════
        self.images_frame = tk.LabelFrame(
            self.right_container, text="   已选择图像  ",
            bg=R_FRAME, fg=T_GOLD,
            font=("Microsoft YaHei UI", 9, "bold"),
            bd=1, relief="flat", padx=8, pady=8)
        self.images_canvas = tk.Canvas(
            self.images_frame, bg=R_CANVAS, highlightthickness=0)
        self.images_scrollbar_x = ttk.Scrollbar(
            self.images_frame, orient="horizontal", command=self.images_canvas.xview)
        self.images_scrollbar_y = ttk.Scrollbar(
            self.images_frame, orient="vertical", command=self.images_canvas.yview)
        self.images_canvas.configure(
            xscrollcommand=self.images_scrollbar_x.set,
            yscrollcommand=self.images_scrollbar_y.set)

        self.result_frame = tk.LabelFrame(
            self.right_container, text="   拼接结果  ",
            bg=R_FRAME, fg=T_GOLD,
            font=("Microsoft YaHei UI", 9, "bold"),
            bd=1, relief="flat", padx=8, pady=8)
        self.result_canvas = tk.Canvas(
            self.result_frame, bg=R_CANVAS, highlightthickness=0)
        self.result_scrollbar_x = ttk.Scrollbar(
            self.result_frame, orient="horizontal", command=self.result_canvas.xview)
        self.result_scrollbar_y = ttk.Scrollbar(
            self.result_frame, orient="vertical", command=self.result_canvas.yview)
        self.result_canvas.configure(
            xscrollcommand=self.result_scrollbar_x.set,
            yscrollcommand=self.result_scrollbar_y.set)

    def _update_seam_band_value(self, *args):
        self.seam_band_value_label.config(text=str(self.seam_band_var.get()))

    def _update_feather_radius_value(self, *args):
        self.feather_radius_value_label.config(text=str(self.feather_radius_var.get()))

    def _on_gc_mode_changed(self, *args):
        mode = self.gc_mode_var.get()
        if mode == "professional":
            self.status_var.set("🔧 精准模式：耗时较长，效果更锐利")
        else:
            self.status_var.set("⚡ 快速模式：迅猛出击，效果良好")

    def _create_menu(self):
        self.menu_bar  = tk.Menu(self.root)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="选择图像", command=self.select_images)
        self.file_menu.add_command(label="保存结果", command=self.save_result, state=tk.DISABLED)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="退出", command=self.on_close)
        self.menu_bar.add_cascade(label="文件", menu=self.file_menu)

        self.edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.edit_menu.add_command(label="清除数据", command=self.clear_data)
        self.menu_bar.add_cascade(label="编辑", menu=self.edit_menu)

        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(label="关于", command=self.show_about)
        self.menu_bar.add_cascade(label="帮助", menu=self.help_menu)
        self.root.config(menu=self.menu_bar)

    def _layout_widgets(self):
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # ── 左面板 ─────────────────────────────────────────────────────────────
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.control_frame.pack_propagate(False)
        self.control_frame.configure(width=272)

        # 品牌头部
        self.brand_frame.pack(fill=tk.X, pady=(0, 12))
        self.brand_top_bar.pack(fill=tk.X)
        self.brand_title_row.pack(fill=tk.X, pady=(8, 2))
        self.brand_icon.pack(side=tk.LEFT, padx=(4, 6))
        self.brand_name.pack(side=tk.LEFT)
        self.brand_sub.pack(fill=tk.X, padx=4, pady=(0, 8))
        self.brand_divider.pack(fill=tk.X)

        # 按钮
        btn_kw = dict(fill=tk.X, pady=5, padx=2)
        self.select_images_btn.pack(**btn_kw)
        self.stitch_btn.pack(**btn_kw)
        self.clear_btn.pack(**btn_kw)
        self.save_btn.pack(**btn_kw)

        # 参数区
        self.params_frame.pack(fill=tk.X, pady=(12, 0), padx=2)

        self.seam_band_label.grid(       row=0, column=0, sticky=tk.W, pady=(4, 2))
        self.seam_band_scale.grid(       row=1, column=0, sticky=tk.EW, pady=2)
        self.seam_band_value_label.grid( row=1, column=1, sticky=tk.E, padx=(6, 0), pady=2)

        self.feather_radius_label.grid(       row=2, column=0, sticky=tk.W, pady=(10, 2))
        self.feather_radius_scale.grid(       row=3, column=0, sticky=tk.EW, pady=2)
        self.feather_radius_value_label.grid( row=3, column=1, sticky=tk.E, padx=(6, 0), pady=2)

        self.gc_mode_label.grid( row=4, column=0, sticky=tk.W, pady=(10, 2))
        self.gc_mode_combo.grid( row=5, column=0, columnspan=2, sticky=tk.EW, pady=2)

        self.params_divider.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=(10, 6))

        self.stitching_time_label.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))

        self.progress_frame_inner.grid(row=8, column=0, columnspan=2, sticky=tk.EW, pady=(0, 4))
        self.progress_bar.pack(fill=tk.X, pady=2)
        self.progress_label.pack(anchor=tk.W)

        self.params_frame.columnconfigure(0, weight=1)

        # 雄鹰区（底部）
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM, pady=(8, 0))
        self.animal_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 4))
        self.animal_top_line.pack(fill=tk.X)
        self.animal_label.pack(pady=(6, 2))
        self.animal_hint_label.pack(pady=(0, 6))

        # ── 金色竖向分隔线 ──────────────────────────────────────────────────────
        self.sep_outer.pack(side=tk.LEFT, fill=tk.Y)
        self.sep_line.place(relx=0.5, rely=0.05, relheight=0.9, anchor="n")

        # ── 右侧区域 ───────────────────────────────────────────────────────────
        self.right_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.images_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        self.images_frame.configure(height=220)
        self.images_frame.pack_propagate(False)

        self.result_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 0))

        self.images_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.images_scrollbar_y.pack(side=tk.RIGHT,  fill=tk.Y)
        self.images_canvas.pack(fill=tk.BOTH, expand=True)

        self.result_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.result_scrollbar_y.pack(side=tk.RIGHT,  fill=tk.Y)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)

    # ══ 动画控制 ════════════════════════════════════════════════════════════════
    def _start_animal_animation(self):
        if self._anim_running:
            return
        self._anim_running = True
        self.animal_hint_label.config(text="雄鹰翱翔，全力拼接中...🦅")
        self._anim_index = 0
        self._anim_loop()

    def _anim_loop(self):
        if not self._anim_running:
            return
        emoji = self.animal_emojis[self._anim_index % len(self.animal_emojis)]
        self.animal_label.config(text=emoji)
        self._anim_index += 1
        self._anim_job_id = self.root.after(150, self._anim_loop)

    def _celebrate_animal(self):
        self._anim_running = False
        if self._anim_job_id:
            self.root.after_cancel(self._anim_job_id)
            self._anim_job_id = None
        self.animal_label.config(text="🏆")
        self.animal_hint_label.config(text="拼接完成！雄鹰凯旋！✦")
        self._celebrate_loop(0)

    def _celebrate_loop(self, count):
        if count >= 5:
            self.animal_label.config(text="🦅")
            self.animal_hint_label.config(text='完成！点击"清除数据"重新起飞~')
            return
        emojis = ["⚡", "✦", "🏆", "🎯", "🦅"]
        self.animal_label.config(text=emojis[count % len(emojis)])
        self.root.after(200, lambda: self._celebrate_loop(count + 1))

    # ══ 核心功能 ════════════════════════════════════════════════════════════════
    def select_images(self):
        if self.is_processing:
            messagebox.showinfo("提示", "当前正在拼接，请等待当前任务完成。")
            return
        file_paths = filedialog.askopenfilenames(
            title="选择多张图像进行拼接",
            filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("所有文件", "*.*")]
        )
        if file_paths:
            self.image_paths = list(file_paths)
            self._display_selected_images(self.image_paths)
            self.stitch_btn.config(state=tk.NORMAL if len(self.image_paths) >= 2 else tk.DISABLED)
            logger.info(f"用户选择了 {len(self.image_paths)} 张图像")

    def _display_selected_images(self, file_paths):
        self.images_canvas.delete("all")
        self.thumbnail_refs = []
        thumbnail_size = (150, 100)
        margin = 10
        self.images_frame_inner = tk.Frame(self.images_canvas, bg=R_CANVAS)
        self.images_canvas.create_window((0, 0), window=self.images_frame_inner, anchor=tk.NW)

        for i, file_path in enumerate(file_paths):
            try:
                row = (i // 3) * 2
                col = i % 3
                image = Image.open(file_path)
                image.thumbnail(thumbnail_size, Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.thumbnail_refs.append(photo)
                img_label = tk.Label(self.images_frame_inner, image=photo,
                    bg=R_CANVAS, bd=1, relief="flat")
                img_label.grid(row=row, column=col, padx=margin, pady=margin)
                file_name = os.path.basename(file_path)
                name_label = tk.Label(self.images_frame_inner, text=file_name,
                    wraplength=thumbnail_size[0], bg=R_CANVAS, fg=T_MED,
                    font=("Microsoft YaHei UI", 8))
                name_label.grid(row=row + 1, column=col, padx=margin, pady=2)
                num_label = tk.Label(self.images_frame_inner, text=f"{i + 1}",
                    bg=GOLD_DARK, fg=GOLD_LIGHT, width=2, font=("Arial", 8, "bold"))
                num_label.grid(row=row, column=col, padx=margin, pady=margin, sticky=tk.NW)
            except Exception as e:
                logger.error(f"无法显示图像 {file_path}: {str(e)}")
        self.images_frame_inner.update_idletasks()
        self.images_canvas.configure(scrollregion=self.images_canvas.bbox("all"))

    def start_stitching(self):
        if self.is_processing or len(self.image_paths) < 2: return
        self._set_processing_state(True)
        self.progress_var.set(0)
        self.progress_label.config(text="准备启动...")
        self.status_var.set("🦅 正在拼接...")
        self.root.title("Eagle Eye Stitcher (正在处理)")
        self._start_animal_animation()
        self.root.update_idletasks()

        config = {
            'SEAM_BAND':     self.seam_band_var.get(),
            'FEATHER_RADIUS': self.feather_radius_var.get(),
            'GC_MODE':       self.gc_mode_var.get()
        }
        output_dir = Path(__file__).parent.parent.parent / 'outputs'
        output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_result_path = str(
            output_dir / f"mp_result_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")

        self.progress_queue = mp.Queue()
        self.result_queue   = mp.Queue()
        self.worker_process = mp.Process(
            target=run_stitching_worker,
            args=(list(self.image_paths), config,
                  self.progress_queue, self.result_queue, self.temp_result_path))
        self.worker_process.start()
        logger.info(f"拼接子进程已启动，PID={self.worker_process.pid}")
        self._poll_worker_messages()

    def _poll_worker_messages(self):
        self._drain_progress_queue()
        result_message = self._get_result_message()
        if result_message:
            self._handle_result_message(result_message)
            return
        if self.is_processing:
            self.root.after(100, self._poll_worker_messages)

    def _drain_progress_queue(self):
        try:
            while True:
                msg = self.progress_queue.get_nowait()
                if msg and msg[0] == "progress":
                    self.progress_var.set(msg[1])
                    self.progress_label.config(text=msg[2])
        except (queue.Empty, AttributeError): pass

    def _get_result_message(self):
        try: return self.result_queue.get_nowait()
        except (queue.Empty, AttributeError): return None

    def _handle_result_message(self, message):
        if message[0] == "done":  self._handle_worker_success(message[1], message[2])
        elif message[0] == "error": self._handle_worker_error(message[1])

    def _handle_worker_success(self, output_path, stitching_time):
        self.root.title("Eagle Eye Stitcher")
        self.stitching_time_label.config(text=f"⏱  拼接时间: {stitching_time:.2f}s")
        result = cv_imread(output_path)
        if result is not None:
            self.result_image = result
            self._display_result(self.result_image)
            self.status_var.set("✅ 拼接完成")
            self.progress_var.set(100)
            self.progress_label.config(text="完成")
            self.save_btn.config(state=tk.NORMAL)
        self._celebrate_animal()
        self._set_processing_state(False)
        self._cleanup_worker_resources(remove_temp_file=False)
        self._cleanup_temp_result()
        messagebox.showinfo("成功", "图像拼接完成！🦅 雄鹰锐视，全景已成！")

    def _handle_worker_error(self, error_message):
        self.root.title("Eagle Eye Stitcher")
        self.status_var.set("❌ 拼接出错")
        self.progress_label.config(text="失败")
        self.animal_label.config(text="🦅")
        self.animal_hint_label.config(text="拼接受阻，调整参数再试！")
        self._set_processing_state(False)
        self._cleanup_worker_resources(remove_temp_file=True)
        logger.error(f"拼接失败: {error_message}")
        messagebox.showerror("错误", f"拼接过程中发生错误:\n{error_message}")

    def _set_processing_state(self, is_processing):
        self.is_processing = is_processing
        if is_processing:
            self.stitch_btn.config(state=tk.DISABLED)
            self.select_images_btn.config(state=tk.DISABLED)
            self.clear_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)
        else:
            self.select_images_btn.config(state=tk.NORMAL)
            self.clear_btn.config(state=tk.NORMAL)
            if len(self.image_paths) >= 2: self.stitch_btn.config(state=tk.NORMAL)
            else: self.stitch_btn.config(state=tk.DISABLED)

    def _cleanup_worker_resources(self, remove_temp_file=False):
        process = self.worker_process
        if process is not None:
            try:
                if process.is_alive(): process.join(timeout=0.2)
            except: pass
        self.worker_process = None
        for q in (self.progress_queue, self.result_queue):
            if q is None: continue
            try: q.close()
            except: pass
            try: q.cancel_join_thread()
            except: pass
        self.progress_queue = None
        self.result_queue   = None
        if remove_temp_file: self._cleanup_temp_result()

    def _cleanup_temp_result(self):
        if self.temp_result_path and os.path.exists(self.temp_result_path):
            try: os.remove(self.temp_result_path)
            except OSError as e:
                logger.warning(f"无法删除临时结果文件 {self.temp_result_path}: {e}")
        self.temp_result_path = None

    def _display_result(self, result_image):
        try:
            self.result_canvas.delete("all")
            if isinstance(result_image, np.ndarray):
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                result_image = Image.fromarray(result_image)
            self.result_canvas.update_idletasks()
            cw = max(self.result_canvas.winfo_width()  - 20, 100)
            ch = max(self.result_canvas.winfo_height() - 20, 100)
            iw, ih = result_image.size
            scale = min(cw / iw, ch / ih, 1.0)
            nw, nh = max(int(iw * scale), 1), max(int(ih * scale), 1)
            display_image = result_image.resize((nw, nh), Image.LANCZOS)
            photo = ImageTk.PhotoImage(display_image)
            self.result_photo = photo
            self.result_canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER)
            self.result_canvas.configure(scrollregion=(0, 0, nw, nh))
            logger.info(f"显示拼接结果，尺寸: {nw}x{nh}")
        except Exception as e:
            logger.error(f"显示结果时出错: {str(e)}")
            messagebox.showerror("错误", f"无法显示拼接结果:\n{str(e)}")

    def save_result(self):
        if self.is_processing or self.result_image is None: return
        file_path = filedialog.asksaveasfilename(
            title="保存拼接结果", defaultextension=".png",
            filetypes=[("PNG图像", "*.png"), ("JPEG图像", "*.jpg;*.jpeg"),
                       ("BMP图像", "*.bmp"), ("所有文件", "*.*")])
        if file_path:
            try:
                if not cv_imwrite(file_path, self.result_image): raise RuntimeError("OpenCV 编码失败")
                self.status_var.set(f"✅ 已保存至: {file_path}")
                messagebox.showinfo("成功", f"结果已成功保存到:\n{file_path}")
                logger.info(f"用户保存结果到: {file_path}")
            except Exception as e:
                logger.error(f"保存结果时出错: {str(e)}")
                messagebox.showerror("错误", f"保存结果失败:\n{str(e)}")

    def clear_data(self):
        if self.is_processing: return
        if messagebox.askyesno("确认", "确定要清除所有数据吗？"):
            self.image_paths, self.thumbnail_refs = [], []
            self.images_canvas.delete("all")
            self.result_image  = None
            self.result_canvas.delete("all")
            self.result_photo  = None
            self.progress_var.set(0)
            self.progress_label.config(text="就绪")
            self.stitching_time_label.config(text="⏱  拼接时间: 0.00s")
            self.save_btn.config(state=tk.DISABLED)
            self.stitch_btn.config(state=tk.DISABLED)
            self.status_var.set("🦅 待命")
            self.animal_label.config(text="🦅")
            self.animal_hint_label.config(text='点击"开始拼接"雄鹰起飞！')
            logger.info("已清除所有数据")

    def show_about(self):
        about_text = "Eagle Eye Stitcher\n\nVersion: 1.2\nFeature: Multi-image sequential stitching, multi-process backend\n\n© 2024"
        messagebox.showinfo("关于", about_text)

    def on_close(self):
        if self.is_processing and self.worker_process is not None and self.worker_process.is_alive():
            should_exit = messagebox.askyesno("确认退出", "当前仍有拼接任务在运行，退出将终止后台进程。确定退出吗？")
            if not should_exit: return
            try:
                self.worker_process.terminate()
                self.worker_process.join(timeout=1)
            except Exception as e: logger.warning(f"终止后台进程时出错: {e}")
        self._cleanup_worker_resources(remove_temp_file=True)
        self.root.destroy()


def main():
    root = tk.Tk()
    app = StitchingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
