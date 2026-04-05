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
        ctypes.windll.kernel32.SetConsoleTitleW("图像拼接应用 - 控制台")
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


class StitchingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("图像自动拼接系统—sigmoid 🌸")

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
        self.root.configure(bg="#F7FAF5")

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

        # 🐰 动画状态
        self._anim_running = False
        self._anim_job_id = None
        self.animal_emojis = ["🐇", "🐰", "🌸", "✨"]
        
        self._create_widgets()
        self._layout_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        logger.info("GUI界面已初始化（春日萌兔版）")

    def _load_button_icons(self):
        icons = {}
        icon_dir = Path(__file__).parent.parent.parent / 'assets' / 'icons'
        icon_files = {
            'select': 'select_images.png',
            'stitch': 'stitch.png',
            'clear': 'clear.png',
            'save': 'save.png'
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

        self.main_frame = ttk.Frame(self.root, padding="15")
        self.control_frame = ttk.Frame(self.main_frame, padding="12")

        # ========== 🌸 春日配色与样式配置 ==========
        style = ttk.Style()
        style.theme_use('clam')

        FONT_MAIN = ("Microsoft YaHei UI", 10)
        FONT_BOLD = ("Microsoft YaHei UI", 10, "bold")
        FONT_SMALL = ("Microsoft YaHei UI", 9)

        # 全局背景与文字
        style.configure("TFrame", background="#F7FAF5")
        style.configure("TLabel", background="#FFFFFF", foreground="#2C4A3E", font=FONT_SMALL)
        style.configure("TLabelFrame", background="#FFFFFF", foreground="#2C4A3E", font=FONT_BOLD)

        # 🌿 按钮样式 - 春日四色
        style.configure("Select.TButton", padding=10, relief="flat", background="#A8D5BA", foreground="#1A3A2A", font=FONT_BOLD, borderwidth=0)
        style.map("Select.TButton", background=[('active', '#8FC4A3'), ('pressed', '#76B08C'), ('disabled', '#D9E8DF')])

        style.configure("Stitch.TButton", padding=10, relief="flat", background="#F4C2C2", foreground="#4A2A2A", font=FONT_BOLD, borderwidth=0)
        style.map("Stitch.TButton", background=[('active', '#E8A8A8'), ('pressed', '#D98F8F'), ('disabled', '#F9E0E0')])

        style.configure("Clear.TButton", padding=10, relief="flat", background="#F8E1A8", foreground="#4A3A1A", font=FONT_BOLD, borderwidth=0)
        style.map("Clear.TButton", background=[('active', '#F0D088'), ('pressed', '#E8C06A'), ('disabled', '#FCF3DC')])

        style.configure("Save.TButton", padding=10, relief="flat", background="#B5C9E8", foreground="#1A2A4A", font=FONT_BOLD, borderwidth=0)
        style.map("Save.TButton", background=[('active', '#9FB8DC'), ('pressed', '#89A8D0'), ('disabled', '#DCE6F4')])

        # 参数区样式
        style.configure("Param.TFrame", background="#F0F7F2")
        style.configure("Param.TLabel", background="#F0F7F2", foreground="#4A6A5A", font=FONT_SMALL)

        # 滑块与进度条 - 春草绿
        style.configure("TScale", background="#F0F7F2", troughcolor="#D4E8D8", sliderrelief="flat", sliderlength=18)
        style.map("TScale", slidercolor=[('active', '#88C9A1')])
        style.configure("Custom.Horizontal.TProgressbar", troughcolor="#D4E8D8", background="#88C9A1", thickness=10, borderwidth=0)

        # ========== 按钮创建（带图标）==========
        self.select_images_btn = ttk.Button(
            self.control_frame, text="选择图像", command=self.select_images,
            image=self.button_icons.get('select'), compound=tk.LEFT, style="Select.TButton"
        )
        self.stitch_btn = ttk.Button(
            self.control_frame, text="开始拼接", command=self.start_stitching, state=tk.DISABLED,
            image=self.button_icons.get('stitch'), compound=tk.LEFT, style="Stitch.TButton"
        )
        self.clear_btn = ttk.Button(
            self.control_frame, text="清除数据", command=self.clear_data,
            image=self.button_icons.get('clear'), compound=tk.LEFT, style="Clear.TButton"
        )
        self.save_btn = ttk.Button(
            self.control_frame, text="保存结果", command=self.save_result, state=tk.DISABLED,
            image=self.button_icons.get('save'), compound=tk.LEFT, style="Save.TButton"
        )

        # ========== 参数设置区域 ==========
        self.params_frame = ttk.LabelFrame(self.control_frame, text=" ⚙️ 参数调节 ", padding="12")
        self.params_inner_frame = ttk.Frame(self.params_frame, style="Param.TFrame")

        self.seam_band_label = ttk.Label(self.params_inner_frame, text="融合带宽:", style="Param.TLabel")
        self.seam_band_var = tk.IntVar(value=9)
        self.seam_band_scale = ttk.Scale(self.params_inner_frame, from_=1, to=20, orient='horizontal', variable=self.seam_band_var, length=160)
        self.seam_band_value_label = tk.Label(self.params_inner_frame, text="9", background="#88C9A1", foreground="#FFFFFF", font=("Arial", 9, "bold"), width=3, relief="flat", padx=6)
        self.seam_band_var.trace_add('write', self._update_seam_band_value)

        self.feather_radius_label = ttk.Label(self.params_inner_frame, text="羽化半径:", style="Param.TLabel")
        self.feather_radius_var = tk.IntVar(value=11)
        self.feather_radius_scale = ttk.Scale(self.params_inner_frame, from_=1, to=30, orient='horizontal', variable=self.feather_radius_var, length=160)
        self.feather_radius_value_label = tk.Label(self.params_inner_frame, text="11", background="#B5C9E8", foreground="#FFFFFF", font=("Arial", 9, "bold"), width=3, relief="flat", padx=6)
        self.feather_radius_var.trace_add('write', self._update_feather_radius_value)

        self.stitching_time_label = ttk.Label(self.params_inner_frame, text="⏱️ 拼接时间: 0.00s", font=("Arial", 9), foreground="#6B8A7A", background="#F0F7F2")

        self.progress_frame = ttk.Frame(self.params_inner_frame, style="Param.TFrame")
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100, mode='determinate', style="Custom.Horizontal.TProgressbar")
        self.progress_label = ttk.Label(self.progress_frame, text="就绪", foreground="#6B8A7A", background="#F0F7F2")

        # 状态栏
        self.status_var = tk.StringVar(value="✨ 就绪")
        self.status_label = tk.Label(
            self.control_frame, textvariable=self.status_var, relief="flat", anchor=tk.W,
            padx=10, pady=8, bg="#E8F0E8", fg="#2C4A3E", font=FONT_SMALL
        )

        # ========== 🐰 左下角小动物区域 ==========
        self.animal_frame = tk.Frame(self.control_frame, bg="#F7FAF5", height=80)
        self.animal_label = tk.Label(
            self.animal_frame,
            text="🐰",
            font=("Segoe UI Emoji", 36),
            bg="#F7FAF5",
            anchor="center"
        )
        # 👇 修复了这里的中文引号嵌套错误
        self.animal_hint_label = tk.Label(
            self.animal_frame,
            text="点击“开始拼接”看我动起来！",
            font=("Microsoft YaHei UI", 8),
            bg="#F7FAF5",
            fg="#6B8A7A"
        )

        # ========== 右侧容器 ==========
        self.right_container = ttk.Frame(self.main_frame)

        # 已选择图像
        self.images_frame = ttk.LabelFrame(self.right_container, text=" 📸 已选择图像 ", padding="10")
        self.images_canvas = tk.Canvas(self.images_frame, bg="#FFFFFF", highlightthickness=0)
        self.images_scrollbar_x = ttk.Scrollbar(self.images_frame, orient="horizontal", command=self.images_canvas.xview)
        self.images_scrollbar_y = ttk.Scrollbar(self.images_frame, orient="vertical", command=self.images_canvas.yview)
        self.images_canvas.configure(xscrollcommand=self.images_scrollbar_x.set, yscrollcommand=self.images_scrollbar_y.set)

        # 拼接结果
        self.result_frame = ttk.LabelFrame(self.right_container, text=" 🖼️ 拼接结果 ", padding="10")
        self.result_canvas = tk.Canvas(self.result_frame, bg="#FFFFFF", highlightthickness=0)
        self.result_scrollbar_x = ttk.Scrollbar(self.result_frame, orient="horizontal", command=self.result_canvas.xview)
        self.result_scrollbar_y = ttk.Scrollbar(self.result_frame, orient="vertical", command=self.result_canvas.yview)
        self.result_canvas.configure(xscrollcommand=self.result_scrollbar_x.set, yscrollcommand=self.result_scrollbar_y.set)

    def _update_seam_band_value(self, *args):
        self.seam_band_value_label.config(text=str(self.seam_band_var.get()))

    def _update_feather_radius_value(self, *args):
        self.feather_radius_value_label.config(text=str(self.feather_radius_var.get()))

    def _create_menu(self):
        self.menu_bar = tk.Menu(self.root)
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
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15), pady=10)
        self.control_frame.configure(width=270)
        self.control_frame.pack_propagate(False)

        # 按钮布局
        button_style = {'fill': tk.X, 'pady': 6, 'padx': 3}
        self.select_images_btn.pack(**button_style)
        self.stitch_btn.pack(**button_style)
        self.clear_btn.pack(**button_style)
        self.save_btn.pack(**button_style)

        # 参数区域
        self.params_frame.pack(fill=tk.X, pady=12, padx=3)
        self.params_inner_frame.pack(fill=tk.BOTH, expand=True)

        self.seam_band_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=(8, 2))
        self.seam_band_scale.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=3)
        self.seam_band_value_label.grid(row=2, column=2, sticky=tk.E, padx=5, pady=3)

        self.feather_radius_label.grid(row=3, column=0, sticky=tk.W, padx=5, pady=(12, 2))
        self.feather_radius_scale.grid(row=4, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=3)
        self.feather_radius_value_label.grid(row=4, column=2, sticky=tk.E, padx=5, pady=3)

        self.params_inner_frame.columnconfigure(1, weight=1)

        self.stitching_time_label.grid(row=6, column=0, columnspan=3, pady=(12, 5), sticky=tk.W, padx=5)

        self.progress_frame.grid(row=7, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        self.progress_bar.pack(fill=tk.X, pady=2)
        self.progress_label.pack(fill=tk.X)

        # 状态栏
        self.status_label.pack(fill=tk.X, pady=(15, 0), side=tk.BOTTOM)

        # 🐰 小动物区域（放在状态栏上方）
        self.animal_frame.pack(fill=tk.X, pady=(10, 0), side=tk.BOTTOM)
        self.animal_label.pack(pady=5)
        self.animal_hint_label.pack(pady=(0, 5))

        # ========== 右侧布局 ==========
        self.right_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.images_frame.pack(side=tk.TOP, fill=tk.X, padx=0, pady=(0, 10))
        self.images_frame.configure(height=220)
        self.images_frame.pack_propagate(False)

        self.result_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=0, pady=(0, 5))

        # Canvas 滚动条
        self.images_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.images_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.images_canvas.pack(fill=tk.BOTH, expand=True)

        self.result_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.result_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)

    # ========== 🐰 小动物动画控制 ==========
    def _start_animal_animation(self):
        """开始奔跑动画"""
        if self._anim_running:
            return
        self._anim_running = True
        self.animal_hint_label.config(text="小兔子正在努力拼接中...🏃")
        self._anim_index = 0
        self._anim_loop()

    def _anim_loop(self):
        """动画循环"""
        if not self._anim_running:
            return
        emoji = self.animal_emojis[self._anim_index % len(self.animal_emojis)]
        self.animal_label.config(text=emoji)
        self._anim_index += 1
        self._anim_job_id = self.root.after(150, self._anim_loop)

    def _celebrate_animal(self):
        """庆祝动画（撒花）"""
        self._anim_running = False
        if self._anim_job_id:
            self.root.after_cancel(self._anim_job_id)
            self._anim_job_id = None
        
        self.animal_label.config(text="🎉")
        self.animal_hint_label.config(text="拼接完成！太棒啦！✨")
        self._celebrate_loop(0)

    def _celebrate_loop(self, count):
        """撒花循环（播放5次）"""
        if count >= 5:
            self.animal_label.config(text="🐰💐")
            # 👇 修复了这里的中文引号嵌套错误
            self.animal_hint_label.config(text="完成！点击“清除数据”重新开始~")
            return
        
        emojis = ["🌸", "✨", "🎉", "🌼", "💐"]
        self.animal_label.config(text=emojis[count % len(emojis)])
        self.root.after(200, lambda: self._celebrate_loop(count + 1))

    # ========== 核心功能 ==========
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
        self.images_frame_inner = ttk.Frame(self.images_canvas)
        self.images_canvas.create_window((0, 0), window=self.images_frame_inner, anchor=tk.NW)

        for i, file_path in enumerate(file_paths):
            try:
                row = (i // 3) * 2
                col = i % 3
                image = Image.open(file_path)
                image.thumbnail(thumbnail_size, Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.thumbnail_refs.append(photo)
                img_label = ttk.Label(self.images_frame_inner, image=photo)
                img_label.grid(row=row, column=col, padx=margin, pady=margin)
                file_name = os.path.basename(file_path)
                name_label = ttk.Label(self.images_frame_inner, text=file_name, wraplength=thumbnail_size[0])
                name_label.grid(row=row + 1, column=col, padx=margin, pady=2)
                num_label = tk.Label(self.images_frame_inner, text=f"{i + 1}", background="#88C9A1", foreground="#FFFFFF", width=2, font=("Arial", 8, "bold"))
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
        self.status_var.set("🔄 正在拼接...")
        self.root.title("图像自动拼接系统—sigmoid (正在处理)")
        
        # 🐰 启动小动物动画
        self._start_animal_animation()
        
        self.root.update_idletasks()

        config = {'SEAM_BAND': self.seam_band_var.get(), 'FEATHER_RADIUS': self.feather_radius_var.get()}
        output_dir = Path(__file__).parent.parent.parent / 'outputs'
        output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_result_path = str(output_dir / f"mp_result_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")

        self.progress_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.worker_process = mp.Process(target=run_stitching_worker, args=(list(self.image_paths), config, self.progress_queue, self.result_queue, self.temp_result_path))
        self.worker_process.start()
        logger.info(f"拼接子进程已启动，PID={self.worker_process.pid}")
        self._poll_worker_messages()

    def _poll_worker_messages(self):
        self._drain_progress_queue()
        result_message = self._get_result_message()
        if result_message:
            self._handle_result_message(result_message)
            return
        if self.is_processing: self.root.after(100, self._poll_worker_messages)

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
        if message[0] == "done": self._handle_worker_success(message[1], message[2])
        elif message[0] == "error": self._handle_worker_error(message[1])

    def _handle_worker_success(self, output_path, stitching_time):
        self.root.title("图像自动拼接系统—sigmoid")
        self.stitching_time_label.config(text=f"⏱️ 拼接时间: {stitching_time:.2f}s")
        result = cv_imread(output_path)
        if result is not None:
            self.result_image = result
            self._display_result(self.result_image)
            self.status_var.set("✅ 拼接完成")
            self.progress_var.set(100)
            self.progress_label.config(text="完成")
            self.save_btn.config(state=tk.NORMAL)
        
        # 🐰 拼接成功，撒花庆祝！
        self._celebrate_animal()
        
        self._set_processing_state(False)
        self._cleanup_worker_resources(remove_temp_file=False)
        self._cleanup_temp_result()
        messagebox.showinfo("成功", "图像拼接完成! 🌸🐰")

    def _handle_worker_error(self, error_message):
        self.root.title("图像自动拼接系统—sigmoid")
        self.status_var.set("❌ 拼接出错")
        self.progress_label.config(text="失败")
        
        # 🐰 出错了，小兔子伤心
        self.animal_label.config(text="😢")
        self.animal_hint_label.config(text="拼接出错了...再试一次吧！")
        
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
        self.result_queue = None
        if remove_temp_file: self._cleanup_temp_result()

    def _cleanup_temp_result(self):
        if self.temp_result_path and os.path.exists(self.temp_result_path):
            try: os.remove(self.temp_result_path)
            except OSError as e: logger.warning(f"无法删除临时结果文件 {self.temp_result_path}: {e}")
        self.temp_result_path = None

    def _display_result(self, result_image):
        try:
            self.result_canvas.delete("all")
            if isinstance(result_image, np.ndarray):
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                result_image = Image.fromarray(result_image)
            self.result_canvas.update_idletasks()
            cw = max(self.result_canvas.winfo_width() - 20, 100)
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
        file_path = filedialog.asksaveasfilename(title="保存拼接结果", defaultextension=".png", filetypes=[("PNG图像", "*.png"), ("JPEG图像", "*.jpg;*.jpeg"), ("BMP图像", "*.bmp"), ("所有文件", "*.*")])
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
            self.result_image = None
            self.result_canvas.delete("all")
            self.result_photo = None
            self.progress_var.set(0)
            self.progress_label.config(text="就绪")
            self.stitching_time_label.config(text="⏱️ 拼接时间: 0.00s")
            self.save_btn.config(state=tk.DISABLED)
            self.stitch_btn.config(state=tk.DISABLED)
            self.status_var.set("✨ 就绪")
            
            # 🐰 重置小动物
            self.animal_label.config(text="🐰")
            # 👇 修复了这里的中文引号嵌套错误
            self.animal_hint_label.config(text="点击“开始拼接”动起来！")
            
            logger.info("已清除所有数据")

    def show_about(self):
        about_text = "图像自动拼接系统\n\n版本: 1.2 (春日萌兔版 🐰)\n功能: 多图顺序自动拼接，多进程后台执行\n\n🌸 特别功能：\n拼接时左下角小兔子会奔跑\n完成后会撒花庆祝哦！\n\n© 2024"
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