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

from stitcher.common.logger import get_logger, setup_logger
from stitcher.io.image_io import cv_imread, cv_imwrite
from stitcher.workers import run_stitching_worker

logs_dir = Path(__file__).parent.parent.parent / 'logs'
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / f"img_stitcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
setup_logger('stitcher', 'INFO', str(log_file), True)
logger = get_logger(__name__)


class StitchingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("图像自动拼接系统—sigmoid")

        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets', 'icons', 'cat.ico')
            logger.info(f"尝试加载图标: {icon_path}")

            if os.path.exists(icon_path):
                logger.info(f"图标文件存在，大小: {os.path.getsize(icon_path)} 字节")

                try:
                    win_icon_path = icon_path.replace('/', '\\')
                    self.root.iconbitmap(default=win_icon_path)
                    logger.info("窗口图标通过iconbitmap设置成功")
                except Exception as e1:
                    logger.warning(f"iconbitmap方法失败: {e1}")

                try:
                    icon_image = Image.open(icon_path)
                    icon_image = icon_image.resize((32, 32), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(icon_image)
                    self.root.wm_iconphoto(True, photo)
                    self.icon_photo = photo
                    logger.info("窗口图标通过wm_iconphoto设置成功")
                except Exception as e2:
                    logger.warning(f"wm_iconphoto方法失败: {e2}")

                try:
                    if hasattr(self, 'icon_photo'):
                        self.root.tk.call('wm', 'iconphoto', self.root._w, self.icon_photo)
                        logger.info("窗口图标通过tk.call设置成功")
                except Exception as e3:
                    logger.warning(f"tk.call方法失败: {e3}")
            else:
                logger.error(f"图标文件不存在: {icon_path}")
        except Exception as e:
            logger.error(f"无法设置窗口图标: {e}")

        self.root.geometry("1000x700")
        self.root.minsize(800, 600)

        self.image_paths = []
        self.result_image = None
        self.thumbnail_refs = []
        self.result_photo = None

        self.worker_process = None
        self.progress_queue = None
        self.result_queue = None
        self.is_processing = False
        self.temp_result_path = None
        self.last_stitching_time = 0.0

        self._create_widgets()
        self._layout_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        logger.info("GUI界面已初始化")

    def _create_widgets(self):
        self._create_menu()

        self.main_frame = ttk.Frame(self.root, padding="10")
        self.control_frame = ttk.LabelFrame(self.main_frame, text="控制面板", padding="10")

        self.select_images_btn = ttk.Button(
            self.control_frame,
            text="选择图像",
            command=self.select_images
        )

        self.stitch_btn = ttk.Button(
            self.control_frame,
            text="开始拼接",
            command=self.start_stitching,
            state=tk.DISABLED
        )

        self.clear_btn = ttk.Button(
            self.control_frame,
            text="清除数据",
            command=self.clear_data
        )

        self.save_btn = ttk.Button(
            self.control_frame,
            text="保存结果",
            command=self.save_result,
            state=tk.DISABLED
        )

        self.params_frame = ttk.LabelFrame(self.control_frame, text="参数设置", padding="10")

        self.seam_band_label = ttk.Label(self.params_frame, text="融合带宽:")
        self.seam_band_var = tk.IntVar(value=9)
        self.seam_band_scale = ttk.Scale(
            self.params_frame,
            from_=1, to=20, orient='horizontal',
            variable=self.seam_band_var, length=200
        )
        self.seam_band_value_label = ttk.Label(self.params_frame, text=str(self.seam_band_var.get()))
        self.seam_band_var.trace_add('write', self._update_seam_band_value)

        self.feather_radius_label = ttk.Label(self.params_frame, text="羽化半径:")
        self.feather_radius_var = tk.IntVar(value=11)
        self.feather_radius_scale = ttk.Scale(
            self.params_frame,
            from_=1, to=30, orient='horizontal',
            variable=self.feather_radius_var, length=200
        )
        self.feather_radius_value_label = ttk.Label(self.params_frame, text=str(self.feather_radius_var.get()))
        self.feather_radius_var.trace_add('write', self._update_feather_radius_value)

        self.stitching_time_label = ttk.Label(
            self.params_frame,
            text="拼接时间: 0.00s",
            font=("Arial", 9)
        )
        self.stitching_time_label.grid(row=6, column=0, columnspan=3, pady=(10, 0), sticky=tk.W, padx=5)

        self.progress_frame = ttk.Frame(self.params_frame)
        self.progress_frame.grid(row=7, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=2)

        self.progress_label = ttk.Label(self.progress_frame, text="就绪", foreground="#666")
        self.progress_label.pack(fill=tk.X)

        self.status_var = tk.StringVar(value="就绪")
        self.status_label = ttk.Label(
            self.control_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=5
        )

        self.images_frame = ttk.LabelFrame(self.main_frame, text="已选择图像", padding="10")
        self.images_canvas = tk.Canvas(self.images_frame)
        self.images_scrollbar_x = ttk.Scrollbar(self.images_frame, orient="horizontal", command=self.images_canvas.xview)
        self.images_scrollbar_y = ttk.Scrollbar(self.images_frame, orient="vertical", command=self.images_canvas.yview)
        self.images_canvas.configure(xscrollcommand=self.images_scrollbar_x.set, yscrollcommand=self.images_scrollbar_y.set)

        self.result_frame = ttk.LabelFrame(self.main_frame, text="拼接结果", padding="10")
        self.result_canvas = tk.Canvas(self.result_frame)
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

        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.control_frame.configure(width=250)
        self.control_frame.pack_propagate(False)

        button_style = {'fill': tk.X, 'pady': 5, 'padx': 5}
        self.select_images_btn.pack(**button_style)
        self.stitch_btn.pack(**button_style)
        self.clear_btn.pack(**button_style)
        self.save_btn.pack(**button_style)

        self.params_frame.pack(fill=tk.X, pady=10, padx=5)

        self.seam_band_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=(5, 0))
        self.seam_band_scale.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5)
        self.seam_band_value_label.grid(row=2, column=2, sticky=tk.W, padx=5)

        self.feather_radius_label.grid(row=3, column=0, sticky=tk.W, padx=5, pady=(10, 0))
        self.feather_radius_scale.grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5)
        self.feather_radius_value_label.grid(row=4, column=2, sticky=tk.W, padx=5)

        self.params_frame.columnconfigure(1, weight=1)
        self.params_frame.columnconfigure(0, weight=0)
        self.params_frame.columnconfigure(2, weight=0)
        self.params_frame.columnconfigure(3, weight=0)

        self.status_label.pack(fill=tk.X, pady=10, side=tk.BOTTOM, ipady=3)

        self.images_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))
        self.result_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        self.images_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.images_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.images_canvas.pack(fill=tk.BOTH, expand=True)

        self.result_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.result_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)

    def select_images(self):
        if self.is_processing:
            messagebox.showinfo("提示", "当前正在拼接，请等待当前任务完成。")
            return

        file_paths = filedialog.askopenfilenames(
            title="选择多张图像进行拼接",
            filetypes=[
                ("图像文件", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"),
                ("所有文件", "*.*")
            ]
        )

        if file_paths:
            self.image_paths = list(file_paths)
            self._display_selected_images(self.image_paths)
            if len(self.image_paths) >= 2:
                self.stitch_btn.config(state=tk.NORMAL)
            else:
                self.stitch_btn.config(state=tk.DISABLED)

            logger.info(f"用户选择了 {len(self.image_paths)} 张图像（按当前顺序依次拼接）")

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

                num_label = ttk.Label(
                    self.images_frame_inner,
                    text=f"{i + 1}",
                    background="#4a7a8c",
                    foreground="white",
                    width=2
                )
                num_label.grid(row=row, column=col, padx=margin, pady=margin, sticky=tk.NW)

                self._create_tooltip(img_label, f"图像 {i + 1}: {file_name}")
            except Exception as e:
                logger.error(f"无法显示图像 {file_path}: {str(e)}")

        self.images_frame_inner.update_idletasks()
        self.images_canvas.configure(scrollregion=self.images_canvas.bbox("all"))

    def _create_tooltip(self, widget, text):
        tooltip_window = [None]

        def enter(event):
            x = widget.winfo_rootx() + widget.winfo_width()
            y = widget.winfo_rooty()

            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{x + 5}+{y}")

            label = tk.Label(
                tooltip,
                text=text,
                background="#ffffe0",
                relief=tk.SOLID,
                borderwidth=1,
                padx=5,
                pady=2
            )
            label.pack()

            tooltip_window[0] = tooltip

        def leave(event):
            if tooltip_window[0] is not None:
                tooltip_window[0].destroy()
                tooltip_window[0] = None

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def start_stitching(self):
        if self.is_processing:
            messagebox.showinfo("提示", "当前已有拼接任务正在运行。")
            return

        if len(self.image_paths) < 2:
            messagebox.showwarning("警告", "请至少选择两张图像进行拼接")
            return

        self._set_processing_state(True)
        self.progress_var.set(0)
        self.progress_label.config(text="准备启动拼接进程...")
        self.status_var.set("正在拼接图像...")
        self.root.title("图像自动拼接系统—sigmoid (正在处理)")
        self.root.update_idletasks()

        config = {
            'SEAM_BAND': self.seam_band_var.get(),
            'FEATHER_RADIUS': self.feather_radius_var.get()
        }

        output_dir = Path(__file__).parent.parent.parent / 'outputs'
        output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_result_path = str(output_dir / f"mp_result_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")

        self.progress_queue = mp.Queue()
        self.result_queue = mp.Queue()

        self.worker_process = mp.Process(
            target=run_stitching_worker,
            args=(list(self.image_paths), config, self.progress_queue, self.result_queue, self.temp_result_path)
        )
        self.worker_process.start()
        logger.info(f"拼接子进程已启动，PID={self.worker_process.pid}")

        self._poll_worker_messages()

    def _poll_worker_messages(self):
        self._drain_progress_queue()

        result_message = self._get_result_message()
        if result_message is not None:
            self._handle_result_message(result_message)
            return

        if self.is_processing and self.worker_process is not None and not self.worker_process.is_alive():
            exit_code = self.worker_process.exitcode
            self._handle_worker_error(f"拼接子进程已退出，但没有返回结果。退出码: {exit_code}")
            return

        if self.is_processing:
            self.root.after(100, self._poll_worker_messages)

    def _drain_progress_queue(self):
        if self.progress_queue is None:
            return

        try:
            while True:
                msg = self.progress_queue.get_nowait()
                if not msg:
                    continue
                if msg[0] == "progress":
                    _, progress, message = msg
                    self.progress_var.set(progress)
                    self.progress_label.config(text=message)
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"检查进度队列时出错: {str(e)}")

    def _get_result_message(self):
        if self.result_queue is None:
            return None
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"检查结果队列时出错: {str(e)}")
            return ("error", f"读取结果队列失败: {e}")

    def _handle_result_message(self, message):
        msg_type = message[0]
        if msg_type == "done":
            _, output_path, preview_path, stitching_time = message
            self.last_stitching_time = stitching_time
            self._handle_worker_success(output_path, preview_path, stitching_time)
        elif msg_type == "error":
            _, error_message = message
            self._handle_worker_error(error_message)
        else:
            self._handle_worker_error(f"收到未知结果消息: {message}")

    def _handle_worker_success(self, output_path, preview_path, stitching_time):
        self.root.title("图像自动拼接系统—sigmoid")
        self.stitching_time_label.config(text=f"拼接时间: {stitching_time:.2f}s")

        # 读取完整结果（用于保存）
        full_result = cv_imread(output_path)
        if full_result is None:
            self._handle_worker_error(f"结果文件存在但无法读取: {output_path}")
            return

        # 读取预览图（用于显示）
        preview_result = cv_imread(preview_path)
        if preview_result is None:
            self._handle_worker_error(f"预览文件存在但无法读取: {preview_path}")
            return

        self.result_image = full_result
        self._display_result(preview_result)
        self.status_var.set("拼接完成")
        self.progress_var.set(100)
        self.progress_label.config(text="拼接完成")

        self.save_btn.config(state=tk.NORMAL)
        self.file_menu.entryconfig(1, state=tk.NORMAL)

        self._set_processing_state(False)
        self._cleanup_worker_resources(remove_temp_file=False)
        self._cleanup_temp_result()
        
        # 清理预览图临时文件
        if preview_path and os.path.exists(preview_path):
            try:
                os.remove(preview_path)
            except OSError as e:
                logger.warning(f"无法删除预览临时文件 {preview_path}: {e}")

        messagebox.showinfo("成功", "图像拼接完成!")

    def _handle_worker_error(self, error_message):
        self.root.title("图像自动拼接系统—sigmoid")
        self.status_var.set("拼接出错")
        self.progress_label.config(text="拼接失败")

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
            self.file_menu.entryconfig(1, state=tk.DISABLED)
        else:
            self.select_images_btn.config(state=tk.NORMAL)
            self.clear_btn.config(state=tk.NORMAL)
            if len(self.image_paths) >= 2:
                self.stitch_btn.config(state=tk.NORMAL)
            else:
                self.stitch_btn.config(state=tk.DISABLED)

    def _cleanup_worker_resources(self, remove_temp_file=False):
        process = self.worker_process
        if process is not None:
            try:
                if process.is_alive():
                    process.join(timeout=0.2)
            except Exception:
                pass
        self.worker_process = None

        for q in (self.progress_queue, self.result_queue):
            if q is None:
                continue
            try:
                q.close()
            except Exception:
                pass
            try:
                q.cancel_join_thread()
            except Exception:
                pass

        self.progress_queue = None
        self.result_queue = None

        if remove_temp_file:
            self._cleanup_temp_result()

    def _cleanup_temp_result(self):
        if self.temp_result_path and os.path.exists(self.temp_result_path):
            try:
                os.remove(self.temp_result_path)
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
            canvas_width = max(self.result_canvas.winfo_width() - 20, 100)
            canvas_height = max(self.result_canvas.winfo_height() - 20, 100)

            img_width, img_height = result_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)

            new_width = max(int(img_width * scale), 1)
            new_height = max(int(img_height * scale), 1)

            display_image = result_image.resize((new_width, new_height), Image.LANCZOS)

            photo = ImageTk.PhotoImage(display_image)
            self.result_photo = photo

            self.result_canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=photo,
                anchor=tk.CENTER
            )

            self.result_canvas.configure(scrollregion=(0, 0, new_width, new_height))

            logger.info(f"显示拼接结果，尺寸: {new_width}x{new_height}")

        except Exception as e:
            logger.error(f"显示结果时出错: {str(e)}")
            messagebox.showerror("错误", f"无法显示拼接结果:\n{str(e)}")

    def save_result(self):
        if self.is_processing:
            messagebox.showwarning("警告", "当前正在拼接，请等待任务完成后再保存结果")
            return

        if self.result_image is None:
            messagebox.showwarning("警告", "没有可保存的拼接结果")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存拼接结果",
            defaultextension=".png",
            filetypes=[
                ("PNG图像", "*.png"),
                ("JPEG图像", "*.jpg;*.jpeg"),
                ("BMP图像", "*.bmp"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            try:
                if not cv_imwrite(file_path, self.result_image):
                    raise RuntimeError("OpenCV 编码失败")
                self.status_var.set(f"结果已保存至: {file_path}")
                messagebox.showinfo("成功", f"结果已成功保存到:\n{file_path}")
                logger.info(f"用户保存结果到: {file_path}")
            except Exception as e:
                logger.error(f"保存结果时出错: {str(e)}")
                messagebox.showerror("错误", f"保存结果失败:\n{str(e)}")

    def clear_data(self):
        if self.is_processing:
            messagebox.showwarning("警告", "当前正在拼接，无法清除数据。")
            return

        if messagebox.askyesno("确认", "确定要清除所有数据吗？已选择的图像和拼接结果将被移除。"):
            self.image_paths = []
            self.thumbnail_refs = []
            self.images_canvas.delete("all")
            self.result_image = None
            self.result_canvas.delete("all")
            self.result_photo = None
            self.progress_var.set(0)
            self.progress_label.config(text="就绪")
            self.stitching_time_label.config(text="拼接时间: 0.00s")
            self.save_btn.config(state=tk.DISABLED)
            self.file_menu.entryconfig(1, state=tk.DISABLED)
            self.stitch_btn.config(state=tk.DISABLED)
            self.status_var.set("就绪")
            logger.info("已清除所有数据")

    def show_about(self):
        about_text = "图像自动拼接系统\n\n" \
                   "版本: 1.2\n" \
                   "功能: 多图顺序自动拼接，GUI 采用多进程后台执行，减少界面未响应现象\n\n" \
                   "© 2023 图像拼接项目组"

        messagebox.showinfo("关于", about_text)

    def on_close(self):
        if self.is_processing and self.worker_process is not None and self.worker_process.is_alive():
            should_exit = messagebox.askyesno("确认退出", "当前仍有拼接任务在运行，退出将终止后台进程。确定退出吗？")
            if not should_exit:
                return
            try:
                self.worker_process.terminate()
                self.worker_process.join(timeout=1)
            except Exception as e:
                logger.warning(f"终止后台进程时出错: {e}")

        self._cleanup_worker_resources(remove_temp_file=True)
        self.root.destroy()


def main():
    root = tk.Tk()
    app = StitchingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
