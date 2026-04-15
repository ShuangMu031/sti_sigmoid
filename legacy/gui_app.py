# -*- coding: utf-8 -*-
"""
图像拼接应用程序 - GUI界面
提供用户友好的图形界面，用于选择图像、设置参数和显示结果
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2  # 添加OpenCV导入，用于色彩空间转换
import logging
import time  # 添加时间模块导入，用于计算拼接耗时
from pathlib import Path
import threading  # 导入多线程模块
from datetime import datetime  # 修复导入冲突问题，直接导入datetime类
import queue  # 导入队列模块用于线程通信

# 在Windows平台下隐藏控制台窗口
if sys.platform == 'win32':
    try:
        import ctypes
        # 获取当前进程句柄
        ctypes.windll.kernel32.SetConsoleTitleW("图像拼接应用 - 控制台")
        # 尝试隐藏控制台窗口
        # 注意：这在直接运行Python脚本时有效，但在打包为可执行文件时需要配合PyInstaller的--noconsole参数
        # 这里的代码主要是为了在开发过程中也能有较好的用户体验
    except Exception as e:
        # 如果隐藏控制台失败，不影响程序功能
        pass

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入主应用逻辑
from main_app import StitchingApp

# 设置日志
# 创建logs目录（如果不存在）
logs_dir = Path(__file__).parent / 'logs'
logs_dir.mkdir(exist_ok=True)

# 配置日志系统
log_file = logs_dir / f"img_stitcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # 输出到文件
        logging.FileHandler(str(log_file), encoding='utf-8'),
        # 输出到控制台（仅在开发模式下，可根据需要启用/禁用）
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StitchingGUI:
    """
    图像拼接应用程序的图形用户界面类
    """
    
    def __init__(self, root):
        """
        初始化GUI界面
        
        Args:
            root: tkinter的主窗口对象
        """
        self.root = root
        # 设置窗口标题为"图像自动拼接系统—sigmoid"
        self.root.title("图像自动拼接系统—sigmoid")
        # 设置窗口图标
        try:
            # 确保使用绝对路径
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ico', 'cat.ico')
            print(f"[DEBUG] 尝试加载图标: {icon_path}")
            logger.info(f"尝试加载图标: {icon_path}")
            
            if os.path.exists(icon_path):
                print(f"[DEBUG] 图标文件存在，大小: {os.path.getsize(icon_path)} 字节")
                logger.info(f"图标文件存在，大小: {os.path.getsize(icon_path)} 字节")
                
                # 在Windows上，尝试使用不同的方式强制设置图标
                try:
                    # 转换为Windows格式的路径
                    win_icon_path = icon_path.replace('/', '\\')
                    print(f"[DEBUG] 使用Windows路径: {win_icon_path}")
                    
                    # 尝试使用iconbitmap方法，在Windows上可能需要使用特定格式
                    self.root.iconbitmap(default=win_icon_path)
                    print(f"[DEBUG] 窗口图标通过iconbitmap(default=...)设置成功")
                    logger.info("窗口图标通过iconbitmap(default=...)设置成功")
                except Exception as e1:
                    print(f"[DEBUG] iconbitmap方法失败: {e1}")
                    logger.warning(f"iconbitmap方法失败: {e1}")
                
                # 尝试使用wm iconphoto方法 (更现代的方式)
                try:
                    # 使用PIL加载图标
                    icon_image = Image.open(icon_path)
                    # 调整图标大小为标准尺寸 (可选)
                    icon_image = icon_image.resize((32, 32), Image.LANCZOS)
                    # 转换为适合Tkinter的格式
                    photo = ImageTk.PhotoImage(icon_image)
                    # 使用wm iconphoto设置图标，这个方法更可靠
                    self.root.wm_iconphoto(True, photo)
                    self.icon_photo = photo  # 保持引用避免被垃圾回收
                    print(f"[DEBUG] 窗口图标通过wm_iconphoto设置成功")
                    logger.info("窗口图标通过wm_iconphoto设置成功")
                except Exception as e2:
                    print(f"[DEBUG] wm_iconphoto方法失败: {e2}")
                    logger.warning(f"wm_iconphoto方法失败: {e2}")
                
                # 尝试使用tk.call方法作为最后的备用
                try:
                    if 'photo' in locals():
                        self.root.tk.call('wm', 'iconphoto', self.root._w, photo)
                        print(f"[DEBUG] 窗口图标通过tk.call设置成功")
                        logger.info("窗口图标通过tk.call设置成功")
                except Exception as e3:
                    print(f"[DEBUG] tk.call方法失败: {e3}")
                    logger.warning(f"tk.call方法失败: {e3}")
                
                # 清除可能存在的默认图标
                try:
                    # 这个方法在某些Tkinter版本中可能有效
                    self.root.tk.call('wm', 'iconmask', self.root._w, '')
                    print(f"[DEBUG] 尝试清除默认图标掩码")
                except:
                    pass
            else:
                print(f"[DEBUG] 图标文件不存在: {icon_path}")
                logger.error(f"图标文件不存在: {icon_path}")
        except Exception as e:
            print(f"[DEBUG] 无法设置窗口图标: {e}")
            logger.error(f"无法设置窗口图标: {e}")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # 存储选择的图像路径
        self.image_paths = []
        # 存储当前拼接结果
        self.result_image = None
        
        # 创建界面组件
        self._create_widgets()
        self._layout_widgets()
        
        logger.info("GUI界面已初始化")
    
    def _create_widgets(self):
        """
        创建所有界面组件
        """
        # 创建菜单栏
        self._create_menu()
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        
        # 创建左侧控制面板
        self.control_frame = ttk.LabelFrame(self.main_frame, text="控制面板", padding="10")
        
        # 创建图像选择按钮
        self.select_images_btn = ttk.Button(
            self.control_frame, 
            text="选择图像", 
            command=self.select_images
        )
        
        # 创建拼接按钮
        self.stitch_btn = ttk.Button(
            self.control_frame, 
            text="开始拼接", 
            command=self.start_stitching
        )
        
        # 创建清除数据按钮
        self.clear_btn = ttk.Button(
            self.control_frame, 
            text="清除数据", 
            command=self.clear_data
        )
        
        # 创建保存按钮
        self.save_btn = ttk.Button(
            self.control_frame, 
            text="保存结果", 
            command=self.save_result,
            state=tk.DISABLED
        )
        
        # 创建参数设置区域
        self.params_frame = ttk.LabelFrame(self.control_frame, text="参数设置", padding="10")
        
        # 参数设置 - 仅保留实际使用的参数
        self.seam_band_label = ttk.Label(self.params_frame, text="融合带宽:")
        self.seam_band_var = tk.IntVar(value=9)  # 从main.py借鉴的默认值
        self.seam_band_scale = ttk.Scale(
            self.params_frame,
            from_=1, to=20, orient='horizontal',
            variable=self.seam_band_var, length=200
        )
        self.seam_band_value_label = ttk.Label(self.params_frame, text=str(self.seam_band_var.get()))
        # 移除最小值标签
        self.seam_band_var.trace_add('write', self._update_seam_band_value)
        
        self.feather_radius_label = ttk.Label(self.params_frame, text="羽化半径:")
        self.feather_radius_var = tk.IntVar(value=11)  # 从main.py借鉴的默认值
        self.feather_radius_scale = ttk.Scale(
            self.params_frame,
            from_=1, to=30, orient='horizontal',
            variable=self.feather_radius_var, length=200
        )
        self.feather_radius_value_label = ttk.Label(self.params_frame, text=str(self.feather_radius_var.get()))
        # 移除最小值标签
        self.feather_radius_var.trace_add('write', self._update_feather_radius_value)

        # 拼接时间和状态
        self.stitching_time_label = ttk.Label(
            self.params_frame,
            text="拼接时间: 0.00s",
            font=("Arial", 9)
        )
        self.stitching_time_label.grid(row=6, column=0, columnspan=3, pady=(10, 0), sticky=tk.W, padx=5)

        # 创建进度条
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

        # 创建状态标签
        self.status_var = tk.StringVar(value="就绪")
        self.status_label = ttk.Label(
            self.control_frame, 
            textvariable=self.status_var,
            relief=tk.SUNKEN, 
            anchor=tk.W,
            padding=5
        )
        
        # 创建右侧图像预览区域
        self.images_frame = ttk.LabelFrame(self.main_frame, text="已选择图像", padding="10")
        self.images_canvas = tk.Canvas(self.images_frame)
        self.images_scrollbar_x = ttk.Scrollbar(self.images_frame, orient="horizontal", command=self.images_canvas.xview)
        self.images_scrollbar_y = ttk.Scrollbar(self.images_frame, orient="vertical", command=self.images_canvas.yview)
        self.images_canvas.configure(xscrollcommand=self.images_scrollbar_x.set, yscrollcommand=self.images_scrollbar_y.set)
        
        # 创建结果显示区域
        self.result_frame = ttk.LabelFrame(self.main_frame, text="拼接结果", padding="10")
        self.result_canvas = tk.Canvas(self.result_frame)
        self.result_scrollbar_x = ttk.Scrollbar(self.result_frame, orient="horizontal", command=self.result_canvas.xview)
        self.result_scrollbar_y = ttk.Scrollbar(self.result_frame, orient="vertical", command=self.result_canvas.yview)
        self.result_canvas.configure(xscrollcommand=self.result_scrollbar_x.set, yscrollcommand=self.result_scrollbar_y.set)
        
        # 存储缩略图引用，防止被垃圾回收
        self.thumbnail_refs = []
    
    def _update_seam_band_value(self, *args):
        """更新融合带宽显示值"""
        self.seam_band_value_label.config(text=str(self.seam_band_var.get()))
    
    def _update_feather_radius_value(self, *args):
        """更新羽化半径显示值"""
        self.feather_radius_value_label.config(text=str(self.feather_radius_var.get()))
    
    def _create_menu(self):
        """
        创建菜单栏
        """
        self.menu_bar = tk.Menu(self.root)
        
        # 文件菜单
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="选择图像", command=self.select_images)
        self.file_menu.add_command(label="保存结果", command=self.save_result, state=tk.DISABLED)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="退出", command=self.root.quit)
        self.menu_bar.add_cascade(label="文件", menu=self.file_menu)
        
        # 编辑菜单
        self.edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.edit_menu.add_command(label="清除数据", command=self.clear_data)
        self.menu_bar.add_cascade(label="编辑", menu=self.edit_menu)
        
        # 帮助菜单
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(label="关于", command=self.show_about)
        self.menu_bar.add_cascade(label="帮助", menu=self.help_menu)
        
        # 设置菜单栏
        self.root.config(menu=self.menu_bar)
    
    def _layout_widgets(self):
        """
        布局界面组件
        """
        # 放置主框架
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 放置左侧控制面板 - 设置固定宽度使其更规整
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.control_frame.configure(width=250)  # 设置固定宽度
        self.control_frame.pack_propagate(False)  # 防止内容改变框架大小
        
        # 放置按钮 - 统一间距和样式
        button_style = {'fill': tk.X, 'pady': 5, 'padx': 5}
        self.select_images_btn.pack(**button_style)
        self.stitch_btn.pack(**button_style)
        self.clear_btn.pack(**button_style)
        self.save_btn.pack(**button_style)
        
        # 放置参数设置区域
        self.params_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # 使用grid布局重新设计参数区域，使布局更规整
        # 融合带宽布局
        # 布局设置
        self.seam_band_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=(5, 0))
        self.seam_band_scale.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5)
        self.seam_band_value_label.grid(row=2, column=2, sticky=tk.W, padx=5)
        
        self.feather_radius_label.grid(row=3, column=0, sticky=tk.W, padx=5, pady=(10, 0))
        self.feather_radius_scale.grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5)
        self.feather_radius_value_label.grid(row=4, column=2, sticky=tk.W, padx=5)

        # 设置列权重，使滑块能够拉伸
        self.params_frame.columnconfigure(1, weight=1)
        self.params_frame.columnconfigure(0, weight=0)
        self.params_frame.columnconfigure(2, weight=0)
        self.params_frame.columnconfigure(3, weight=0)
        
        # 放置状态标签 - 固定在底部并设置适当高度
        self.status_label.pack(fill=tk.X, pady=10, side=tk.BOTTOM, ipady=3)  # ipady增加内部填充
        
        # 放置右侧图像预览和结果区域
        self.images_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))
        self.result_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
        # 放置图像预览区域的滚动条
        self.images_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.images_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.images_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 放置结果显示区域的滚动条
        self.result_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.result_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
    
    def select_images(self):
        """
        选择要拼接的图像文件
        """
        # 打开文件选择对话框
        file_paths = filedialog.askopenfilenames(
            title="选择要拼接的图像",
            filetypes=[
                ("图像文件", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"),
                ("所有文件", "*.*")
            ]
        )
        
        # 如果用户选择了文件
        if file_paths:
            # 保存选择的文件路径
            self.image_paths = list(file_paths)
            # 显示选择的图像
            self._display_selected_images(self.image_paths)
            # 启用拼接按钮
            if len(self.image_paths) >= 2:
                self.stitch_btn.config(state=tk.NORMAL)
            else:
                self.stitch_btn.config(state=tk.DISABLED)
            
            logger.info(f"用户选择了 {len(self.image_paths)} 张图像")
    
    def _display_selected_images(self, file_paths):
        """
        显示用户选择的图像的缩略图
        
        Args:
            file_paths: 图像文件路径列表
        """
        # 清空现有内容
        self.images_canvas.delete("all")
        self.thumbnail_refs = []  # 清空引用列表
        
        # 设置缩略图大小和布局
        thumbnail_size = (150, 100)
        margin = 10
        
        # 创建一个内部框架来放置缩略图
        self.images_frame_inner = ttk.Frame(self.images_canvas)
        self.images_canvas.create_window((0, 0), window=self.images_frame_inner, anchor=tk.NW)
        
        # 显示每个图像的缩略图
        for i, file_path in enumerate(file_paths):
            try:
                # 计算网格位置
                row = i // 3  # 每行显示3个缩略图
                col = i % 3
                
                # 打开图像并创建缩略图
                image = Image.open(file_path)
                image.thumbnail(thumbnail_size, Image.LANCZOS)
                
                # 转换为Tkinter格式
                photo = ImageTk.PhotoImage(image)
                self.thumbnail_refs.append(photo)  # 保存引用
                
                # 创建图像标签
                img_label = ttk.Label(self.images_frame_inner, image=photo)
                img_label.grid(row=row, column=col, padx=margin, pady=margin)
                
                # 创建图像名称标签
                file_name = os.path.basename(file_path)
                name_label = ttk.Label(self.images_frame_inner, text=file_name, wraplength=thumbnail_size[0])
                name_label.grid(row=row+1, column=col, padx=margin, pady=2)
                
                # 创建图像序号标签
                num_label = ttk.Label(
                    self.images_frame_inner, 
                    text=f"{i+1}", 
                    background="#4a7a8c", 
                    foreground="white",
                    width=2
                )
                num_label.grid(row=row, column=col, padx=margin, pady=margin, sticky=tk.NW)
                
                # 添加工具提示
                self._create_tooltip(img_label, f"图像 {i+1}: {file_name}")
            except Exception as e:
                logger.error(f"无法显示图像 {file_path}: {str(e)}")
        
        # 更新滚动区域
        self.images_frame_inner.update_idletasks()  # 确保子组件已布局
        self.images_canvas.configure(
            scrollregion=self.images_canvas.bbox("all")
        )
    
    def _create_tooltip(self, widget, text):
        """
        为控件创建工具提示
        
        Args:
            widget: 目标控件
            text: 工具提示文本
        """
        def enter(event):
            # 计算工具提示位置
            x = widget.winfo_rootx() + widget.winfo_width()
            y = widget.winfo_rooty()
            
            # 创建工具提示窗口
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)  # 无边框窗口
            tooltip.wm_geometry(f"+{x+5}+{y}")
            
            # 创建标签显示文本
            label = tk.Label(
                tooltip,
                text=text, 
                background="#ffffe0", 
                relief=tk.SOLID, 
                borderwidth=1, 
                padx=5,  # 使用padx替代padding的水平值
                pady=2   # 使用pady替代padding的垂直值
            )
            label.pack()
        
        def leave(event):
            # 查找并销毁工具提示窗口
            for child in widget.winfo_children():
                if isinstance(child, tk.Toplevel):
                    child.destroy()
        
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
    
    def start_stitching(self):
        """
        开始图像拼接过程
        """
        # 检查是否选择了足够的图像
        if len(self.image_paths) < 2:
            messagebox.showwarning("警告", "请至少选择两张图像进行拼接")
            return
        
        # 禁用按钮，防止重复点击
        self.stitch_btn.config(state=tk.DISABLED)
        self.select_images_btn.config(state=tk.DISABLED)
        
        # 更新状态
        self.status_var.set("正在拼接图像...")
        self.root.title("图像自动拼接系统—sigmoid (正在处理)")
        
        # 获取当前参数值
        config = {
            'SEAM_BAND': self.seam_band_var.get(),
            'FEATHER_RADIUS': self.feather_radius_var.get()
        }
        
        # 创建队列用于线程间通信
        self.progress_queue = queue.Queue()
        
        # 创建新线程执行拼接，避免阻塞UI
        def do_stitching():
            try:
                # 记录开始时间
                t_start = time.time()
                
                # 创建StitchingApp实例
                app = StitchingApp()
                
                # 使用update_config方法更新配置
                app.update_config(config)
                
                # 加载图像
                app.load_images(self.image_paths)
                
                # 设置进度回调函数
                def progress_callback(step, total, message):
                    progress = int((step / total) * 100)
                    self.progress_queue.put((progress, message))
                
                app.set_progress_callback(progress_callback)
                
                # 调用拼接功能
                result = app.run_stitching()
                
                # 计算拼接时间
                stitching_time = time.time() - t_start
                
                # 使用主线程更新UI
                def update_ui():
                    # 恢复窗口标题
                    self.root.title("图像自动拼接系统—sigmoid")
                    
                    # 更新拼接时间显示
                    self.stitching_time_label.config(text=f"拼接时间: {stitching_time:.2f}s")
                    
                    if result is not False:  # 修改判断条件，因为现在result是图像数组而不是True/False
                        # 保存结果引用
                        self.result_image = result
                        # 显示结果
                        self._display_result(self.result_image)
                        self.status_var.set("拼接完成")
                        
                        # 启用保存按钮
                        self.save_btn.config(state=tk.NORMAL)
                        self.file_menu.entryconfig(1, state=tk.NORMAL)  # 启用保存菜单
                        
                        messagebox.showinfo("成功", "图像拼接完成!")
                    else:
                        self.status_var.set("拼接失败")
                        messagebox.showerror("错误", "图像拼接失败，请检查日志了解详情")
                    
                    # 恢复按钮状态
                    self.stitch_btn.config(state=tk.NORMAL)
                    self.select_images_btn.config(state=tk.NORMAL)
                
                # 在主线程中执行UI更新
                self.root.after(0, update_ui)
                
            except Exception as e:
                logger.error(f"拼接过程中发生未捕获的错误: {str(e)}")
                # 错误情况下也需要恢复UI状态
                def error_update_ui():
                    # 恢复窗口标题
                    self.root.title("图像自动拼接系统—sigmoid")
                    
                    # 更新状态
                    self.status_var.set("拼接出错")
                    messagebox.showerror("错误", f"拼接过程中发生错误:\n{str(e)}")
                    self.stitch_btn.config(state=tk.NORMAL)
                    self.select_images_btn.config(state=tk.NORMAL)
                
                self.root.after(0, error_update_ui)
        
        # 创建并启动线程
        stitching_thread = threading.Thread(target=do_stitching)
        stitching_thread.daemon = True  # 设置为守护线程，主线程结束时自动结束
        stitching_thread.start()
        
        # 启动进度更新检查
        self._check_progress_queue()
    
    def _check_progress_queue(self):
        """
        检查进度队列并更新UI（防止GUI冻结）
        """
        try:
            while True:
                try:
                    progress, message = self.progress_queue.get_nowait()
                    self.progress_var.set(progress)
                    self.progress_label.config(text=message)
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"检查进度队列时出错: {str(e)}")

        if self.stitch_btn.cget('state') == 'disabled':
            self.root.update()
            self.root.after(50, self._check_progress_queue)
    
    def _display_result(self, result_image):
        """
        显示拼接结果
        
        Args:
            result_image: numpy数组
        """
        try:
            # 清空现有内容
            self.result_canvas.delete("all")
            
            # 确保是PIL图像对象
            if isinstance(result_image, np.ndarray):
                # 重要：OpenCV使用BGR色彩空间，而PIL使用RGB色彩空间
                # 需要将BGR转换为RGB，否则颜色会显示异常
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                result_image = Image.fromarray(result_image)
            
            # 计算适合显示的大小
            canvas_width = self.result_canvas.winfo_width() - 20
            canvas_height = self.result_canvas.winfo_height() - 20
            
            # 调整图像大小
            img_width, img_height = result_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # 创建缩略图
            display_image = result_image.resize((new_width, new_height), Image.LANCZOS)
            
            # 转换为Tkinter格式
            photo = ImageTk.PhotoImage(display_image)
            self.result_photo = photo  # 保存引用
            
            # 显示图像
            self.result_canvas.create_image(
                canvas_width // 2, 
                canvas_height // 2, 
                image=photo,
                anchor=tk.CENTER
            )
            
            # 更新滚动区域
            self.result_canvas.configure(
                scrollregion=(0, 0, new_width, new_height)
            )
            
            logger.info(f"显示拼接结果，尺寸: {new_width}x{new_height}")
            
        except Exception as e:
            logger.error(f"显示结果时出错: {str(e)}")
            messagebox.showerror("错误", f"无法显示拼接结果:\n{str(e)}")
    
    def save_result(self):
        """
        保存拼接结果
        """
        if self.result_image is None:
            messagebox.showwarning("警告", "没有可保存的拼接结果")
            return
        
        # 打开保存对话框
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
                # 保存结果
                cv2.imwrite(file_path, self.result_image)
                self.status_var.set(f"结果已保存至: {file_path}")
                messagebox.showinfo("成功", f"结果已成功保存到:\n{file_path}")
                logger.info(f"用户保存结果到: {file_path}")
            except Exception as e:
                logger.error(f"保存结果时出错: {str(e)}")
                messagebox.showerror("错误", f"保存结果失败:\n{str(e)}")
    
    def clear_data(self):
        """
        清除所有数据，包括已选择的图像和拼接结果
        """
        # 确认清除操作
        if messagebox.askyesno("确认", "确定要清除所有数据吗？已选择的图像和拼接结果将被移除。"):
            # 清空图像列表
            self.image_paths = []
            
            # 清空缩略图引用
            self.thumbnail_refs = []
            
            # 清空图像预览区域
            self.images_canvas.delete("all")
            
            # 清空结果
            self.result_image = None
            self.result_canvas.delete("all")
            
            # 禁用保存按钮
            self.save_btn.config(state=tk.DISABLED)
            self.file_menu.entryconfig(1, state=tk.DISABLED)  # 禁用保存菜单
            
            # 更新状态栏
            self.status_var.set("就绪")
            
            logger.info("已清除所有数据")
    
    def show_about(self):
        """
        显示关于对话框
        """
        about_text = "图像自动拼接系统\n\n" \
                   "版本: 1.0\n" \
                   "功能: 多张图像自动拼接，支持图割接缝选择和局部梯度融合\n\n" \
                   "© 2023 图像拼接项目组"
        
        messagebox.showinfo("关于", about_text)


def main():
    """
    应用程序主入口
    """
    # 创建Tkinter根窗口
    root = tk.Tk()
    
    # 创建GUI应用实例
    app = StitchingGUI(root)
    
    # 启动主事件循环
    root.mainloop()


if __name__ == "__main__":
    main()