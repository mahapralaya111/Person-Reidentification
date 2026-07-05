#!/usr/bin/env python
"""
ReID系统 - 图形化启动器

提供软件式的图形界面入口，集成所有功能模块。
支持一键启动训练、推理、Web界面、API服务等。

使用方法:
    python launcher.py          # 启动图形界面
    python launcher.py --mini   # 启动迷你模式
"""

import sys
import os
import subprocess
import threading
import webbrowser
from pathlib import Path
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext, filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False
    print("[错误] 需要 tkinter 支持，请安装: pip install tk")

# =============================================================================
# 配置常量
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
PYTHON_EXE = sys.executable
STREAMLIT_PORT = 8501
API_PORT = 5000


# =============================================================================
# 日志重定向（将控制台输出显示到GUI）
# =============================================================================

class TextRedirector:
    """将print输出重定向到tkinter文本框"""
    def __init__(self, text_widget, tag="stdout"):
        self.text_widget = text_widget
        self.tag = tag
    
    def write(self, string):
        if self.text_widget:
            self.text_widget.insert(tk.END, string, self.tag)
            self.text_widget.see(tk.END)
            self.text_widget.update_idletasks()
    
    def flush(self):
        pass


# =============================================================================
# 主启动器窗口
# =============================================================================

class ReIDLauncher:
    """ReID系统图形化启动器"""
    
    # 应用版本
    VERSION = "2.0.0"
    
    # 功能按钮配置
    FUNCTIONS = [
        {
            "id": "check",
            "name": "🔍 环境检查",
            "desc": "检查Python/依赖/GPU/模型文件",
            "icon": "🔍",
            "color": "#4CAF50",
            "category": "系统工具"
        },
        {
            "id": "web",
            "name": "🌐 Web界面",
            "desc": "启动可视化Web操作界面",
            "icon": "🌐",
            "color": "#2196F3",
            "category": "核心功能"
        },
        {
            "id": "api",
            "name": "⚡ API服务",
            "desc": "启动REST API推理服务",
            "icon": "⚡",
            "color": "#FF9800",
            "category": "核心功能"
        },
        {
            "id": "search",
            "name": "🔎 人员检索",
            "desc": "上传图片在底库中搜索匹配人员",
            "icon": "🔎",
            "color": "#9C27B0",
            "category": "核心功能"
        },
        {
            "id": "feature",
            "name": "📸 特征提取",
            "desc": "提取单张/批量图片特征向量",
            "icon": "📸",
            "color": "#00BCD4",
            "category": "核心功能"
        },
        {
            "id": "gallery",
            "name": "📚 构建底库",
            "desc": "从图片目录构建底库特征库",
            "icon": "📚",
            "color": "#607D8B",
            "category": "核心功能"
        },
        {
            "id": "train",
            "name": "🎯 模型训练",
            "desc": "启动完整模型训练流程",
            "icon": "🎯",
            "color": "#F44336",
            "category": "训练"
        },
        {
            "id": "quick_train",
            "name": "⚡ 一键快速训练",
            "desc": "自动整理底库→开始训练(推荐)",
            "icon": "⚡",
            "color": "#FF1744",
            "category": "训练"
        },
        {
            "id": "augment",
            "name": "🎨 底库增强",
            "desc": "生成多视角多环境增强样本",
            "icon": "🎨",
            "color": "#E91E63",
            "category": "训练"
        },
        {
            "id": "eval",
            "name": "📊 模型评估",
            "desc": "评估模型Rank-1/mAP性能",
            "icon": "📊",
            "color": "#3F51B5",
            "category": "训练"
        },
        {
            "id": "video",
            "name": "🎬 视频ReID",
            "desc": "视频流实时行人检测与跟踪",
            "icon": "🎬",
            "color": "#009688",
            "category": "视频处理"
        },
        {
            "id": "demo",
            "name": "🎪 演示模式",
            "desc": "快速体验搜救识别全流程",
            "icon": "🎪",
            "color": "#FF5722",
            "category": "演示"
        },
        {
            "id": "test",
            "name": "🧪 运行测试",
            "desc": "运行单元测试验证系统",
            "icon": "🧪",
            "color": "#795548",
            "category": "系统工具"
        },
    ]
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"行人重识别系统 v{self.VERSION}")
        self.root.geometry("900x680")
        self.root.minsize(800, 600)
        
        # 设置图标
        try:
            self.root.iconbitmap(default=self._get_icon_path())
        except Exception:
            pass
        
        # 进程管理
        self.running_process = None
        self.process_thread = None
        self.status_var = tk.StringVar(value="就绪 - 请选择功能")
        
        # 构建界面
        self._build_ui()
        
        # 窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _get_icon_path(self):
        """获取图标路径"""
        # 尝试使用Python自带的图标
        import tempfile
        icon_path = os.path.join(tempfile.gettempdir(), "reid_icon.ico")
        return icon_path
    
    def _build_ui(self):
        """构建用户界面"""
        # 主布局: 顶部标题 + 中间内容 + 底部状态栏
        self._build_header()
        self._build_content()
        self._build_statusbar()
    
    def _build_header(self):
        """构建顶部标题栏"""
        header = tk.Frame(self.root, bg="#1a237e", height=80)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        
        # 标题文字
        tk.Label(
            header,
            text="🚨  抗恶劣环境行人重识别系统",
            font=("微软雅黑", 18, "bold"),
            bg="#1a237e",
            fg="white"
        ).pack(side=tk.LEFT, padx=20, pady=15)
        
        # 版本号
        tk.Label(
            header,
            text=f"v{self.VERSION}",
            font=("微软雅黑", 10),
            bg="#1a237e",
            fg="#90CAF9"
        ).pack(side=tk.LEFT, padx=5, pady=15)
        
        # 右侧信息
        tk.Label(
            header,
            text="面向灾害应急搜救",
            font=("微软雅黑", 10),
            bg="#1a237e",
            fg="#B0BEC5"
        ).pack(side=tk.RIGHT, padx=20, pady=15)
    
    def _build_content(self):
        """构建中间内容区域"""
        content = tk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：功能按钮区域
        left_frame = tk.Frame(content, width=480)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_frame.pack_propagate(False)
        
        # 分类标题
        categories = ["核心功能", "训练", "视频处理", "演示", "系统工具"]
        canvas = tk.Canvas(left_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 按分类排列按钮
        current_row = 0
        for cat in categories:
            # 分类标题
            tk.Label(
                scrollable_frame,
                text=f"── {cat} ──",
                font=("微软雅黑", 11, "bold"),
                fg="#1565C0",
                anchor="w"
            ).grid(row=current_row, column=0, columnspan=3, sticky="ew", pady=(8, 2), padx=5)
            current_row += 1
            
            # 该类下的功能按钮（每行3个）
            cat_funcs = [f for f in self.FUNCTIONS if f["category"] == cat]
            for i, func in enumerate(cat_funcs):
                col = i % 3
                if col == 0 and i > 0:
                    current_row += 1
                self._create_function_button(scrollable_frame, func, current_row, col)
            current_row += 1
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 鼠标滚轮支持
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # 右侧：日志和控制台
        right_frame = tk.Frame(content, width=380)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # 控制台标题
        console_header = tk.Frame(right_frame, bg="#263238", height=35)
        console_header.pack(fill=tk.X, side=tk.TOP)
        console_header.pack_propagate(False)
        tk.Label(
            console_header,
            text="📋 控制台输出",
            font=("微软雅黑", 10, "bold"),
            bg="#263238",
            fg="#ECEFF1"
        ).pack(side=tk.LEFT, padx=10, pady=5)
        
        # 日志文本框
        self.console = scrolledtext.ScrolledText(
            right_frame,
            font=("Consolas", 9),
            bg="#1a1a2e",
            fg="#00ff00",
            insertbackground="white",
            wrap=tk.WORD,
            height=20
        )
        self.console.pack(fill=tk.BOTH, expand=True)
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, "系统就绪，请选择功能...\n")
        self.console.config(state=tk.DISABLED)
        
        # 标签配置
        self.console.tag_config("stdout", foreground="#00ff00")
        self.console.tag_config("stderr", foreground="#ff4444")
        self.console.tag_config("info", foreground="#64B5F6")
        self.console.tag_config("warning", foreground="#FFD54F")
        self.console.tag_config("success", foreground="#69F0AE")
    
    def _create_function_button(self, parent, func, row, col):
        """创建功能按钮"""
        frame = tk.Frame(parent, relief=tk.RAISED, bd=1, cursor="hand2")
        frame.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        
        # 配置网格权重
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_columnconfigure(2, weight=1)
        
        # 图标
        tk.Label(
            frame,
            text=func["icon"],
            font=("Segoe UI Emoji", 16),
            fg=func["color"]
        ).pack(pady=(6, 0))
        
        # 功能名
        tk.Label(
            frame,
            text=func["name"],
            font=("微软雅黑", 10, "bold"),
            fg="#37474F"
        ).pack()
        
        # 描述
        tk.Label(
            frame,
            text=func["desc"],
            font=("微软雅黑", 7),
            fg="#78909C",
            wraplength=140
        ).pack(pady=(0, 6))
        
        # 点击事件
        cmd_id = func["id"]
        frame.bind("<Button-1>", lambda e, c=cmd_id: self._on_function_click(c))
        for child in frame.winfo_children():
            child.bind("<Button-1>", lambda e, c=cmd_id: self._on_function_click(c))
        
        # 悬停效果
        frame.bind("<Enter>", lambda e, f=frame: f.configure(bg="#E3F2FD"))
        frame.bind("<Leave>", lambda e, f=frame: f.configure(bg="SystemButtonFace"))
    
    def _build_statusbar(self):
        """构建底部状态栏"""
        statusbar = tk.Frame(self.root, bg="#ECEFF1", height=30)
        statusbar.pack(fill=tk.X, side=tk.BOTTOM)
        statusbar.pack_propagate(False)
        
        # 状态文字
        tk.Label(
            statusbar,
            textvariable=self.status_var,
            font=("微软雅黑", 9),
            bg="#ECEFF1",
            fg="#546E7A",
            anchor="w"
        ).pack(side=tk.LEFT, padx=10)
        
        # 进程控制按钮
        self.stop_btn = tk.Button(
            statusbar,
            text="⏹ 停止",
            font=("微软雅黑", 9),
            bg="#EF5350",
            fg="white",
            relief=tk.FLAT,
            command=self._stop_process,
            state=tk.DISABLED,
            padx=10
        )
        self.stop_btn.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # 清空控制台按钮
        tk.Button(
            statusbar,
            text="🗑 清空",
            font=("微软雅黑", 9),
            bg="#78909C",
            fg="white",
            relief=tk.FLAT,
            command=self._clear_console,
            padx=8
        ).pack(side=tk.RIGHT, padx=5)
    
    def _on_function_click(self, func_id):
        """功能按钮点击事件"""
        self._clear_console()
        self._log(f"[系统] 启动功能: {func_id}\n", "info")
        
        if func_id == "check":
            self._run_check()
        elif func_id == "web":
            self._run_web()
        elif func_id == "api":
            self._run_api()
        elif func_id == "search":
            self._run_search()
        elif func_id == "feature":
            self._run_feature()
        elif func_id == "gallery":
            self._run_gallery()
        elif func_id == "train":
            self._run_train()
        elif func_id == "quick_train":
            self._run_quick_train()
        elif func_id == "augment":
            self._run_augment()
        elif func_id == "eval":
            self._run_eval()
        elif func_id == "video":
            self._run_video()
        elif func_id == "demo":
            self._run_demo()
        elif func_id == "test":
            self._run_test()
    
    # =========================================================================
    # 功能执行方法
    # =========================================================================
    
    def _run_command(self, cmd_args, shell=False):
        """在后台线程中运行命令"""
        def target():
            try:
                self.root.after(0, lambda: self.stop_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.status_var.set("正在运行..."))
                
                # 设置环境变量：强制无缓冲输出，确保日志实时显示
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"

                self.running_process = subprocess.Popen(
                    cmd_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    shell=shell,
                    cwd=str(PROJECT_ROOT),
                    env=env
                )
                
                for line in iter(self.running_process.stdout.readline, ""):
                    if line:
                        self.root.after(0, lambda l=line: self._log(l))
                    if not self.running_process or self.running_process.poll() is not None:
                        break
                
                return_code = self.running_process.wait() if self.running_process else -1
                self.root.after(0, lambda: self.status_var.set(
                    f"完成 (退出码: {return_code})" if return_code == 0 else f"失败 (退出码: {return_code})"
                ))
                if return_code == 0:
                    self.root.after(0, lambda: self._log("[系统] ✅ 执行成功\n", "success"))
                else:
                    self.root.after(0, lambda: self._log(f"[系统] ❌ 进程退出码: {return_code}\n", "stderr"))
            except Exception as e:
                self.root.after(0, lambda: self._log(f"[错误] {e}\n", "stderr"))
            finally:
                self.running_process = None
                self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
        
        self.process_thread = threading.Thread(target=target, daemon=True)
        self.process_thread.start()
    
    def _run_check(self):
        """环境检查"""
        self._run_command([PYTHON_EXE, "main.py", "check"])
    
    def _run_web(self):
        """启动Web界面"""
        self._log("[系统] 正在启动Web界面...\n", "info")
        self._log(f"[系统] 访问地址: http://localhost:{STREAMLIT_PORT}\n", "info")
        
        def target():
            try:
                self.root.after(0, lambda: self.stop_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.status_var.set("Web服务运行中..."))
                
                web_dir = PROJECT_ROOT / "web"
                self.running_process = subprocess.Popen(
                    [PYTHON_EXE, "-m", "streamlit", "run", "app.py",
                     f"--server.port={STREAMLIT_PORT}",
                     "--server.headless=true",   # 禁用 streamlit 自动开浏览器，由下方 webbrowser.open 统一控制
                     "--browser.gatherUsageStats=false",
                     "--server.fileWatcherType=none"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(web_dir)
                )
                
                # 等待服务启动
                import time
                time.sleep(3)
                webbrowser.open(f"http://localhost:{STREAMLIT_PORT}")
                self.root.after(0, lambda: self._log("[系统] 🌐 浏览器已打开\n", "success"))
                
                for line in iter(self.running_process.stdout.readline, ""):
                    if line:
                        self.root.after(0, lambda l=line: self._log(l))
            except Exception as e:
                self.root.after(0, lambda: self._log(f"[错误] {e}\n", "stderr"))
            finally:
                self.running_process = None
                self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
        
        self.process_thread = threading.Thread(target=target, daemon=True)
        self.process_thread.start()
    
    def _run_api(self):
        """启动API服务"""
        self._log("[系统] 正在启动API服务...\n", "info")
        self._log(f"[系统] 地址: http://localhost:{API_PORT}\n", "info")
        self._log("[系统] 接口: GET /health, POST /feature, POST /search\n", "info")
        
        def target():
            try:
                self.root.after(0, lambda: self.stop_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.status_var.set("API服务运行中..."))
                
                api_dir = PROJECT_ROOT / "web"
                self.running_process = subprocess.Popen(
                    [PYTHON_EXE, "api.py", f"--port={API_PORT}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(api_dir)
                )
                
                for line in iter(self.running_process.stdout.readline, ""):
                    if line:
                        self.root.after(0, lambda l=line: self._log(l))
            except Exception as e:
                self.root.after(0, lambda: self._log(f"[错误] {e}\n", "stderr"))
            finally:
                self.running_process = None
                self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
        
        self.process_thread = threading.Thread(target=target, daemon=True)
        self.process_thread.start()
    
    def _run_search(self):
        """人员检索"""
        self._start_interactive("search")
    
    def _run_feature(self):
        """特征提取"""
        self._start_interactive("feature")
    
    def _run_gallery(self):
        """构建底库"""
        self._start_interactive("gallery")
    
    def _start_interactive(self, mode):
        """启动交互式功能（弹窗输入参数）"""
        if mode == "search":
            title = "人员检索"
            query = self._ask_input("请输入查询图片路径:", "查询图片")
            gallery = self._ask_input("请输入底库目录路径 (如 rescue_gallery):", "底库目录")
            if query and gallery:
                self._run_command([
                    PYTHON_EXE, "main.py", "inference",
                    "--mode", "search", "--query", query, "--images", gallery
                ])
        elif mode == "feature":
            title = "特征提取"
            path = self._ask_input("请选择图片或目录:", "图片路径")
            output = self._ask_input("输出路径 (选目录自动生成 features.npy):", "输出文件", default="")
            if path:
                cmd = [PYTHON_EXE, "main.py", "inference", "--mode", "feature", "--image", path]
                if output:
                    cmd.extend(["--output", output])
                self._run_command(cmd)
        elif mode == "gallery":
            title = "构建底库"
            gallery_dir = self._ask_input("请输入图片目录 (如 rescue_gallery):", "图片目录")
            output = self._ask_input("请输入输出文件 (默认: gallery.npz):", "输出文件", default="gallery.npz")
            if gallery_dir:
                self._run_command([
                    PYTHON_EXE, "main.py", "inference",
                    "--mode", "gallery", "--images", gallery_dir, "--output", output
                ])
    
    def _ask_input(self, prompt, title, default=""):
        """弹窗获取用户输入（支持粘贴路径或点击浏览选择文件夹）"""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("580x180")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 居中
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 580) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 180) // 2
        dialog.geometry(f"+{x}+{y}")
        
        result = tk.StringVar(value=default)
        
        tk.Label(dialog, text=prompt, font=("微软雅黑", 10)).pack(pady=(10, 5))
        
        # 输入框 + 浏览按钮放在一行
        entry_frame = tk.Frame(dialog)
        entry_frame.pack(padx=15, pady=5, fill=tk.X)
        
        entry = tk.Entry(entry_frame, textvariable=result, font=("微软雅黑", 10))
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        def on_browse():
            folder = filedialog.askdirectory(title="选择文件夹")
            if folder:
                result.set(folder)
        
        tk.Button(entry_frame, text="浏览...", command=on_browse,
                 font=("微软雅黑", 9), padx=8).pack(side=tk.RIGHT, padx=(5, 0))
        
        tk.Label(dialog, text="提示: 可直接粘贴路径，或点击「浏览」选择文件夹",
                font=("微软雅黑", 8), fg="gray").pack()
        
        entry.focus_set()
        
        def on_ok():
            dialog.destroy()
        
        def on_cancel():
            result.set("")
            dialog.destroy()
        
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="确定", command=on_ok, bg="#4CAF50", fg="white",
                 font=("微软雅黑", 10), padx=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="取消", command=on_cancel, bg="#9E9E9E", fg="white",
                 font=("微软雅黑", 10), padx=15).pack(side=tk.LEFT, padx=5)
        
        entry.bind("<Return>", lambda e: on_ok())
        entry.bind("<Escape>", lambda e: on_cancel())
        
        dialog.wait_window()
        value = result.get().strip()
        # 去除 Windows 资源管理器复制路径时自动添加的引号
        if len(value) > 1 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1]
        return value
    
    def _run_train(self):
        """模型训练"""
        dialog = tk.Toplevel(self.root)
        dialog.title("模型训练配置")
        dialog.geometry("420x360")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 420) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 360) // 2
        dialog.geometry(f"+{x}+{y}")
        
        tk.Label(dialog, text="训练参数配置", font=("微软雅黑", 12, "bold"),
                fg="#1565C0").pack(pady=(10, 5))
        
        params = {}
        fields = [
            ("backbone", "骨干网络:", "osnet", ["osnet", "resnet50", "resnet34", "agw", "resnet_ibn", "pcb", "nnlrnet"]),
            ("epochs", "训练轮数:", "60"),
            ("batch_size", "批次大小:", "64"),
            ("lr", "学习率:", "0.0001"),
            ("use_circle", "Circle Loss:", True),
            ("use_weather", "天气增强:", True),
            ("use_cosine", "余弦退火:", True),
        ]
        
        for key, label, default, *extra in fields:
            row = tk.Frame(dialog)
            row.pack(fill=tk.X, padx=20, pady=3)
            tk.Label(row, text=label, font=("微软雅黑", 10), width=12, anchor="w").pack(side=tk.LEFT)
            
            if extra and isinstance(extra[0], list):
                # 下拉选择
                var = tk.StringVar(value=default)
                ttk.Combobox(row, textvariable=var, values=extra[0], width=15).pack(side=tk.LEFT)
                params[key] = var
            elif isinstance(default, bool):
                # 复选框
                var = tk.BooleanVar(value=default)
                tk.Checkbutton(row, variable=var).pack(side=tk.LEFT)
                params[key] = var
            else:
                # 文本框
                var = tk.StringVar(value=default)
                tk.Entry(row, textvariable=var, width=15).pack(side=tk.LEFT)
                params[key] = var
        
        def start_train():
            cmd = [PYTHON_EXE, "main.py", "train",
                   "--backbone", params["backbone"].get(),
                   "--epochs", params["epochs"].get(),
                   "--batch_size", params["batch_size"].get(),
                   "--lr", params["lr"].get()]
            if params["use_circle"].get():
                cmd.append("--use_circle_loss")
            if params["use_weather"].get():
                cmd.append("--use_weather_aug")
            if params["use_cosine"].get():
                cmd.append("--use_cosine_annealing")
            
            self._log(f"[系统] 训练命令: {' '.join(cmd)}\n", "info")
            self._run_command(cmd)
            dialog.destroy()
        
        tk.Button(dialog, text="开始训练", command=start_train,
                 bg="#F44336", fg="white", font=("微软雅黑", 11, "bold"),
                 padx=30, pady=5).pack(pady=15)
        tk.Button(dialog, text="取消", command=dialog.destroy,
                 bg="#9E9E9E", fg="white", font=("微软雅黑", 10)).pack()
    
    def _run_quick_train(self):
        """一键快速训练"""
        self._clear_console()
        self._log("[系统] ⚡ 一键快速训练启动\n", "info")
        
        # 打开训练配置窗口（内嵌底库选择）
        self._run_train_with_preset()

    def _run_train_with_preset(self):
        """训练配置窗口（带一键推荐参数 + 底库目录选择）"""
        dialog = tk.Toplevel(self.root)
        dialog.title("快速训练配置")
        dialog.geometry("520x460")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 520) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 460) // 2
        dialog.geometry(f"+{x}+{y}")

        tk.Label(dialog, text="⚡ 一键快速训练 - 推荐参数已填好", font=("微软雅黑", 12, "bold"),
                fg="#FF1744").pack(pady=(10, 5))

        params = {}
        # 底库目录（新增！让用户选择原始照片目录）
        gallery_frame = tk.Frame(dialog)
        gallery_frame.pack(fill=tk.X, padx=20, pady=3)
        tk.Label(gallery_frame, text="底库目录:", font=("微软雅黑", 10), width=12, anchor="w").pack(side=tk.LEFT)
        gallery_var = tk.StringVar(value=str(PROJECT_ROOT / "rescue_gallery"))
        tk.Entry(gallery_frame, textvariable=gallery_var, font=("微软雅黑", 10), width=28).pack(side=tk.LEFT, padx=2)
        def on_browse_gallery():
            folder = filedialog.askdirectory(title="选择人物照片目录（每人一个子文件夹）")
            if folder:
                gallery_var.set(folder)
        tk.Button(gallery_frame, text="浏览...", command=on_browse_gallery,
                 font=("微软雅黑", 9), padx=5).pack(side=tk.LEFT)
        tk.Label(dialog, text="   每人一个子文件夹，如 rescue_gallery/张三/1.jpg",
                font=("微软雅黑", 8), fg="gray", anchor="w").pack(fill=tk.X, padx=32, pady=(0, 5))

        # 分隔线
        ttk.Separator(dialog, orient='horizontal').pack(fill=tk.X, padx=20, pady=5)

        # 防过拟合参数：小底库(2人27张)用少量轮次、更低学习率、强力数据增强
        fields = [
            ("backbone", "骨干网络:", "nnlrnet", ["nnlrnet", "osnet", "resnet50"]),
            ("epochs", "训练轮数:", "10"),
            ("batch_size", "批次大小:", "8"),
            ("lr", "学习率:", "0.00005"),
            ("data_dir", "数据目录:", "./data/rescue_train"),
        ]


        for key, label, default, *extra in fields:
            row = tk.Frame(dialog)
            row.pack(fill=tk.X, padx=20, pady=3)
            tk.Label(row, text=label, font=("微软雅黑", 10), width=12, anchor="w").pack(side=tk.LEFT)
            if extra and isinstance(extra[0], list):
                var = tk.StringVar(value=default)
                ttk.Combobox(row, textvariable=var, values=extra[0], width=18).pack(side=tk.LEFT)
                params[key] = var
            else:
                var = tk.StringVar(value=default)
                tk.Entry(row, textvariable=var, width=18).pack(side=tk.LEFT)
                params[key] = var

        # 选项
        opt_frame = tk.Frame(dialog)
        opt_frame.pack(pady=5)
        use_erase_var = tk.BooleanVar(value=True)
        use_weather_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opt_frame, text="随机擦除(RandomErasing)", variable=use_erase_var,
                      font=("微软雅黑", 9)).pack(anchor="w")
        tk.Checkbutton(opt_frame, text="天气增强(WeatherAug)", variable=use_weather_var,
                      font=("微软雅黑", 9)).pack(anchor="w")

        def start_quick():
            cmd = [PYTHON_EXE, "main.py", "train",
                   "--backbone", params["backbone"].get(),
                   "--epochs", params["epochs"].get(),
                   "--batch_size", params["batch_size"].get(),
                   "--lr", params["lr"].get(),
                   "--data_dir", params["data_dir"].get(),
                   "--use_circle_loss",
                   "--use_cosine_annealing",
                   "--use_random_erasing" if use_erase_var.get() else None,
                   "--use_weather_aug" if use_weather_var.get() else None,
                   "--finetune", "checkpoints/market1501_pretrained.pth"]
            cmd = [c for c in cmd if c is not None]
            
            self._log(f"[系统] 快速训练命令: {' '.join(cmd)}\n", "info")
            self._log("[系统] 训练开始...耐心等待完成\n", "info")
            self._log("[系统] 训练完成后会自动保存到 checkpoints/best_model.pth\n", "info")
            self._run_command(cmd)
            dialog.destroy()

        def prepare_data_first():
            """先整理数据再训练（使用用户选择的底库目录）"""
            gallery_path = gallery_var.get().strip()
            if not gallery_path or not os.path.isdir(gallery_path):
                self._log("[错误] 底库目录不存在或为空，请重新选择\n", "stderr")
                return
            self._log(f"[系统] 正在从 {gallery_path} 整理数据...\n", "info")
            prepare_cmd = [
                PYTHON_EXE, str(PROJECT_ROOT / "tools" / "prepare_train_data.py"),
                "--source", gallery_path
            ]
            self._run_command(prepare_cmd)
            self._log("[系统] 数据整理完成，请点击「开始训练」继续\n", "success")
            dialog.destroy()

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="📦 先整理数据", command=prepare_data_first,
                 bg="#FF9800", fg="white", font=("微软雅黑", 9), padx=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🚀 开始训练", command=start_quick,
                 bg="#FF1744", fg="white", font=("微软雅黑", 11, "bold"), padx=25, pady=5).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="取消", command=dialog.destroy,
                 bg="#9E9E9E", fg="white", font=("微软雅黑", 10)).pack(side=tk.LEFT, padx=5)

        # 提示
        tk.Label(dialog, text="💡 底库目录 = 你的原始照片（每人一个子文件夹） → 点「先整理数据」→ 再点「开始训练」",
                font=("微软雅黑", 8), fg="#FF6F00").pack(pady=(2, 0))
        tk.Label(dialog, text="💡 数据目录 = 整理后的训练数据（一般保持默认 ./data 即可）",
                font=("微软雅黑", 8), fg="#546E7A").pack()

    def _run_augment(self):
        """底库增强"""
        input_dir = self._ask_input("请输入原始底库目录 (如 rescue_gallery):", "输入目录")
        output_dir = self._ask_input("请输入增强输出目录:", "输出目录", default="rescue_gallery_augmented")
        num_aug = self._ask_input("每张图片增强数量 (默认5):", "增强数量", default="5")
        
        if input_dir and output_dir:
            self._run_command([
                PYTHON_EXE, "main.py", "augment-gallery",
                "--input", input_dir, "--output", output_dir,
                "--num_aug", num_aug
            ])
    
    def _run_eval(self):
        """模型评估（支持选择底库目录）"""
        dialog = tk.Toplevel(self.root)
        dialog.title("模型评估")
        dialog.geometry("520x220")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 520) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 220) // 2
        dialog.geometry(f"+{x}+{y}")

        model_var = tk.StringVar(value="checkpoints/best_model.pth")
        gallery_var = tk.StringVar()
        use_custom = tk.BooleanVar(value=True)

        def on_browse_model():
            f = filedialog.askopenfilename(title="选择模型文件", filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")])
            if f: model_var.set(f)

        def on_browse_gallery():
            f = filedialog.askdirectory(title="选择底库目录")
            if f: gallery_var.set(f)

        def on_start():
            m = model_var.get().strip()
            g = gallery_var.get().strip()
            if use_custom.get() and g:
                self._run_command([PYTHON_EXE, "tools/evaluate_custom.py", "--model_path", m, "--gallery", g, "--output", "eval_results"])
            else:
                self._run_command([PYTHON_EXE, "main.py", "eval", "--model_path", m])
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        tk.Label(dialog, text="模型评估配置", font=("微软雅黑", 11, "bold"), fg="#1565C0").pack(pady=(8, 10))

        # 模型路径
        row1 = tk.Frame(dialog); row1.pack(fill=tk.X, padx=20, pady=3)
        tk.Label(row1, text="模型路径:", font=("微软雅黑", 10), width=10, anchor="w").pack(side=tk.LEFT)
        tk.Entry(row1, textvariable=model_var, font=("微软雅黑", 10), width=33).pack(side=tk.LEFT, padx=5)
        tk.Button(row1, text="浏览", command=on_browse_model).pack(side=tk.LEFT)

        # 评估方式选择
        row2 = tk.Frame(dialog); row2.pack(fill=tk.X, padx=20, pady=3)
        tk.Label(row2, text="评估方式:", font=("微软雅黑", 10), width=10, anchor="w").pack(side=tk.LEFT)
        tk.Radiobutton(row2, text="自定义底库（推荐）", variable=use_custom, value=True, font=("微软雅黑", 9)).pack(side=tk.LEFT)
        tk.Radiobutton(row2, text="Market-1501标准", variable=use_custom, value=False, font=("微软雅黑", 9)).pack(side=tk.LEFT)

        # 底库目录
        row3 = tk.Frame(dialog); row3.pack(fill=tk.X, padx=20, pady=3)
        tk.Label(row3, text="底库目录:", font=("微软雅黑", 10), width=10, anchor="w").pack(side=tk.LEFT)
        tk.Entry(row3, textvariable=gallery_var, font=("微软雅黑", 10), width=33).pack(side=tk.LEFT, padx=5)
        tk.Button(row3, text="浏览", command=on_browse_gallery).pack(side=tk.LEFT)

        # 提示
        tk.Label(dialog, text="自定义底库评估输出: Rank-1/mAP/混淆矩阵/相似度分布图",
                font=("微软雅黑", 8), fg="gray").pack(pady=(0, 5))

        # 按钮
        btn_frame = tk.Frame(dialog); btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="开始评估", command=on_start, bg="#4CAF50", fg="white", font=("微软雅黑", 10), padx=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="取消", command=on_cancel, bg="#9E9E9E", fg="white", font=("微软雅黑", 10), padx=15).pack(side=tk.LEFT, padx=5)
    
    def _run_video(self):
        """视频处理"""
        # 使用文件选择对话框让用户选择视频文件
        dialog = tk.Toplevel(self.root)
        dialog.title("选择视频源")
        dialog.geometry("500x340")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 居中
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 500) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 340) // 2
        dialog.geometry(f"+{x}+{y}")
        
        source_var = tk.StringVar(value="webcam")
        gallery_var = tk.StringVar()
        model_var = tk.StringVar(value="checkpoints/best_model.pth")
        debug_var = tk.BooleanVar(value=False)
        result = {'source': None, 'gallery': None, 'model': None, 'debug': False, 'ready': False}
        
        def on_browse_video():
            filename = filedialog.askopenfilename(
                title="选择视频文件",
                filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
            )
            if filename:
                source_var.set(filename)
        
        def on_browse_gallery():
            folder = filedialog.askdirectory(title="选择底库目录（可选）")
            if folder:
                gallery_var.set(folder)
        
        def on_start():
            result['source'] = source_var.get().strip()
            result['gallery'] = gallery_var.get().strip()
            result['model'] = model_var.get().strip()
            result['debug'] = debug_var.get()
            result['ready'] = True
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        # 标题
        tk.Label(dialog, text="视频ReID - 选择视频源", font=("微软雅黑", 12, "bold"), fg="#1565C0").pack(pady=(10, 15))
        
        # 视频源选择
        tk.Label(dialog, text="视频源:", font=("微软雅黑", 10), anchor="w").place(x=20, y=55, width=80, height=25)
        tk.Entry(dialog, textvariable=source_var, font=("微软雅黑", 10)).place(x=100, y=55, width=250, height=28)
        tk.Button(dialog, text="浏览...", command=on_browse_video, font=("微软雅黑", 9), padx=8).place(x=355, y=55, width=70, height=28)
        tk.Label(dialog, text="(支持: MP4/AVI视频文件, webcam, rtsp://...)", font=("微软雅黑", 8), fg="gray").place(x=100, y=80, width=320, height=18)
        
        # 底库目录（可选）
        tk.Label(dialog, text="底库目录:", font=("微软雅黑", 10), anchor="w").place(x=20, y=105, width=80, height=25)
        tk.Entry(dialog, textvariable=gallery_var, font=("微软雅黑", 10)).place(x=100, y=105, width=250, height=28)
        tk.Button(dialog, text="浏览...", command=on_browse_gallery, font=("微软雅黑", 9), padx=8).place(x=355, y=105, width=70, height=28)
        tk.Label(dialog, text="(可选，用于识别视频中的人员身份)", font=("微软雅黑", 8), fg="gray").place(x=100, y=130, width=280, height=18)
        
        # 模型路径
        tk.Label(dialog, text="模型路径:", font=("微软雅黑", 10), anchor="w").place(x=20, y=160, width=80, height=25)
        tk.Entry(dialog, textvariable=model_var, font=("微软雅黑", 10)).place(x=100, y=160, width=325, height=28)
        
        # 调试模式
        debug_check = tk.Checkbutton(
            dialog, text="调试模式（跳过YOLO，直接对全图做ReID）",
            variable=debug_var, font=("微软雅黑", 9),
            anchor="w", justify="left", padx=5
        )
        debug_check.place(x=20, y=200, width=430, height=25)
        tk.Label(dialog, text="(用于测试ReID识别效果，无需YOLO正确检测到人)",
                font=("微软雅黑", 8), fg="gray").place(x=40, y=225, width=380, height=18)
        
        # 提示
        tk.Label(dialog, text="提示：处理真实监控视频时，请确保YOLO能检测到人",
                font=("微软雅黑", 8), fg="orange").place(x=20, y=255, width=430, height=18)
        
        # 按钮
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(side=tk.BOTTOM, pady=15)
        tk.Button(btn_frame, text="开始处理", command=on_start, bg="#4CAF50", fg="white", 
                 font=("微软雅黑", 11, "bold"), padx=20, pady=5).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="取消", command=on_cancel, bg="#9E9E9E", fg="white", 
                 font=("微软雅黑", 10), padx=15, pady=5).pack(side=tk.LEFT, padx=10)
        
        dialog.wait_window()
        
        if result['ready'] and result['source']:
            cmd = [PYTHON_EXE, "main.py", "video", "--source", result['source'], "--model_path", result['model']]
            if result['gallery']:
                cmd.extend(["--gallery", result['gallery']])
            if result['debug']:
                cmd.append("--debug")
            # 视频模式需要独立控制台窗口（OpenCV GUI需要真实控制台，不能pipe）
            import subprocess as _sp
            try:
                _sp.Popen(cmd, creationflags=_sp.CREATE_NEW_CONSOLE, cwd=str(PROJECT_ROOT))
            except Exception:
                # 回退到普通方式
                self._run_command(cmd)
    
    def _run_demo(self):
        """演示模式"""
        self._run_command([PYTHON_EXE, "main.py", "demo"])
    
    def _run_test(self):
        """运行测试"""
        dialog = tk.Toplevel(self.root)
        dialog.title("运行测试")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="选择测试模块", font=("微软雅黑", 12, "bold"),
                fg="#1565C0").pack(pady=(10, 5))
        
        modules = [("全部", ""), ("核心", "test_core"), ("推理", "test_inference"),
                   ("增强", "test_augmentation"), ("工具", "test_utils")]
        
        for name, module in modules:
            tk.Button(
                dialog, text=f"运行 {name} 测试", font=("微软雅黑", 10),
                bg="#795548", fg="white", padx=20,
                command=lambda m=module: self._start_test(m, dialog)
            ).pack(pady=2)
    
    def _start_test(self, module, dialog=None):
        if dialog:
            dialog.destroy()
        cmd = [PYTHON_EXE, "tests/run_tests.py", "-v"]
        if module:
            cmd.extend(["--module", module])
        self._run_command(cmd)
    
    def _stop_process(self):
        """停止当前进程"""
        if self.running_process:
            self._log("[系统] ⏹ 正在停止进程...\n", "warning")
            try:
                self.running_process.terminate()
                self.running_process.wait(timeout=3)
            except Exception:
                try:
                    self.running_process.kill()
                except Exception:
                    pass
            self.running_process = None
            self._log("[系统] ✅ 进程已停止\n", "warning")
            self.status_var.set("已停止")
            self.stop_btn.config(state=tk.DISABLED)
    
    def _log(self, message, tag="stdout"):
        """向控制台输出日志"""
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, message, tag)
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)
    
    def _clear_console(self):
        """清空控制台"""
        self.console.config(state=tk.NORMAL)
        self.console.delete("1.0", tk.END)
        self.console.config(state=tk.DISABLED)
    
    def _on_close(self):
        """窗口关闭事件"""
        if self.running_process:
            if messagebox.askyesno("确认退出", "有进程正在运行，确定要退出吗？"):
                self._stop_process()
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """启动主循环"""
        self.root.mainloop()


# =============================================================================
# 程序入口
# =============================================================================

def main():
    if not HAS_TK:
        print("[错误] tkinter 不可用，请使用: python main.py 命令行模式")
        print("  或者安装 tk: pip install tk")
        sys.exit(1)
    
    # 解析参数
    if "--mini" in sys.argv:
        # 迷你模式：仅显示精简版菜单
        print("迷你模式启动中...")
    
    app = ReIDLauncher()
    app.run()


if __name__ == "__main__":
    main()
