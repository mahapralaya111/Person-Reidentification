#!/usr/bin/env python
"""
抗恶劣环境行人重识别系统 - 统一入口

一个程序入口整合所有功能，支持以下子命令：
1. train         训练模型
2. inference     特征提取与行人检索
3. augment-gallery 底库增强（生成多视角/多环境样本）
4. demo          图形界面演示
5. web           启动Web界面 (Streamlit)
6. api           启动API服务 (Flask)
7. eval          模型评估
8. video         视频处理与行人跟踪
9. check         环境检查与依赖验证

使用方法：
    python main.py <子命令> [选项]

示例：
    python main.py train --backbone osnet --epochs 80
    python main.py inference --mode search --query drone_captures/xxx.png --images rescue_gallery
    python main.py augment-gallery --input rescue_gallery --output rescue_gallery_augmented --num_aug 5
    python main.py web
    python main.py check
"""

import sys
import os
import argparse
from pathlib import Path

# 添加src目录到Python路径，确保模块可以导入
sys.path.insert(0, str(Path(__file__).parent / "src"))


def train_command(args):
    """训练模型"""
    # 延迟导入以避免依赖问题
    try:
        from train import main as train_main
        # 设置sys.argv以模拟命令行参数
        sys.argv = ['train'] + args.extra_args
        train_main()
    except ImportError as e:
        print(f"[错误] 无法导入训练模块: {e}")
        print("请确保src/train.py存在且依赖已安装")
        return 1
    return 0


def inference_command(args):
    """推理与检索"""
    try:
        from inference import main as inference_main
        sys.argv = ['inference'] + args.extra_args
        inference_main()
    except ImportError as e:
        print(f"[错误] 无法导入推理模块: {e}")
        return 1
    return 0


def augment_gallery_command(args):
    """底库增强"""
    try:
        from augment_gallery import main as augment_main
        sys.argv = ['augment-gallery'] + args.extra_args
        augment_main()
    except ImportError as e:
        print(f"[错误] 无法导入底库增强模块: {e}")
        print("请确保src/augment_gallery.py存在")
        return 1
    return 0


def demo_command(args):
    """图形界面演示"""
    try:
        from demos.demo_real_images import main as demo_main
        sys.argv = ['demo'] + args.extra_args
        demo_main()
    except ImportError as e:
        print(f"[错误] 无法导入演示模块: {e}")
        print("请确保demos/demo_real_images.py存在")
        return 1
    return 0


def web_command(args):
    """启动Web界面"""
    try:
        # 切换到web目录，避免路径问题
        web_dir = Path(__file__).parent / "web"
        os.chdir(web_dir)
        
        # 导入并运行streamlit应用
        import subprocess
        import webbrowser
        
        print("正在启动Web界面...")
        print("请稍等，Streamlit服务启动中...")
        print(f"启动后请访问: http://localhost:8501")
        
        # 使用streamlit命令行启动
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except Exception as e:
        print(f"[错误] 启动Web界面失败: {e}")
        return 1
    return 0


def api_command(args):
    """启动API服务"""
    try:
        # 切换到web目录
        web_dir = Path(__file__).parent / "web"
        os.chdir(web_dir)
        
        import subprocess
        print("正在启动API服务...")
        print(f"API地址: http://localhost:5000")
        print("接口文档:")
        print("  GET  /health     健康检查")
        print("  POST /feature    提取特征")
        print("  POST /search     行人检索")
        
        # 优先使用 gpu_env 的 Python（其中已安装 Flask）
        gpu_python = Path(__file__).parent / "gpu_env" / "Scripts" / "python.exe"
        python_exe = str(gpu_python) if gpu_python.exists() else sys.executable
        
        subprocess.run([python_exe, "api.py"])
    except Exception as e:
        print(f"[错误] 启动API服务失败: {e}")
        return 1
    return 0


def eval_command(args):
    """模型评估"""
    try:
        from eval import main as eval_main
        sys.argv = ['eval'] + args.extra_args
        eval_main()
    except ImportError as e:
        print(f"[错误] 无法导入评估模块: {e}")
        print("请确保src/eval.py存在且包含main函数")
        return 1
    return 0


def video_command(args):
    """视频处理"""
    try:
        from video_processor import main as video_main
        sys.argv = ['video'] + args.extra_args
        video_main()
    except ImportError as e:
        print(f"[错误] 无法导入视频处理模块: {e}")
        print("请确保src/video_processor.py存在")
        return 1
    return 0


def check_command(args):
    """环境检查"""
    print("正在检查系统环境...")
    
    # Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查关键依赖
    required = ['torch', 'torchvision', 'numpy', 'cv2', 'PIL', 'streamlit', 'flask']
    optional = ['matplotlib', 'scipy', 'tqdm']
    
    # 导入名 -> 包分发名映射（用于版本查询）
    _PKG_MAP = {
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
    }

    def _get_version(import_name, module):
        import importlib.metadata as md
        dist_name = _PKG_MAP.get(import_name, import_name)
        try:
            return md.version(dist_name)
        except (md.PackageNotFoundError, Exception):
            return getattr(module, '__version__', '未知')

    print("\n必需依赖:")
    for lib in required:
        try:
            module = __import__(lib)
            version = _get_version(lib, module)
            print(f"  [OK] {lib:20s} {version}")
        except ImportError:
            print(f"  [NO] {lib:20s} 未安装")

    print("\n可选依赖:")
    for lib in optional:
        try:
            module = __import__(lib)
            version = _get_version(lib, module)
            print(f"  [OK] {lib:20s} {version}")
        except ImportError:
            print(f"  [--] {lib:20s} 未安装")
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\nGPU可用: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
        else:
            print("\nGPU不可用，将使用CPU")
    except ImportError:
        print("\nPyTorch未安装，无法检查GPU状态")
    except Exception as e:
        print(f"\nGPU检查失败: {e}")
    
    # 检查数据目录
    data_dirs = ['checkpoints', 'data', 'drone_captures', 'rescue_gallery', 'test_video']
    print("\n目录检查:")
    for d in data_dirs:
        path = Path(d)
        if path.exists():
            print(f"  [OK] {d:20s} 存在")
        else:
            print(f"  - {d:20s} 不存在 (可能影响部分功能)")
    
    print("\n环境检查完成。")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='抗恶劣环境行人重识别系统 - 统一入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # train 子命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='传递给train.py的额外参数')
    
    # inference 子命令
    infer_parser = subparsers.add_parser('inference', help='特征提取与行人检索')
    infer_parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='传递给inference.py的额外参数')
    
    # augment-gallery 子命令
    aug_parser = subparsers.add_parser('augment-gallery', help='底库增强')
    aug_parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='传递给augment_gallery.py的额外参数')
    
    # demo 子命令
    demo_parser = subparsers.add_parser('demo', help='图形界面演示')
    demo_parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='传递给demo_real_images.py的额外参数')
    
    # web 子命令
    web_parser = subparsers.add_parser('web', help='启动Web界面')
    
    # api 子命令
    api_parser = subparsers.add_parser('api', help='启动API服务')
    
    # eval 子命令（用 parse_known_args 防止 -- 参数被父级拦截）
    eval_parser = subparsers.add_parser('eval', help='模型评估')
    eval_parser.add_argument('extra_args', nargs=argparse.REMAINDER, 
                            help='传递给eval.py的额外参数 (例如: --model_path checkpoints/best_model.pth)')
    
    # video 子命令
    video_parser = subparsers.add_parser('video', help='视频处理')
    video_parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='传递给video_processor.py的额外参数')
    
    # check 子命令
    check_parser = subparsers.add_parser('check', help='环境检查')
    
    # 使用 parse_known_args 防止子命令的参数被父级拦截
    args, unknown = parser.parse_known_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # 未知参数（如 --mode）可能被父级截胡，需要前置合并到 extra_args
    if hasattr(args, 'extra_args'):
        args.extra_args = unknown + args.extra_args
    
    # 执行对应命令
    commands = {
        'train': train_command,
        'inference': inference_command,
        'augment-gallery': augment_gallery_command,
        'demo': demo_command,
        'web': web_command,
        'api': api_command,
        'eval': eval_command,
        'video': video_command,
        'check': check_command,
    }
    
    func = commands.get(args.command)
    if func:
        return func(args)
    else:
        print(f"未知命令: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())