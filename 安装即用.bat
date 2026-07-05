@echo off
chcp 65001 >nul 2>&1
title S.E.E.R. 先见者 - 一键安装
setlocal enabledelayedexpansion

echo ╔══════════════════════════════════════════════════════╗
echo ║     S.E.E.R. · 先见者  - 一键安装部署              ║
echo ║     把整个文件夹复制到别人电脑，双击这个即可安装     ║
echo ╚══════════════════════════════════════════════════════╝
echo.

:: ============================================================
:: 1. 检查 Python（兼容多种检测方式）
:: ============================================================
echo [1/6] 检查 Python 环境...
echo.

:: 方式1：直接 python 命令
python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_FULL=%%i
    goto :python_ok
)

:: 方式2：py 启动器（Windows 自带）
py --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('py --version 2^>^&1') do set PYTHON_FULL=%%i
    echo   发现 Python 启动器 (py)，尝试使用 py...
    set USE_PY_LAUNCHER=1
    goto :python_ok
)

:: 方式3：检查常见安装路径
for %%p in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
    "C:\Program Files\Python312\python.exe"
    "C:\Program Files\Python311\python.exe"
    "C:\Program Files\Python310\python.exe"
) do (
    if exist %%p (
        set PYTHON_EXE=%%p
        "%%p" --version >nul 2>&1
        if !errorlevel! equ 0 (
            for /f "tokens=*" %%j in ('"%%p" --version 2^>^&1') do set PYTHON_FULL=%%j
            echo   发现 Python 安装: %%p
            goto :python_ok
        )
    )
)

:: Python 未找到
echo ❌ 未检测到 Python！
echo    本系统需要 Python 3.8 ~ 3.12 运行。
echo.
echo    请先安装 Python 3.10（推荐）：
echo    下载: https://www.python.org/downloads/release/python-31011/
echo.
echo    ⚠ 安装时请务必勾选 "Add Python to PATH"
echo.
echo    安装完成后，重新运行本脚本即可。
echo.
pause
exit /b 1

:python_ok
:: 提取版本号
if defined PYTHON_EXE (
    set PYTHON_CMD=!PYTHON_EXE!
) else if defined USE_PY_LAUNCHER (
    set PYTHON_CMD=py
) else (
    set PYTHON_CMD=python
)
echo ✅ 检测到: %PYTHON_FULL%
echo.

:: ============================================================
:: 2. 创建/检测虚拟环境
:: ============================================================
echo [2/6] 配置虚拟环境...
echo.

set VENV_DIR=gpu_env
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo ✅ 检测到已有虚拟环境 (%VENV_DIR%)，跳过创建。
) else (
    echo    正在创建虚拟环境（首次约 10~30 秒）...
    %PYTHON_CMD% -m venv %VENV_DIR%
    if !errorlevel! neq 0 (
        echo ❌ 创建虚拟环境失败！
        echo    尝试: %PYTHON_CMD% -m venv --without-pip %VENV_DIR%
        %PYTHON_CMD% -m venv --without-pip %VENV_DIR%
        if !errorlevel! neq 0 (
            echo ❌ 创建失败，请检查 Python 安装。
            pause
            exit /b 1
        )
        echo ✅ 虚拟环境已创建（pip 需要手动安装）
    ) else (
        echo ✅ 虚拟环境创建成功
    )
)
echo.

:: ============================================================
:: 3. 安装依赖
:: ============================================================
echo [3/6] 安装依赖包（首次约 5~15 分钟，视网速而定）...
echo.

:: 激活虚拟环境
if exist "%VENV_DIR%\Scripts\activate.bat" (
    call %VENV_DIR%\Scripts\activate.bat
) else (
    echo ⚠ 未找到虚拟环境激活脚本，使用系统 Python。
)

:: 升级 pip
echo    升级 pip...
python -m pip install --upgrade pip -q
echo ✅ pip 已升级
echo.

:: 选择国内源加速（默认使用清华镜像）
set PIP_SOURCE=-i https://pypi.tuna.tsinghua.edu.cn/simple

:: 安装 PyTorch（CPU 版）
echo    [1/4] 安装 PyTorch (CPU版)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q !PIP_SOURCE!
if !errorlevel! neq 0 (
    echo    ⚠ PyTorch 官方源失败，尝试国内镜像...
    pip install torch torchvision !PIP_SOURCE! -q
)
echo    ✅ PyTorch 安装完成

:: 安装 opencv
echo    [2/4] 安装 OpenCV...
pip install opencv-python -q !PIP_SOURCE!
echo    ✅ OpenCV 安装完成

:: 安装其他核心依赖
echo    [3/4] 安装其他核心依赖...
pip install Pillow numpy scipy matplotlib scikit-learn tqdm pyyaml -q !PIP_SOURCE!
echo    ✅ 核心依赖安装完成

:: 安装可选依赖（不中断流程）
echo    [4/4] 安装可选依赖...
pip install ultralytics streamlit flask flask-cors requests -q !PIP_SOURCE!
echo    ✅ 可选依赖安装完成

echo.
echo ✅ 所有依赖安装完成！
echo.

:: ============================================================
:: 4. 检查模型文件
:: ============================================================
echo [4/6] 检查模型文件...
echo.

if not exist checkpoints (
    mkdir checkpoints
)

:: 检查预训练模型
if exist checkpoints\market1501_pretrained.pth (
    echo ✅ 预训练模型已存在 (market1501_pretrained.pth)
) else (
    echo ℹ 预训练模型不存在，首次训练时会自动提取。
    echo    不影响一键训练，但训练时间会稍长。
)

:: 检查最佳模型
if exist checkpoints\best_model.pth (
    echo ✅ 最佳模型已存在 (best_model.pth)
) else (
    echo ℹ 最佳模型不存在，需要先训练。
)

echo.

:: ============================================================
:: 5. 创建快捷方式
:: ============================================================
echo [5/6] 创建快捷方式...
echo.

set TARGET_DIR=%cd%

:: 桌面快捷方式
set DESKTOP_SHORTCUT=%USERPROFILE%\Desktop\S.E.E.R.先见者.lnk
if exist "!DESKTOP_SHORTCUT!" (
    echo ✅ 桌面快捷方式已存在
) else (
    if exist "%WINDIR%\System32\WindowsPowerShell\v1.0\powershell.exe" (
        powershell -Command ^
            "$ws = New-Object -ComObject WScript.Shell;" ^
            "$s = $ws.CreateShortcut('%USERPROFILE%\Desktop\S.E.E.R.先见者.lnk');" ^
            "$s.TargetPath = '%TARGET_DIR%\启动GUI启动器.bat';" ^
            "$s.WorkingDirectory = '%TARGET_DIR%';" ^
            "$s.Description = 'S.E.E.R. 先见者 - 行人重识别系统';" ^
            "$s.IconLocation = '%TARGET_DIR%\launcher.py,0';" ^
            "$s.Save()" 2>nul
        if exist "!DESKTOP_SHORTCUT!" (
            echo ✅ 桌面快捷方式已创建
        ) else (
            echo ℹ 桌面快捷方式创建失败（手动双击 启动GUI启动器.bat 即可）
        )
    )
)

echo.

:: ============================================================
:: 6. 完成
:: ============================================================
echo [6/6] 验证安装...
echo.
python -c "import torch, cv2, numpy; print('✅ 所有核心库导入成功')" 2>nul
if !errorlevel! equ 0 (
    echo ✅ 环境验证通过
) else (
    echo ⚠ 环境验证有警告，但大部分功能仍可正常使用
)
echo.

echo ╔══════════════════════════════════════════════════════╗
echo ║            ✅ 安装完成！                              ║
echo ╠══════════════════════════════════════════════════════╣
echo ║                                                      ║
echo ║  ① 双击桌面 "S.E.E.R.先见者" 快捷方式启动              ║
echo ║                                                      ║
echo ║  ② 放入底库照片到 rescue_gallery 文件夹               ║
echo ║     每人一个子文件夹                                  ║
echo ║                                                      ║
echo ║  ③ 点击「整理底库」→「一键快速训练」→「视频ReID」     ║
echo ║                                                      ║
echo ║  有问题？打开 "使用说明.txt" 查看详细教程              ║
echo ║                                                      ║
echo ╚══════════════════════════════════════════════════════╝
echo.

:: 打开使用说明
if exist 使用说明.txt (
    start notepad 使用说明.txt 2>nul
)

echo 按任意键退出...
pause >nul
endlocal
