@echo off
chcp 65001 >nul 2>&1
title S.E.E.R. 先见者 - 启动器

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║        S.E.E.R. · 先见者  - 图形启动器              ║
echo ╚══════════════════════════════════════════════════════╝
echo.

REM 检查虚拟环境
set PYTHON=python
if exist "%~dp0gpu_env\Scripts\python.exe" (
    set PYTHON=%~dp0gpu_env\Scripts\python.exe
    echo ✅ 使用虚拟环境: gpu_env
) else (
    echo ℹ 使用系统 Python（未发现虚拟环境）
)
echo.

REM 检查 Python 是否可用
%PYTHON% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未检测到 Python 环境！
    echo.
    echo   请先运行 "安装即用.bat" 完成环境配置。
    echo.
    pause
    exit /b 1
)

%PYTHON% --version
echo.

REM 检查 tkinter
%PYTHON% -c "import tkinter" 2>nul
if %errorlevel% neq 0 (
    echo ⚠ 未检测到 tkinter 模块，尝试安装...
    %PYTHON% -m pip install tk -q 2>nul
    %PYTHON% -c "import tkinter" 2>nul
    if !errorlevel! neq 0 (
        echo ❌ tkinter 安装失败！
        echo   方案1: 重新安装 Python，勾选 "tcl/tk and IDLE"
        echo   方案2: 使用 Web 界面代替（双击 启动Web界面.bat）
        pause
        exit /b 1
    )
)

echo  使用 Python: %PYTHON%
echo  启动 launcher.py...
echo.

REM 启动图形界面
start "" "%PYTHON%" "%~dp0launcher.py"

if %errorlevel% equ 0 (
    echo ✅ 启动器已启动，请在窗口中操作
) else (
    echo ❌ 启动失败
    pause
)
