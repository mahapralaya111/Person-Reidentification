@echo off
chcp 65001 >nul
title 行人重识别系统 - Web界面/云服务器

cd /d "%~dp0"

:menu
cls
echo ============================================
echo   行人重识别系统 - 启动菜单
echo ============================================
echo.
echo  1. Streamlit Web界面（交互式，默认）
echo     访问: http://localhost:8501
echo.
echo  2. Flask API 云服务器（REST API）
echo     访问: http://localhost:5000
echo     适合远程调用、部署到云服务器
echo.
echo  3. 云服务器 - Gunicorn（多进程）
echo     需要安装 gunicorn 和 flask-cors
echo.
echo  0. 退出
echo.
echo ============================================
echo.

set /p choice="请选择 (0-3): "

REM 检查虚拟环境
if exist "gpu_env\Scripts\python.exe" (
    set PYTHON=gpu_env\Scripts\python.exe
) else (
    set PYTHON=python
)

if "%choice%"=="1" goto streamlit
if "%choice%"=="2" goto flask_api
if "%choice%"=="3" goto gunicorn
if "%choice%"=="0" goto end

echo.
echo [错误] 无效选项！
timeout /t 2 >nul
goto menu

:streamlit
cls
echo ============================================
echo   启动 Streamlit Web界面
echo ============================================
echo.
echo 访问: http://localhost:8501
echo 按 Ctrl+C 停止服务
echo.
%PYTHON% -m streamlit run web/app.py --server.port 8501 --server.headless true --browser.gatherUsageStats=false
goto after_run

:flask_api
cls
echo ============================================
echo   启动 Flask API 云服务器
echo ============================================
echo.
echo  API 文档: http://localhost:5000/
echo  健康检查: http://localhost:5000/health
echo.
echo  从其他设备访问时，请使用本机的实际 IP 地址
echo  需要在防火墙中放行 5000 端口
echo.
echo  按 Ctrl+C 停止服务
echo.
%PYTHON% -m web.api --host 0.0.0.0 --port 5000 --gallery rescue_gallery_augmented
goto after_run

:gunicorn
cls
echo ============================================
echo   启动 Gunicorn 生产服务器
echo ============================================
echo.
echo  API 文档: http://localhost:5000/
echo  健康检查: http://localhost:5000/health
echo.
echo  检查 gunicorn 是否安装...
%PYTHON% -c "import gunicorn" 2>nul
if %errorlevel% neq 0 (
    echo [错误] gunicorn 未安装，请执行:
    echo   pip install gunicorn flask-cors
    pause
    goto menu
)
echo.
echo  按 Ctrl+C 停止服务
echo.
%PYTHON% -m gunicorn -c deploy/gunicorn.conf.py web.api:app
goto after_run

:after_run
echo.
echo 服务已停止。
echo.
pause
goto menu

:end
exit /b 0
