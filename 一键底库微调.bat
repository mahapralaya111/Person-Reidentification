@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo ================================================================
echo   一键底库微调 (rescue_gallery -> data/rescue_train)
echo   参数: nnlrnet, 30 epochs, batch 16, lr 0.0001, dropout 0.3
echo ================================================================
echo.

if not exist "rescue_gallery" (
    echo [错误] 底库目录 rescue_gallery 不存在！
    pause
    exit /b 1
)

echo [1/3] 整理底库数据到 data/rescue_train ...
python tools/prepare_train_data.py --source rescue_gallery
if errorlevel 1 (
    echo [错误] 数据整理失败
    pause
    exit /b 1
)

echo.
echo [2/3] 开始训练 (进度会实时显示) ...
python main.py train --data_dir ./data/rescue_train --backbone nnlrnet --epochs 30 --batch_size 16 --lr 0.0001 --use_circle_loss --use_cosine_annealing --use_random_erasing --finetune checkpoints/market1501_pretrained.pth
if errorlevel 1 (
    echo [错误] 训练失败
    pause
    exit /b 1
)

echo.
echo [3/3] 训练完成！模型已保存到 checkpoints/best_model.pth
echo 建议下一步：运行启动GUI启动器.bat 或 启动Web界面.bat 进行视频识别测试。
pause
