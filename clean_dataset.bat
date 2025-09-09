@echo off
echo 清理数据集并重新映射类别
echo ================================

REM 设置环境变量
set PYTHONPATH=%CD%

REM 检查Python环境
python --version
if %errorlevel% neq 0 (
    echo Python未安装或未添加到PATH
    pause
    exit /b 1
)

REM 运行清理脚本
echo 开始清理数据集...
python scripts/clean_and_remap_dataset.py

if %errorlevel% neq 0 (
    echo 清理过程中出现错误
    pause
    exit /b 1
)

echo 数据集清理完成！
echo 清理后的数据集位于: data/cleaned_dataset/
pause
