@echo off
chcp 65001 >nul
title YOLO-World S模型缺失文件下载脚本

echo.
echo 🚀 开始下载YOLO-World S模型所需的缺失文件...
echo.

REM 检查是否安装了huggingface-cli
where huggingface-cli >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ 未检测到huggingface-cli，请先安装：
    echo    pip install huggingface-hub
    echo    或者：conda install -c conda-forge huggingface-hub
    echo.
    pause
    exit /b 1
)
echo ✅ 检测到huggingface-cli已安装
echo.

REM 设置镜像
set HF_ENDPOINT=https://hf-mirror.com
echo ✅ 已设置Hugging Face镜像: %HF_ENDPOINT%
echo.

REM 下载CLIP模型
echo 📥 正在下载CLIP文本编码器模型...
echo    模型: openai/clip-vit-base-patch32
echo    大小: 约150-200MB
echo    存储位置: ./weights/clip-vit-base-patch32/
echo.

huggingface-cli download --resume-download openai/clip-vit-base-patch32 --local-dir ./weights/clip-vit-base-patch32 --local-dir-use-symlinks False
if exist "./weights/clip-vit-base-patch32" (
    echo ✅ CLIP模型下载完成
) else (
    echo ❌ CLIP模型下载失败
    pause
    exit /b 1
)

echo.

REM 创建必要目录
echo 📁 正在创建必要目录...
if not exist "./data/coco/lvis" mkdir "./data/coco/lvis"
if not exist "./data/texts" mkdir "./data/texts"
echo ✅ 目录创建完成
echo.

REM 下载LVIS验证集标注文件
echo 📥 正在下载LVIS验证集标注文件...
echo    文件: lvis_v1_minival_inserted_image_name.json
echo    大小: 几MB
echo    存储位置: ./data/coco/lvis/
echo.

huggingface-cli download GLIPModel/GLIP lvis_v1_minival_inserted_image_name.json --local-dir ./data/coco/lvis --local-dir-use-symlinks False --resume-download
if exist "./data/coco/lvis/lvis_v1_minival_inserted_image_name.json" (
    echo ✅ LVIS标注文件下载完成
) else (
    echo ❌ LVIS标注文件下载失败
    pause
    exit /b 1
)

echo.

REM 下载Objects365类别文本文件
echo 📥 正在下载Objects365类别文本文件...
echo    文件: obj365v1_class_texts.json
echo    大小: 几MB
echo    存储位置: ./data/texts/
echo.

huggingface-cli download wondervictor/YOLO-World data/texts/obj365v1_class_texts.json --local-dir ./data/texts --local-dir-use-symlinks False --resume-download
if exist "./data/texts/obj365v1_class_texts.json" (
    echo ✅ Objects365类别文本下载完成
) else (
    echo ❌ Objects365类别文本下载失败
    pause
    exit /b 1
)

echo.

REM 下载LVIS类别文本文件
echo 📥 正在下载LVIS类别文本文件...
echo    文件: lvis_v1_class_texts.json
echo    大小: 几MB
echo    存储位置: ./data/texts/
echo.

huggingface-cli download wondervictor/YOLO-World data/texts/lvis_v1_class_texts.json --local-dir ./data/texts --local-dir-use-symlinks False --resume-download
if exist "./data/texts/lvis_v1_class_texts.json" (
    echo ✅ LVIS类别文本下载完成
) else (
    echo ❌ LVIS类别文本下载失败
    pause
    exit /b 1
)

echo.
echo 🎉 所有文件下载完成！
echo.

REM 显示下载的文件列表
echo 📋 下载的文件列表：
echo.

if exist "./weights/clip-vit-base-patch32" (
    echo   ✅ CLIP文本编码器模型
    echo      路径: ./weights/clip-vit-base-patch32/
) else (
    echo   ❌ CLIP文本编码器模型
    echo      路径: ./weights/clip-vit-base-patch32/
)

if exist "./data/coco/lvis/lvis_v1_minival_inserted_image_name.json" (
    echo   ✅ LVIS验证集标注文件
    echo      路径: ./data/coco/lvis/lvis_v1_minival_inserted_image_name.json
) else (
    echo   ❌ LVIS验证集标注文件
    echo      路径: ./data/coco/lvis/lvis_v1_minival_inserted_image_name.json
)

if exist "./data/texts/obj365v1_class_texts.json" (
    echo   ✅ Objects365类别文本文件
    echo      路径: ./data/texts/obj365v1_class_texts.json
) else (
    echo   ❌ Objects365类别文本文件
    echo      路径: ./data/texts/obj365v1_class_texts.json
)

if exist "./data/texts/lvis_v1_class_texts.json" (
    echo   ✅ LVIS类别文本文件
    echo      路径: ./data/texts/lvis_v1_class_texts.json
) else (
    echo   ❌ LVIS类别文本文件
    echo      路径: ./data/texts/lvis_v1_class_texts.json
)

echo.
echo 🔧 下一步操作：
echo    1. 确保你的YOLO-World S模型权重文件在 weights/ 目录中
echo    2. 运行: python demo/simple_demo.py
echo.

echo 💡 提示：如果下载过程中断，可以重新运行此脚本，它会自动续传
echo.

pause
