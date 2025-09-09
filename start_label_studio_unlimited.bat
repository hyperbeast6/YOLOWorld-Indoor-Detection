@echo off
echo === 启动Label Studio (无限制版本) ===
echo.

echo 设置上传限制...
set LABEL_STUDIO_DATA_UPLOAD_MAX_NUMBER_FILES=5000
set LABEL_STUDIO_DATA_UPLOAD_MAX_MEMORY_SIZE=1073741824
set LABEL_STUDIO_DATA_UPLOAD_MAX_FILE_SIZE=104857600
set LABEL_STUDIO_DATA_UPLOAD_MAX_TOTAL_SIZE=2147483648

echo 设置数据目录...
set LABEL_STUDIO_DATA_DIR=E:\code\YOLO-World\label_studio_data

echo 激活conda环境...
call conda activate yolo_world

echo.
echo 当前设置:
echo - 最大文件数量: %LABEL_STUDIO_DATA_UPLOAD_MAX_NUMBER_FILES%
echo - 最大内存使用: %LABEL_STUDIO_DATA_UPLOAD_MAX_MEMORY_SIZE% bytes
echo - 单文件最大: %LABEL_STUDIO_DATA_UPLOAD_MAX_FILE_SIZE% bytes
echo - 数据目录: %LABEL_STUDIO_DATA_DIR%
echo.

echo 启动Label Studio...
echo 启动后请访问: http://localhost:8080
echo 按 Ctrl+C 停止服务
echo.

label-studio start --data-dir "%LABEL_STUDIO_DATA_DIR%"

pause
