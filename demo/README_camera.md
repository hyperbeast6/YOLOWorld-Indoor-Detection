# YOLO-World 实时摄像头检测

本目录包含了用于实时摄像头检测的代码文件，基于原有的 `simple_demo.py` 开发，无需修改原有代码。

## 文件说明

### 1. `camera_demo.py` - 基础版本
- 基本的实时摄像头检测功能
- 支持实时显示检测结果
- 按键控制：'q' 退出，'s' 保存图像

### 2. `camera_demo_advanced.py` - 高级版本
- 包含FPS计算和性能优化
- 支持跳帧处理以提高性能
- 更多交互功能：
  - 'q' - 退出
  - 's' - 保存当前帧
  - 'r' - 重置FPS计数
  - '1-9' - 调整置信度阈值 (0.1-0.9)
  - '+'/'-' - 调整跳帧数

## 使用方法

### 环境要求
确保已安装以下依赖：
```bash
pip install opencv-python torch mmcv mmdet mmengine
```

### 运行基础版本
```bash
python demo/camera_demo.py
```

### 运行高级版本
```bash
python demo/camera_demo_advanced.py
```

## 功能特性

### 实时检测
- 调用摄像头进行实时目标检测
- 支持多种常见物体类别
- 实时显示检测结果和置信度

### 性能优化
- 自动设备检测 (GPU/CPU)
- 可调节的置信度阈值
- 跳帧处理以提高FPS
- 摄像头分辨率优化

### 交互控制
- 实时FPS显示
- 检测结果数量统计
- 推理时间显示
- 多种按键控制选项

## 检测类别

默认支持的检测类别包括：
- 人物：person
- 交通工具：car, bus, truck, bicycle, motorcycle
- 动物：dog, cat
- 家具：chair, table
- 物品：cup, bottle, book, phone, laptop, keyboard, mouse, tv, remote

## 注意事项

1. **配置文件路径**：确保配置文件路径正确
2. **权重文件**：确保权重文件存在且可访问
3. **摄像头权限**：确保系统允许访问摄像头
4. **性能考虑**：在CPU环境下，建议使用跳帧功能提高性能

## 故障排除

### 常见问题

1. **摄像头无法打开**
   - 检查摄像头是否被其他程序占用
   - 确认摄像头驱动正常

2. **检测速度慢**
   - 降低摄像头分辨率
   - 增加跳帧数
   - 使用GPU加速（如果可用）

3. **检测结果不准确**
   - 调整置信度阈值
   - 检查光照条件
   - 确认目标物体在检测类别中

## 自定义配置

### 修改检测类别
在代码中修改 `texts` 变量：
```python
texts = [['your_class1'], ['your_class2'], ...]
```

### 调整性能参数
- `score_threshold`：置信度阈值
- `max_detections`：最大检测数量
- `skip_frames`：跳帧数量

### 修改摄像头设置
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # 设置高度
cap.set(cv2.CAP_PROP_FPS, 60)             # 设置帧率
```

## 扩展功能

可以基于现有代码添加更多功能：
- 录制视频
- 多摄像头支持
- 检测结果保存
- 网络流传输
- 移动端部署

## 技术支持

如遇到问题，请检查：
1. 依赖包版本兼容性
2. 配置文件路径正确性
3. 权重文件完整性
4. 系统环境配置
