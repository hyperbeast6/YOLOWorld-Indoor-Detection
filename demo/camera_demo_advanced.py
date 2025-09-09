# Copyright (c) Tencent Inc. All rights reserved.
import os.path as osp
import sys
import cv2
import torch
import numpy as np
import time
from collections import deque
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg


class FPS:
    """FPS计算类"""
    def __init__(self, avg_frames=30):
        self.fps = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps_times = deque(maxlen=avg_frames)
    
    def update(self):
        """更新FPS"""
        self.fps_counter += 1
        current_time = time.time()
        self.fps_times.append(current_time - self.fps_start_time)
        
        if len(self.fps_times) >= 2:
            self.fps = len(self.fps_times) / (self.fps_times[-1] - self.fps_times[0])
        
        return self.fps


def inference_camera_optimized(model, image, texts, test_pipeline, score_thr=0.3, max_dets=100):
    """优化的实时摄像头推理函数"""
    # 转换BGR到RGB
    image_rgb = image[:, :, [2, 1, 0]]
    data_info = dict(img=image_rgb, img_id=0, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])
    
    with torch.no_grad():
        output = model.test_step(data_batch)[0]
    
    pred_instances = output.pred_instances
    # score thresholding
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    # max detections
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    boxes = pred_instances['bboxes']
    labels = pred_instances['labels']
    scores = pred_instances['scores']
    label_texts = [texts[x][0] for x in labels]
    return boxes, labels, label_texts, scores


def draw_detections_enhanced(image, boxes, labels, label_texts, scores, fps=0):
    """增强的检测结果绘制函数"""
    # 绘制FPS
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # 绘制检测数量
    cv2.putText(image, f"Detections: {len(boxes)}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # 绘制检测结果
    for i, (box, label, label_text, score) in enumerate(zip(boxes, labels, label_texts, scores)):
        x1, y1, x2, y2 = map(int, box)
        
        # 根据置信度选择颜色
        if score > 0.8:
            color = (0, 255, 0)  # 绿色 - 高置信度
        elif score > 0.5:
            color = (0, 255, 255)  # 黄色 - 中等置信度
        else:
            color = (0, 165, 255)  # 橙色 - 低置信度
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签文本
        label_text = f"{label_text}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # 绘制文本背景
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # 绘制文本
        cv2.putText(image, label_text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
    
    # 绘制操作提示
    cv2.putText(image, "Press 'q' to quit, 's' to save, 'r' to reset FPS", 
                (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image


def camera_detection_advanced():
    """高级摄像头检测函数"""
    # 使用不需要CLIP模型的配置文件
    config_file = "E:\code\YOLO-World\configs\pretrain\yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
    checkpoint = "E:\code\YOLO-World\weights\yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth"

    # 检查文件是否存在
    if not osp.exists(config_file):
        print(f"配置文件不存在: {config_file}")
        return
    
    if not osp.exists(checkpoint):
        print(f"权重文件不存在: {checkpoint}")
        return

    print("正在初始化模型...")
    cfg = Config.fromfile(config_file)
    cfg.work_dir = osp.join('./work_dirs')
    
    # 初始化模型
    cfg.load_from = checkpoint
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    model = init_detector(cfg, checkpoint=checkpoint, device=device)
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)

    # 定义检测类别 - 可以根据需要调整
    texts = [['person'], ['car'], ['bus'], ['truck'], ['bicycle'], ['motorcycle'], 
             ['dog'], ['cat'], ['chair'], ['table'], ['cup'], ['bottle'], ['book'],
             ['phone'], ['laptop'], ['keyboard'], ['mouse'], ['tv'], ['remote']]

    print("正在打开摄像头...")
    # 打开摄像头 (0表示默认摄像头)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 设置摄像头分辨率以提高性能
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("摄像头已打开")
    print("检测类别:", [text[0] for text in texts])
    print("操作说明:")
    print("  - 按 'q' 键退出")
    print("  - 按 's' 键保存当前帧")
    print("  - 按 'r' 键重置FPS计数")
    print("  - 按 '1-9' 键调整置信度阈值")

    # 初始化FPS计算器
    fps_calculator = FPS()
    
    # 可调整的参数
    score_threshold = 0.3
    max_detections = 50
    skip_frames = 0  # 跳帧数，用于提高性能
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break

            frame_count += 1
            
            # 跳帧处理以提高性能
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                cv2.imshow('YOLO-World Real-time Detection', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            # 进行推理
            try:
                start_time = time.time()
                results = inference_camera_optimized(model, frame, texts, test_pipeline, 
                                                  score_thr=score_threshold, max_dets=max_detections)
                inference_time = time.time() - start_time
                
                boxes, labels, label_texts, scores = results
                
                # 更新FPS
                fps = fps_calculator.update()
                
                # 在图像上绘制检测结果
                frame_with_detections = draw_detections_enhanced(
                    frame.copy(), boxes, labels, label_texts, scores, fps)
                
                # 显示推理时间
                cv2.putText(frame_with_detections, f"Inference: {inference_time*1000:.1f}ms", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow('YOLO-World Real-time Detection', frame_with_detections)
                
            except Exception as e:
                print(f"推理过程中出现错误: {e}")
                cv2.imshow('YOLO-World Real-time Detection', frame)

            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户退出")
                break
            elif key == ord('s'):
                # 保存当前帧
                try:
                    filename = f"camera_capture_{int(time.time())}.jpg"
                    filepath = osp.join("demo/sample_images", filename)
                    cv2.imwrite(filepath, frame_with_detections)
                    print(f"已保存图像到: {filepath}")
                except:
                    print("保存图像失败")
            elif key == ord('r'):
                # 重置FPS计数
                fps_calculator = FPS()
                print("FPS计数已重置")
            elif key in [ord(str(i)) for i in range(1, 10)]:
                # 调整置信度阈值 (1-9对应0.1-0.9)
                score_threshold = key - ord('0') * 0.1
                print(f"置信度阈值调整为: {score_threshold:.1f}")
            elif key == ord('+') or key == ord('='):
                # 增加跳帧数
                skip_frames = min(skip_frames + 1, 5)
                print(f"跳帧数调整为: {skip_frames}")
            elif key == ord('-'):
                # 减少跳帧数
                skip_frames = max(skip_frames - 1, 0)
                print(f"跳帧数调整为: {skip_frames}")

    except KeyboardInterrupt:
        print("程序被中断")
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头已关闭")
        print(f"最终FPS: {fps_calculator.fps:.1f}")


if __name__ == "__main__":
    camera_detection_advanced()
