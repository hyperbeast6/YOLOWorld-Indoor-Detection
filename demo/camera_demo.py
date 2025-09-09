# Copyright (c) Tencent Inc. All rights reserved.
import os.path as osp
import sys
import cv2
import torch
import numpy as np
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg


def inference_camera(model, image, texts, test_pipeline, score_thr=0.3, max_dets=100):
    """实时摄像头推理函数"""
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


def draw_detections(image, boxes, labels, label_texts, scores):
    """在图像上绘制检测结果"""
    for i, (box, label, label_text, score) in enumerate(zip(boxes, labels, label_texts, scores)):
        x1, y1, x2, y2 = map(int, box)
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签文本
        label_text = f"{label_text}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # 绘制文本背景
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
        
        # 绘制文本
        cv2.putText(image, label_text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
    
    return image


def camera_detection():
    """主摄像头检测函数"""
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

    # 定义检测类别
    texts = [['person'], ['car'], ['bus'], ['truck'], ['bicycle'], ['motorcycle'], 
             ['dog'], ['cat'], ['chair'], ['table'], ['cup'], ['bottle']]

    print("正在打开摄像头...")
    # 打开摄像头 (0表示默认摄像头)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("摄像头已打开，按 'q' 键退出，按 's' 键保存当前帧")
    print("检测类别:", [text[0] for text in texts])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break

            # 进行推理
            try:
                results = inference_camera(model, frame, texts, test_pipeline, score_thr=0.3, max_dets=50)
                boxes, labels, label_texts, scores = results
                
                # 在图像上绘制检测结果
                frame_with_detections = draw_detections(frame.copy(), boxes, labels, label_texts, scores)
                
                # 显示检测结果数量
                cv2.putText(frame_with_detections, f"Detections: {len(boxes)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # 显示FPS信息
                cv2.putText(frame_with_detections, "Press 'q' to quit, 's' to save", 
                           (10, frame_with_detections.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
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
                filename = f"camera_capture_{len(os.listdir('demo/sample_images')) + 1}.jpg"
                filepath = osp.join("demo/sample_images", filename)
                cv2.imwrite(filepath, frame_with_detections)
                print(f"已保存图像到: {filepath}")

    except KeyboardInterrupt:
        print("程序被中断")
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头已关闭")


if __name__ == "__main__":
    camera_detection()
