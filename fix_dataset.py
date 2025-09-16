#!/usr/bin/env python3
"""
修复损坏图像文件的脚本
"""
import os
import cv2
import json
import shutil
from PIL import Image
import numpy as np

def fix_image(img_path, output_path):
    """尝试修复单个图像文件"""
    try:
        # 方法1: 尝试用PIL读取并转换
        with Image.open(img_path) as img:
            # 转换为RGB模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 保存为新的JPEG文件
            img.save(output_path, 'JPEG', quality=95)
            return True
    except Exception as e1:
        try:
            # 方法2: 尝试用OpenCV读取
            img = cv2.imread(img_path)
            if img is not None:
                cv2.imwrite(output_path, img)
                return True
        except Exception as e2:
            print(f"无法修复 {img_path}: PIL错误={e1}, OpenCV错误={e2}")
            return False
    return False

def create_fixed_dataset(data_root, ann_file, output_root, output_ann_file):
    """创建修复后的数据集"""
    print(f"处理数据集: {data_root}")
    
    # 读取原始标注文件
    with open(os.path.join(data_root, ann_file), 'r') as f:
        data = json.load(f)
    
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(output_root, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'annotations'), exist_ok=True)
    
    fixed_images = []
    corrupted_images = []
    
    for i, img_info in enumerate(data['images']):
        if i % 100 == 0:
            print(f"处理进度: {i}/{len(data['images'])}")
            
        file_name = img_info['file_name']
        original_path = os.path.join(data_root, file_name)
        output_path = os.path.join(output_root, file_name)
        
        # 检查原始文件是否存在
        if not os.path.exists(original_path):
            print(f"原始文件不存在: {file_name}")
            corrupted_images.append(img_info)
            continue
        
        # 尝试修复图像
        if fix_image(original_path, output_path):
            fixed_images.append(img_info)
        else:
            print(f"无法修复: {file_name}")
            corrupted_images.append(img_info)
    
    # 更新标注文件，只保留修复成功的图像
    fixed_data = data.copy()
    fixed_data['images'] = fixed_images
    
    # 更新annotations，只保留修复成功图像的标注
    fixed_image_ids = {img['id'] for img in fixed_images}
    fixed_data['annotations'] = [
        ann for ann in data['annotations'] 
        if ann['image_id'] in fixed_image_ids
    ]
    
    # 保存修复后的标注文件
    with open(os.path.join(output_root, output_ann_file), 'w') as f:
        json.dump(fixed_data, f, indent=2)
    
    print(f"\n修复完成!")
    print(f"成功修复: {len(fixed_images)} 个图像")
    print(f"无法修复: {len(corrupted_images)} 个图像")
    print(f"输出目录: {output_root}")
    
    return len(fixed_images), len(corrupted_images)

if __name__ == "__main__":
    # 配置路径
    original_root = "data/indoor_dataset_cocostyle"
    fixed_root = "data/indoor_dataset_cocostyle_fixed"
    
    print("=== 修复训练集 ===")
    train_fixed, train_failed = create_fixed_dataset(
        original_root, 
        "annotations/instances_train.json",
        fixed_root,
        "annotations/instances_train.json"
    )
    
    print("\n=== 修复验证集 ===")
    val_fixed, val_failed = create_fixed_dataset(
        original_root,
        "annotations/instances_val.json", 
        fixed_root,
        "annotations/instances_val.json"
    )
    
    print(f"\n最终结果:")
    print(f"训练集 - 修复成功: {train_fixed}, 修复失败: {train_failed}")
    print(f"验证集 - 修复成功: {val_fixed}, 修复失败: {val_failed}")
    print(f"修复后的数据集保存在: {fixed_root}")
