#!/usr/bin/env python3
"""
检查图像文件是否损坏的脚本
"""
import os
import cv2
from PIL import Image
import json

def check_images(data_root, ann_file):
    """检查图像文件是否可读"""
    print(f"检查数据集: {data_root}")
    print(f"标注文件: {ann_file}")
    
    # 读取标注文件
    with open(os.path.join(data_root, ann_file), 'r') as f:
        data = json.load(f)
    
    images = data['images']
    print(f"总图像数量: {len(images)}")
    
    corrupted_files = []
    valid_files = []
    
    for i, img_info in enumerate(images):
        if i % 100 == 0:
            print(f"已检查: {i}/{len(images)}")
            
        file_name = img_info['file_name']
        img_path = os.path.join(data_root, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(img_path):
            print(f"文件不存在: {file_name}")
            corrupted_files.append(file_name)
            continue
            
        # 尝试用OpenCV读取
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            # 尝试用PIL读取
            try:
                img_pil = Image.open(img_path)
                img_pil.verify()  # 验证图像
                print(f"OpenCV无法读取但PIL可以: {file_name}")
                valid_files.append(file_name)
            except Exception as e:
                print(f"文件损坏: {file_name} - {e}")
                corrupted_files.append(file_name)
        else:
            valid_files.append(file_name)
    
    print(f"\n检查完成!")
    print(f"有效文件: {len(valid_files)}")
    print(f"损坏文件: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\n损坏的文件列表:")
        for f in corrupted_files[:10]:  # 只显示前10个
            print(f"  - {f}")
        if len(corrupted_files) > 10:
            print(f"  ... 还有 {len(corrupted_files) - 10} 个文件")
    
    return valid_files, corrupted_files

if __name__ == "__main__":
    data_root = "data/indoor_dataset_cocostyle"
    train_ann = "annotations/instances_train.json"
    val_ann = "annotations/instances_val.json"
    
    print("=== 检查训练集 ===")
    train_valid, train_corrupted = check_images(data_root, train_ann)
    
    print("\n=== 检查验证集 ===")
    val_valid, val_corrupted = check_images(data_root, val_ann)
    
    print(f"\n总结:")
    print(f"训练集 - 有效: {len(train_valid)}, 损坏: {len(train_corrupted)}")
    print(f"验证集 - 有效: {len(val_valid)}, 损坏: {len(val_corrupted)}")
