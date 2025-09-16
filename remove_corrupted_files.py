#!/usr/bin/env python3
"""
删除损坏的图像文件和对应的标注数据
"""
import os
import json
import shutil

def remove_corrupted_files(data_root, corrupted_files):
    """删除损坏的图像文件和对应的标注数据"""
    
    # 损坏的文件列表
    corrupted_file_names = [
        "images/3698e135-000000474004.jpg",
        "images/247e0a7d-000000460722.jpg", 
        "images/54da8fdc-000000491408.jpg",
        "images/efb32bcf-000000193193.jpg"
    ]
    
    print(f"开始删除损坏的文件...")
    print(f"数据根目录: {data_root}")
    
    # 1. 删除图像文件
    deleted_images = []
    for file_name in corrupted_file_names:
        file_path = os.path.join(data_root, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            deleted_images.append(file_name)
            print(f"已删除图像文件: {file_name}")
        else:
            print(f"图像文件不存在: {file_name}")
    
    # 2. 处理训练集标注文件
    train_ann_file = os.path.join(data_root, "annotations", "instances_train.json")
    if os.path.exists(train_ann_file):
        print(f"\n处理训练集标注文件: {train_ann_file}")
        
        # 备份原文件
        backup_file = train_ann_file + ".backup"
        shutil.copy2(train_ann_file, backup_file)
        print(f"已备份原文件到: {backup_file}")
        
        # 读取标注文件
        with open(train_ann_file, 'r') as f:
            train_data = json.load(f)
        
        # 找到要删除的图像ID
        corrupted_image_ids = []
        for img_info in train_data['images']:
            if img_info['file_name'] in corrupted_file_names:
                corrupted_image_ids.append(img_info['id'])
                print(f"找到损坏图像ID: {img_info['id']} - {img_info['file_name']}")
        
        # 删除损坏的图像信息
        train_data['images'] = [
            img for img in train_data['images'] 
            if img['file_name'] not in corrupted_file_names
        ]
        
        # 删除对应的标注信息
        original_ann_count = len(train_data['annotations'])
        train_data['annotations'] = [
            ann for ann in train_data['annotations'] 
            if ann['image_id'] not in corrupted_image_ids
        ]
        removed_ann_count = original_ann_count - len(train_data['annotations'])
        
        # 保存更新后的标注文件
        with open(train_ann_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        print(f"训练集 - 删除图像: {len(corrupted_image_ids)} 个")
        print(f"训练集 - 删除标注: {removed_ann_count} 个")
        print(f"训练集 - 剩余图像: {len(train_data['images'])} 个")
        print(f"训练集 - 剩余标注: {len(train_data['annotations'])} 个")
    
    # 3. 处理验证集标注文件
    val_ann_file = os.path.join(data_root, "annotations", "instances_val.json")
    if os.path.exists(val_ann_file):
        print(f"\n处理验证集标注文件: {val_ann_file}")
        
        # 备份原文件
        backup_file = val_ann_file + ".backup"
        shutil.copy2(val_ann_file, backup_file)
        print(f"已备份原文件到: {backup_file}")
        
        # 读取标注文件
        with open(val_ann_file, 'r') as f:
            val_data = json.load(f)
        
        # 找到要删除的图像ID
        corrupted_image_ids = []
        for img_info in val_data['images']:
            if img_info['file_name'] in corrupted_file_names:
                corrupted_image_ids.append(img_info['id'])
                print(f"找到损坏图像ID: {img_info['id']} - {img_info['file_name']}")
        
        # 删除损坏的图像信息
        val_data['images'] = [
            img for img in val_data['images'] 
            if img['file_name'] not in corrupted_file_names
        ]
        
        # 删除对应的标注信息
        original_ann_count = len(val_data['annotations'])
        val_data['annotations'] = [
            ann for ann in val_data['annotations'] 
            if ann['image_id'] not in corrupted_image_ids
        ]
        removed_ann_count = original_ann_count - len(val_data['annotations'])
        
        # 保存更新后的标注文件
        with open(val_ann_file, 'w') as f:
            json.dump(val_data, f, indent=2)
        
        print(f"验证集 - 删除图像: {len(corrupted_image_ids)} 个")
        print(f"验证集 - 删除标注: {removed_ann_count} 个")
        print(f"验证集 - 剩余图像: {len(val_data['images'])} 个")
        print(f"验证集 - 剩余标注: {len(val_data['annotations'])} 个")
    
    print(f"\n清理完成!")
    print(f"已删除的图像文件: {len(deleted_images)} 个")
    for img in deleted_images:
        print(f"  - {img}")

if __name__ == "__main__":
    data_root = "data/indoor_dataset_cocostyle"
    
    # 检查数据目录是否存在
    if not os.path.exists(data_root):
        print(f"错误: 数据目录不存在: {data_root}")
        exit(1)
    
    # 执行删除操作
    remove_corrupted_files(data_root, [])
