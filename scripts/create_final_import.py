#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据Label Studio实际路径创建最终的导入文件
"""

import json
import os

def create_final_import():
    """创建最终的导入文件"""
    
    # 检查实际路径文件
    actual_paths_file = "data/labelstudio_import/actual_paths.json"
    if not os.path.exists(actual_paths_file):
        print("❌ 实际路径文件不存在")
        print("请先运行: python scripts/export_labelstudio_tasks.py")
        return
    
    # 读取实际路径
    with open(actual_paths_file, 'r', encoding='utf-8') as f:
        actual_paths = json.load(f)
    
    print(f"📊 找到 {len(actual_paths)} 个实际路径")
    
    # 创建路径映射
    path_mapping = {}
    for path_info in actual_paths:
        task_id = path_info['task_id']
        image_path = path_info['image_path']
        path_mapping[task_id] = image_path
    
    # 读取COCO数据
    coco_file = "data/filtered_coco_dataset/annotations/instances_filtered.json"
    
    with open(coco_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 创建类别映射
    categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    
    # 创建图片映射
    images = {img['id']: img for img in coco_data.get('images', [])}
    
    # 按图片ID分组标注
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # 生成Label Studio格式
    labelstudio_items = []
    matched_count = 0
    
    # 按任务ID顺序处理
    for task_id in sorted(path_mapping.keys(), key=int):
        image_path = path_mapping[task_id]
        
        # 这里需要根据任务ID找到对应的图片信息
        # 由于Label Studio的任务ID可能与COCO的图片ID不同，
        # 我们需要一个映射关系
        
        # 暂时使用前N个图片的标注
        if matched_count < len(images):
            image_info = list(images.values())[matched_count]
            image_id = image_info['id']
            
            if image_id in annotations_by_image:
                # 该图片的所有标注
                annotations = annotations_by_image[image_id]
                results = []
                
                for ann in annotations:
                    category_name = categories.get(ann['category_id'], 'unknown')
                    bbox = ann['bbox']  # [x, y, width, height]
                    
                    # 转换为百分比坐标
                    x_percent = (bbox[0] / image_info['width']) * 100
                    y_percent = (bbox[1] / image_info['height']) * 100
                    width_percent = (bbox[2] / image_info['width']) * 100
                    height_percent = (bbox[3] / image_info['height']) * 100
                    
                    result = {
                        "id": f"bbox_{ann['id']}",
                        "from_name": "objects",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "x": x_percent,
                            "y": y_percent,
                            "width": width_percent,
                            "height": height_percent,
                            "rotation": 0,
                            "rectanglelabels": [category_name]
                        },
                        "original_width": image_info['width'],
                        "original_height": image_info['height'],
                        "image_rotation": 0
                    }
                    results.append(result)
                
                # 创建Label Studio项目项
                item = {
                    "data": {
                        "image": image_path
                    },
                    "annotations": [
                        {
                            "result": results,
                            "ground_truth": False
                        }
                    ]
                }
                labelstudio_items.append(item)
                matched_count += 1
    
    print(f"📊 匹配统计:")
    print(f"   - 成功匹配: {matched_count} 张图片")
    print(f"   - 总标注: {len(labelstudio_items)} 个项目")
    
    # 保存文件
    output_file = "data/labelstudio_import/final_import.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(labelstudio_items, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 创建最终导入文件: {output_file}")
    
    # 创建测试版本
    test_data = labelstudio_items[:3]
    test_file = "data/labelstudio_import/test_final.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 创建测试文件: {test_file}")

if __name__ == "__main__":
    create_final_import()
