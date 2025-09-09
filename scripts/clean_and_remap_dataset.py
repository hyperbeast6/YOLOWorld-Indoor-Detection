#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理数据集并重新映射类别的脚本
将78个类别缩减为35个核心室内物体类别
"""

import os
import shutil
import json
from pathlib import Path
import glob

def create_class_mapping():
    """创建类别映射关系"""
    
    # 你需要的35个核心类别
    target_classes = [
        'person', 'chair', 'couch', 'bed', 'dining table', 'tv', 'refrigerator',
        'cell phone', 'cup', 'book', 'bottle', 'bowl', 'fork', 'knife', 'spoon',
        'remote', 'mouse', 'keyboard', 'laptop', 'handbag', 'backpack',
        'clock', 'vase', 'plant', 'mirror', 'lamp', 'curtain', 'window',
        'door', 'wall', 'floor', 'ceiling', 'carpet', 'rug', 'other'
    ]
    
    # 原始数据集中的类别
    with open('data/mydataset/classes.txt', 'r') as f:
        original_classes = [line.strip() for line in f.readlines() if line.strip()]
    
    # 创建映射关系：目标类别 -> 原始类别ID
    class_mapping = {}
    reverse_mapping = {}  # 原始ID -> 新ID
    
    print("创建类别映射关系...")
    print("=" * 60)
    
    for new_id, target_class in enumerate(target_classes):
        found = False
        for old_id, original_class in enumerate(original_classes):
            # 精确匹配
            if target_class.lower() == original_class.lower():
                class_mapping[target_class] = old_id
                reverse_mapping[old_id] = new_id
                print(f"✓ {target_class:15} -> 原始ID {old_id:2d} -> 新ID {new_id:2d}")
                found = True
                break
            # 模糊匹配
            elif target_class in original_class.lower() or original_class.lower() in target_class:
                class_mapping[target_class] = old_id
                reverse_mapping[old_id] = new_id
                print(f"≈ {target_class:15} -> 原始ID {old_id:2d} -> 新ID {new_id:2d} (模糊匹配: {original_class})")
                found = True
                break
        
        if not found:
            print(f"✗ {target_class:15} -> 未找到对应类别")
    
    print("=" * 60)
    print(f"成功映射 {len(class_mapping)} 个类别")
    print(f"未映射 {len(target_classes) - len(class_mapping)} 个类别")
    
    return class_mapping, reverse_mapping, target_classes

def clean_annotations(reverse_mapping, output_dir):
    """清理标注文件，只保留映射的类别"""
    
    print("\n清理标注文件...")
    
    # 创建输出目录
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    total_annotations = 0
    kept_annotations = 0
    removed_annotations = 0
    class_stats = {}
    
    # 处理所有标注文件
    label_files = glob.glob('data/mydataset/labels/*.txt')
    
    for label_file in label_files:
        filename = os.path.basename(label_file)
        output_file = output_dir / "labels" / filename
        
        kept_lines = []
        
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        old_class_id = int(parts[0])
                        total_annotations += 1
                        
                        # 检查是否在映射中
                        if old_class_id in reverse_mapping:
                            new_class_id = reverse_mapping[old_class_id]
                            # 更新类别ID
                            parts[0] = str(new_class_id)
                            kept_lines.append(' '.join(parts))
                            kept_annotations += 1
                            
                            # 统计类别使用情况
                            class_stats[new_class_id] = class_stats.get(new_class_id, 0) + 1
                        else:
                            removed_annotations += 1
                            print(f"移除类别ID {old_class_id} 的标注")
            
            # 保存清理后的标注文件
            with open(output_file, 'w') as f:
                for line in kept_lines:
                    f.write(line + '\n')
                    
        except Exception as e:
            print(f"处理文件 {label_file} 时出错: {e}")
    
    print(f"\n标注清理完成:")
    print(f"  总标注数: {total_annotations}")
    print(f"  保留标注数: {kept_annotations}")
    print(f"  移除标注数: {removed_annotations}")
    print(f"  保留率: {kept_annotations/total_annotations*100:.1f}%")
    
    return class_stats

def copy_images(output_dir):
    """复制图片文件"""
    
    print("\n复制图片文件...")
    
    # 创建输出目录
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    
    # 复制所有图片
    image_files = glob.glob('data/mydataset/images/*.jpg')
    
    for image_file in image_files:
        filename = os.path.basename(image_file)
        output_file = output_dir / "images" / filename
        shutil.copy2(image_file, output_file)
    
    print(f"复制了 {len(image_files)} 张图片")

def create_new_classes_file(target_classes, output_dir):
    """创建新的类别文件"""
    
    print("\n创建新的类别文件...")
    
    classes_file = output_dir / "classes.txt"
    
    with open(classes_file, 'w') as f:
        for class_name in target_classes:
            f.write(class_name + '\n')
    
    print(f"创建了包含 {len(target_classes)} 个类别的 classes.txt")

def create_metadata(target_classes, class_stats, output_dir):
    """创建元数据文件"""
    
    print("\n创建元数据文件...")
    
    # 创建类别信息
    categories = []
    for i, class_name in enumerate(target_classes):
        categories.append({
            "id": i,
            "name": class_name,
            "count": class_stats.get(i, 0)
        })
    
    metadata = {
        "info": {
            "description": "Cleaned Indoor Object Detection Dataset",
            "version": "2.0",
            "year": 2024,
            "contributor": "Custom Dataset - Cleaned",
            "total_classes": len(target_classes),
            "total_images": len(glob.glob(str(output_dir / "images/*.jpg"))),
            "total_annotations": sum(class_stats.values())
        },
        "categories": categories,
        "class_statistics": class_stats
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"创建了元数据文件: {metadata_file}")

def create_yolo_world_texts(target_classes, output_dir):
    """创建YOLO-World需要的类别文本文件"""
    
    print("\n创建YOLO-World类别文本文件...")
    
    # 为每个类别创建多个描述
    class_texts = []
    for class_name in target_classes:
        # 为每个类别创建多个描述变体
        descriptions = [class_name]
        
        # 添加一些常见的同义词
        synonyms = {
            'person': ['human', 'people', 'individual'],
            'chair': ['seat', 'stool'],
            'couch': ['sofa', 'settee'],
            'bed': ['mattress', 'sleeping area'],
            'dining table': ['table', 'dinner table'],
            'tv': ['television', 'monitor', 'screen'],
            'refrigerator': ['fridge', 'cooler'],
            'cell phone': ['mobile phone', 'smartphone', 'phone'],
            'cup': ['mug', 'glass'],
            'book': ['books', 'novel', 'magazine'],
            'bottle': ['container', 'vessel'],
            'bowl': ['dish', 'plate'],
            'fork': ['utensil'],
            'knife': ['blade', 'cutting tool'],
            'spoon': ['utensil', 'ladle'],
            'remote': ['remote control', 'controller'],
            'mouse': ['computer mouse', 'pointing device'],
            'keyboard': ['keypad', 'input device'],
            'laptop': ['computer', 'notebook', 'pc'],
            'handbag': ['purse', 'bag'],
            'backpack': ['rucksack', 'bag'],
            'clock': ['timepiece', 'watch'],
            'vase': ['container', 'pot'],
            'plant': ['vegetation', 'flower', 'tree'],
            'mirror': ['looking glass', 'reflection'],
            'lamp': ['light', 'lighting'],
            'curtain': ['drape', 'blind'],
            'window': ['opening', 'glass'],
            'door': ['entrance', 'exit'],
            'wall': ['surface', 'barrier'],
            'floor': ['ground', 'surface'],
            'ceiling': ['roof', 'top'],
            'carpet': ['rug', 'mat'],
            'rug': ['carpet', 'mat'],
            'other': ['miscellaneous', 'unknown']
        }
        
        if class_name in synonyms:
            descriptions.extend(synonyms[class_name])
        
        class_texts.append(descriptions)
    
    # 保存为YOLO-World格式
    texts_file = output_dir / "yolo_world_texts.json"
    with open(texts_file, 'w', encoding='utf-8') as f:
        json.dump(class_texts, f, ensure_ascii=False, indent=2)
    
    print(f"创建了YOLO-World文本文件: {texts_file}")

def main():
    """主函数"""
    
    print("=" * 60)
    print("数据集清理和类别重新映射工具")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = Path("data/indoor_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 创建类别映射
    class_mapping, reverse_mapping, target_classes = create_class_mapping()
    
    if len(class_mapping) == 0:
        print("错误: 没有找到任何可映射的类别，请检查类别名称")
        return
    
    # 2. 清理标注文件
    class_stats = clean_annotations(reverse_mapping, output_dir)
    
    # 3. 复制图片
    copy_images(output_dir)
    
    # 4. 创建新的类别文件
    create_new_classes_file(target_classes, output_dir)
    
    # 5. 创建元数据
    create_metadata(target_classes, class_stats, output_dir)
    
    # 6. 创建YOLO-World文本文件
    create_yolo_world_texts(target_classes, output_dir)
    
    print("\n" + "=" * 60)
    print("数据集清理完成！")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"类别数量: {len(target_classes)}")
    print(f"图片数量: {len(glob.glob(str(output_dir / 'images/*.jpg')))}")
    print(f"标注数量: {sum(class_stats.values())}")
    print("\n生成的文件:")
    print(f"  - classes.txt (类别定义)")
    print(f"  - images/ (图片文件)")
    print(f"  - labels/ (标注文件)")
    print(f"  - metadata.json (元数据)")
    print(f"  - yolo_world_texts.json (YOLO-World文本)")
    print("\n现在可以使用清理后的数据集进行训练了！")

if __name__ == "__main__":
    main()
