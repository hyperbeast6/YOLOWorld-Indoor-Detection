#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细检查数据集中类别使用情况的脚本
"""

import json
import os
from collections import defaultdict

def analyze_categories_detailed(annotations_dir):
    """详细分析各类别在数据集中的使用情况"""
    
    # 存储各类别信息
    train_categories = set()
    val_categories = set()
    relation_categories = set()
    all_categories = set()
    
    category_id_to_name = {}
    category_name_to_id = {}
    
    print("=" * 60)
    print("详细分析数据集中的类别使用情况")
    print("=" * 60)
    
    # 1. 分析训练集
    train_file = os.path.join(annotations_dir, 'instances_train.json')
    if os.path.exists(train_file):
        print(f"\n1. 分析训练集: {train_file}")
        with open(train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取类别映射
        if 'categories' in data:
            for cat in data['categories']:
                if 'id' in cat and 'name' in cat:
                    category_id_to_name[cat['id']] = cat['name']
                    category_name_to_id[cat['name']] = cat['id']
        
        # 分析实际使用的类别
        if 'annotations' in data:
            for ann in data['annotations']:
                if 'category_id' in ann:
                    train_categories.add(ann['category_id'])
                    all_categories.add(ann['category_id'])
        
        print(f"   训练集中使用的类别数: {len(train_categories)}")
        print(f"   训练集类别ID: {sorted(train_categories)}")
    
    # 2. 分析验证集
    val_file = os.path.join(annotations_dir, 'instances_val.json')
    if os.path.exists(val_file):
        print(f"\n2. 分析验证集: {val_file}")
        with open(val_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取类别映射（如果训练集中没有）
        if not category_id_to_name and 'categories' in data:
            for cat in data['categories']:
                if 'id' in cat and 'name' in cat:
                    category_id_to_name[cat['id']] = cat['name']
                    category_name_to_id[cat['name']] = cat['id']
        
        # 分析实际使用的类别
        if 'annotations' in data:
            for ann in data['annotations']:
                if 'category_id' in ann:
                    val_categories.add(ann['category_id'])
                    all_categories.add(ann['category_id'])
        
        print(f"   验证集中使用的类别数: {len(val_categories)}")
        print(f"   验证集类别ID: {sorted(val_categories)}")
    
    # 3. 分析关系标注
    relation_file = os.path.join(annotations_dir, 'relation_annotations.json')
    if os.path.exists(relation_file):
        print(f"\n3. 分析关系标注: {relation_file}")
        with open(relation_file, 'r', encoding='utf-8') as f:
            relations = json.load(f)
        
        for rel in relations:
            if 'from_category' in rel:
                cat_name = rel['from_category']
                if cat_name in category_name_to_id:
                    cat_id = category_name_to_id[cat_name]
                    relation_categories.add(cat_id)
                    all_categories.add(cat_id)
            
            if 'to_category' in rel:
                cat_name = rel['to_category']
                if cat_name in category_name_to_id:
                    cat_id = category_name_to_id[cat_name]
                    relation_categories.add(cat_id)
                    all_categories.add(cat_id)
        
        print(f"   关系标注中使用的类别数: {len(relation_categories)}")
        print(f"   关系标注类别ID: {sorted(relation_categories)}")
    
    # 4. 分析增强版数据集
    enhanced_file = os.path.join(annotations_dir, 'dataset_with_relations_enhanced.json')
    if os.path.exists(enhanced_file):
        print(f"\n4. 分析增强版数据集: {enhanced_file}")
        with open(enhanced_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        enhanced_categories = set()
        if 'images' in data:
            for img in data['images']:
                if 'relations' in img:
                    for rel in img['relations']:
                        if 'from_category' in rel:
                            cat_name = rel['from_category']
                            if cat_name in category_name_to_id:
                                cat_id = category_name_to_id[cat_name]
                                enhanced_categories.add(cat_id)
                                all_categories.add(cat_id)
                        
                        if 'to_category' in rel:
                            cat_name = rel['to_category']
                            if cat_name in category_name_to_id:
                                cat_id = category_name_to_id[cat_name]
                                enhanced_categories.add(cat_id)
                                all_categories.add(cat_id)
        
        print(f"   增强版数据集中使用的类别数: {len(enhanced_categories)}")
        print(f"   增强版数据集类别ID: {sorted(enhanced_categories)}")
    
    # 5. 详细分析结果
    print(f"\n" + "=" * 60)
    print("详细分析结果")
    print("=" * 60)
    
    print(f"\n各类别使用情况:")
    print(f"  训练集: {len(train_categories)} 个类别")
    print(f"  验证集: {len(val_categories)} 个类别")
    print(f"  关系标注: {len(relation_categories)} 个类别")
    print(f"  总计: {len(all_categories)} 个类别")
    
    # 只在关系标注中出现的类别
    relation_only = relation_categories - train_categories - val_categories
    print(f"\n只在关系标注中出现的类别 ({len(relation_only)} 个):")
    for cat_id in sorted(relation_only):
        if cat_id in category_id_to_name:
            print(f"  ID {cat_id}: {category_id_to_name[cat_id]}")
    
    # 只在训练集中出现的类别
    train_only = train_categories - val_categories - relation_categories
    print(f"\n只在训练集中出现的类别 ({len(train_only)} 个):")
    for cat_id in sorted(train_only):
        if cat_id in category_id_to_name:
            print(f"  ID {cat_id}: {category_id_to_name[cat_id]}")
    
    # 只在验证集中出现的类别
    val_only = val_categories - train_categories - relation_categories
    print(f"\n只在验证集中出现的类别 ({len(val_only)} 个):")
    for cat_id in sorted(val_only):
        if cat_id in category_id_to_name:
            print(f"  ID {cat_id}: {category_id_to_name[cat_id]}")
    
    # 在训练集和验证集中都出现的类别
    train_val_common = train_categories & val_categories
    print(f"\n在训练集和验证集中都出现的类别 ({len(train_val_common)} 个):")
    for cat_id in sorted(train_val_common):
        if cat_id in category_id_to_name:
            print(f"  ID {cat_id}: {category_id_to_name[cat_id]}")
    
    # 建议的类别列表（基于实际使用的类别）
    print(f"\n建议的类别列表 (基于实际使用的 {len(all_categories)} 个类别):")
    for cat_id in sorted(all_categories):
        if cat_id in category_id_to_name:
            print(f"  ID {cat_id}: {category_id_to_name[cat_id]}")
    
    return all_categories, category_id_to_name

def main():
    annotations_dir = "annotations"
    
    if not os.path.exists(annotations_dir):
        print(f"错误：目录 {annotations_dir} 不存在！")
        return
    
    analyze_categories_detailed(annotations_dir)

if __name__ == "__main__":
    main()
