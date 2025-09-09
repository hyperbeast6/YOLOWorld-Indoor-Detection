#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据COCO标注文件生成本地路径的Label Studio导入文件
"""

import json
import os
import glob

def find_coco_images():
    """查找COCO图片文件"""
    print("🔍 查找COCO图片文件...")
    
    # 可能的COCO图片路径
    possible_paths = [  # 用户提到的完整COCO路径
        "data/coco/train2017",
        "data/coco/val2017", 
        "data/coco/images",
    ]
    
    image_files = []
    found_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ 找到图片目录: {path}")
            found_path = path
            
            # 查找所有图片文件
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(glob.glob(os.path.join(path, ext)))
                image_files.extend(glob.glob(os.path.join(path, '**', ext), recursive=True))
            
            break
    
    if not image_files:
        print("❌ 没有找到COCO图片文件")
        return None, []
    
    print(f"📸 找到 {len(image_files)} 张图片")
    
    # 创建文件名到路径的映射
    filename_to_path = {}
    for img_path in image_files:
        filename = os.path.basename(img_path)
        filename_to_path[filename] = img_path
    
    print(f"📊 创建了 {len(filename_to_path)} 个文件名映射")
    
    return found_path, filename_to_path

def find_coco_annotations():
    """查找COCO标注文件"""
    print("\n🔍 查找COCO标注文件...")
    
    # 可能的COCO标注文件路径
    possible_files = [
        "data/filtered_coco_dataset/annotations/instances_filtered.json",
        "data/coco/annotations/instances_train2017.json",
        "data/coco/annotations/instances_val2017.json",
        "mydataset/cocomini/annotations/instances_train2017.json"
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            print(f"✅ 找到标注文件: {file_path}")
            return file_path
    
    print("❌ 没有找到COCO标注文件")
    return None

def create_labelstudio_import(coco_file, filename_to_path, max_images=None):
    """创建Label Studio导入文件"""
    print(f"\n🔄 创建Label Studio导入文件...")
    
    # 读取COCO数据
    with open(coco_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    print(f"📊 COCO数据信息:")
    print(f"   - 图片数量: {len(coco_data.get('images', []))}")
    print(f"   - 标注数量: {len(coco_data.get('annotations', []))}")
    print(f"   - 类别数量: {len(coco_data.get('categories', []))}")
    
    # 创建类别映射
    categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    print(f"   - 类别映射: {len(categories)} 个类别")
    
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
    processed_count = 0
    skipped_count = 0
    
    for image_id, image_info in images.items():
        if max_images and processed_count >= max_images:
            break
            
        if image_id not in annotations_by_image:
            continue
        
        original_filename = image_info['file_name']
        
        # 查找对应的本地文件路径
        if original_filename in filename_to_path:
            local_path = filename_to_path[original_filename]
            processed_count += 1
        else:
            print(f"⚠️  未找到图片: {original_filename}")
            skipped_count += 1
            continue
        
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
                "image": local_path  # 使用本地文件路径
            },
            "annotations": [
                {
                    "result": results,
                    "ground_truth": False
                }
            ]
        }
        labelstudio_items.append(item)
    
    print(f"📊 处理统计:")
    print(f"   - 成功处理: {processed_count} 张图片")
    print(f"   - 跳过: {skipped_count} 张图片")
    print(f"   - 总标注: {len(labelstudio_items)} 个项目")
    
    return labelstudio_items

def save_import_files(labelstudio_items, coco_file):
    """保存导入文件"""
    print(f"\n💾 保存导入文件...")
    
    # 创建输出目录
    os.makedirs("data/labelstudio_import", exist_ok=True)
    
    # 生成文件名
    base_name = os.path.splitext(os.path.basename(coco_file))[0]
    
    # 保存完整文件
    full_file = f"data/labelstudio_import/{base_name}_local_path.json"
    with open(full_file, 'w', encoding='utf-8') as f:
        json.dump(labelstudio_items, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 完整文件: {full_file}")
    
    # 保存测试文件（前10个）
    test_items = labelstudio_items[:10]
    test_file = f"data/labelstudio_import/{base_name}_local_path_test.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_items, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 测试文件: {test_file}")
    
    return full_file, test_file

def create_import_guide(full_file, test_file):
    """创建导入指南"""
    print(f"\n📖 创建导入指南...")
    
    guide = f"""# 本地路径Label Studio导入指南

## 🎉 成功创建本地路径导入文件！

### 📊 文件信息
- **完整导入文件**: `{full_file}`
- **测试文件**: `{test_file}`

### 📋 导入步骤

#### 步骤1：测试导入
1. 在Label Studio中进入"Data Import"页面
2. 选择"Import"标签页
3. 点击"Upload"按钮
4. 选择文件：`{test_file}`
5. 点击"Import"
6. 确认图片和标注都正确显示

#### 步骤2：完整导入
1. 如果测试成功，导入完整文件：`{full_file}`
2. 等待导入完成
3. 检查所有图片的标注是否正确

### 🔍 验证方法
1. **图片显示**：确认图片正常显示，没有红色错误框
2. **边界框显示**：确认边界框正确显示在图片上
3. **标注信息**：点击边界框，右侧面板显示标注信息
4. **类别正确**：确认物体类别标注正确

### 🚨 注意事项
1. **文件路径**：确保图片文件路径正确且可访问
2. **文件权限**：确保Label Studio有权限读取图片文件
3. **路径格式**：使用绝对路径或相对于Label Studio工作目录的路径

### 📈 下一步
1. 确认标注正确后，开始进行空间关系标注
2. 使用Relations功能标注物体间的空间关系
3. 添加场景类型和自然语言描述
4. 定期导出标注结果进行备份

## 🎯 预期结果
- ✅ 图片正常显示
- ✅ 边界框正确显示
- ✅ 标注信息完整
- ✅ 可以正常进行空间关系标注
"""
    
    # 保存指南
    guide_file = "data/labelstudio_import/local_path_import_guide.md"
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print(f"✅ 创建导入指南: {guide_file}")

def main():
    print("🔄 开始创建本地路径Label Studio导入文件...")
    
    # 1. 查找COCO图片文件
    found_path, filename_to_path = find_coco_images()
    if not found_path:
        print("❌ 无法继续，没有找到图片文件")
        return
    
    # 2. 查找COCO标注文件
    coco_file = find_coco_annotations()
    if not coco_file:
        print("❌ 无法继续，没有找到标注文件")
        return
    
    # 3. 创建Label Studio导入文件
    labelstudio_items = create_labelstudio_import(coco_file, filename_to_path, max_images=100)  # 限制100张图片
    
    if not labelstudio_items:
        print("❌ 没有生成任何导入项目")
        return
    
    # 4. 保存导入文件
    full_file, test_file = save_import_files(labelstudio_items, coco_file)
    
    # 5. 创建导入指南
    create_import_guide(full_file, test_file)
    
    print(f"\n🎉 本地路径导入文件创建完成！")
    print(f"📊 创建的文件:")
    print(f"   - 完整导入文件: {full_file}")
    print(f"   - 测试文件: {test_file}")
    print(f"   - 导入指南: data/labelstudio_import/local_path_import_guide.md")
    
    print(f"\n📋 下一步操作:")
    print("1. 在Label Studio中导入测试文件进行验证")
    print("2. 确认图片和标注都正确显示")
    print("3. 如果成功，导入完整文件")
    print("4. 开始进行空间关系标注")

if __name__ == "__main__":
    main()

