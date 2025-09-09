#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态匹配Label Studio重命名文件的解决方案
"""

import json
import os
import shutil
import glob
from pathlib import Path

def analyze_labelstudio_import():
    """分析Label Studio导入后的文件结构"""
    print("🔍 分析Label Studio导入后的文件结构...")
    
    # 检查Label Studio数据目录
    labelstudio_dir = "label_studio_data"
    if not os.path.exists(labelstudio_dir):
        print("❌ Label Studio数据目录不存在，请先导入图片到Label Studio")
        return None
    
    # 查找导入的图片文件
    upload_dir = os.path.join(labelstudio_dir, "media", "upload")
    if not os.path.exists(upload_dir):
        print("❌ 上传目录不存在")
        return None
    
    # 收集所有图片文件及其路径
    image_files = []
    for root, dirs, files in os.walk(upload_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.relpath(os.path.join(root, file), upload_dir)
                # 转换为Label Studio路径格式
                ls_path = f"/data/upload/{rel_path.replace(os.sep, '/')}"
                image_files.append({
                    'original_name': file,
                    'ls_path': ls_path,
                    'full_path': os.path.join(root, file)
                })
    
    print(f"✅ 找到 {len(image_files)} 个图片文件")
    
    # 显示前几个文件作为示例
    for i, img in enumerate(image_files[:5]):
        print(f"   {i+1}. {img['original_name']} -> {img['ls_path']}")
    
    return image_files

def create_matched_import_file(image_files):
    """创建匹配Label Studio文件名的导入文件"""
    print("\n🔄 创建匹配的导入文件...")
    
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
    
    # 创建文件名到Label Studio路径的映射
    # 假设Label Studio保持了原始文件名的部分信息
    filename_mapping = {}
    for img in image_files:
        original_name = img['original_name']
        ls_path = img['ls_path']
        
        # 尝试从重命名后的文件名中提取原始文件名
        # 例如: a18667a0-000000002347.jpg -> 000000002347.jpg
        if '-' in original_name:
            parts = original_name.split('-')
            if len(parts) >= 2:
                # 取最后一部分作为原始文件名
                original_filename = parts[-1]
                filename_mapping[original_filename] = ls_path
                print(f"   映射: {original_filename} -> {ls_path}")
    
    # 生成Label Studio格式
    labelstudio_items = []
    matched_count = 0
    unmatched_count = 0
    
    for image_id, image_info in images.items():
        if image_id not in annotations_by_image:
            continue
        
        original_filename = image_info['file_name']
        
        # 查找对应的Label Studio路径
        if original_filename in filename_mapping:
            image_path = filename_mapping[original_filename]
            matched_count += 1
        else:
            # 如果找不到匹配，尝试其他方法
            print(f"⚠️  未找到匹配: {original_filename}")
            unmatched_count += 1
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
    
    print(f"📊 匹配统计:")
    print(f"   - 成功匹配: {matched_count} 张图片")
    print(f"   - 未匹配: {unmatched_count} 张图片")
    print(f"   - 总标注: {len(labelstudio_items)} 个项目")
    
    # 保存文件
    output_file = "data/labelstudio_import/matched_import.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(labelstudio_items, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 创建匹配导入文件: {output_file}")
    
    # 创建测试版本
    test_data = labelstudio_items[:3]
    test_file = "data/labelstudio_import/test_matched.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 创建测试文件: {test_file}")
    
    return output_file, test_file

def create_manual_mapping_guide():
    """创建手动映射指南"""
    print("\n📖 创建手动映射指南...")
    
    guide = """# Label Studio 文件名映射指南

## 🔍 问题分析
Label Studio在导入图片时会自动重命名文件，导致原始文件名与标注文件中的路径不匹配。

## 🚀 解决方案

### 方法1：自动匹配（推荐）
1. 运行脚本: `python scripts/fix_dynamic_matching.py`
2. 脚本会自动分析Label Studio的文件结构
3. 创建匹配的导入文件: `data/labelstudio_import/matched_import.json`

### 方法2：手动映射
如果自动匹配失败，可以手动创建映射：

1. **查看Label Studio中的实际文件名**
   - 进入Label Studio项目
   - 查看任意任务的图片路径
   - 例如: `/data/upload/3/a18667a0-000000002347.jpg`

2. **创建映射文件**
   ```json
   {
     "000000002347.jpg": "/data/upload/3/a18667a0-000000002347.jpg",
     "000000002348.jpg": "/data/upload/4/b18667a0-000000002348.jpg"
   }
   ```

3. **使用映射创建导入文件**

### 方法3：重新导入策略
1. **先导入图片**：将图片文件夹导入到Label Studio
2. **导出任务列表**：从Label Studio导出任务列表，获取实际的文件路径
3. **创建匹配的标注文件**：根据导出的路径创建标注文件
4. **重新导入标注**：将匹配的标注文件导入到Label Studio

## 📋 操作步骤

### 步骤1：分析文件结构
```bash
python scripts/fix_dynamic_matching.py
```

### 步骤2：检查匹配结果
- 查看控制台输出的匹配统计
- 确认匹配率是否足够高

### 步骤3：导入匹配的文件
1. 在Label Studio中进入"Data Import"
2. 导入: `data/labelstudio_import/test_matched.json`
3. 确认图片和标注都正确显示
4. 如果成功，导入完整文件: `data/labelstudio_import/matched_import.json`

## 🔧 故障排除

### 如果匹配率很低：
- 检查Label Studio的文件命名规则
- 可能需要调整匹配算法
- 考虑使用手动映射方法

### 如果仍有问题：
- 尝试重新导入图片到Label Studio
- 检查Label Studio的版本和配置
- 考虑使用其他标注工具

## 📊 预期结果
- ✅ 图片路径正确匹配
- ✅ 边界框正确显示
- ✅ 标注信息完整
- ✅ 可以正常进行空间关系标注
"""
    
    # 保存指南
    guide_file = "data/labelstudio_import/mapping_guide.md"
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print(f"✅ 创建映射指南: {guide_file}")

def create_export_script():
    """创建导出Label Studio任务列表的脚本"""
    print("\n📝 创建导出脚本...")
    
    export_script = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
导出Label Studio任务列表，获取实际的文件路径
\"\"\"

import json
import os

def export_labelstudio_tasks():
    \"\"\"导出Label Studio任务列表\"\"\"
    
    # Label Studio数据目录
    labelstudio_dir = "label_studio_data"
    
    # 查找项目文件
    projects_dir = os.path.join(labelstudio_dir, "projects")
    if not os.path.exists(projects_dir):
        print("❌ 项目目录不存在")
        return
    
    # 查找第一个项目
    project_dirs = [d for d in os.listdir(projects_dir) if os.path.isdir(os.path.join(projects_dir, d))]
    if not project_dirs:
        print("❌ 没有找到项目")
        return
    
    project_dir = os.path.join(projects_dir, project_dirs[0])
    print(f"📁 找到项目: {project_dirs[0]}")
    
    # 查找任务文件
    tasks_file = os.path.join(project_dir, "tasks.json")
    if not os.path.exists(tasks_file):
        print("❌ 任务文件不存在")
        return
    
    # 读取任务文件
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    print(f"📊 找到 {len(tasks)} 个任务")
    
    # 提取图片路径
    image_paths = []
    for task in tasks:
        if 'data' in task and 'image' in task['data']:
            image_paths.append({
                'task_id': task.get('id', 'unknown'),
                'image_path': task['data']['image']
            })
    
    # 保存到文件
    output_file = "data/labelstudio_import/actual_paths.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(image_paths, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 导出实际路径到: {output_file}")
    print(f"📊 包含 {len(image_paths)} 个图片路径")
    
    # 显示前几个路径
    for i, path_info in enumerate(image_paths[:5]):
        print(f"   {i+1}. 任务 {path_info['task_id']}: {path_info['image_path']}")

if __name__ == "__main__":
    export_labelstudio_tasks()
"""
    
    # 保存脚本
    script_file = "scripts/export_labelstudio_tasks.py"
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(export_script)
    
    print(f"✅ 创建导出脚本: {script_file}")

def main():
    print("🔄 开始动态匹配Label Studio文件...")
    
    # 1. 分析Label Studio文件结构
    image_files = analyze_labelstudio_import()
    
    if image_files is None:
        print("\n❌ 无法分析Label Studio文件结构")
        print("请先导入图片到Label Studio，然后重新运行此脚本")
        return
    
    # 2. 创建匹配的导入文件
    import_file, test_file = create_matched_import_file(image_files)
    
    # 3. 创建映射指南
    create_manual_mapping_guide()
    
    # 4. 创建导出脚本
    create_export_script()
    
    print(f"\n🎉 动态匹配完成！")
    print(f"📊 创建的文件:")
    print(f"   - 匹配导入文件: {import_file}")
    print(f"   - 测试文件: {test_file}")
    print(f"   - 映射指南: data/labelstudio_import/mapping_guide.md")
    print(f"   - 导出脚本: scripts/export_labelstudio_tasks.py")
    
    print(f"\n📋 下一步操作:")
    print("1. 在Label Studio中导入: data/labelstudio_import/test_matched.json")
    print("2. 确认图片和标注都正确显示")
    print("3. 如果成功，导入完整文件: data/labelstudio_import/matched_import.json")
    print("4. 如果仍有问题，运行: python scripts/export_labelstudio_tasks.py")

if __name__ == "__main__":
    main()
