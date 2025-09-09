#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出Label Studio任务列表，获取实际的文件路径
"""

import json
import os

def export_labelstudio_tasks():
    """导出Label Studio任务列表"""
    
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
