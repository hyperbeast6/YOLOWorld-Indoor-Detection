# Label Studio 文件名映射指南

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
