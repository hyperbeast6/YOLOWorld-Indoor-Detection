import os
_base_ = '../pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py'

# 路径根据你的数据实际位置修改（建议使用绝对路径，避免 Windows 工作目录导致的问题）
data_root = os.path.abspath(os.path.join('data', 'indoor_dataset_cocostyle'))
train_ann = os.path.join('annotations', 'instances_train.json')
val_ann = os.path.join('annotations', 'instances_val.json')

# 数据集类别数量（保持预训练模型的类别数量以最大化利用预训练权重）
num_classes = 1203  # 保持预训练模型的类别数量
num_training_classes = 59  # 训练时只使用59个室内类别

# 类别文本文件路径 - 使用增强版本以提高匹配效果
class_text_path = os.path.join(data_root, 'annotations', 'enhanced_class_texts.json')

# 定义类别名称列表（与annotation文件中的类别顺序一致）
classes = [
	'backpack', 'bed', 'book', 'bottle', 'bowl', 'cell phone', 'chair', 'clock', 'couch', 'cup',
	'dining table', 'fork', 'handbag', 'keyboard', 'knife', 'lamp', 'laptop', 'mouse', 'person', 'refrigerator',
	'remote', 'spoon', 'tv', 'vase', 'microwave', 'sink', 'suitcase', 'oven', 'potted plant', 'umbrella',
	'sports ball', 'apple', 'cat', 'wine glass', 'banana', 'scissors', 'teddy bear', 'toaster', 'bench', 'car',
	'sheep', 'orange', 'dog', 'toilet', 'cake', 'horse', 'bicycle', 'bird', 'hot dog', 'donut',
	'frisbee', 'boat', 'tie', 'carrot', 'motorcycle', 'truck', 'sandwich', 'tennis racket', 'pizza'
]

# 增强版训练pipeline - 添加丰富的数据增强策略
train_pipeline = [
	# 图像与标注
	dict(type='LoadImageFromFile', to_float32=True),
	dict(type='LoadAnnotations', with_bbox=True),
	
	# 几何变换增强
	dict(type='mmdet.RandomFlip', prob=0.5),
	dict(type='mmdet.RandomResize', 
		 scale=(640, 640), 
		 ratio_range=(0.8, 1.2),
		 keep_ratio=True),
	
	# 强化光度变换 - 针对室内场景优化
	dict(type='mmdet.PhotoMetricDistortion',
		 brightness_delta=32,        # 增加亮度变化范围
		 contrast_range=(0.5, 1.5),  # 扩大对比度变化
		 saturation_range=(0.5, 1.5), # 扩大饱和度变化
		 hue_delta=18),              # 增加色调变化
	
	# 高级数据增强
	dict(type='mmdet.RandomCrop', crop_size=(640, 640)),  # 随机裁剪
	
	# 统一到 640 边长并填充
	dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
	dict(type='mmdet.Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
	
	# 文本 - 优化配置以最大化利用预训练模型
	dict(type='RandomLoadText',
		 num_neg_samples=(59, 59),  # 负样本数量匹配室内类别数量
		 max_num_samples=59,        # 最大样本数量
		 padding_to_max=True,
		 padding_value=''),
	
	# 打包
	dict(type='mmdet.PackDetInputs',
		 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction', 'texts'))
]

test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
	dict(type='mmdet.Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
	dict(type='LoadText'),
	dict(type='mmdet.PackDetInputs',
		 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param', 'texts'))
]

train_dataloader = dict(
	_delete_=True,
	batch_size=8,
	num_workers=4,
	persistent_workers=True,
	collate_fn=dict(type='yolow_collate', use_ms_training=False),
	dataset=dict(
		type='MultiModalDataset',
		dataset=dict(
			type='mmdet.CocoDataset',
			data_root=data_root,
			ann_file=train_ann,
			data_prefix=dict(img='images'),
			filter_cfg=dict(filter_empty_gt=True, min_size=1),
			metainfo=dict(classes=classes)
		),
		# 使用自定义类别文本文件
		class_text_path=class_text_path,
		# 使用本地定义的训练 pipeline
		pipeline=train_pipeline,
	)
)

# 验证集配置
val_dataloader = dict(
	_delete_=True,
	batch_size=4,
	num_workers=2,
	persistent_workers=True,
	collate_fn=dict(type='yolow_collate', use_ms_training=False),
	dataset=dict(
		type='MultiModalDataset',
		dataset=dict(
			type='mmdet.CocoDataset',
			data_root=data_root,
			ann_file=val_ann,
			data_prefix=dict(img='images'),
			metainfo=dict(classes=classes)
		),
		class_text_path=class_text_path,
		test_mode=True,
		# 使用本地定义的测试/验证 pipeline
		pipeline=test_pipeline,
	)
)

test_dataloader = val_dataloader

# 验证评估器配置 - 训练完成后使用
val_evaluator = dict(
	type='mmdet.CocoMetric',
	ann_file=os.path.join(data_root, val_ann),
	metric='bbox',
	classwise=False,  # 关闭类别评估，避免类别ID超出范围
	format_only=False,
	proposal_nums=(100, 300, 1000)
)

test_evaluator = val_evaluator

# 模型配置：保持预训练类别数量，优化微调策略
model = dict(
	num_train_classes=num_training_classes,
	num_test_classes=num_training_classes,  # 推理时也使用59个类别，避免索引超出范围
	bbox_head=dict(
		head_module=dict(
			num_classes=num_training_classes,
			# 冻结文本编码器，只微调视觉部分
			freeze_all=False
		)
	),
	train_cfg=dict(
		assigner=dict(num_classes=num_training_classes)
	),
	# 添加测试时配置，确保类别ID在有效范围内
	test_cfg=dict(
		score_thr=0.01,
		nms=dict(type='nms', iou_threshold=0.5),
		max_per_img=300
	)
)

# 关键修复：确保模型输出类别ID在0-58范围内
model['bbox_head']['head_module']['num_classes'] = num_training_classes
model['num_test_classes'] = num_training_classes

# 训练策略：80 个 epoch，达到最佳性能后停止
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=20)

# 优化器配置 - 针对数据增强优化
optim_wrapper = dict(
	optimizer=dict(
		_delete_=True,
		type='AdamW', 
		lr=3e-4,  # 提高学习率以应对增强数据
		weight_decay=0.005  # 减少正则化
	)
)

# 学习率调度器配置 - 多阶段调度
param_scheduler = [
	# 预热阶段
	dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=10),
	# 余弦退火
	dict(type='CosineAnnealingLR', T_max=90, eta_min=1e-7, by_epoch=True, begin=10, end=100)
]

default_hooks = dict(
	checkpoint=dict(
		type='CheckpointHook', 
		interval=5,  # 每5轮保存一次checkpoint
		max_keep_ckpts=5,
		save_best='coco/bbox_mAP',
		rule='greater'
	)
)

# 从预训练权重加载进行微调（路径按你的实际文件修改）
load_from = r'E:\code\YOLO-World\weights\yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth'

# 单卡训练常用设置（如需多卡请改用 tools/dist_train.sh 或相应启动方式）
env_cfg = dict(cudnn_benchmark=True)


