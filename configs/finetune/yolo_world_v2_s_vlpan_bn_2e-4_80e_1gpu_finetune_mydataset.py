_base_ = ['../pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py']

# 路径根据你的数据实际位置修改（建议使用绝对路径，避免 Windows 工作目录导致的问题）
data_root = 'E:/code/YOLO-World/data/indoor_dataset_cocostyle/'
train_ann = 'annotations/instances_train.json'
val_ann = 'annotations/instances_val.json'

# 数据集类别数量（根据实际数据集调整）
num_classes = 78
num_training_classes = 78

# 类别文本文件路径
class_text_path = 'E:/code/YOLO-World/data/indoor_dataset_cocostyle/annotations/class_texts.json'

train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='MultiModalDataset',
        dataset=dict(
            type='mmdet.CocoDataset',
            data_root=data_root,
            ann_file=train_ann,
            data_prefix=dict(img='images/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=1)
        ),
        # 使用自定义类别文本文件
        class_text_path=class_text_path,
    )
)

val_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type='MultiModalDataset',
        dataset=dict(
            type='mmdet.CocoDataset',
            data_root=data_root,
            ann_file=val_ann,
            data_prefix=dict(img='images/')
        ),
        class_text_path=class_text_path,
        test_mode=True,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + val_ann,
    metric='bbox'
)

test_evaluator = val_evaluator

# 模型配置：更新类别数量
model = dict(
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    bbox_head=dict(
        head_module=dict(num_classes=num_training_classes)
    ),
    train_cfg=dict(
        assigner=dict(num_classes=num_training_classes)
    )
)

# 训练策略：80 个 epoch，使用较小学习率进行微调
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=1)

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='SGD', lr=2e-4, momentum=0.9, weight_decay=5e-4
    )
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3)
)

# 从预训练权重加载进行微调（路径按你的实际文件修改）
load_from = 'E:/code/YOLO-World/weights/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth'

# 单卡训练常用设置（如需多卡请改用 tools/dist_train.sh 或相应启动方式）
env_cfg = dict(cudnn_benchmark=True)


