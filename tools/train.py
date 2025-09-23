# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import sys

# 添加 third_party 路径，优先使用本地的 mmyolo
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
third_party_path = os.path.join(project_root, 'third_party')
# 确保路径存在且不在sys.path中
if os.path.exists(third_party_path) and third_party_path not in sys.path:
    sys.path.insert(0, third_party_path)

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower


def get_enhanced_augmentation_pipeline(intensity='medium'):
    """获取增强版数据增强pipeline"""
    
    # 基础pipeline
    base_pipeline = [
        dict(type='LoadImageFromFile', to_float32=True),
        dict(type='LoadAnnotations', with_bbox=True),
    ]
    
    # 根据强度配置数据增强
    if intensity == 'light':
        # 轻度增强
        augmentation_steps = [
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(type='mmdet.RandomResize', scale=(640, 640), ratio_range=(0.8, 1.2), keep_ratio=True),
            dict(type='mmdet.PhotoMetricDistortion',
                 brightness_delta=16,
                 contrast_range=(0.8, 1.2),
                 saturation_range=(0.8, 1.2),
                 hue_delta=8),
        ]
    elif intensity == 'medium':
        # 中度增强 - 使用多尺度训练
        augmentation_steps = [
            dict(type='mmdet.RandomFlip', prob=0.5),
            # 多尺度训练 - 随机选择尺寸
            dict(type='mmdet.RandomChoice',
                 transforms=[
                     dict(type='mmdet.RandomResize', scale=(512, 512), ratio_range=(0.8, 1.2), keep_ratio=True),
                     dict(type='mmdet.RandomResize', scale=(640, 640), ratio_range=(0.8, 1.2), keep_ratio=True),
                     dict(type='mmdet.RandomResize', scale=(768, 768), ratio_range=(0.8, 1.2), keep_ratio=True),
                 ]),
            dict(type='mmdet.PhotoMetricDistortion',
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18),
            dict(type='mmdet.RandomCrop', crop_size=(640, 640)),
        ]
    else:  # heavy
        # 重度增强 - 更丰富的多尺度训练
        augmentation_steps = [
            dict(type='mmdet.RandomFlip', prob=0.5),
            # 更丰富的多尺度训练
            dict(type='mmdet.RandomChoice',
                 transforms=[
                     dict(type='mmdet.RandomResize', scale=(480, 480), ratio_range=(0.8, 1.2), keep_ratio=True),
                     dict(type='mmdet.RandomResize', scale=(512, 512), ratio_range=(0.8, 1.2), keep_ratio=True),
                     dict(type='mmdet.RandomResize', scale=(640, 640), ratio_range=(0.8, 1.2), keep_ratio=True),
                     dict(type='mmdet.RandomResize', scale=(768, 768), ratio_range=(0.8, 1.2), keep_ratio=True),
                     dict(type='mmdet.RandomResize', scale=(1024, 1024), ratio_range=(0.8, 1.2), keep_ratio=True),
                 ]),
            dict(type='mmdet.PhotoMetricDistortion',
                 brightness_delta=40,
                 contrast_range=(0.3, 1.7),
                 saturation_range=(0.3, 1.7),
                 hue_delta=25),
            dict(type='mmdet.RandomCrop', crop_size=(640, 640)),
        ]
    
    # 标准化步骤
    normalization_steps = [
        dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
        dict(type='mmdet.Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    ]
    
    # 文本处理步骤
    text_steps = [
        dict(type='RandomLoadText',
             num_neg_samples=(59, 59),
             max_num_samples=59,
             padding_to_max=True,
             padding_value=''),
    ]
    
    # 打包步骤
    pack_steps = [
        dict(type='mmdet.PackDetInputs',
             meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction', 'texts'))
    ]
    
    # 组合完整pipeline
    full_pipeline = base_pipeline + augmentation_steps + normalization_steps + text_steps + pack_steps
    
    return full_pipeline


def apply_enhanced_training_config(cfg, args):
    """应用增强版训练配置"""
    
    if not args.enable_augmentation:
        return cfg
    
    print_log("🚀 启用增强版数据增强策略...", logger='current', level=logging.INFO)
    print_log(f"📊 增强强度: {args.aug_intensity}", logger='current', level=logging.INFO)
    
    # 获取增强版pipeline
    enhanced_pipeline = get_enhanced_augmentation_pipeline(args.aug_intensity)
    
    # 更新训练pipeline
    if 'train_dataloader' in cfg and 'dataset' in cfg.train_dataloader:
        cfg.train_dataloader.dataset.pipeline = enhanced_pipeline
        print_log("✅ 训练数据增强已更新", logger='current', level=logging.INFO)
    
    # 调整训练参数
    if args.aug_intensity in ['medium', 'heavy']:
        # 延长训练轮数
        if 'train_cfg' in cfg and 'max_epochs' in cfg.train_cfg:
            original_epochs = cfg.train_cfg.max_epochs
            cfg.train_cfg.max_epochs = min(original_epochs * 1.5, 200)  # 最多200轮
            print_log(f"📈 训练轮数: {original_epochs} → {cfg.train_cfg.max_epochs}", 
                     logger='current', level=logging.INFO)
        
        # 调整学习率
        if 'optim_wrapper' in cfg and 'optimizer' in cfg.optim_wrapper:
            if 'lr' in cfg.optim_wrapper.optimizer:
                original_lr = cfg.optim_wrapper.optimizer.lr
                cfg.optim_wrapper.optimizer.lr = original_lr * 1.5  # 提高学习率
                print_log(f"📈 学习率: {original_lr} → {cfg.optim_wrapper.optimizer.lr}", 
                         logger='current', level=logging.INFO)
        
        # 调整验证间隔
        if 'train_cfg' in cfg and 'val_interval' in cfg.train_cfg:
            cfg.train_cfg.val_interval = 25  # 更频繁的验证
            print_log("📈 验证间隔: 每25轮", logger='current', level=logging.INFO)
    
    # 添加早停机制
    if 'default_hooks' not in cfg:
        cfg.default_hooks = {}
    
    if 'checkpoint' not in cfg.default_hooks:
        cfg.default_hooks['checkpoint'] = {}
    
    cfg.default_hooks['checkpoint'].update({
        'type': 'CheckpointHook',
        'interval': 10,
        'max_keep_ckpts': 5,
        'save_best': 'coco/bbox_mAP',
        'rule': 'greater'
    })
    
    print_log("✅ 增强版训练配置已应用", logger='current', level=logging.INFO)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    # 数据增强相关参数
    parser.add_argument(
        '--enable-augmentation',
        action='store_true',
        default=False,
        help='enable enhanced data augmentation for better performance')
    parser.add_argument(
        '--aug-intensity',
        type=str,
        choices=['light', 'medium', 'heavy'],
        default='medium',
        help='data augmentation intensity: light, medium, heavy')
    parser.add_argument(
        '--target-mAP',
        type=float,
        default=0.25,
        help='target mAP for early stopping (default: 0.25)')
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    # replace the ${key} with the value of cfg.key
    # cfg = replace_cfg_vals(cfg)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # 应用增强版训练配置
    cfg = apply_enhanced_training_config(cfg, args)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        if args.config.startswith('projects/'):
            config = args.config[len('projects/'):]
            config = config.replace('/configs/', '/')
            cfg.work_dir = osp.join('./work_dirs', osp.splitext(config)[0])
        else:
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # Determine whether the custom metainfo fields are all lowercase
    is_metainfo_lower(cfg)

    # 显示训练信息
    if args.enable_augmentation:
        print_log("=" * 60, logger='current', level=logging.INFO)
        print_log("🎯 增强版训练配置信息:", logger='current', level=logging.INFO)
        print_log(f"📊 数据增强强度: {args.aug_intensity}", logger='current', level=logging.INFO)
        print_log(f"🎯 目标mAP: {args.target_mAP}", logger='current', level=logging.INFO)
        if 'train_cfg' in cfg and 'max_epochs' in cfg.train_cfg:
            print_log(f"📈 训练轮数: {cfg.train_cfg.max_epochs}", logger='current', level=logging.INFO)
        if 'optim_wrapper' in cfg and 'optimizer' in cfg.optim_wrapper and 'lr' in cfg.optim_wrapper.optimizer:
            print_log(f"📈 学习率: {cfg.optim_wrapper.optimizer.lr}", logger='current', level=logging.INFO)
        print_log("=" * 60, logger='current', level=logging.INFO)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
