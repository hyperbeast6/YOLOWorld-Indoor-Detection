# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
from mmengine.dataset import COLLATE_FUNCTIONS
from mmdet.structures import DetDataSample


@COLLATE_FUNCTIONS.register_module()
def yolow_collate(data_batch: Sequence,
                  use_ms_training: bool = False) -> dict:
    """Rewrite collate_fn to return DetDataSample list format for standard MMDet compatibility.

    Args:
       data_batch (Sequence): Batch of data from pipeline.
       use_ms_training (bool): Whether to use multi-scale training.
    
    Returns:
        dict: Contains 'inputs' (tensor) and 'data_samples' (list[DetDataSample])
    """
    batch_imgs = []
    data_samples = []
    
    for i, item in enumerate(data_batch):
        # 获取图像数据
        inputs = item['inputs']
        batch_imgs.append(inputs)
        
        # 获取原始数据样本
        datasample = item['data_samples']
        
        # 创建新的 DetDataSample
        det_sample = DetDataSample()
        
        # 复制 metainfo
        if hasattr(datasample, 'metainfo'):
            det_sample.set_metainfo(datasample.metainfo)
        
        # 复制 gt_instances
        if hasattr(datasample, 'gt_instances'):
            det_sample.gt_instances = datasample.gt_instances
        
        # 复制 ignored_instances
        if hasattr(datasample, 'ignored_instances'):
            det_sample.ignored_instances = datasample.ignored_instances
        
        # 复制其他属性
        if hasattr(datasample, 'pred_instances'):
            det_sample.pred_instances = datasample.pred_instances
        
        data_samples.append(det_sample)
    
    # 准备返回结果
    if use_ms_training:
        collated_inputs = batch_imgs
    else:
        collated_inputs = torch.stack(batch_imgs, 0)
    
    return {
        'inputs': collated_inputs,
        'data_samples': data_samples
    }
