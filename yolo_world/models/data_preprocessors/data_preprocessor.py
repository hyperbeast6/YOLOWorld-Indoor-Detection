# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.structures import BaseDataElement

from mmyolo.registry import MODELS

CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str,
                 None]


@MODELS.register_module()
class YOLOWDetDataPreprocessor(DetDataPreprocessor):
    """Rewrite collate_fn to get faster training speed.

    Note: It must be used together with `mmyolo.datasets.utils.yolow_collate`
    """

    def __init__(self, *args, non_blocking: Optional[bool] = True, **kwargs):
        super().__init__(*args, non_blocking=non_blocking, **kwargs)

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``DetDataPreprocessorr``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        if not training:
            return super().forward(data, training)

        data = self.cast_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        # 现在 data_samples 是 DetDataSample 列表，不是字典

        # TODO: Supports multi-scale training
        if self._channel_conversion and inputs.shape[1] == 3:
            inputs = inputs[:, [2, 1, 0], ...]
        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std

        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        # 更新每个 DetDataSample 的 metainfo 中的 batch_input_shape
        for i, sample in enumerate(data_samples):
            if sample.metainfo is None:
                sample.set_metainfo({})
            sample.metainfo['batch_input_shape'] = inputs.shape[2:]

        return {'inputs': inputs, 'data_samples': data_samples}
