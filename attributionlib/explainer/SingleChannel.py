import time

import numpy as np
import torch

from attributionlib import AttributionResult
from distilllib.engine.utils import predict, individual_predict


class SingleChannelExplainer(object):
    def __init__(self, dataset: str, label_names: list,  # dataset information
                 model: torch.nn,  # model information
                 mark_baseline: float or str = 0):  # mark baseline
        # dataset information
        self.dataset = dataset
        self.label_names = label_names
        self.classes = len(label_names)
        # model information
        self.model = model
        self.model_name = model.__class__.__name__
        # mark baseline
        self.mark_baseline = mark_baseline
        # explainer name
        self.explainer_name = "SingleChannel"

    def __call__(self, sample_id: int, origin_input: np.ndarray, truth_label: int):
        assert len(origin_input.shape) == 2
        channels, points = origin_input.shape
        if self.mark_baseline == 'mean':
            self.mark_baseline = origin_input.mean()
        origin_pred = individual_predict(self.model, origin_input)
        origin_pred = origin_pred.cpu().numpy()
        origin_pred_label = np.argmax(origin_pred)
        time_start = time.perf_counter()
        # 初始化扰动数据，生成样本数等于通道数
        perturbation_data = np.full((channels, channels, points), self.mark_baseline, dtype=np.float32)
        # 填充生成扰动数据
        for channel in range(channels):
            perturbation_data[channel, channel, :] = origin_input[channel, :]
        # 预测扰动数据，计算每个通道的权重值
        heatmap_channel = predict(self.model, perturbation_data, eval=True).cpu().numpy()
        # 生成热力图
        heatmap = np.expand_dims(heatmap_channel, 1)
        heatmap = np.repeat(heatmap, points, 1)
        time_end = time.perf_counter()  # 记录结束时间
        run_time = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        result = AttributionResult(self.dataset, self.label_names, sample_id, origin_input, truth_label,
                                   self.model_name, origin_pred, origin_pred_label,
                                   self.explainer_name, heatmap, run_time)
        return result