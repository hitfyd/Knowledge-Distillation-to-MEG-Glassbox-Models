import time

import mne
import numpy as np
import ray
import torch
from math import ceil, floor
from matplotlib import pyplot as plt, gridspec, colors, colorbar
from matplotlib.collections import LineCollection
from sklearn import metrics
from tqdm import tqdm

from TorchUtil import predict, individual_predict


class AttributionResult(object):
    def __init__(self, dataset: str, label_names: list,  # dataset information
                 sample_id: int, origin_input: np.ndarray, truth_label: int,  # sample information
                 model_name: str, pred: list, pred_label: int,  # model information and model predictions for the sample
                 attribution_method: str,  # feature attribution method
                 attribution_maps: np.ndarray,  # feature attribution maps
                 run_time: float = 0.0):  # the run time of feature attribution method
        # input check
        assert len(attribution_maps.shape) == 3
        assert len(origin_input.shape) == 2
        assert origin_input.shape == attribution_maps.shape[:2]
        assert len(label_names) == len(pred) == attribution_maps.shape[2]
        # dataset information
        self.dataset = dataset
        self.label_names = label_names
        # sample information
        self.sample_id = sample_id
        self.origin_input = origin_input  # which shape is [channels, points]
        self.truth_label = truth_label
        self.channels = origin_input.shape[0]
        self.points = origin_input.shape[1]
        # model information and model predictions for the sample
        self.model_name = model_name
        self.pred = pred
        self.pred_label = pred_label
        # feature attribution method
        self.attribution_method = attribution_method
        # feature attribution maps, which shape is [channels, points, len(pred)]
        self.attribution_maps = attribution_maps
        # the run time of feature attribution method
        self.run_time = run_time
        # Automatically generated result ID for retrieval after persistence
        self.result_id = "{}_{}_{}_{}".format(dataset, sample_id, model_name, attribution_method)


class AttributionExplainer(object):
    def __init__(self, dataset: str, label_names: list,  # dataset information
                 model: torch.nn,  # model information
                 reference_dataset: np.ndarray,
                 # The original reference dataset. ‘reference_num’ reference samples are randomly selected when no reference dataset filter is used
                 reference_num: int = 100, window_length: int = 5, M: int = 256,  # The interpreter hyperparameters
                 reference_filter: bool = True, antithetic_variables: bool = False,
                 # The parameters of the variant algorithm, used for comparative ablation experiments
                 parallel: bool = True):  # Whether to use parallel acceleration
        # input check
        assert len(reference_dataset.shape) == 3
        assert len(reference_dataset) >= reference_num
        # dataset information
        self.dataset = dataset
        self.label_names = label_names
        self.classes = len(label_names)
        # model information
        self.model = model
        self.model_name = model.__class__.__name__
        # The original reference dataset
        self.reference_dataset = reference_dataset  # which shape is [channels, points]
        # The interpreter hyperparameters
        self.reference_num = reference_num  # n
        self.window_length = window_length  # w
        self.M = M  # m
        # The parameters of the variant algorithm
        self.reference_filter = reference_filter
        self.antithetic_variables = antithetic_variables
        # Whether to use parallel acceleration
        self.parallel = parallel
        # Automatically generated attribution explainer name
        self.explainer_name = "attribution_{}_{}_{}_{}_{}_{}".format(
            reference_num, window_length, M, reference_filter, antithetic_variables, parallel)

        if parallel and not ray.is_initialized():
            # 多线程初始化
            ray.init(num_gpus=1, num_cpus=16,   # 计算资源
                     local_mode=False,           # 是否启动串行模型，用于调试
                     ignore_reinit_error=True,  # 重复启动不视为错误
                     include_dashboard=False,   # 是否启动仪表盘
                     configure_logging=False,   # 不配置日志
                     log_to_driver=False,       # 日志记录不配置到driver
                     )

    def __call__(self, sample_id: int, origin_input: np.ndarray, truth_label: int):
        origin_pred, origin_pred_label = individual_predict(self.model, torch.from_numpy(origin_input))
        time_start = time.perf_counter()
        if self.parallel:
            assert self.parallel and ray.is_initialized()
            attribution_maps = feature_attribution_parallel(self.model, self.classes, origin_input,
                                                            self.reference_dataset,
                                                            self.reference_num, self.window_length, self.M,
                                                            self.reference_filter, self.antithetic_variables)
        else:
            attribution_maps = feature_attribution(self.model, self.classes, origin_input,
                                                   self.reference_dataset,
                                                   self.reference_num, self.window_length, self.M,
                                                   self.reference_filter, self.antithetic_variables)
        time_end = time.perf_counter()  # 记录结束时间
        run_time = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        result = AttributionResult(self.dataset, self.label_names, sample_id, origin_input, truth_label,
                                   self.model_name, origin_pred, origin_pred_label,
                                   self.explainer_name, attribution_maps, run_time)
        return result

    def multi_models(self, sample_id: int, origin_input: np.ndarray, truth_label: int, model_list: list, M_list: list):
        assert self.parallel and ray.is_initialized()
        # 多模型多采样数下的计时仅供参考
        time_start = time.perf_counter()
        attribution_maps = feature_attribution_parallel_multi_models(model_list, self.classes, origin_input,
                                                                     self.reference_dataset,
                                                                     self.reference_num, self.window_length, M_list,
                                                                     self.reference_filter, self.antithetic_variables)
        time_end = time.perf_counter()  # 记录结束时间
        run_time = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        result_list = []
        for model_id in range(len(model_list)):
            result_list.append([])
            model = model_list[model_id]
            origin_pred, origin_pred_label = individual_predict(model, torch.from_numpy(origin_input))
            for M_id in range(len(M_list)):
                M = M_list[M_id]
                explainer_name = "attribution_{}_{}_{}_{}_{}_{}".format(
                    self.reference_num, self.window_length, M, self.reference_filter, self.antithetic_variables,
                    self.parallel)
                result = AttributionResult(self.dataset, self.label_names, sample_id, origin_input, truth_label,
                                           model.__class__.__name__, origin_pred, origin_pred_label,
                                           explainer_name, attribution_maps[:, :, model_id, M_id, :], run_time)
                result_list[model_id].append(result)
        return result_list

    # def __del__(self):
    #     if self.parallel and ray.is_initialized():
    #         ray.shutdown()
    #         print("stop_parallel: ray.shutdown")


def feature_segment(channels, points, window_length):
    channel_windows_num = int(points / window_length)  # 需要计算的通道特征数和时间特征数，总特征数为c_features x p_features
    features_num = channels * channel_windows_num
    channel_list, point_start_list = [], []
    for feature_id in range(features_num):
        channel_list.append(int(feature_id / channel_windows_num))
        point_start_list.append(int(feature_id % channel_windows_num * window_length))
    return features_num, channel_list, point_start_list


def reference_dataset_filter(origin_input: np.ndarray, reference_dataset: np.ndarray, model: torch.nn, threshold: int):
    reference_dataset_num = len(reference_dataset)
    dist = np.zeros(reference_dataset_num, dtype=np.float32)
    origin_pred, origin_pred_label = individual_predict(model, torch.from_numpy(origin_input))
    reference_predictions, reference_prediction_labels = predict(model, torch.from_numpy(reference_dataset))
    # for reference_id in range(reference_dataset_num):
    #     dist[reference_id] = np.linalg.norm(reference_predictions[reference_id])
    # sort_index = dist.argsort()
    # optimized_reference_dataset = reference_dataset[sort_index[:threshold]]

    channels, points = origin_input.shape
    point_slice, channel_slice = points // 2, channels // 2
    for i in range(reference_dataset_num):
        reference = reference_dataset[i]
        temp_dataset = np.zeros((4, channels, points), dtype=np.float32)
        temp_dataset[0, :, :point_slice] = origin_input[:, :point_slice]
        temp_dataset[0, :, point_slice:] = reference[:, point_slice:]
        temp_dataset[1, :, :point_slice] = reference[:, :point_slice]
        temp_dataset[1, :, point_slice:] = origin_input[:, point_slice:]
        temp_dataset[2, :channel_slice, :] = origin_input[:channel_slice, :]
        temp_dataset[2, channel_slice:, :] = reference[channel_slice:, :]
        temp_dataset[3, :channel_slice, :] = reference[:channel_slice, :]
        temp_dataset[3, channel_slice:, :] = origin_input[channel_slice:, :]
        preds, pred_labels = predict(model, torch.from_numpy(temp_dataset))
        # print(i, preds.sum(axis=0), preds.sum(axis=0) - origin_pred, np.linalg.norm(preds.sum(axis=0) - origin_pred))
        dist[i] = np.linalg.norm(preds.sum(axis=0) - origin_pred)
    sort_index = dist.argsort()
    optimized_reference_dataset = reference_dataset[sort_index[:threshold]]

    # print(origin_pred, origin_pred_label)
    # for index in sort_index:
    #     print(index, reference_predictions[index], reference_prediction_labels[index])

    return optimized_reference_dataset


def feature_attribution(model: torch.nn, classes: int, origin_input: np.ndarray, reference_dataset: np.ndarray,
                        reference_num: int = 100, window_length: int = 5, M: int = 256,
                        reference_filter: bool = False, antithetic_variables: bool = False):
    assert len(origin_input.shape) == 2
    assert len(reference_dataset.shape) == 3
    channels, points = origin_input.shape
    assert 0 < window_length <= points
    assert points % window_length == 0  # 特征的大小可以被原始输入数据的大小整除
    assert len(reference_dataset) >= reference_num

    features_num, channel_list, point_start_list = feature_segment(channels, points, window_length)

    if reference_filter:
        reference_dataset = reference_dataset_filter(origin_input, reference_dataset, model, reference_num)
    else:
        # reference_dataset = reference_dataset[-reference_num:]
        reference_dataset = reference_dataset[np.random.randint(len(reference_dataset), size=reference_num)]

    if antithetic_variables:
        S1 = np.zeros((M * 2, channels, points), dtype=np.float32)
        S2 = np.zeros((M * 2, channels, points), dtype=np.float32)
    else:
        S1 = np.zeros((M, channels, points), dtype=np.float32)
        S2 = np.zeros((M, channels, points), dtype=np.float32)
    # S1 = np.zeros((M, channels, points), dtype=np.float32)
    # S2 = np.zeros((M, channels, points), dtype=np.float32)
    # if antithetic_variables:
    #     S1_anti = np.zeros((M, channels, points), dtype=np.float32)
    #     S2_anti = np.zeros((M, channels, points), dtype=np.float32)

    attribution_maps = np.zeros((channels, points, classes), dtype=np.float32)

    for feature in tqdm(range(features_num)):
        for m in range(M):
            # 从参考数据集中随机选择一个参考样本，用于替换不考虑的特征核;或者直接选择空数据集
            reference_input = reference_dataset[np.random.randint(reference_num)]

            feature_mark = np.random.randint(0, 2, features_num,
                                             dtype=np.bool)  # 直接生成0，1数组，最后确保feature位满足要求，并且将数据类型改为Boolean型减少后续矩阵点乘计算量
            # feature_mark = np.random.rand(features_num) > 0.5     # 在window_length为5，2，1时反而稍慢
            feature_mark[feature] = 0
            feature_mark = np.repeat(feature_mark, window_length)
            feature_mark = np.reshape(feature_mark, origin_input.shape)  # reshape是view，resize是copy
            # # 矩阵点乘计算不包含该特征的样本
            # S2[m] = feature_mark * origin_input + ~feature_mark * reference_input
            # # 生成包含该特征核的集合的替换矩阵
            # feature_mark[channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length] = 1
            # # 矩阵点乘计算包含该特征的样本
            # S1[m] = feature_mark * origin_input + ~feature_mark * reference_input

            S1[m] = S2[m] = feature_mark * origin_input + ~feature_mark * reference_input
            S1[m][channel_list[feature],
            point_start_list[feature]:point_start_list[feature] + window_length] = \
                origin_input[channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length]
            if antithetic_variables:
                S1[m + M] = S2[m + M] = ~feature_mark * origin_input + feature_mark * reference_input
                S2[m + M][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length] = \
                    reference_input[channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length]
            # if antithetic_variables:
            #     S1_anti[m] = S2_anti[m] = ~feature_mark * origin_input + feature_mark * reference_input
            #     S2_anti[m][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length] = reference_input[channel_list[feature], point_start_list[feature]: point_start_list[feature] + window_length]
        # 计算S1和S2的预测差值
        S1_preds, _ = predict(model, torch.from_numpy(S1))
        S2_preds, _ = predict(model, torch.from_numpy(S2))
        feature_weight = (S1_preds - S2_preds).sum(axis=0) / len(S1_preds)  # TODO: 计算中间采样结果并返回
        # if antithetic_variables:
        #     S1_anti_preds, _ = predict(model, torch.from_numpy(S1_anti))
        #     S2_anti_preds, _ = predict(model, torch.from_numpy(S2_anti))
        #     feature_weight = ((S1_anti_preds - S2_anti_preds).sum(axis=0) / len(S1_anti_preds) + feature_weight) / 2
        attribution_maps[channel_list[feature],
        point_start_list[feature]:point_start_list[feature] + window_length] = feature_weight

    return attribution_maps


def feature_attribution_parallel(model: torch.nn, classes: int, origin_input: np.ndarray, reference_dataset: np.ndarray,
                                 reference_num: int = 100, window_length: int = 5, M: int = 256,
                                 reference_filter: bool = False, antithetic_variables: bool = False):
    assert len(origin_input.shape) == 2
    assert len(reference_dataset.shape) == 3
    channels, points = origin_input.shape
    assert 0 < window_length <= points
    assert points % window_length == 0  # 特征的大小可以被原始输入数据的大小整除
    assert len(reference_dataset) >= reference_num

    features_num, channel_list, point_start_list = feature_segment(channels, points, window_length)

    if reference_filter:
        reference_dataset = reference_dataset_filter(origin_input, reference_dataset, model, reference_num)
    else:
        # reference_dataset = reference_dataset[-reference_num:]
        reference_dataset = reference_dataset[np.random.randint(len(reference_dataset), size=reference_num)]

    attribution_maps = np.zeros((channels, points, classes), dtype=np.float32)

    @ray.remote(num_gpus=0.0625)
    def feature_contribution(feature):
        # if antithetic_variables:
        #     S1 = np.zeros((M*2, channels, points), dtype=np.float32)
        #     S2 = np.zeros((M*2, channels, points), dtype=np.float32)
        # else:
        #     S1 = np.zeros((M, channels, points), dtype=np.float32)
        #     S2 = np.zeros((M, channels, points), dtype=np.float32)
        S1 = np.zeros((M, channels, points), dtype=np.float32)
        S2 = np.zeros((M, channels, points), dtype=np.float32)
        if antithetic_variables:
            S1_anti = np.zeros((M, channels, points), dtype=np.float32)
            S2_anti = np.zeros((M, channels, points), dtype=np.float32)
        for m in range(M):
            # 从参考数据集中随机选择一个参考样本，用于替换不考虑的特征核;或者直接选择空数据集
            reference_input = reference_dataset[np.random.randint(reference_num)]

            feature_mark = np.random.randint(0, 2, features_num, dtype=np.bool)  # 直接生成0，1数组，最后确保feature位满足要求
            # feature_mark = np.random.rand(features_num) > 0.5     # 在window_length为5，2，1时反而稍慢
            feature_mark[feature] = 0
            feature_mark = np.repeat(feature_mark, window_length)
            feature_mark = np.reshape(feature_mark, origin_input.shape)  # reshape是view，resize是copy
            # # 矩阵点乘计算不包含该特征的样本
            # S2[m] = feature_mark * origin_input + ~feature_mark * reference_input
            # # 生成包含该特征核的集合的替换矩阵
            # feature_mark[channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length] = 1
            # # 矩阵点乘计算包含该特征的样本
            # S1[m] = feature_mark * origin_input + ~feature_mark * reference_input

            S1[m] = S2[m] = feature_mark * origin_input + ~feature_mark * reference_input
            S1[m][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length] = \
                origin_input[channel_list[feature],
                point_start_list[feature]:point_start_list[feature] + window_length]
            # if antithetic_variables:
            #     S1[m+M] = S2[m+M] = ~feature_mark * origin_input + feature_mark * reference_input
            #     S2[m+M][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length] = reference_input[channel_list[feature], point_start_list[feature]: point_start_list[feature] + window_length]
            if antithetic_variables:
                S1_anti[m] = S2_anti[m] = ~feature_mark * origin_input + feature_mark * reference_input
                S2_anti[m][channel_list[feature],
                point_start_list[feature]:point_start_list[feature] + window_length] = \
                    reference_input[channel_list[feature],
                    point_start_list[feature]:point_start_list[feature] + window_length]
        # 计算S1和S2的预测差值
        S1_preds, _ = predict(model, torch.from_numpy(S1))
        S2_preds, _ = predict(model, torch.from_numpy(S2))
        feature_weight = (S1_preds - S2_preds).sum(axis=0) / len(S1_preds)
        if antithetic_variables:
            S1_anti_preds, _ = predict(model, torch.from_numpy(S1_anti))
            S2_anti_preds, _ = predict(model, torch.from_numpy(S2_anti))
            feature_weight = ((S1_anti_preds - S2_anti_preds).sum(axis=0) / len(S1_anti_preds) + feature_weight) / 2
        return feature, feature_weight

    rs = [feature_contribution.remote(feature) for feature in range(features_num)]
    rs_list = ray.get(rs)
    for feature, feature_weight in rs_list:
        attribution_maps[channel_list[feature],
        point_start_list[feature]:point_start_list[feature] + window_length] = feature_weight

    return attribution_maps


# 输入的模型和采样数为列表形式，即同时计算多个模型和多种采样的结果
# M_list应为递增形式
def feature_attribution_parallel_multi_models(model_list: list, classes: int, origin_input: np.ndarray,
                                              reference_dataset: np.ndarray,
                                              reference_num: int = 100, window_length: int = 5, M_list: list = [256],
                                              reference_filter: bool = False, antithetic_variables: bool = False):
    assert len(origin_input.shape) == 2
    assert len(reference_dataset.shape) == 3
    channels, points = origin_input.shape
    assert 0 < window_length <= points
    assert points % window_length == 0  # 特征的大小可以被原始输入数据的大小整除
    assert len(reference_dataset) >= reference_num

    features_num, channel_list, point_start_list = feature_segment(channels, points, window_length)

    if reference_filter:
        reference_dataset = reference_dataset_filter(origin_input, reference_dataset, model_list[0], reference_num)
    else:
        # reference_dataset = reference_dataset[-reference_num:]
        reference_dataset = reference_dataset[np.random.randint(len(reference_dataset), size=reference_num)]

    attribution_maps = np.zeros((channels, points, len(model_list), len(M_list), classes), dtype=np.float32)
    M_max = M_list[-1]

    @ray.remote(num_gpus=0.0625)
    def feature_contribution(feature):
        S1 = np.zeros((M_max, channels, points), dtype=np.float32)
        S2 = np.zeros((M_max, channels, points), dtype=np.float32)
        if antithetic_variables:
            S1_anti = np.zeros((M_max, channels, points), dtype=np.float32)
            S2_anti = np.zeros((M_max, channels, points), dtype=np.float32)
        for m in range(M_max):
            # 从参考数据集中随机选择一个参考样本，用于替换不考虑的特征核;或者直接选择空数据集
            reference_input = reference_dataset[np.random.randint(reference_num)]

            feature_mark = np.random.randint(0, 2, features_num, dtype=np.bool)  # 直接生成0，1数组，最后确保feature位满足要求
            # feature_mark = np.random.rand(features_num) > 0.5     # 在window_length为5，2，1时反而稍慢
            feature_mark[feature] = 0
            feature_mark = np.repeat(feature_mark, window_length)
            feature_mark = np.reshape(feature_mark, origin_input.shape)  # reshape是view，resize是copy

            S1[m] = S2[m] = feature_mark * origin_input + ~feature_mark * reference_input
            S1[m][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length] = \
                origin_input[channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length]
            if antithetic_variables:
                S1_anti[m] = S2_anti[m] = ~feature_mark * origin_input + feature_mark * reference_input
                S2_anti[m][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length] = \
                    reference_input[channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length]
        # 计算S1和S2的预测差值
        feature_weight = np.zeros((len(model_list), len(M_list), classes), dtype=np.float32)
        for model_id in range(len(model_list)):
            model = model_list[model_id]
            S1_preds, _ = predict(model, torch.from_numpy(S1))
            S2_preds, _ = predict(model, torch.from_numpy(S2))
            for M_id in range(len(M_list)):
                M = M_list[M_id]
                feature_weight[model_id, M_id] = (S1_preds[:M, :] - S2_preds[:M, :]).sum(axis=0) / M  # 计算中间采样结果并返回
            if antithetic_variables:
                S1_anti_preds, _ = predict(model, torch.from_numpy(S1_anti))
                S2_anti_preds, _ = predict(model, torch.from_numpy(S2_anti))
                for M_id in range(len(M_list)):
                    M = M_list[M_id]
                    feature_weight[model_id, M_id] = ((S1_anti_preds[:M, :] - S2_anti_preds[:M, :]).sum(axis=0) / M +
                                                      feature_weight[model_id, M_id]) / 2
        return feature, feature_weight

    rs = [feature_contribution.remote(feature) for feature in range(features_num)]
    rs_list = ray.get(rs)
    for feature, feature_weight in rs_list:
        attribution_maps[channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length] = feature_weight

    return attribution_maps


def deletion_test(model: torch.nn, attribution: AttributionResult,
                  deletion_max: int = 30, deletion_step: int = 1, deletion_baseline: float or str = 0):
    assert attribution.model_name == model.__class__.__name__

    pred_label = attribution.pred_label
    attribution_map = attribution.attribution_maps[:, :, attribution.pred_label]
    origin_input = attribution.origin_input

    if deletion_baseline == 'mean':
        deletion_baseline = origin_input.mean()

    sort_index = attribution_map.reshape(-1).argsort()[::-1]  # 将获取的升序索引反转为降维排列
    index_num = len(sort_index)
    # 根据权重排序依次计算删除1%采样点后的预测置信度
    delete_batch, deletion_percent_list = [], []
    deletion_percent = 0
    while deletion_percent <= deletion_max:
        threshold = int(index_num * deletion_percent / 100)  # deletion_percent改为整型，避免浮点数运算固有误差导致的采样数不足
        delete_points = sort_index[:threshold]
        delete_input = origin_input.copy().reshape(-1)
        delete_input[delete_points] = deletion_baseline
        delete_input = delete_input.reshape(origin_input.shape)
        # print(delete_input[delete_input == deletion_baseline].size)

        delete_batch.append(delete_input)
        deletion_percent_list.append(deletion_percent / 100)

        deletion_percent += deletion_step

    # 将delete_batch处理为torch形式，并进行预测
    delete_batch = torch.from_numpy(np.array(delete_batch))
    delete_predictions, delete_pred_labels = predict(model, delete_batch)

    auc = metrics.auc(deletion_percent_list, delete_predictions[:, pred_label].squeeze())

    return deletion_percent_list, delete_predictions, delete_pred_labels, auc


def deletion_channel_test(model: torch.nn, attribution: AttributionResult,
                          deletion_max=30, deletion_step=1, deletion_baseline=0):
    assert attribution.model_name == model.__class__.__name__

    pred_label = attribution.pred_label
    attribution_channel = attribution.attribution_maps[:, :, attribution.pred_label].sum(axis=1)
    origin_input = attribution.origin_input
    channels = attribution.channels

    if deletion_baseline == 'mean':
        deletion_baseline = origin_input.mean()

    sort_index = attribution_channel.argsort()[::-1]  # 将获取的升序索引反转为降维排列
    # 根据权重排序依次计算删除1%采样点后的预测置信度
    delete_batch, deletion_percent_list = [], []
    deletion_percent = 0
    while deletion_percent <= deletion_max:
        threshold = int(channels * deletion_percent / 100)  # deletion_percent改为整型，避免浮点数运算固有误差导致的采样数不足
        delete_points = sort_index[:threshold]
        delete_input = origin_input.copy()
        delete_input[delete_points, :] = deletion_baseline

        delete_batch.append(delete_input)
        deletion_percent_list.append(deletion_percent / 100)

        deletion_percent += deletion_step

    # 将delete_batch处理为torch形式，并进行预测
    delete_batch = torch.from_numpy(np.array(delete_batch))
    delete_predictions, delete_pred_labels = predict(model, delete_batch)

    auc = metrics.auc(deletion_percent_list, delete_predictions[:, pred_label].squeeze())

    return deletion_percent_list, delete_predictions, delete_pred_labels, auc


def generate_plot(model: torch.nn, sample_explanation, channels_info):
    """
    input:
        model
        sample_explanation
        channels_info
    """
    heatmap = sample_explanation.attribution_maps[:, :, sample_explanation.pred_label]
    origin_input = sample_explanation.origin_input
    channels = sample_explanation.channels
    points = sample_explanation.points
    title = 'Dataset: {}    Sample ID: {}    Label: {}\nModel: {}'.format(
        sample_explanation.dataset, sample_explanation.sample_id,
        sample_explanation.label_names[sample_explanation.truth_label], sample_explanation.model_name)
    for i in range(len(sample_explanation.label_names)):
        title += '    $P_{{{}}}={:.4f}$'.format(sample_explanation.label_names[i], sample_explanation.pred[i])
    title += '\nInterpretation: {}'.format(sample_explanation.attribution_method)

    heatmap_channel = heatmap.sum(axis=1)
    heatmap_time = heatmap.sum(axis=0)
    heatmap = (heatmap - np.mean(heatmap)) / (np.std(heatmap))
    heatmap_channel = (heatmap_channel - np.mean(heatmap_channel)) / (np.std(heatmap_channel))
    heatmap_time = (heatmap_time - np.mean(heatmap_time)) / (np.std(heatmap_time))

    fig = plt.figure(figsize=(12, 12))
    gridlayout = gridspec.GridSpec(ncols=20, nrows=8, figure=fig, top=0.92, wspace=0.1, hspace=0.1)
    axs0 = fig.add_subplot(gridlayout[:, :8])
    axs1 = fig.add_subplot(gridlayout[:6, 8:])
    axs2 = fig.add_subplot(gridlayout[6:, 10:19])
    # axs3 = fig.add_subplot(gridlayout[5:7, 1])
    # axs4 = fig.add_subplot(gridlayout[7, 1])

    fontsize = 26
    linewidth = 2
    cmap = 'plasma'  # 'viridis' 'plasma' 'inferno' 'magma' 'RdBu_r' 'bwr'

    time_xticks = [0, 25, 50, 75, 100]
    time_xticklabels = ['-0.2', '0', '0.2', '0.4', '0.6(s)']

    fig.suptitle(title, y=0.99, fontsize=fontsize)

    # 绘制时间曲线图
    thespan = np.percentile(origin_input, 98)
    xx = np.arange(1, points + 1)

    for channel in range(channels):
        y = origin_input[channel, :] + thespan * (channels - 1 - channel)
        dydx = heatmap[channel, :]

        img_points = np.array([xx, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([img_points[:-1], img_points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(-1, 1), linewidths=(1,))
        lc.set_array(dydx)
        axs0.add_collection(lc)

    axs0.set_xlim([0, points + 1])
    axs0.set_xticks(time_xticks)
    axs0.set_xticklabels(time_xticklabels, fontsize=fontsize)
    axs0.set_xlabel('Time', fontsize=fontsize)
    axs0.set_title("(a)Contribution Map", fontsize=fontsize)

    inversechannelnames = []
    for channel in range(channels):
        inversechannelnames.append(channels_info.ch_names[channels - 1 - channel])

    yttics = np.zeros(channels)
    for gi in range(channels):
        yttics[gi] = gi * thespan

    axs0.set_ylim([-thespan, thespan * channels])
    plt.sca(axs0)
    plt.yticks(yttics, inversechannelnames, fontsize='x-small')

    # 绘制地形图
    mne.viz.plot_topomap(heatmap_channel, channels_info, ch_type='grad', cmap=cmap, axes=axs1, outlines='head',
                         show=False, names=channels_info.ch_names[::2])
    axs1.set_title("(b)Channel Contribution\n(Topomap)", y=0.9, fontsize=fontsize)

    # 绘制时间贡献曲线
    xx = np.arange(1, points + 1)
    img_points = np.array([xx, heatmap_time]).T.reshape(-1, 1, 2)
    segments = np.concatenate([img_points[:-1], img_points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidths=(linewidth+1,))
    lc.set_array(heatmap_time)
    axs2.set_title("(c)Time Contribution", fontsize=fontsize)
    axs2.add_collection(lc)
    axs2.set_ylim(floor(heatmap_time.min()), ceil(heatmap_time.max()))
    axs2.set_xticks(time_xticks)
    axs2.set_xticklabels(time_xticklabels, fontsize=fontsize)
    axs2.set_ylabel('Contribution', fontsize=fontsize)
    axs2.set_xlabel('Time', fontsize=fontsize)

    # # 绘制deletion test折线图
    # # get the results of the deletion test
    # deletion_max, deletion_step, deletion_baseline = 30, 1, 0
    # percents, preds, preds_labels, auc = deletion_test(model, sample_explanation, deletion_max, deletion_step,
    #                                                    deletion_baseline)
    # axs3.plot(percents, preds[:, explanation_label], color='r', linewidth=linewidth, label='Deletion Test(point)')
    #
    # percents_cha, preds_cha, preds_labels_cha, auc_cha = \
    #     deletion_channel_test(model, sample_explanation, deletion_max, deletion_step, deletion_baseline)
    # axs3.plot(percents_cha, preds_cha[:, explanation_label], color='g', linewidth=linewidth,
    #           label='Deletion Test(channel)')
    # axs3.legend(fontsize=fontsize)
    #
    # axs4.xaxis.set_ticks([])
    # axs4.yaxis.set_ticks([])
    # axs4.spines['top'].set_visible(False)
    # axs4.spines['right'].set_visible(False)
    # axs4.spines['bottom'].set_visible(False)
    # axs4.spines['left'].set_visible(False)
    #
    # axs4.text(0.01, 0.9, 'Top-10 contribution channels:', horizontalalignment='left', fontsize=fontsize)
    # top_channel_names = compute_top_channel_names(heatmap_channel, channels_info.ch_names, 10)
    # axs4.text(0.01, 0.75, '    ' + top_channel_names[:int(len(top_channel_names) / 2)], horizontalalignment='left',
    #           fontsize=fontsize)
    # axs4.text(0.01, 0.6, '    ' + top_channel_names[int(len(top_channel_names) / 2):], horizontalalignment='left',
    #           fontsize=fontsize)
    # axs4.text(0.01, 0.45, 'Top-10 contribution points:'.format(compute_top_point_names(heatmap_time, 10)),
    #           horizontalalignment='left', fontsize=fontsize)
    # axs4.text(0.01, 0.3, '    ' + compute_top_point_names(heatmap_time, 10), horizontalalignment='left',
    #           fontsize=fontsize)
    # axs4.text(0.01, 0.15, 'AUDC(point): {:.6f}'.format(auc), horizontalalignment='left', fontsize=fontsize)
    # axs4.text(0.01, 0, 'AUDC(channel): {:.6f}'.format(auc_cha), horizontalalignment='left', fontsize=fontsize)

    plt.show()
    return fig


def topomap_plot(sample_explanation, channels_info):
    heatmap = sample_explanation.attribution_maps[:, :, sample_explanation.pred_label]
    heatmap_channel = heatmap.sum(axis=1)

    # heatmap_channel = (heatmap_channel - np.mean(heatmap_channel)) / (np.std(heatmap_channel))

    fig, axs = plt.subplots(figsize=(16, 16))
    cmap = 'plasma'  # 'viridis' 'plasma' 'inferno' 'magma' 'RdBu_r' 'bwr'

    top_num = 10
    top_index = np.argsort(-heatmap_channel)[:top_num]
    top_channels = np.zeros(len(heatmap_channel), dtype=bool)
    top_channels[top_index] = True

    for i in range(top_num):
        print('top-{}'.format(i), channels_info.ch_names[top_index[i]], heatmap_channel[top_index[i]])

    # 绘制地形图
    mne.viz.plot_topomap(heatmap_channel, channels_info, ch_type='grad', cmap=cmap, axes=axs,
                         show=False, names=channels_info.ch_names[::2])
    return fig


def class_mean_plot(sample_explanations, channels_info, top_channels=10, attribution_method=None):
    assert isinstance(sample_explanations, list)
    sample_explanation = sample_explanations[0]
    channels = sample_explanation.channels
    points = sample_explanation.points
    title = 'Dataset: {}   Label: {}    Model: {}'.format(
        sample_explanation.dataset, sample_explanation.label_names[sample_explanation.truth_label],
        sample_explanation.model_name)
    if attribution_method is None:
        attribution_method = sample_explanation.attribution_method
    title += '\nInterpretation: {}'.format(attribution_method)

    heatmap_list = []
    origin_input_list = []  # 绘制归因贡献图的原始数据需要平均
    for sample_explanation in sample_explanations:
        heatmap_list.append(sample_explanation.attribution_maps[:, :, sample_explanation.pred_label])
        origin_input_list.append(sample_explanation.origin_input)
    heatmap_list = np.array(heatmap_list)
    origin_input_list = np.array(origin_input_list)

    origin_input = origin_input_list.mean(axis=0)
    heatmap = heatmap_list.mean(axis=0)
    heatmap_channel = heatmap.mean(axis=1)
    heatmap_time = heatmap.mean(axis=0)
    heatmap = (heatmap - np.mean(heatmap)) / (np.std(heatmap))
    heatmap_channel = (heatmap_channel - np.mean(heatmap_channel)) / (np.std(heatmap_channel))
    heatmap_time = (heatmap_time - np.mean(heatmap_time)) / (np.std(heatmap_time))

    # 计算地形图中需要突出显示的通道及名称，注意：由于在绘制地形图时两两合并为一个位置，需要保证TOP通道的名称一定显示，其余通道对显示第一个通道的名称
    mask_list = np.zeros(channels//2, dtype=bool)   # 由于通道类型为Grad，在绘制地形图时两两合并为一个位置
    top_channel_index = np.argsort(-heatmap_channel)[:top_channels]
    names_list = []     # 两两合并后对应的通道名称
    for channel_index in range(channels//2):
        if 2*channel_index in top_channel_index:
            mask_list[channel_index] = True
            names_list.append(channels_info.ch_names[2 * channel_index] + '\n')     # 避免显示标记遮挡通道名称
            if 2 * channel_index + 1 in top_channel_index:
                names_list[channel_index] += channels_info.ch_names[2 * channel_index+1] + '\n\n'
        elif 2*channel_index+1 in top_channel_index:
            mask_list[channel_index] = True
            names_list.append(channels_info.ch_names[2 * channel_index+1] + '\n')
        else:
            names_list.append(channels_info.ch_names[2*channel_index])

    # 打印TOP通道及其名称、贡献值
    print("index\tchannel name\tcontribution value")
    id = 0
    for index in top_channel_index:
        print(id, index, channels_info.ch_names[index], heatmap_channel[index])
        id += 1

    fig = plt.figure(figsize=(12, 12))
    gridlayout = gridspec.GridSpec(ncols=48, nrows=12, figure=fig, top=0.92, wspace=None, hspace=0.2)
    axs0 = fig.add_subplot(gridlayout[:, :20])
    axs1 = fig.add_subplot(gridlayout[:9, 20:47])
    axs1_colorbar = fig.add_subplot(gridlayout[2:8, 47])
    axs2 = fig.add_subplot(gridlayout[9:, 24:47])

    fontsize = 16
    linewidth = 2
    # 配色方案
    # 贡献由大到小颜色由深变浅：'plasma' 'viridis'
    # 有浅变深：'summer' 'YlGn' 'YlOrRd'
    # 'Oranges'
    cmap = 'Oranges'
    plt.rcParams['font.size'] = fontsize
    time_xticks = [0, 25, 50, 75, 100]
    time_xticklabels = ['-0.2', '0', '0.2', '0.4', '0.6(s)']

    fig.suptitle(title, y=0.99, fontsize=fontsize)

    # 绘制时间曲线图
    thespan = np.percentile(origin_input, 98)
    xx = np.arange(1, points + 1)

    for channel in range(channels):
        y = origin_input[channel, :] + thespan * (channels - 1 - channel)
        dydx = heatmap[channel, :]

        img_points = np.array([xx, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([img_points[:-1], img_points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(-1, 1), linewidths=(1,))
        lc.set_array(dydx)
        axs0.add_collection(lc)

    axs0.set_xlim([0, points + 1])
    axs0.set_xticks(time_xticks)
    axs0.set_xticklabels(time_xticklabels, fontsize=fontsize)
    axs0.set_xlabel('Time', fontsize=fontsize)
    axs0.set_title("(a)Contribution Map", fontsize=fontsize)

    inversechannelnames = []
    for channel in range(channels):
        inversechannelnames.append(channels_info.ch_names[channels - 1 - channel])

    yttics = np.zeros(channels)
    for gi in range(channels):
        yttics[gi] = gi * thespan

    axs0.set_ylim([-thespan, thespan * channels])
    plt.sca(axs0)
    plt.yticks(yttics, inversechannelnames, fontsize=fontsize//3)

    # 绘制地形图
    # 地形图中TOP通道的显示参数
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=4)
    mne.viz.plot_topomap(heatmap_channel, channels_info, ch_type='grad', cmap=cmap, axes=axs1, outlines='head',
                         show=False, names=names_list, mask=mask_list, mask_params=mask_params)
    axs1.set_title("(b)Channel Contribution\n(Topomap)", y=0.9, fontsize=fontsize)
    # 设置颜色条带
    norm = colors.Normalize(vmin=heatmap_channel.min(), vmax=heatmap_channel.max())
    colorbar.ColorbarBase(axs1_colorbar, cmap=cmap, norm=norm)

    # 绘制时间贡献曲线
    xx = np.arange(1, points + 1)
    img_points = np.array([xx, heatmap_time]).T.reshape(-1, 1, 2)
    segments = np.concatenate([img_points[:-1], img_points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidths=(linewidth+1,))
    lc.set_array(heatmap_time)
    axs2.set_title("(c)Time Contribution", fontsize=fontsize)
    axs2.add_collection(lc)
    axs2.set_ylim(floor(heatmap_time.min()), ceil(heatmap_time.max()))
    axs2.set_xticks(time_xticks)
    axs2.set_xticklabels(time_xticklabels, fontsize=fontsize)
    axs2.set_ylabel('Contribution', fontsize=fontsize)
    axs2.set_xlabel('Time', fontsize=fontsize)
    axs2.patch.set_facecolor('lightgreen')

    plt.show()
    return fig, heatmap, heatmap_channel, heatmap_time


def compute_top_channel_names(heatmap_channel, ch_names, top=5):
    top_index = np.argsort(-heatmap_channel)[:top]
    top_names = ''
    for i in range(top):
        top_names += ch_names[top_index[i]] + ' '
    return top_names


def compute_top_point_names(heatmap_time, top=5):
    top_index = np.argsort(-heatmap_time)[:top]
    top_names = ''
    for i in range(top):
        top_names += str(top_index[i]) + ' '
    return top_names
