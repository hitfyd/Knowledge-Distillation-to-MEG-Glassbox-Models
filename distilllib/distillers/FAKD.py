import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def NMSE(input_data, target):
    # input_data = F.normalize(input_data, p=2)
    # target = F.normalize(target, p=2)
    nmse_loss = F.mse_loss(input_data, target, reduction="mean")
    return nmse_loss

def SingleChannel(origin_input, model):
    channels, points = origin_input.size()
    origin_input = origin_input.cpu().numpy()
    # 初始化扰动数据，生成样本数等于通道数
    perturbation_data = np.zeros((channels, channels, points), dtype=np.float32)
    # 填充生成扰动数据
    for channel in range(channels):
        perturbation_data[channel, channel, :] = origin_input[channel, :]
    # 计算每个通道的权重值
    features = model(torch.from_numpy(perturbation_data).cuda())
    return features.view(-1)


def feature_segment(channels, points, window_length):
    channel_windows_num = int(points / window_length)  # 需要计算的通道特征数和时间特征数，总特征数为c_features x p_features
    features_num = channels * channel_windows_num
    channel_list, point_start_list = [], []
    for feature_id in range(features_num):
        channel_list.append(int(feature_id / channel_windows_num))
        point_start_list.append(int(feature_id % channel_windows_num * window_length))
    return features_num, channel_list, point_start_list


def feature_attribution(origin_input, model, num_classes: int, reference_dataset, window_length: int = 5, M: int = 8):
    assert len(origin_input.size()) == 2
    assert len(reference_dataset.size()) == 3
    channels, points = origin_input.size()
    assert 0 < window_length <= points
    assert points % window_length == 0  # 特征的大小可以被原始输入数据的大小整除

    features_num, channel_list, point_start_list = feature_segment(channels, points, window_length)

    S1 = torch.zeros((M, channels, points), dtype=torch.float32).cuda()
    S2 = torch.zeros((M, channels, points), dtype=torch.float32).cuda()

    attribution_maps = torch.zeros((channels, num_classes), dtype=torch.float32).cuda()

    for feature in range(features_num):
        for m in range(M):
            # 从参考数据集中随机选择一个参考样本，用于替换不考虑的特征核;或者直接选择空数据集
            reference_input = reference_dataset[np.random.randint(len(reference_dataset))]

            feature_mark = np.random.randint(0, 2, features_num, dtype=np.bool_)  # 直接生成0，1数组，最后确保feature位满足要求，并且将数据类型改为Boolean型减少后续矩阵点乘计算量
            feature_mark[feature] = 0
            feature_mark = np.repeat(feature_mark, window_length)
            feature_mark = np.reshape(feature_mark, origin_input.size())  # reshape是view，resize是copy
            feature_mark = torch.from_numpy(feature_mark).cuda()

            S1[m] = S2[m] = feature_mark * origin_input + ~feature_mark * reference_input
            S1[m][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length] = \
                origin_input[channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length]
        # 计算S1和S2的预测差值
        S1_preds = model(S1)
        S2_preds = model(S2)
        feature_weight = (S1_preds - S2_preds).sum(axis=0) / len(S1_preds)  # TODO: 计算中间采样结果并返回
        attribution_maps[channel_list[feature]] = feature_weight

    return attribution_maps.view(-1)


def fakd_loss(data, num_classes, student, teacher):
    batch_size, channels, points = data.size()
    num_features = channels * num_classes
    features_student = torch.zeros((batch_size, num_features), dtype=torch.float32).cuda()
    features_teacher = torch.zeros((batch_size, num_features), dtype=torch.float32).cuda()

    # import sys
    # import time
    # from line_profiler import line_profiler
    #
    # func = SingleChannel
    # time_start = time.time()
    # prof = line_profiler.LineProfiler(func)  # 把函数传递到性能分析器中
    # prof.enable()  # 开始性能分析
    for i in range(batch_size):
        features_student[i] = SingleChannel(data[i], student)
        features_teacher[i] = SingleChannel(data[i], teacher)

        # reference_dataset = data[random.sample(range(batch_size), 10)]
        # features_student[i] = feature_attribution(data[i], student, num_classes, reference_dataset, points)
        # features_teacher[i] = feature_attribution(data[i], teacher, num_classes, reference_dataset, points)

    # prof.disable()  # 停止性能分析
    # prof.print_stats(sys.stdout)  # 打印性能分析结果
    # prof.print_stats(open('line_profiler', 'w'))  # 打印性能分析结果
    # time_end = time.time()  # 记录结束时间
    # run_time = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    # print(run_time)

    loss_fakd = NMSE(features_student, features_teacher)
    return loss_fakd


def sc_fakd_loss(data, num_classes, student, teacher):
    batch_size, channels, points = data.size()
    data = data.cpu().numpy()
    # 初始化扰动数据，生成样本数等于通道数
    perturbation_data = np.zeros((batch_size, channels, channels, points), dtype=np.float32)
    # 填充生成扰动数据
    for channel in range(channels):
        perturbation_data[:, channel, channel, :] = data[:, channel, :]
    # 计算每个通道的权重值
    with torch.no_grad():
        features_student = student(torch.from_numpy(perturbation_data.reshape(batch_size*channels, channels, points)).cuda())
        features_teacher = teacher(torch.from_numpy(perturbation_data.reshape(batch_size*channels, channels, points)).cuda())
        # if teacher.__class__.__name__ == 'HGRN':
        #     features_student = F.log_softmax(features_student)
        loss_fakd = NMSE(features_student, features_teacher)
        # if teacher.__class__.__name__ == 'HGRN':
        #     loss_fakd /= 100
    return loss_fakd


class FAKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(FAKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.FAKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.FAKD.LOSS.KD_WEIGHT
        self.num_classes = cfg.DATASET.NUM_CLASSES

    def forward_train(self, data, target, **kwargs):
        logits_student, penalty = self.student(data, is_training_data=True)
        with torch.no_grad():
            logits_teacher = self.teacher(data)

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student, target) + penalty)
        loss_kd = self.kd_loss_weight * fakd_loss(data, self.num_classes, self.student, self.teacher)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
