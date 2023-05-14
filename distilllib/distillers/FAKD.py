import gc
import random
from sys import getsizeof

import numpy as np
import torch
import torch.nn.functional as F

from ._base import Distiller


def NMSE(input_data, target):
    input_data = F.normalize(input_data, p=2)
    target = F.normalize(target, p=2)
    nmse_loss = F.mse_loss(input_data, target, reduction="mean")
    return nmse_loss


def feature_segment(channels, points, window_length):
    channel_windows_num = int(points / window_length)  # 需要计算的通道特征数和时间特征数，总特征数为c_features x p_features
    features_num = channels * channel_windows_num
    channel_list, point_start_list = [], []
    for feature_id in range(features_num):
        channel_list.append(int(feature_id / channel_windows_num))
        point_start_list.append(int(feature_id % channel_windows_num * window_length))
    return features_num, channel_list, point_start_list


def feature_attribution(origin_input, model, num_classes: int, reference_dataset, window_length: int = 5, M: int = 4):
    assert len(origin_input.shape) == 2
    assert len(reference_dataset.shape) == 3
    channels, points = origin_input.shape
    assert 0 < window_length <= points
    assert points % window_length == 0  # 特征的大小可以被原始输入数据的大小整除

    features_num, channel_list, point_start_list = feature_segment(channels, points, window_length)

    S1 = np.zeros((M, channels, points), dtype=np.float32)
    S2 = np.zeros((M, channels, points), dtype=np.float32)

    attribution_maps = torch.zeros((channels, num_classes), dtype=torch.float32)

    for feature in range(features_num):
        for m in range(M):
            # 从参考数据集中随机选择一个参考样本，用于替换不考虑的特征核;或者直接选择空数据集
            reference_input = reference_dataset[np.random.randint(len(reference_dataset))]

            feature_mark = np.random.randint(0, 2, features_num, dtype=np.bool_)  # 直接生成0，1数组，最后确保feature位满足要求，并且将数据类型改为Boolean型减少后续矩阵点乘计算量
            feature_mark[feature] = 0
            feature_mark = np.repeat(feature_mark, window_length)
            feature_mark = np.reshape(feature_mark, (channels, points))  # reshape是view，resize是copy

            S1[m] = S2[m] = feature_mark * origin_input + ~feature_mark * reference_input
            S1[m][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length] = \
                origin_input[channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length]
        # 计算S1和S2的预测差值
        S1_preds = model(torch.from_numpy(S1).cuda())
        S2_preds = model(torch.from_numpy(S2).cuda())
        feature_weight = (S1_preds - S2_preds).sum(axis=0) / len(S1_preds)  # TODO: 计算中间采样结果并返回
        attribution_maps[channel_list[feature]] = feature_weight

    return attribution_maps


S1_list = []
S2_list = []


def shapley_fakd_loss(data, num_classes, student, teacher, epoch, data_itx):
    current_device = torch.cuda.current_device()
    devices_num = torch.cuda.device_count()
    batch_size, channels, points = data.size()
    num_features = channels * num_classes
    window_length = points
    M = 8
    features_num, channel_list, point_start_list = feature_segment(channels, points, window_length)
    # if data_itx >=10:
    #     return torch.zeros(1).cuda()
    if epoch >= 1:
        data = data.cpu().numpy()
        reference_dataset = data[random.sample(range(batch_size), int(batch_size/2))]
        # features_student = torch.zeros((batch_size, num_features), dtype=torch.float32).cuda()
        # features_teacher = torch.zeros((batch_size, num_features), dtype=torch.float32).cuda()
        #
        # for i in range(batch_size):
        #     features_student[i] = feature_attribution(data[i], student, num_classes, reference_dataset, points)
        #     features_teacher[i] = feature_attribution(data[i], teacher, num_classes, reference_dataset, points)

        S1 = np.zeros((batch_size, features_num, M, channels, points), dtype=np.float32)
        S2 = np.zeros((batch_size, features_num, M, channels, points), dtype=np.float32)

        for feature in range(features_num):
            for m in range(M):
                # 从参考数据集中随机选择一个参考样本，用于替换不考虑的特征核;或者直接选择空数据集
                reference_input = reference_dataset[np.random.randint(len(reference_dataset))]

                feature_mark = np.random.randint(0, 2, features_num, dtype=np.bool_)  # 直接生成0，1数组，最后确保feature位满足要求，并且将数据类型改为Boolean型减少后续矩阵点乘计算量
                feature_mark[feature] = 0
                feature_mark = np.repeat(feature_mark, window_length)
                feature_mark = np.reshape(feature_mark, (channels, points))  # reshape是view，resize是copy

                for index in range(batch_size):
                    S1[index, feature, m] = S2[:, feature, m] = feature_mark * data[index] + ~feature_mark * reference_input
                    S1[index, feature, m][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length] = \
                        data[index][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length]
        # 计算S1和S2的预测差值
        S1 = S1.reshape(-1, channels, points)
        S2 = S2.reshape(-1, channels, points)
        with torch.no_grad():
            S1_preds = student(torch.from_numpy(S1).cuda())
            S2_preds = student(torch.from_numpy(S2).cuda())
            features_student = (S1_preds.view(batch_size, features_num, M, -1) - S2_preds.view(batch_size, features_num, M, -1)).sum(axis=(2)) / M
            S1_preds = teacher(torch.from_numpy(S1).cuda())
            S2_preds = teacher(torch.from_numpy(S2).cuda())
            features_teacher = (S1_preds.view(batch_size, features_num, M, -1) - S2_preds.view(batch_size, features_num, M, -1)).sum(axis=(2)) / M
            loss_fakd = NMSE(features_student, features_teacher)

        # S1_list.append(S1)
        # S2_list.append(S2)
        # features_teacher_list.append(features_teacher)
    else:
        list_index = data_itx*devices_num+current_device
        with torch.no_grad():
            S1_preds = student(torch.from_numpy(S1_list[list_index]).cuda())
            S2_preds = student(torch.from_numpy(S2_list[list_index]).cuda())
            features_student = (S1_preds.view(batch_size, features_num, M, -1) - S2_preds.view(batch_size, features_num, M, -1)).sum(axis=(2)) / M
            loss_fakd = NMSE(features_student, features_teacher_list[list_index].cuda(current_device))
    return loss_fakd


perturbation_data_list = []
features_teacher_list = []



# 不同归因算法的不同在于扰动数据的生成方法不同，这里将所有样本对应的扰动数据只在第一轮生成一次，其余轮不再重新生成；教师模型的归因特征矩阵也只在第一轮计算
def sc_fakd_loss(data, num_classes, student, teacher, epoch, data_itx):
    current_device = torch.cuda.current_device()
    devices_num = torch.cuda.device_count()
    batch_size, channels, points = data.size()
    if epoch == 1:
        data = data.cpu().numpy()
        # 初始化扰动数据，生成样本数等于通道数
        perturbation_data = np.zeros((batch_size, channels, channels, points), dtype=np.float32)
        # 填充生成扰动数据
        for channel in range(channels):
            perturbation_data[:, channel, channel, :] = data[:, channel, :]
        perturbation_data = perturbation_data.reshape(batch_size * channels, channels, points)
        # 计算每个通道的权重值
        with torch.no_grad():
            features_student = student(torch.from_numpy(perturbation_data).cuda())
            features_teacher = teacher(torch.from_numpy(perturbation_data).cuda())
            loss_fakd = NMSE(features_student.view(batch_size, channels, -1), features_teacher.view(batch_size, channels, -1))

        perturbation_data_list.append(perturbation_data)
        features_teacher_list.append(features_teacher)
    else:
        list_index = data_itx*devices_num+current_device
        with torch.no_grad():
            features_student = student(torch.from_numpy(perturbation_data_list[list_index]).cuda())
            loss_fakd = NMSE(features_student.view(batch_size, channels, -1),
                             features_teacher_list[list_index].view(batch_size, channels, -1).cuda(current_device))
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
        loss_kd = self.kd_loss_weight * shapley_fakd_loss(data, self.num_classes, self.student, self.teacher, kwargs["epoch"], kwargs["data_itx"])
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
