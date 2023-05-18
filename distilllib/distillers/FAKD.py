import numpy as np
import torch
import torch.nn.functional as F

from .KD import kd_loss
from ._base import Distiller
from ..engine.utils import predict


# fa_loss的值太小了：1、改成归一化后再对比；2、直接使用KL散度
def fa_loss(input_data, target):
    input_data = F.normalize(input_data, p=2)
    target = F.normalize(target, p=2)
    loss_kd = F.mse_loss(input_data, target, reduction="mean")
    return loss_kd


def feature_segment(channels, points, window_length):
    channel_windows_num = int(points / window_length)  # 需要计算的通道特征数和时间特征数，总特征数为c_features x p_features
    features_num = channels * channel_windows_num
    channel_list, point_start_list = [], []
    for feature_id in range(features_num):
        channel_list.append(int(feature_id / channel_windows_num))
        point_start_list.append(int(feature_id % channel_windows_num * window_length))
    return features_num, channel_list, point_start_list


S1_list = []
S2_list = []


def shapley_fakd_loss(data, student, teacher, **kwargs):
    epoch, data_itx = kwargs["epoch"], kwargs["data_itx"]
    current_device = torch.cuda.current_device()
    devices_num = torch.cuda.device_count()
    batch_size, channels, points = data.size()
    window_length = points
    M = 4
    features_num, channel_list, point_start_list = feature_segment(channels, points, window_length)
    if epoch == 1:
        data = data.cpu().numpy()

        S1 = np.zeros((batch_size, features_num, M, channels, points), dtype=np.float16)
        S2 = np.zeros((batch_size, features_num, M, channels, points), dtype=np.float16)

        for feature in range(features_num):
            for m in range(M):
                for index in range(batch_size):
                    feature_mark = np.random.randint(0, 2, features_num, dtype=np.bool_)  # 直接生成0，1数组，最后确保feature位满足要求，并且将数据类型改为Boolean型减少后续矩阵点乘计算量
                    feature_mark[feature] = 0
                    feature_mark = np.repeat(feature_mark, window_length)
                    feature_mark = np.reshape(feature_mark, (channels, points))  # reshape是view，resize是copy
                    # 随机选择一个参考样本，用于替换不考虑的特征核
                    reference_index = (index + np.random.randint(1, batch_size)) % batch_size
                    assert index != reference_index # 参考样本不能是样本本身
                    reference_input = data[reference_index]
                    S1[index, feature, m] = S2[index, feature, m] = feature_mark * data[index] + ~feature_mark * reference_input
                    S1[index, feature, m][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length] = \
                        data[index][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length]

        # 计算S1和S2的预测差值
        S1 = S1.reshape(-1, channels, points)
        S2 = S2.reshape(-1, channels, points)
        S1_student_preds = predict(student, S1)
        S2_student_preds = predict(student, S2)
        features_student = (S1_student_preds.view(batch_size, features_num, M, -1) -
                            S2_student_preds.view(batch_size, features_num, M, -1)).sum(axis=(2)) / M
        S1_preds = predict(teacher, S1, eval=True)
        S2_preds = predict(teacher, S2, eval=True)
        features_teacher = (S1_preds.view(batch_size, features_num, M, -1) - S2_preds.view(batch_size, features_num, M,
                                                                                           -1)).sum(axis=(2)) / M
        loss_fakd = fa_loss(features_student, features_teacher)

        S1_list.append(S1)
        S2_list.append(S2)
        features_teacher_list.append(features_teacher)
    else:
        list_index = data_itx*devices_num+current_device
        S1_student_preds = predict(student, S1_list[list_index])
        S2_student_preds = predict(student, S2_list[list_index])
        features_student = (S1_student_preds.view(batch_size, features_num, M, -1) -
                            S2_student_preds.view(batch_size, features_num, M, -1)).sum(axis=(2)) / M
        loss_fakd = fa_loss(features_student, features_teacher_list[list_index].cuda(current_device))
    return loss_fakd


perturbation_data_list = []
features_teacher_list = []



# 不同归因算法的不同在于扰动数据的生成方法不同，这里将所有样本对应的扰动数据只在第一轮生成一次，其余轮不再重新生成；教师模型的归因特征矩阵也只在第一轮计算
def sc_fakd_loss(data, student, teacher, **kwargs):
    epoch, data_itx = kwargs["epoch"], kwargs["data_itx"]
    current_device = torch.cuda.current_device()
    devices_num = torch.cuda.device_count()
    batch_size, channels, points = data.size()
    if epoch == 1:
        data = data.cpu().numpy()
        # 初始化扰动数据，生成样本数等于通道数
        perturbation_data = np.zeros((batch_size, channels, channels, points), dtype=np.float16)
        # 填充生成扰动数据
        for channel in range(channels):
            perturbation_data[:, channel, channel, :] = data[:, channel, :]
        perturbation_data = perturbation_data.reshape(batch_size * channels, channels, points)
        # 计算每个通道的权重值
        features_student = predict(student, perturbation_data)
        features_teacher = predict(teacher, perturbation_data, eval=True)
        loss_fakd = fa_loss(features_student.view(batch_size, channels, -1), features_teacher.view(batch_size, channels, -1))

        perturbation_data_list.append(perturbation_data)
        features_teacher_list.append(features_teacher)
    else:
        list_index = data_itx*devices_num+current_device
        features_student = predict(student, perturbation_data_list[list_index])
        loss_fakd = fa_loss(features_student.view(batch_size, channels, -1),
                            features_teacher_list[list_index].view(batch_size, channels, -1).cuda(current_device))
    return loss_fakd


class FAKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(FAKD, self).__init__(student, teacher)
        self.temperature = cfg.FAKD.TEMPERATURE
        self.ce_loss_weight = cfg.FAKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.FAKD.LOSS.KD_WEIGHT
        self.fa_loss_weight = cfg.FAKD.LOSS.FA_WEIGHT

    def forward_train(self, data, target, **kwargs):
        logits_student, penalty = self.student(data, is_training_data=True)
        with torch.no_grad():
            logits_teacher = self.teacher(data)

        # import sys
        # import time
        # from line_profiler import line_profiler
        #
        # prof = line_profiler.LineProfiler(shapley_fakd_loss)  # 把函数传递到性能分析器中
        # prof.enable()  # 开始性能分析

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student, target) + penalty)
        loss_kd = self.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.temperature)
        loss_fa = self.fa_loss_weight * sc_fakd_loss(data, self.student, self.teacher, **kwargs)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_fa": loss_fa,
        }

        # prof.disable()  # 停止性能分析
        # prof.print_stats(sys.stdout)  # 打印性能分析结果
        # prof.print_stats(open('../record/line_profiler', 'w'))  # 打印性能分析结果

        return logits_student, losses_dict
