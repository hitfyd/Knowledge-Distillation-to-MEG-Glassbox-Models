import numpy as np
import torch
import torch.nn.functional as F
# import ray

from .KD import kd_loss
from .SCFAKD import fa_loss
from ._base import Distiller
from ..engine.utils import predict


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
features_teacher_list = []


def shapley_fakd_loss(data, student, teacher, M, **kwargs):
    epoch, data_itx = kwargs["epoch"], kwargs["data_itx"]
    current_device = torch.cuda.current_device()
    devices_num = torch.cuda.device_count()
    batch_size, channels, points = data.size()
    window_length = points
    features_num, channel_list, point_start_list = feature_segment(channels, points, window_length)
    if epoch >= 0:
        data = data.cpu().numpy()

        S1 = np.zeros((batch_size, features_num, M, channels, points), dtype=np.float16)
        S2 = np.zeros((batch_size, features_num, M, channels, points), dtype=np.float16)

        for feature in range(features_num):
            for m in range(M):
                # 直接生成0，1数组，最后确保feature位满足要求，并且将数据类型改为Boolean型减少后续矩阵点乘计算量
                feature_mark = np.random.randint(0, 2, features_num, dtype=np.int8)
                feature_mark[feature] = 0
                feature_mark = np.repeat(feature_mark, window_length)
                feature_mark = np.reshape(feature_mark, (channels, points))  # reshape是view，resize是copy
                for index in range(batch_size):
                    # 随机选择一个参考样本，用于替换不考虑的特征核
                    reference_index = (index + np.random.randint(1, batch_size)) % batch_size
                    # assert index != reference_index # 参考样本不能是样本本身
                    # reference_input = data[reference_index]
                    S1[index, feature, m] = S2[index, feature, m] = feature_mark * data[index] + ~feature_mark * data[reference_index]
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

        # S1_list.append(S1)
        # S2_list.append(S2)
        # features_teacher_list.append(features_teacher)
    else:
        list_index = data_itx*devices_num+current_device
        S1_student_preds = predict(student, S1_list[list_index])
        S2_student_preds = predict(student, S2_list[list_index])
        features_student = (S1_student_preds.view(batch_size, features_num, M, -1) -
                            S2_student_preds.view(batch_size, features_num, M, -1)).sum(axis=(2)) / M
        loss_fakd = fa_loss(features_student, features_teacher_list[list_index].cuda(current_device))
    return loss_fakd


def shapley_fakd_loss_parallel(data, student, teacher, M, **kwargs):
    batch_size, channels, points = data.size()
    window_length = points
    features_num, channel_list, point_start_list = feature_segment(channels, points, window_length)
    data = data.cpu().numpy()

    S1 = np.zeros((batch_size, features_num, M, channels, points), dtype=np.float16)
    S2 = np.zeros((batch_size, features_num, M, channels, points), dtype=np.float16)

    @ray.remote
    def run(feature, data_):
        S1_r = np.zeros((batch_size, M, channels, points), dtype=np.float16)
        S2_r = np.zeros((batch_size, M, channels, points), dtype=np.float16)
        for m in range(M):
            # 直接生成0，1数组，最后确保feature位满足要求，并且将数据类型改为Boolean型减少后续矩阵点乘计算量
            feature_mark = np.random.randint(0, 2, features_num, dtype=np.int8)
            feature_mark[feature] = 0
            feature_mark = np.repeat(feature_mark, window_length)
            feature_mark = np.reshape(feature_mark, (channels, points))  # reshape是view，resize是copy
            for index in range(batch_size):
                # 随机选择一个参考样本，用于替换不考虑的特征核
                reference_index = (index + np.random.randint(1, batch_size)) % batch_size
                # assert index != reference_index # 参考样本不能是样本本身
                # reference_input = data[reference_index]
                S1_r[index, m] = S2_r[index, m] = feature_mark * data_[index] + ~feature_mark * data_[reference_index]
                S2_r[index, m][channel_list[feature],
                point_start_list[feature]:point_start_list[feature] + window_length] = \
                    data_[index][channel_list[feature],
                    point_start_list[feature]:point_start_list[feature] + window_length]
        return feature, S1_r, S2_r

    data_ = ray.put(data)
    rs = [run.remote(feature, data_) for feature in range(features_num)]
    rs_list = ray.get(rs)
    for feature, S1_r, S2_r in rs_list:
        S1[:, feature] = S1_r
        S2[:, feature] = S2_r

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
    return loss_fakd


class ShapleyFAKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(ShapleyFAKD, self).__init__(student, teacher)
        self.with_kd = cfg.ShapleyFAKD.WITH_KD
        self.M = cfg.ShapleyFAKD.M
        self.temperature = cfg.ShapleyFAKD.TEMPERATURE
        self.ce_loss_weight = cfg.ShapleyFAKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.ShapleyFAKD.LOSS.KD_WEIGHT
        self.fa_loss_weight = cfg.ShapleyFAKD.LOSS.FA_WEIGHT

        # if not ray.is_initialized():
        #     ray.init(num_gpus=0, num_cpus=16,  # 计算资源
        #              local_mode=False,  # 是否启动串行模型，用于调试
        #              ignore_reinit_error=True,  # 重复启动不视为错误
        #              include_dashboard=False,  # 是否启动仪表盘
        #              configure_logging=False,  # 不配置日志
        #              log_to_driver=False,  # 日志记录不配置到driver
        #              )

    def forward_train(self, data, target, **kwargs):
        logits_student, penalty = self.student(data, is_training_data=True)
        with torch.no_grad():
            logits_teacher = self.teacher(data)

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student, target) + penalty)
        loss_fa = self.fa_loss_weight * shapley_fakd_loss(data, self.student, self.teacher, self.M, **kwargs)

        if self.with_kd:
            loss_kd = self.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.temperature)
            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_kd,
                "loss_fa": loss_fa,
            }
        else:
            losses_dict = {
                "loss_ce": loss_ce,
                "loss_fa": loss_fa,
            }

        return logits_student, losses_dict
