import numpy as np
import torch
import torch.nn.functional as F

from .KD import kd_loss
from ._base import Distiller
from ..engine.utils import predict


def fa_loss(input_data, target):
    input_data = F.normalize(input_data, p=2)
    target = F.normalize(target, p=2)
    loss_fa = F.mse_loss(input_data, target, reduction="mean") \
              # + F.mse_loss(input_data.permute(1, 0, 2), target.permute(1, 0, 2), reduction="mean")    # 样本间类关系知识
    return loss_fa


perturbation_data_list = []
features_teacher_list = []


# 不同归因算法的不同在于扰动数据的生成方法不同，这里将所有样本对应的扰动数据只在第一轮生成一次，其余轮不再重新生成；教师模型的归因特征矩阵也只在第一轮计算
def sc_fakd_loss(data, student, teacher, **kwargs):
    epoch, data_itx = kwargs["epoch"], kwargs["data_itx"]
    current_device = torch.cuda.current_device()
    devices_num = torch.cuda.device_count()
    batch_size, channels, points = data.size()
    if epoch == 0:
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
        loss_fakd = fa_loss(features_student.view(batch_size, channels, -1),
                            features_teacher.view(batch_size, channels, -1))

        perturbation_data_list.append(perturbation_data)
        features_teacher_list.append(features_teacher)
    else:
        list_index = data_itx * devices_num + current_device
        features_student = predict(student, perturbation_data_list[list_index])
        loss_fakd = fa_loss(features_student.view(batch_size, channels, -1),
                            features_teacher_list[list_index].view(batch_size, channels, -1).cuda(current_device))
    return loss_fakd


class SCFAKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(SCFAKD, self).__init__(student, teacher)
        self.with_kd = cfg.SCFAKD.WITH_KD
        self.temperature = cfg.SCFAKD.TEMPERATURE
        self.ce_loss_weight = cfg.SCFAKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.SCFAKD.LOSS.KD_WEIGHT
        self.fa_loss_weight = cfg.SCFAKD.LOSS.FA_WEIGHT

    def forward_train(self, data, target, **kwargs):
        logits_student, penalty = self.student(data, is_training_data=True)
        with torch.no_grad():
            logits_teacher = self.teacher(data)

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student, target) + penalty)
        loss_fa = self.fa_loss_weight * sc_fakd_loss(data, self.student, self.teacher, **kwargs)

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
