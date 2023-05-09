import torch
import torch.nn as nn
import torch.nn.functional as F

from .KD import kd_loss
from ._base import Distiller


class GKD(Distiller):
    """Channel Distillation: Channel-Wise Attention for Knowledge Distillation"""

    def __init__(self, student, teacher, cfg):
        super(GKD, self).__init__(student, teacher)
        self.temperature = cfg.GKD.TEMPERATURE
        self.ce_loss_weight = cfg.GKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.GKD.LOSS.KD_WEIGHT

    def forward_train(self, data, target, **kwargs):
        logits_student, penalty = self.student(data, is_training_data=True)
        with torch.no_grad():
            logits_teacher = self.teacher(data)

        # 找到教师模型预测正确的样本编号
        teacher_pred_labels = logits_teacher.max(1, keepdim=True)[1].view(-1)
        true_teacher_indexes = teacher_pred_labels == target

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student, target) + penalty)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student[true_teacher_indexes], logits_teacher[true_teacher_indexes], self.temperature
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
