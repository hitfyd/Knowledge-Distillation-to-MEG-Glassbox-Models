import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def MSE_loss(logits_student, logits_teacher):
    mse_loss = F.mse_loss(logits_student, logits_teacher, reduction="mean")
    return mse_loss


class MSEKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(MSEKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.MSEKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.MSEKD.LOSS.KD_WEIGHT

    def forward_train(self, data, target, **kwargs):
        logits_student, penalty = self.student(data, is_training_data=True)
        with torch.no_grad():
            logits_teacher = self.teacher(data)

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student, target) + penalty)
        loss_kd = self.kd_loss_weight * MSE_loss(logits_student, logits_teacher)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
