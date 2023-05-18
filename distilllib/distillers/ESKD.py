import torch
import torch.nn as nn
import torch.nn.functional as F

from .KD import kd_loss
from ._base import Distiller


class ESKD(Distiller):
    """On the Efficacy of Knowledge Distillation"""

    def __init__(self, student, teacher, cfg):
        super(ESKD, self).__init__(student, teacher)
        self.temperature = cfg.ESKD.TEMPERATURE
        self.stop_epoch = cfg.ESKD.STOP_EPOCH
        self.ce_loss_weight = cfg.ESKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.ESKD.LOSS.KD_WEIGHT

    def forward_train(self, data, target, **kwargs):
        logits_student, penalty = self.student(data, is_training_data=True)
        with torch.no_grad():
            logits_teacher = self.teacher(data)

        # losses
        if kwargs['epoch'] < self.stop_epoch:
            loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student, target) + penalty)
            loss_kd = self.kd_loss_weight * kd_loss(
                logits_student, logits_teacher, self.temperature
            )
            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_kd,
            }
        else:
            loss_ce = F.cross_entropy(logits_student, target) + penalty
            losses_dict = {
                "loss_ce": loss_ce,
            }
        return logits_student, losses_dict
