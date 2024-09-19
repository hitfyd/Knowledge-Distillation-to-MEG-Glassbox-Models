from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


class WTTM(Distiller):
    def __init__(self, student, teacher, cfg):
        super(WTTM, self).__init__(student, teacher)
        self.l = cfg.WTTM.L
        self.ce_loss_weight = cfg.WTTM.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.WTTM.LOSS.KD_WEIGHT

    def forward_train(self, data, target, **kwargs):
        y_s, penalty = self.student(data, is_training_data=True)
        with torch.no_grad():
            y_t = self.teacher(data)


        p_s = F.log_softmax(y_s, dim=1)
        p_t = torch.pow(torch.softmax(y_t, dim=1), self.l)
        norm = torch.sum(p_t, dim=1)
        p_t = p_t / norm.unsqueeze(1)
        KL = torch.sum(F.kl_div(p_s, p_t, reduction='none'), dim=1)
        loss = torch.mean(norm*KL)

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(y_s, target) + penalty)
        loss_kd = self.kd_loss_weight * loss
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return y_s, losses_dict
