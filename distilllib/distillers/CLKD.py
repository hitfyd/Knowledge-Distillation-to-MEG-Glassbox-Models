import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def NMSE(input_data, target):
    input_data = F.normalize(input_data, p=2)
    target = F.normalize(target, p=2)
    nmse_loss = F.mse_loss(input_data, target, reduction="mean")
    return nmse_loss


def ins_loss(logits_student, logits_teacher):
    loss_ins = NMSE(logits_student, logits_teacher)
    return loss_ins


def cla_loss(logits_student, logits_teacher):
    # norm_T_logits_student = F.normalize(logits_student, p=2).T
    # norm_T_logits_teacher = F.normalize(logits_teacher, p=2).T
    # loss_cla = NMSE(norm_T_logits_student, norm_T_logits_teacher)

    loss_cla = NMSE(logits_student.T, logits_teacher.T)
    return loss_cla


def cc_loss(logits_student, logits_teacher):
    num_classes = logits_student.size(1)
    # mean_logits_student = torch.mean(logits_student, dim=1)
    # mean_logits_teacher = torch.mean(logits_teacher, dim=1)
    # cc_logits_student, cc_logits_teacher = 0, 0
    # for i in range(num_classes):
    #     temp = logits_student[:, i] - mean_logits_student
    #     temp_T = temp.view(1, -1)
    #     cc_logits_student += temp_T * temp
    #     cc_logits_teacher += (logits_teacher[:, i] - mean_logits_teacher) * (logits_teacher[:, i] - mean_logits_teacher).T
    # cc_logits_student /= (num_classes-1)
    # cc_logits_teacher /= (num_classes-1)

    # norm_logits_student = F.normalize(logits_student, p=2)
    # norm_logits_teacher = F.normalize(logits_teacher, p=2)
    # cc_logits_student = torch.matmul(norm_logits_student.T, norm_logits_student)
    # cc_logits_teacher = torch.matmul(norm_logits_teacher.T, norm_logits_teacher)

    cc_logits_student = torch.matmul(logits_student.T, logits_student)
    cc_logits_teacher = torch.matmul(logits_teacher.T, logits_teacher)
    loss_cc = NMSE(cc_logits_student, cc_logits_teacher) / num_classes**2
    return loss_cc


class CLKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(CLKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.CLKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.CLKD.LOSS.KD_WEIGHT
        self.cla_coefficient = cfg.CLKD.LOSS.CLA_COEFFICIENT
        self.cc_loss_weight = cfg.CLKD.LOSS.CC_WEIGHT

    def forward_train(self, data, target, **kwargs):
        logits_student, penalty = self.student(data, is_training_data=True)
        with torch.no_grad():
            logits_teacher = self.teacher(data)

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student, target) + penalty)
        loss_ins = ins_loss(logits_student, logits_teacher)
        loss_cla = cla_loss(logits_student, logits_teacher)
        loss_kd = self.kd_loss_weight * (loss_ins + self.cla_coefficient * loss_cla)
        loss_cc = self.cc_loss_weight * cc_loss(logits_student, logits_teacher)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_cc": loss_cc,
        }
        return logits_student, losses_dict
