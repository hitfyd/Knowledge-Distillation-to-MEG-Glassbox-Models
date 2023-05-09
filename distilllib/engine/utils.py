import os
import random

import torch
import torch.nn as nn
import numpy as np
import sys
import time
from tqdm import tqdm


# 设置全局随机数种子，同时用于记录实验数据
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 尽可能提高确定性
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


# 配置Pytorch批处理数据集
def get_data_loader(data, label, batch_size=256, shuffle=True):
    assert isinstance(data, np.ndarray) and isinstance(label, np.ndarray) and len(data) == len(label)
    assert data.dtype == np.float32 #and label.dtype == np.longlong
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data), torch.from_numpy(label))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# 从数据集文件读取样本和标签
def get_data_labels_from_dataset(dataset_path):
    dataset = np.load(dataset_path)
    data = dataset['data']
    labels = dataset['labels']
    return data, labels


# 从数据集文件读取，生成数据集
def get_data_loader_from_dataset(dataset_path, batch_size=256, shuffle=True):
    dataset = np.load(dataset_path)
    data = dataset['data']
    labels = dataset['labels']
    return get_data_loader(data, labels, batch_size, shuffle)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, distiller):
    batch_time, losses, top1 = [AverageMeter() for _ in range(3)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (data, target) in enumerate(val_loader):
            data = data.float()
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = distiller(data=data)
            loss = criterion(output, target)
            acc1, _ = accuracy(output, target, topk=(1, 2))
            batch_size = data.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}".format(
                top1=top1
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    return top1.avg, losses.avg


def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.SOLVER.LR_DECAY_STAGES))
    if steps > 0:
        new_lr = cfg.SOLVER.LR * (cfg.SOLVER.LR_DECAY_RATE**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return cfg.SOLVER.LR


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(origin_test_labels, pred_labels):
    assert len(origin_test_labels) == len(pred_labels)
    all_num = len(origin_test_labels)
    # 统计TP（true positive，真正例，即把正例正确预测为正例）、FN（false negative，假负例，即把正例错误预测为负例）、
    # FP（false positive，假正例，即把负例错误预测为正例）、TN（true negative，真负例，即把负例正确预测为负例）
    tp, fn, fp, tn = 0, 0, 0, 0
    for i in range(all_num):
        if origin_test_labels[i] == pred_labels[i]:
            if origin_test_labels[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if origin_test_labels[i] == 1:
                fn += 1
            else:
                fp += 1
    print('all: {}\ttp: {}\tfp: {}\tfn: {}\ttn:{}'.format(all_num, tp, fp, fn, tn))
    accuracy = 1.0 * (tp + tn) / all_num
    precision = 1.0 * tp / (tp + fp)
    recall = 1.0 * tp / (tp + fn)
    F1 = (2 * precision * recall) / (precision + recall)
    return accuracy, precision, recall, F1, tp, fn, fp, tn


def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")
