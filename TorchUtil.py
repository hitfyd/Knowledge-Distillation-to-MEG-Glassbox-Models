import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import tensor
from torch.backends import cudnn
from torch.utils.data import TensorDataset


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print, min_train_epochs=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
            min_train_epochs (int): Minimum train epochs before counting starts.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_model_parameters = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.min_epochs = min_train_epochs

    def __call__(self, val_loss, model, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.update_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # 满足最小训练轮数后再进行早停计数
            if epoch > self.min_epochs:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                # print('early_stop epoch', epoch, 'best_epoch:', self.best_epoch, 'best_score:', self.best_score)
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.update_checkpoint(val_loss, model)
            self.counter = 0

    def update_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_model_parameters = model.state_dict()
        self.val_loss_min = val_loss


class TrainResult(object):
    def __init__(self, result_id: str, dataset: str, channels: int, points: int, classes: int, model_name: str,
                 BATCH_SIZE: int, MAX_TRAIN_EPOCHS: int, EARLY_STOP: bool or int, LEARN_RATE, L1_penalty, L2_penalty,
                 test_accuracy, test_loss,
                 validate_accuracy_list, validate_loss_list, train_accuracy_list, train_loss_list,
                 test_accuracy_list, test_loss_list):
        # result id
        self.result_id = result_id

        # train hyperparameter
        self.dataset = dataset
        self.channels = channels
        self.points = points
        self.classes = classes
        self.model_name = model_name
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_TRAIN_EPOCHS = MAX_TRAIN_EPOCHS
        self.EARLY_STOP = EARLY_STOP
        self.LEARN_RATE = LEARN_RATE
        self.L1_penalty = L1_penalty
        self.L2_penalty = L2_penalty

        # train result
        self.test_accuracy = test_accuracy
        self.test_loss = test_loss
        self.validate_accuracy_list = validate_accuracy_list
        self.validate_loss_list = validate_loss_list
        self.train_accuracy_list = train_accuracy_list
        self.train_loss_list = train_loss_list
        self.test_accuracy_list = test_accuracy_list
        self.test_loss_list = test_loss_list


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


# 默认使用CPU
DEVICE = torch.device("cpu")


# 设置运算硬件
def set_device(cuda=0):
    global DEVICE
    # GPU/CPU 选择
    if cuda < 0:
        DEVICE = torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  ###指定此处为-1即可
    elif torch.cuda.is_available():
        # 防止GPU编号超出硬件数量
        if torch.cuda.device_count() <= cuda:
            cuda = torch.cuda.device_count() - 1
        DEVICE = torch.device("cuda:{}".format(cuda))


criterion = nn.CrossEntropyLoss()


# 获取样本标签分布情况
def get_labels_info(labels):
    info = {'count': len(labels)}
    labels_set = set(labels)
    for label in labels_set:
        info[label] = np.sum(labels == label)
    return info


# 配置Pytorch批处理数据集
def get_data_loader(data, label, batch_size=256, shuffle=True):
    assert isinstance(data, np.ndarray) and isinstance(label, np.ndarray) and len(data) == len(label)
    assert data.dtype == np.float32  # and label.dtype == np.longlong
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


# 同时生成训练集、验证集和测试集
def get_train_validate_test_dataset(train_path, validate_path, test_path, batch_size, shuffle=True):
    train_data_loader = get_data_loader_from_dataset(train_path, batch_size, shuffle=shuffle)
    validate_data_loader = get_data_loader_from_dataset(validate_path, batch_size, shuffle=shuffle)
    test_data_loader = get_data_loader_from_dataset(test_path, batch_size, shuffle=shuffle)
    return train_data_loader, validate_data_loader, test_data_loader


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.GRU):
        m.reset_parameters()


# 增加L1正则化
def __l1_regularization__(model, l1_penalty=3e-4):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(abs(param))  # torch.norm(param, p=1)
    return l1_penalty * regularization_loss


def train(model, train_loader, epoch, lr=3e-4, l1_penalty=0, l2_penalty=0):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_penalty)
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target) + __l1_regularization__(model, l1_penalty)    # 训练集精度不高，取消正则项
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_accuracy = 100. * correct / len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    print('Training Dataset\tEpoch：{}\tAccuracy: [{}/{} ({:.6f}%)]\tAverage Loss: {:.6f}'.format(
        epoch, correct, len(train_loader.dataset), train_accuracy, train_loss))
    return train_accuracy, train_loss


def train_soft(model, train_loader, epoch, lr=3e-4, l1_penalty=0, l2_penalty=0):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_penalty)
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target) + __l1_regularization__(model, l1_penalty)    # 训练集精度不高，取消正则项
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        correct += pred.eq(target.max(1, keepdim=True)[1].view_as(pred)).sum().item()

    train_accuracy = 100. * correct / len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    print('Training Dataset\tEpoch：{}\tAccuracy: [{}/{} ({:.6f}%)]\tAverage Loss: {:.6f}'.format(
        epoch, correct, len(train_loader.dataset), train_accuracy, train_loss))
    return train_accuracy, train_loss


def test(model, test_loader, validate=False):
    model.to(DEVICE)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            data = data.float()
            output = model(data)
            test_loss += criterion(output, target).item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_accuracy = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    if validate:
        print('Validation Dataset\tAccuracy: {}/{} ({:.6f}%)\tAverage loss: {:.6f}'.format(
            correct, len(test_loader.dataset), test_accuracy, test_loss))
    else:
        print('Test Dataset\tAccuracy: {}/{} ({:.6f}%)\tAverage loss: {:.6f}'.format(
            correct, len(test_loader.dataset), test_accuracy, test_loss))
    # 返回测试集精度，损失
    return test_accuracy, test_loss


def test_soft(model, test_loader, validate=False):
    model.to(DEVICE)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            data = data.float()
            output = model(data)
            test_loss += criterion(output, target).item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.max(1, keepdim=True)[1].view_as(pred)).sum().item()

    test_accuracy = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    if validate:
        print('Validation Dataset\tAccuracy: {}/{} ({:.6f}%)\tAverage loss: {:.6f}'.format(
            correct, len(test_loader.dataset), test_accuracy, test_loss))
    else:
        print('Test Dataset\tAccuracy: {}/{} ({:.6f}%)\tAverage loss: {:.6f}'.format(
            correct, len(test_loader.dataset), test_accuracy, test_loss))
    # 返回测试集精度，损失
    return test_accuracy, test_loss


def predict(model, test_data: torch.Tensor, test_batch_size=1024):
    model.to(DEVICE)
    model.eval()
    predictions, pred_labels = [], []  # 预测的置信度和置信度最大的标签编号
    with torch.no_grad():
        data_split = torch.split(test_data, test_batch_size, dim=0)
        for data in data_split:
            data = data.to(DEVICE)
            data = data.float()
            output = model(data)
            predictions.extend(output.cpu().numpy())  # 恢复标签置信度
            pred_labels.extend(output.max(1, keepdim=True)[1].cpu().numpy())  # 找到概率最大的下标
    # 返回每个标签的置信度
    pred_labels = np.array(pred_labels)
    predictions = np.array(predictions)
    if model.__class__.__name__ == 'HGRN':
        predictions = np.exp(predictions)
    else:
        predictions = np.array(torch.softmax(tensor(predictions), dim=1))
    return predictions, pred_labels


def individual_predict(model, test_data: torch.Tensor):
    # model.to(DEVICE)
    # model.eval()
    # with torch.no_grad():
    #     test_data = test_data.to(DEVICE)
    #     test_data = test_data.float().unsqueeze(dim=0)
    #     output = model(test_data)
    # pred = torch.squeeze(output).cpu()     # 恢复标签置信度
    # if model.__class__.__name__ != 'HGRN':
    #     pred = torch.softmax(pred, dim=0)
    # pred = pred.numpy()
    # # if model.__class__.__name__ == 'HGRN':
    # #     pred = np.exp(pred)
    # pred_label = np.argmax(pred)
    # return pred, pred_label
    pred, _ = predict(model, test_data.unsqueeze(dim=0))
    return pred[0], np.argmax(pred[0])


def save_checkpoint(model_parameters, checkpoint_path):
    # 确保checkpoint路径存在
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model_parameters, checkpoint_path)


def read_checkpoint(model, checkpoint_path):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cuda')))
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))


def restore_baseline_checkpoint(model_list, checkpoint_dir, dataset, checkpoint_id):
    if not isinstance(model_list, list):
        model_list = [model_list]
    for model in model_list:
        model_name = model.__class__.__name__
        benchmark_checkpoint = '{}/{}_{}_{}_checkpoint.pt'.format(checkpoint_dir, dataset, model_name, checkpoint_id)
        read_checkpoint(model, benchmark_checkpoint)


def save_figure(fig, save_dir, figure_name, save_dpi=400, format_list=None):
    # EPS format for LaTeX
    # PDF format for LaTeX/Display
    # SVG format for Web
    # JPG format for display
    if format_list is None:
        format_list = ["eps", "pdf", "svg"]
    plt.rcParams['savefig.dpi'] = save_dpi  # 图片保存像素
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)  # 确保路径存在
    for save_format in format_list:
        fig.savefig('{}{}.{}'.format(save_dir, figure_name, save_format), format=save_format,
                    bbox_inches='tight', transparent=False)
