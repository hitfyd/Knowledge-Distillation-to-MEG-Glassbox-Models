import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor
from torch.backends import cudnn
from torch.utils.data import TensorDataset


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
        loss = criterion(output, target) + __l1_regularization__(model, l1_penalty)  # 训练集精度不高，取消正则项
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