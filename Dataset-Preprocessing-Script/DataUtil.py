import math
import os

import numpy as np
from scipy import signal
from scipy.io import loadmat


def get_subject_name(subject_path):
    return os.path.split(subject_path)[-1].split(".")[0]


def butterBandPassFilter(lowcut, highcut, samplerate, order):
    """生成巴特沃斯带通滤波器"""
    semiSampleRate = samplerate * 0.5
    low = lowcut / semiSampleRate
    high = highcut / semiSampleRate
    b, a = signal.butter(order, [low, high], btype='bandpass')
    print("bandpass:", "b.shape:", b.shape, "a.shape:", a.shape, "order=", order)
    return b, a


def preprocess_DecMeg2014_subject(filename, start=0.5, apply_butter_filter=False, select_grad_channels=True):
    print("Loading", filename)
    data = loadmat(filename, squeeze_me=True)
    sfreq = data['sfreq']
    X = data['X'][:, :, int(start * sfreq):]
    y = data['y']
    print("Dataset summary:", X.shape, y.shape)

    # 筛选Grad类型的通道
    if select_grad_channels:
        grad_channels_num = int(X.shape[1] / 3 * 2)
        X_grad = np.zeros((X.shape[0], grad_channels_num, X.shape[2]))
        for ch in range(int(grad_channels_num/2)):
            X_grad[:, 2*ch, :] = X[:, 3*ch, :]
            X_grad[:, 2*ch+1, :] = X[:, 3*ch+1, :]
        print("select_grad_channels:", X_grad.shape)
        X = X_grad

    # 4阶巴特沃斯滤波器
    if apply_butter_filter:
        b, a = butterBandPassFilter(0.1, 20, sfreq, order=4)  # b，a: IIR滤波器的分子（b）和分母（a）多项式系数向量
        X = signal.lfilter(b, a, X)

    X = X.astype(np.float32)
    y = y.astype(np.longlong)

    # 标准化
    X -= X.mean(0)
    X = np.nan_to_num(X / X.std(0))

    return X, y


# 合并和重置标签，传入数组参数X，并且对其数组X中元素改变，会对原数组造成影响
# 但是如果对X进行增加一列，删除一列，重新赋值，都不会对原数组造成影响
def _reset_labels(labels):
    for i in range(labels.size):
        if labels[i] == 9:
            labels[i] = 1
        else:
            labels[i] = 0


def _reset_labels_vision(labels):
    for i in range(labels.size):
        if labels[i] == 6:
            labels[i] = 0
        elif labels[i] == 7:
            labels[i] = 1
        elif labels[i] == 8:
            labels[i] = 2
        else:
            labels[i] = 3


# 进行标准化，即z-score，对基线区间计算均值和标准差，对整个epoch区间应用z-score
def z_score_standardization(epoch, baseline_scale=0.375, crop_baseline=False):
    if len(epoch.shape) == 3:
        epoch = np.squeeze(epoch)
    assert isinstance(epoch, np.ndarray) and len(epoch.shape) == 2
    assert isinstance(baseline_scale, float) and 0 <= baseline_scale <= 1

    baseline_points = math.floor(epoch.shape[1] * baseline_scale)  # 计算基线区间所包含的时间点，向下取整
    baseline_mean = epoch[..., :baseline_points].mean(axis=(0, 1), keepdims=True)
    baseline_sd = epoch[..., :baseline_points].std(axis=(0, 1), keepdims=True)

    epoch -= baseline_mean
    epoch /= baseline_sd

    if crop_baseline:
        epoch = epoch[..., baseline_points:]


def preprocess_CamCAN_subjects(npz_files, vision_combline=True):
    assert isinstance(npz_files, str) or isinstance(npz_files, list)
    if isinstance(npz_files, str):
        npz_files = [npz_files]
    assert len(npz_files) > 0

    data, labels = [], []
    # 读取剩余npz
    for npz_file in npz_files:
        npz = np.load(npz_file)
        npz_data = npz['data']
        npz_labels = npz['labels']
        print("subject_name:", get_subject_name(npz_file))
        print("npz_data:", npz_data.shape)
        print("npz_labels:", npz_labels.shape)

        data.append(npz_data)
        labels.append(npz_labels)

    data = np.vstack(data)
    labels = np.concatenate(labels)
    print("Dataset summary:")
    print("data:", data.shape)
    print("labels:", labels.shape)

    assert data.dtype == np.float32 and labels.dtype == np.longlong

    # 合并和重置标签
    if vision_combline:
        _reset_labels(labels)
    else:
        _reset_labels_vision(labels)

    # 对每一个样本进行Z-Score标准化
    data -= data.mean(0)
    data = np.nan_to_num(data / data.std(0))

    return data, labels


#Class 0: Both Hand Imagery, Class 1: Both Feet Imagery, Class 2: Word generation Imagery, Class 3: Subtraction Imagery
#Their associated triggers in the STIM channels are 4, 8, 16, and 32
def preprocess_MentalImagery_subjects(npz_files):
    assert isinstance(npz_files, str) or isinstance(npz_files, list)
    if isinstance(npz_files, str):
        npz_files = [npz_files]
    assert len(npz_files) > 0

    data, labels = [], []
    # 读取剩余npz
    for npz_file in npz_files:
        npz = np.load(npz_file)
        npz_data = npz['data']
        npz_labels = npz['labels']
        print("subject_name:", get_subject_name(npz_file))
        print("npz_data:", npz_data.shape)
        print("npz_labels:", npz_labels.shape)

        data.append(npz_data)
        labels.append(npz_labels)

    data = np.vstack(data)
    labels = np.concatenate(labels)
    print("Dataset summary:")
    print("data:", data.shape)
    print("labels:", labels.shape)

    assert data.dtype == np.float32 and labels.dtype == np.longlong

    # 重置标签
    for i in range(labels.size):
        if labels[i] == 4:
            labels[i] = 0
        elif labels[i] == 8:
            labels[i] = 1
        elif labels[i] == 16:
            labels[i] = 2
        else:
            labels[i] = 3

    # # 只保留Both Hand Imagery和Word generation Imagery
    # retain_index = (labels == 0) + (labels == 2)
    # data = data[retain_index]
    # labels = labels[retain_index]
    # for i in range(labels.size):
    #     if labels[i] == 2:
    #         labels[i] = 1

    # 对每一个样本进行Z-Score标准化
    data -= data.mean(0)
    data = np.nan_to_num(data / data.std(0))

    return data, labels
