# 根据现有研究的留一法交叉验证实验结果，选择具有最佳表现的受试者4作为测试集，其余受试者1，2，3，5-16为训练集，并进一步按照9：1的比例划分为训练集和验证集
# 0.1-20Hz的4阶巴沃斯特滤波
# 按照采集设备和数据的结构，对其进行简单的通道筛选，只保留Grad通道类型的204通道
# 对样本进行Z-Score标准化
#
#     X : the 3D data matrix, trial x channel x time (580 x 306 x 375).
#     y : the vector of the class labels (580), in the same order of the trials in X. The classes are 0 (Scramble) and 1 (Face).
#     sfreq : the sampling frequency in Hz, i.e. 250.
#     tmin: the time, in seconds, of the beginning of the trial from the stimulus onset, i.e. -0.5
#     tmax: the time, in seconds, of the end of the trial from the stimulus onset, i.e. 1.0.
#
# Reference：
# [1] J. Li, J. Pan, F. Wang, and Z. Yu, “Inter-Subject MEG Decoding for Visual Information with Hybrid Gated Recurrent Network,” Appl. Sci.-Basel, vol. 11, no. 3, Art. no. 3, Jan. 2021, doi: 10.3390/app11031215.
# [2] https://www.kaggle.com/c/decoding-the-human-brain/data

import os

import numpy as np

from DataUtil import preprocess_DecMeg2014_subject

# 数据集名称
dataset = 'DecMeg2014'
# CamCAN经过预处理后的Epochs数据
epochs_path = 'D:/decoding-the-human-brain/data/'
# 划分好的数据集存储路径
dataset_save_path = '../dataset/'
if not os.path.exists(dataset_save_path):
    os.mkdir(dataset_save_path)
# 训练集、验证集、测试集划分参数
train_subjects = list(range(1, 4)) + list(range(5, 17))
test_subjects = [4]

# 读取测试集和训练集
test_data, test_labels = [], []
for subject in test_subjects:
    filename = '%strain_subject%02d.mat' % (epochs_path, subject)
    X, y = preprocess_DecMeg2014_subject(filename, apply_butter_filter=True)
    test_data.append(X)
    test_labels.append(y)
test_data, test_labels = np.vstack(test_data), np.concatenate(test_labels)

train_data, train_labels = [], []
for subject in train_subjects:
    filename = '%strain_subject%02d.mat' % (epochs_path, subject)
    X, y = preprocess_DecMeg2014_subject(filename, apply_butter_filter=True)
    train_data.append(X)
    train_labels.append(y)
train_data, train_labels = np.vstack(train_data), np.concatenate(train_labels)

# 保存训练集
np.savez('{}{}_train'.format(dataset_save_path, dataset), data=train_data, labels=train_labels)
# 保存测试集
np.savez('{}{}_test'.format(dataset_save_path, dataset), data=test_data, labels=test_labels)
