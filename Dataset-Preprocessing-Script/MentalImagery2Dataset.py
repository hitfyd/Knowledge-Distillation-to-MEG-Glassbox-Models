# 划分方式为前200个受试者为训练集，在训练时进一步按照9：1的比例划分为训练集和验证集，201-250个受试者为测试集
# 对每个样本分别进行Z-Score标准化
import os
from glob import glob

import numpy as np

from DataUtil import preprocess_MentalImagery_subjects

# 数据集名称
dataset = 'MentalImagery'
# CamCAN经过预处理后的Epochs数据
epochs_path = 'C:/MentalImagery/'
npz_format = 'sub-*_ses-*_epochs-*.npz'
# 划分好的数据集存储路径
dataset_save_path = '../dataset/'
if not os.path.exists(dataset_save_path):
    os.mkdir(dataset_save_path)
# 训练集、测试集划分参数(验证集在训练集中划分)
split_subject = -6

# 获取所有Subjects的Epochs路径
all_subjects = glob(epochs_path + npz_format)
all_subjects.sort(key=lambda arr: int(os.path.basename(arr).split('_')[0][4:]))
# 读取测试集和训练集
test_data, test_labels = preprocess_MentalImagery_subjects(all_subjects[split_subject:])
train_data, train_labels = preprocess_MentalImagery_subjects(all_subjects[:split_subject])

# 保存训练集
np.savez('{}{}_train'.format(dataset_save_path, dataset), data=train_data, labels=train_labels)
# 保存测试集
np.savez('{}{}_test'.format(dataset_save_path, dataset), data=test_data, labels=test_labels)
