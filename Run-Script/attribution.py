import os
import shelve

import numpy as np
import ray
import torch

from Util import get_project_path
from Util.GeneralUtil import save_figure
from Util.TorchUtil import get_data_labels_from_dataset
from attributionlib import class_mean_plot, AttributionResult
from attributionlib.explainer.SingleChannel import SingleChannelExplainer
from distilllib.engine.utils import setup_seed, load_checkpoint, predict, individual_predict
from distilllib.models import sdt
from distilllib.models.DNNClassifier import hgrn, lfcnn


def feature_segment(channels, points, window_length):
    channel_windows_num = int(points / window_length)  # 需要计算的通道特征数和时间特征数，总特征数为c_features x p_features
    features_num = channels * channel_windows_num
    channel_list, point_start_list = [], []
    for feature_id in range(features_num):
        channel_list.append(int(feature_id / channel_windows_num))
        point_start_list.append(int(feature_id % channel_windows_num * window_length))
    return features_num, channel_list, point_start_list


def shapley_fakd_parallel(data, model_list, M=1):
    batch_size, channels, points = data.shape
    window_length = points
    features_num, channel_list, point_start_list = feature_segment(channels, points, window_length)

    S1 = np.zeros((batch_size, features_num, M, channels, points), dtype=np.float16)
    S2 = np.zeros((batch_size, features_num, M, channels, points), dtype=np.float16)

    @ray.remote
    def run(feature, data_r):
        S1_r = np.zeros((batch_size, M, channels, points), dtype=np.float16)
        S2_r = np.zeros((batch_size, M, channels, points), dtype=np.float16)
        for m in range(M):
            # 直接生成0，1数组，最后确保feature位满足要求，并且将数据类型改为Boolean型减少后续矩阵点乘计算量
            feature_mark = np.random.randint(0, 2, features_num, dtype=np.bool_)    # bool_类型不能改为int8类型
            feature_mark[feature] = 0
            feature_mark = np.repeat(feature_mark, window_length)
            feature_mark = np.reshape(feature_mark, (channels, points))  # reshape是view，resize是copy
            for index in range(batch_size):
                # 随机选择一个参考样本，用于替换不考虑的特征核
                reference_index = (index + np.random.randint(1, batch_size)) % batch_size
                assert index != reference_index # 参考样本不能是样本本身
                reference_input = data_r[reference_index]
                S1_r[index, m] = S2_r[index, m] = feature_mark * data_r[index] + ~feature_mark * reference_input
                S2_r[index, m][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length] = \
                    data_r[index][channel_list[feature], point_start_list[feature]:point_start_list[feature] + window_length]
        return feature, S1_r, S2_r

    data_ = ray.put(data)
    rs = [run.remote(feature, data_) for feature in range(features_num)]
    rs_list = ray.get(rs)
    for feature, S1_r, S2_r in rs_list:
        S1[:, feature] = S1_r
        S2[:, feature] = S2_r

    # 计算S1和S2的预测差值
    S1 = S1.reshape(-1, channels, points)
    S2 = S2.reshape(-1, channels, points)
    features_lists = []
    if not isinstance(model_list, list):
        model_list = [model_list]
    for model in model_list:
        S1_student_preds = predict(model, S1)
        S2_student_preds = predict(model, S2)
        features = (S1_student_preds.view(batch_size, features_num, M, -1) -
                    S2_student_preds.view(batch_size, features_num, M, -1)).sum(axis=(2)) / M
        features_lists.append(features)
    return features_lists


# 超参数
RAND_SEED = 16
GPU_ID = 0
# 固定随机数种子
setup_seed(RAND_SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 要解释的样本数据集
# dataset = 'CamCAN'  # CamCAN DecMeg2014
# channels, points, num_classes = 204, 100, 2
dataset = 'DecMeg2014'  # CamCAN DecMeg2014
channels, points, num_classes = 204, 250, 2
dataset_path = get_project_path() + '/dataset/{}_test.npz'.format(dataset)
origin_data, labels = get_data_labels_from_dataset(dataset_path)
sample_num, channels, points = origin_data.shape
classes = len(set(labels))
label_names = ['audio', 'visual']
if dataset == 'DecMeg2014':
    label_names = ['Scramble', 'Face']

# 恢复预训练模型
# model = lfcnn(channels=channels, points=points, num_classes=num_classes)  # LFCNN(), VARCNN(), HGRN()
model = hgrn(channels=channels, points=points, num_classes=num_classes)  # LFCNN(), VARCNN(), HGRN()
# model = sdt(channels=channels, points=points, num_classes=num_classes)  # LFCNN(), VARCNN(), HGRN()
model_name = model.__class__.__name__
# pretrain_model_path = "../checkpoint/Models_Train/CamCAN_LFCNN_20220616160458_checkpoint.pt"
pretrain_model_path = "../checkpoint/Models_Train/DecMeg2014_HGRN_20220616192753_checkpoint.pt"
# pretrain_model_path = "../checkpoint/Models_Train/DecMeg2014_SDT_student"
# pretrain_model_path = "../checkpoint/Models_Train/DecMeg2014_SDT_SCFAKD"
# pretrain_model_path = "../checkpoint/Models_Train/DecMeg2014_SDT_ShapleyFAKD"
save_model_name = pretrain_model_path.split('/')[-1]
if model_name == "SDT":
    model.load_state_dict(load_checkpoint(pretrain_model_path)['model'])
else:
    model.load_state_dict(load_checkpoint(pretrain_model_path))

db_path = get_project_path() + '/record/{}_benchmark'.format(dataset)
db = shelve.open(db_path)
batch_size = 256
for sample_id in range(0, sample_num, batch_size):
    origin_input, truth_label = origin_data[sample_id:batch_size], labels[sample_id:batch_size]
    origin_pred = predict(model, origin_input)
    origin_pred_label = origin_pred
    features_lists = shapley_fakd_parallel(origin_input, model)
    result = AttributionResult(dataset, label_names, sample_id, origin_input, truth_label,
                               model.__class__.__name__, origin_pred, origin_pred_label,
                               "Shapley_Channel", features_lists[0], 0)
    db[result.result_id] = result

# 读取通道可视化信息
channel_db = shelve.open(get_project_path() + '/dataset/grad_info')
channels_info = channel_db['info']
channel_db.close()

mean_class = "all"
# for mean_class in [0, 1]:
result_list = []
for sample_id in range(sample_num):
    result_id = "{}_{}_{}_{}".format(dataset, sample_id, model_name, explainer.explainer_name)
    result = db[result_id]
    # if result.pred_label == result.truth_label == mean_class: # 根据标签区分预测正确样本的模型平均特征归因
    #     result_list.append(result)
    # if result.pred_label == mean_class:   # 根据标签区分的模型平均特征归因（不分区是否预测正确）
    #     result_list.append(result)
    if result.pred_label == result.truth_label:   # 所有预测正确样本的总体特征归因（不分区标签）
        result_list.append(result)
    # result_list.append(result)  # 模型的总体特征归因（不分区标签）

fig, _, _, _ = class_mean_plot(result_list, channels_info, attribution_method=explainer.explainer_name)
save_figure(fig, get_project_path() + '/plot/heatmap/', '{}_{}_mean'.format(save_model_name, mean_class))
