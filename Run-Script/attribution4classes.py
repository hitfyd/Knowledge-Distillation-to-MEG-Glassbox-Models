import os
import shelve

import PIL
import mne
import numpy as np
from matplotlib import pyplot as plt, gridspec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import r2_score

from attributionlib import class_mean_plot, AttributionResult, shapley_fakd_parallel
from distilllib.engine.utils import setup_seed, load_checkpoint, predict, get_data_labels_from_dataset, save_figure
from distilllib.models import sdt, model_dict
from distilllib.models.DNNClassifier import hgrn, varcnn, lfcnn

# 超参数
RAND_SEED = 16
GPU_ID = 0
# 固定随机数种子
setup_seed(RAND_SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 要解释的样本数据集
dataset = 'CamCAN'  # CamCAN DecMeg2014
# dataset = 'DecMeg2014'  # CamCAN DecMeg2014

dataset_path = '../dataset/{}_test.npz'.format(dataset)
data, labels = get_data_labels_from_dataset(dataset_path)
sample_num, channels, points = data.shape
num_classes = len(set(labels))
label_names = ['audio', 'visual']
if dataset == 'DecMeg2014':
    label_names = ['Scramble', 'Face']

# 恢复预训练模型
lfcnn_teacher = lfcnn(channels, points, num_classes)
varvnn_teacher = varcnn(channels, points, num_classes)
hgrn_teacher = hgrn(channels, points, num_classes)
varvnn_teacher.load_state_dict(load_checkpoint(model_dict["{}_varcnn".format(dataset)][1]))
lfcnn_teacher.load_state_dict(load_checkpoint(model_dict["{}_lfcnn".format(dataset)][1]))
hgrn_teacher.load_state_dict(load_checkpoint(model_dict["{}_hgrn".format(dataset)][1]))

sdt_vanilla = sdt(channels=channels, points=points, num_classes=num_classes)
sdt_vanilla.load_state_dict(load_checkpoint("../checkpoint/Models_Train/{}_SDT_Vanilla".format(dataset)))

sdt_lfcnn_kd = sdt(channels=channels, points=points, num_classes=num_classes)       # KD
sdt_varcnn_kd = sdt(channels=channels, points=points, num_classes=num_classes)
sdt_hgrn_kd = sdt(channels=channels, points=points, num_classes=num_classes)
sdt_lfcnn_fakd_ce = sdt(channels=channels, points=points, num_classes=num_classes)  # r"$\mathcal{L}_{FAKD}$"
sdt_varcnn_fakd_ce = sdt(channels=channels, points=points, num_classes=num_classes)
sdt_hgrn_fakd_ce = sdt(channels=channels, points=points, num_classes=num_classes)
sdt_lfcnn_fakd = sdt(channels=channels, points=points, num_classes=num_classes)     # FAKD
sdt_varcnn_fakd = sdt(channels=channels, points=points, num_classes=num_classes)
sdt_hgrn_fakd = sdt(channels=channels, points=points, num_classes=num_classes)
sdt_lfcnn_kd.load_state_dict(load_checkpoint("../checkpoint/Models_Train/{}_SDT_LFCNN_KD".format(dataset)))
sdt_varcnn_kd.load_state_dict(load_checkpoint("../checkpoint/Models_Train/{}_SDT_VARCNN_KD".format(dataset)))
sdt_hgrn_kd.load_state_dict(load_checkpoint("../checkpoint/Models_Train/{}_SDT_HGRN_KD".format(dataset)))
sdt_lfcnn_fakd_ce.load_state_dict(load_checkpoint("../checkpoint/Models_Train/{}_SDT_LFCNN_FAKD_CE".format(dataset)))
sdt_varcnn_fakd_ce.load_state_dict(load_checkpoint("../checkpoint/Models_Train/{}_SDT_VARCNN_FAKD_CE".format(dataset)))
sdt_hgrn_fakd_ce.load_state_dict(load_checkpoint("../checkpoint/Models_Train/{}_SDT_HGRN_FAKD_CE".format(dataset)))
sdt_lfcnn_fakd.load_state_dict(load_checkpoint("../checkpoint/Models_Train/{}_SDT_LFCNN_FAKD".format(dataset)))
sdt_varcnn_fakd.load_state_dict(load_checkpoint("../checkpoint/Models_Train/{}_SDT_VARCNN_FAKD".format(dataset)))
sdt_hgrn_fakd.load_state_dict(load_checkpoint("../checkpoint/Models_Train/{}_SDT_HGRN_FAKD".format(dataset)))

model_list = [lfcnn_teacher, varvnn_teacher, hgrn_teacher, sdt_vanilla,
              sdt_lfcnn_kd, sdt_varcnn_kd, sdt_hgrn_kd,
              sdt_lfcnn_fakd_ce, sdt_varcnn_fakd_ce, sdt_hgrn_fakd_ce,
              sdt_lfcnn_fakd, sdt_varcnn_fakd, sdt_hgrn_fakd,]
model_name_list = ["LFCNN", "VARCNN", "HGRN", "SDT_Vanilla",
                   "SDT_LFCNN_KD", "SDT_VARCNN_KD", "SDT_HGRN_KD",
                   "SDT_LFCNN_"+r"$\mathcal{L}_{FAKD}$",
                   "SDT_VARCNN_"+r"$\mathcal{L}_{FAKD}$",
                   "SDT_HGRN_"+r"$\mathcal{L}_{FAKD}$",
                   "SDT_LFCNN_FAKD", "SDT_VARCNN_FAKD", "SDT_HGRN_FAKD",]

db_path = '../{}_benchmark4classes'.format(dataset)
db = shelve.open(db_path)
batch_size = 128
M = 32
for sample_id in range(0, sample_num, batch_size):
    origin_input, truth_label = data[sample_id:sample_id + batch_size], labels[sample_id:sample_id + batch_size]
    features_lists = shapley_fakd_parallel(origin_input, model_list, M=M)
    for model_id in range(len(model_list)):
        model = model_list[model_id]
        origin_pred = predict(model, origin_input)
        origin_pred_label = origin_pred.max(1)[1]
        origin_pred = origin_pred.detach().cpu().numpy()
        origin_pred_label = origin_pred_label.detach().cpu().numpy()
        for batch_id in range(0, len(origin_input)):
            result = AttributionResult(dataset, label_names, sample_id + batch_id,
                                       origin_input[batch_id], truth_label[batch_id],
                                       model_name_list[model_id], origin_pred[batch_id], origin_pred_label[batch_id],
                                       features_lists[model_id].detach().cpu().numpy()[batch_id])
            db[result.result_id] = result

# 读取通道可视化信息
channel_db = shelve.open('../dataset/grad_info')
channels_info = channel_db['info']
channel_db.close()

for mean_class in [0, 1]:
    pred_list = []
    heatmap_channel_list = []
    for model_id in range(len(model_list)):
        model = model_list[model_id]
        pred = predict(model, data).detach().cpu().numpy()
        pred_list.append(pred)

        result_list = []
        for sample_id in range(sample_num):
            result_id = "{}_{}_{}".format(dataset, sample_id, model_name_list[model_id])
            result = db[result_id]
            if result.pred_label == result.truth_label == mean_class:  # 根据标签区分预测正确样本的模型平均特征归因
                result_list.append(result)
            # if result.pred_label == mean_class:   # 根据预测标签区分的模型平均特征归因（不分区是否预测正确）
            #     result_list.append(result)
            # if result.truth_label == mean_class:   # 根据真是标签区分的模型平均特征归因（不分区是否预测正确）
            #     result_list.append(result)
            # if result.pred_label == result.truth_label:   # 所有预测正确样本的总体特征归因（不分区标签）
            #     result_list.append(result)
            # result_list.append(result)  # 模型的总体特征归因（不分区标签）

        fig, heatmap_channel, _ = class_mean_plot(result_list, channels_info, top_channel_num=5)
        save_figure(fig, '../plot/heatmap4classes/', '{}_{}_{}_mean'.format(dataset, model_name_list[model_id], mean_class))
        for i in range(len(heatmap_channel_list)):
            cos_sim = cosine_similarity(heatmap_channel_list[i].reshape(1, -1), heatmap_channel.reshape(1, -1))
            r2 = r2_score(pred_list[i], pred)
            print(cos_sim, r2)
        heatmap_channel_list.append(heatmap_channel)
