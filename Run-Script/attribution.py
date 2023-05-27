import os
import shelve

import numpy as np

from Util import get_project_path
from Util.GeneralUtil import save_figure
from Util.TorchUtil import get_data_labels_from_dataset
from attributionlib import class_mean_plot
from attributionlib.explainer.SingleChannel import SingleChannelExplainer
from distilllib.engine.utils import setup_seed, load_checkpoint
from distilllib.models import sdt
from distilllib.models.DNNClassifier import hgrn, lfcnn

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
# model = hgrn(channels=channels, points=points, num_classes=num_classes)  # LFCNN(), VARCNN(), HGRN()
model = sdt(channels=channels, points=points, num_classes=num_classes)  # LFCNN(), VARCNN(), HGRN()
model_name = model.__class__.__name__
# pretrain_model_path = "../checkpoint/Models_Train/CamCAN_LFCNN_20220616160458_checkpoint.pt"
# pretrain_model_path = "../checkpoint/Models_Train/DecMeg2014_HGRN_20220616192753_checkpoint.pt"
# pretrain_model_path = "../checkpoint/Models_Train/DecMeg2014_SDT_student"
pretrain_model_path = "../checkpoint/Models_Train/DecMeg2014_SDT_SCFAKD"
# pretrain_model_path = "../checkpoint/Models_Train/DecMeg2014_SDT_ShapleyFAKD"
save_model_name = pretrain_model_path.split('/')[-1]
if model_name == "SDT":
    model.load_state_dict(load_checkpoint(pretrain_model_path)['model'])
else:
    model.load_state_dict(load_checkpoint(pretrain_model_path))

db_path = get_project_path() + '/record/{}_benchmark'.format(dataset)
db = shelve.open(db_path)
explainer = SingleChannelExplainer(dataset, label_names, model)
run_time_list = np.zeros(sample_num, dtype=np.float32)
auc_list = np.zeros(sample_num, dtype=np.float32)
for sample_id in range(sample_num):
    origin_input, truth_label = origin_data[sample_id], labels[sample_id]
    result = explainer(sample_id, origin_input, truth_label)
    db[result.result_id] = result

# 读取通道可视化信息
channel_db = shelve.open(get_project_path() + '/dataset/grad_info')
channels_info = channel_db['info']
channel_db.close()

for mean_class in [0, 1]:
    result_list = []
    for sample_id in range(sample_num):
        result_id = "{}_{}_{}_{}".format(dataset, sample_id, model_name, explainer.explainer_name)
        result = db[result_id]
        if result.pred_label == result.truth_label == mean_class:
            result_list.append(result)

    fig, _, _, _ = class_mean_plot(result_list, channels_info, attribution_method=explainer.explainer_name)
    save_figure(fig, get_project_path() + '/plot/heatmap/', '{}_{}_mean'.format(save_model_name, mean_class))
