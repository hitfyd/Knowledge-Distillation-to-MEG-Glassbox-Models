import os
import shelve

from attributionlib import class_mean_plot, AttributionResult, save_figure, shapley_fakd_parallel
from distilllib.engine.utils import setup_seed, load_checkpoint, predict, get_data_labels_from_dataset
from distilllib.models import sdt
from distilllib.models.DNNClassifier import hgrn

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
dataset_path = '../dataset/{}_test.npz'.format(dataset)
data, labels = get_data_labels_from_dataset(dataset_path)
sample_num, channels, points = data.shape
classes = len(set(labels))
label_names = ['audio', 'visual']
if dataset == 'DecMeg2014':
    label_names = ['Scramble', 'Face']

# 恢复预训练模型
teacher_model = hgrn(channels=channels, points=points, num_classes=num_classes)     # DecMeg2014数据集上以FAKD效果最好的HGRN为例，CamCAN以VARCNN为例
student_model_vanilla = sdt(channels=channels, points=points, num_classes=num_classes)  # 没有经过知识蒸馏的SDT
student_model_fakd = sdt(channels=channels, points=points, num_classes=num_classes)     # 纯FAKD，无CE损失
student_model_fakd_ce = sdt(channels=channels, points=points, num_classes=num_classes)  # 具有CE

pretrain_teacher_model_path = "../checkpoint/Models_Train/DecMeg2014_HGRN_20220616192753_checkpoint.pt"
pretrain_student_model_vanilla_path = "../checkpoint/Models_Train/DecMeg2014_SDT_student"
pretrain_student_model_fakd_path = "../checkpoint/Models_Train/DecMeg2014_SDT_ShapleyFAKD"
pretrain_student_model_fakd_ce_path = "../checkpoint/Models_Train/DecMeg2014_SDT_ShapleyFAKD_CE"

teacher_model.load_state_dict(load_checkpoint(pretrain_teacher_model_path))
student_model_vanilla.load_state_dict(load_checkpoint(pretrain_student_model_vanilla_path))
student_model_fakd.load_state_dict(load_checkpoint(pretrain_student_model_fakd_path))
student_model_fakd_ce.load_state_dict(load_checkpoint(pretrain_student_model_fakd_ce_path))

model_list = [teacher_model, student_model_vanilla, student_model_fakd, student_model_fakd_ce]
model_name_list = [pretrain_teacher_model_path.split('/')[-1], pretrain_student_model_vanilla_path.split('/')[-1],
                   pretrain_student_model_fakd_path.split('/')[-1], pretrain_student_model_fakd_ce_path.split('/')[-1]]

db_path = '../record/{}_benchmark'.format(dataset)
db = shelve.open(db_path)
batch_size = 64
for sample_id in range(0, sample_num, batch_size):
    origin_input, truth_label = data[sample_id:sample_id + batch_size], labels[sample_id:sample_id + batch_size]
    features_lists = shapley_fakd_parallel(origin_input, model_list)
    for model_id in range(len(model_list)):
        model = model_list[model_id]
        origin_pred = predict(model, origin_input)
        origin_pred_label = origin_pred.max(1)[1]
        origin_pred = origin_pred.detach().cpu().numpy()
        origin_pred_label = origin_pred_label.detach().cpu().numpy()
        for batch_id in range(0, len(origin_input)):
            result = AttributionResult(dataset, label_names, sample_id+batch_id, origin_input[batch_id], truth_label[batch_id],
                                       model_name_list[model_id], origin_pred[batch_id], origin_pred_label[batch_id],
                                       features_lists[0].detach().cpu().numpy()[batch_id])
            db[result.result_id] = result

# 读取通道可视化信息
channel_db = shelve.open('../dataset/grad_info')
channels_info = channel_db['info']
channel_db.close()

for mean_class in [0, 1]:
    for model_id in range(len(model_list)):
        result_list = []
        for sample_id in range(sample_num):
            result_id = "{}_{}_{}".format(dataset, sample_id, model_name_list[model_id])
            result = db[result_id]
            if result.pred_label == result.truth_label == mean_class: # 根据标签区分预测正确样本的模型平均特征归因
                result_list.append(result)
            # if result.pred_label == mean_class:   # 根据标签区分的模型平均特征归因（不分区是否预测正确）
            #     result_list.append(result)
            # if result.pred_label == result.truth_label:   # 所有预测正确样本的总体特征归因（不分区标签）
            #     result_list.append(result)
            # result_list.append(result)  # 模型的总体特征归因（不分区标签）

        fig, _, _ = class_mean_plot(result_list, channels_info, label=mean_class)
        save_figure(fig, '../plot/heatmap/', '{}_{}_mean'.format(model_name_list[model_id], mean_class))
