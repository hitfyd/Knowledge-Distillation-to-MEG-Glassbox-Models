import time

from interpret.glassbox import ClassificationTree
from torch import tensor

from Classifier import init_global_network_parameters, init_models, LFCNN, VARCNN, HGRN
from ExperimentRecord import ExperimentRecord, get_project_path
from Glassbox.mytree import DecisionTreeClass, predict_tree
from TorchUtil import setup_seed, set_device, get_data_labels_from_dataset, restore_baseline_checkpoint, predict

# 从配置文件中读取超参数
RAND_SEED = 0
GPU_ID = 0
dataset = 'CamCAN'
channels = 204
points = 100
classes = 2

# 固定随机数种子
setup_seed(RAND_SEED)
# 设置运算硬件
set_device(GPU_ID)

# 初始化模型
init_global_network_parameters(channels=channels, points=points, classes=classes)
model_list = [LFCNN(), VARCNN(), HGRN()]  # LFCNN(), VARCNN(), HGRN()
baseline_checkpoint = 20220616160458  # DecMeg2014：20220616192753     CamCAN：20220616160458
if dataset == 'DecMeg2014':
    baseline_checkpoint = 20220616192753
restore_baseline_checkpoint(model_list, get_project_path() + '/checkpoint/Models_Train/', dataset, baseline_checkpoint)

# 创建运行记录文件
record = ExperimentRecord()
# 记录超参数
record.append([])

# 根据数据集名称读取预设数据集，分为训练集和测试集
assert dataset == 'CamCAN' or dataset == 'DecMeg2014'
train_path = get_project_path() + '/dataset/{}_train.npz'.format(dataset)
test_path = get_project_path() + '/dataset/{}_test.npz'.format(dataset)
train_data, train_labels = get_data_labels_from_dataset(train_path)
test_data, test_labels = get_data_labels_from_dataset(test_path)

transfer_labels, _ = predict(model_list[0], tensor(train_data))

train_data, test_data = train_data.reshape(len(train_labels), -1), test_data.reshape(len(test_labels), -1)

tree = DecisionTreeClass()
tree.fit(train_data[:1000], train_labels[:1000])
print(predict_tree(tree, test_data, test_labels))

# clf = ClassificationTree(max_depth=20, random_state=RAND_SEED)
#
# time_start = time.perf_counter()
# clf.fit(train_data, train_labels)
# time_end = time.perf_counter()  # 记录结束时间
# run_time = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
# train_accuracy = clf.score(train_data, train_labels)
# test_accuracy = clf.score(test_data, test_labels)
# record.append([channels, run_time, train_accuracy, test_accuracy])
#
# time_start = time.perf_counter()
# clf.fit(train_data, transfer_labels)
# time_end = time.perf_counter()  # 记录结束时间
# run_time = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
# train_accuracy = clf.score(train_data, transfer_labels)
# test_accuracy = clf.score(test_data, test_labels)
# record.append([channels, run_time, train_accuracy, test_accuracy])
