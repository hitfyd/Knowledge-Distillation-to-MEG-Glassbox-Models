# 验证将软标签迁移到降维方法是否有增益

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import tensor

from Classifier import init_global_network_parameters, init_models, LFCNN, VARCNN, HGRN
from EarlyStopping import EarlyStopping
from ExperimentRecord import ExperimentRecord, checkpoint_dir, get_project_path
from TorchUtil import weight_reset, train, test, get_data_loader, setup_seed, set_device, save_checkpoint, \
    get_data_labels_from_dataset, get_labels_info, TrainResult, restore_baseline_checkpoint, predict, train_soft, \
    test_soft

# 从配置文件中读取超参数
RAND_SEED = 0
GPU_ID = 0
dataset = 'CamCAN'
validate_rate = 0.1
channels = 51
points = 25
classes = 2
models_name = ['LFCNN', 'VARCNN', 'HGRN']
BATCH_SIZE = [64, 128]
MAX_TRAIN_EPOCHS = 50
EARLY_STOP = 10
LEARN_RATE = [0.0003, 0.001, 0.003]
L1_penalty = [0, 0.0003]
L2_penalty = [0, 0.000001]
repetition_times = 10

assert isinstance(models_name, list)
assert isinstance(BATCH_SIZE, list) and isinstance(LEARN_RATE, list) and \
       isinstance(L1_penalty, list) and isinstance(L2_penalty, list)

# 固定随机数种子
setup_seed(RAND_SEED)
# 设置运算硬件
set_device(GPU_ID)

# 初始化模型
init_global_network_parameters(channels=204, points=100, classes=classes)
model_list = [LFCNN(), VARCNN(), HGRN()]  # LFCNN(), VARCNN(), HGRN()
baseline_checkpoint = 20220616160458  # DecMeg2014：20220616192753     CamCAN：20220616160458
if dataset == 'DecMeg2014':
    baseline_checkpoint = 20220616192753
restore_baseline_checkpoint(model_list, get_project_path() + '/checkpoint/Models_Train/', dataset, baseline_checkpoint)

init_global_network_parameters(channels=channels, points=points, classes=classes)
models = []
for model in init_models():
    if model.__class__.__name__ in models_name:
        models.append(model)

# 创建运行记录文件
record = ExperimentRecord()
# 记录超参数
record.append([])

# 根据数据集名称读取预设数据集，分为训练集和测试集
assert dataset == 'CamCAN' or dataset == 'DecMeg2014'
train_path = get_project_path() + '/dataset/{}_train.npz'.format(dataset)
test_path = get_project_path() + '/dataset/{}_test.npz'.format(dataset)
train_data, _ = get_data_labels_from_dataset(train_path)
test_data, test_labels = get_data_labels_from_dataset(test_path)

train_labels, _ = predict(model_list[0], tensor(train_data))

# 训练集进一步划分训练集和验证集
train_data, validate_data, train_labels, validate_labels = train_test_split(train_data, train_labels,
                                                                            test_size=validate_rate,
                                                                            random_state=RAND_SEED,
                                                                            shuffle=True,)
                                                                            # stratify=train_labels)

# record.append([get_labels_info(test_labels), get_labels_info(validate_labels), get_labels_info(train_labels)])
record.append(['model', 'BATCH_SIZE', 'LEARN_RATE', 'L1_penalty', 'L2_penalty'])
# 结果汇总
overall = {}
for model in models:
    model_name = model.__class__.__name__
    print(model_name)
    overall[model_name] = []
    best_hyperparams = []
    best_score = float('inf')
    best_checkpoint_path = '{}{}/{}_{}_{}_checkpoint.pt'.format(checkpoint_dir, record.run_py_name,
                                                                dataset, model_name, record.time)

    npz_name = get_project_path() + '/dataset/{}_mean_heatmap.npz'.format(model_name)
    heatmap_db = np.load(npz_name)
    heatmap_channel = heatmap_db['heatmap_channel']
    top_index = np.argsort(-heatmap_channel)[:channels]
    dr_train_data = train_data[:, top_index, :]
    dr_validate_data = validate_data[:, top_index, :]
    dr_test_data = test_data[:, top_index, :]
    select_times = range(50, 75)
    dr_train_data = dr_train_data[:, :, select_times]
    dr_validate_data = dr_validate_data[:, :, select_times]
    dr_test_data = dr_test_data[:, :, select_times]


    test_loader = get_data_loader(dr_test_data, test_labels, BATCH_SIZE[-1], shuffle=False)
    validate_loader = get_data_loader(dr_validate_data, validate_labels, BATCH_SIZE[-1])

    # 进行超参数网格搜索，记录最佳超参数及其训练后的模型
    for batch_size in BATCH_SIZE:
        for learn_rate in LEARN_RATE:
            for l1 in L1_penalty:
                for l2 in L2_penalty:
                    record.append([model_name, batch_size, learn_rate, l1, l2])
                    train_loader = get_data_loader(dr_train_data, train_labels, batch_size)

                    # 开始模型训练
                    validate_accuracy_list, validate_loss_list, train_accuracy_list, train_loss_list = [], [], [], []
                    # 重置模型参数
                    model.zero_grad()
                    model.apply(weight_reset)

                    early_stopping = EarlyStopping(patience=EARLY_STOP, verbose=True)
                    for epoch in range(MAX_TRAIN_EPOCHS):
                        train_accuracy, train_loss = train_soft(model, train_loader, epoch, learn_rate, l1, l2)
                        validate_accuracy, validate_loss = test_soft(model, validate_loader, validate=True)

                        train_accuracy_list.append(train_accuracy)
                        train_loss_list.append(train_loss)
                        validate_accuracy_list.append(validate_accuracy)
                        validate_loss_list.append(validate_loss)

                        early_stopping(validate_loss, model, epoch)
                        # 若满足 early stopping 要求
                        if early_stopping.early_stop:
                            print("Early stopping")
                            # 结束模型训练
                            break

                    # 记录超参数在验证集上的结果
                    best_epoch = early_stopping.best_epoch
                    output = [best_epoch, validate_accuracy_list[best_epoch], validate_loss_list[best_epoch],
                              validate_accuracy_list, validate_loss_list,
                              train_accuracy_list, train_loss_list]
                    record.append(output)
                    # 更新最佳超参数
                    if validate_loss_list[best_epoch] < best_score:
                        best_hyperparams = [batch_size, learn_rate, l1, l2]
                        best_score = validate_loss_list[best_epoch]
                        save_checkpoint(early_stopping.best_model_parameters, best_checkpoint_path)

    # 恢复验证集上的最佳模型，在测试集上执行
    model.load_state_dict(torch.load(best_checkpoint_path))
    test_accuracy, test_loss = test(model, test_loader)
    record.append(['model', 'test_accuracy', 'best_hyperparams', 'best_score', 'best_checkpoint_path'])
    output = [model_name, test_accuracy, best_hyperparams, best_score, best_checkpoint_path]
    record.append(output)

    # 在最佳超参数上多次执行训练、验证和测试，统计平均表现
    record.append(['model', 'best_hyperparams', 'repetition_times'])
    record.append([model_name, best_hyperparams, repetition_times])
    best_test_accuracy = 0
    validate_times, test_times = [], []
    for time in range(repetition_times):
        train_loader = get_data_loader(dr_train_data, train_labels, best_hyperparams[0])

        # 开始模型训练
        validate_accuracy_list, validate_loss_list, train_accuracy_list, train_loss_list = [], [], [], []
        test_accuracy_list, test_loss_list = [], []
        # 重置模型参数
        model.zero_grad()
        model.apply(weight_reset)

        early_stopping = EarlyStopping(patience=EARLY_STOP, verbose=True)
        for epoch in range(MAX_TRAIN_EPOCHS):
            train_accuracy, train_loss = train_soft(model, train_loader, epoch, best_hyperparams[1], best_hyperparams[2], best_hyperparams[3])
            validate_accuracy, validate_loss = test_soft(model, validate_loader, validate=True)
            test_accuracy, test_loss = test(model, test_loader)

            train_accuracy_list.append(train_accuracy)
            train_loss_list.append(train_loss)
            validate_accuracy_list.append(validate_accuracy)
            validate_loss_list.append(validate_loss)
            test_accuracy_list.append(test_accuracy)
            test_loss_list.append(test_loss)

            early_stopping(validate_loss, model, epoch)

        # 恢复验证集上的最佳模型，在测试集上执行
        model.load_state_dict(early_stopping.best_model_parameters)
        best_epoch = early_stopping.best_epoch
        test_accuracy, test_loss = test(model, test_loader)
        validate_times.append(validate_accuracy_list[best_epoch])
        test_times.append(test_accuracy)
        # 记录超参数在验证集上的结果
        output = [time, best_epoch, test_accuracy, test_loss,
                  validate_accuracy_list[best_epoch], validate_loss_list[best_epoch],
                  validate_accuracy_list, validate_loss_list,
                  train_accuracy_list, train_loss_list,
                  test_accuracy_list, test_loss_list]
        record.append(output)
        # 保存最佳测试精度下的模型
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            record.append(['update best_model_parameters', time, best_test_accuracy])
            save_checkpoint(early_stopping.best_model_parameters, best_checkpoint_path)

        # 保存训练结果
        result_id = '{}_{}_{}'.format(dataset, model_name, time)
        result = TrainResult(result_id, dataset, channels, points, classes, model_name,
                             best_hyperparams[0], MAX_TRAIN_EPOCHS, EARLY_STOP, best_hyperparams[1], best_hyperparams[2], best_hyperparams[3],
                             test_accuracy, test_loss,
                             validate_accuracy_list, validate_loss_list, train_accuracy_list, train_loss_list,
                             test_accuracy_list, test_loss_list)

    record.append(['model', 'test_mean', 'test_std', 'validate_mean', 'validate_std', 'test_times', 'validate_times', 'best_checkpoint_path'])
    output = [model_name, np.mean(test_times), np.std(test_times), np.mean(validate_times), np.std(validate_times), test_times,  validate_times, best_checkpoint_path]
    record.append(output)
    overall[model_name] = output

record.append('overall')
for key in overall.keys():
    record.append(overall[key])
