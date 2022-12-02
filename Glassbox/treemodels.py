import os
import time
# from pickle import dump, load
from joblib import dump, load

from interpret.glassbox import ClassificationTree, DecisionListClassifier, LogisticRegression, \
    ExplainableBoostingClassifier
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA

from ExperimentRecord import get_project_path, checkpoint_dir, ExperimentRecord
from TorchUtil import get_data_labels_from_dataset

RAND_SEED = 16
n_jobs = -1
dataset = 'CamCAN'  # 'DecMeg2014':
all_channels = 204
channels_list = [12, 25, 51, 102, 204]
model_list = ['EBM', 'linear', 'tree', 'rule']  # 'EBM', 'linear', 'tree', 'rule'
# 根据数据集名称读取预设数据集，分为训练集和测试集
assert dataset == 'CamCAN' or dataset == 'DecMeg2014'
train_path = get_project_path() + '/dataset/{}_train.npz'.format(dataset)
test_path = get_project_path() + '/dataset/{}_test.npz'.format(dataset)
origin_train_data, train_labels = get_data_labels_from_dataset(train_path)
origin_test_data, test_labels = get_data_labels_from_dataset(test_path)


def predict_tree(model, data, labels):
    predict_labels = model.predict(data)
    correct = 0
    for i in range(len(labels)):
        if predict_labels[i] == labels[i]:
            correct += 1
    accuracy = 100. * correct / len(labels)
    return accuracy


record = ExperimentRecord()
record.append([RAND_SEED, n_jobs, dataset, channels_list, model_list])

for channels in channels_list:
    if channels < all_channels:
        pca_filter = UnsupervisedSpatialFilter(PCA(channels), average=False)
        train_data = pca_filter.fit_transform(origin_train_data)
        test_data = pca_filter.transform(origin_test_data)
    else:
        train_data = origin_train_data
        test_data = origin_test_data

    train_data, test_data = train_data.reshape(len(train_labels), -1), test_data.reshape(len(test_labels), -1)

    for model_name in model_list:
        if model_name == 'linear':
            clf = LogisticRegression(max_iter=5000, random_state=RAND_SEED)
        elif model_name == 'tree':
            clf = ClassificationTree(max_depth=20, random_state=RAND_SEED)
        elif model_name == 'rule':
            clf = DecisionListClassifier(random_state=RAND_SEED, n_jobs=n_jobs)  # 效果最差
        else:
            clf = ExplainableBoostingClassifier(random_state=RAND_SEED, n_jobs=n_jobs)
            # # 未降维时EBM模型内存占用过大，无法执行，因此跳过
            # if channels == all_channels:
            #     continue
        time_start = time.perf_counter()
        clf.fit(train_data, train_labels)
        time_end = time.perf_counter()  # 记录结束时间
        run_time = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

        train_accuracy = predict_tree(clf, train_data, train_labels)
        test_accuracy = predict_tree(clf, test_data, test_labels)

        record.append([channels, model_name, run_time, train_accuracy, test_accuracy])
        checkpoint_name = '{}{}/{}_{}_{}_{}_checkpoint.pt'.format(checkpoint_dir, record.run_py_name,
                                                                  dataset, channels, model_name, record.time)
        os.makedirs(os.path.dirname(checkpoint_name), exist_ok=True)
        dump(clf, checkpoint_name)
        print(channels, model_name)
        model_name = load(checkpoint_name)
        print(predict_tree(model_name, test_data, test_labels))
