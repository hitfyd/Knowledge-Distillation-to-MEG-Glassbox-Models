import time

from sklearn import tree
from sklearn.model_selection import train_test_split
from interpret.glassbox import ClassificationTree, DecisionListClassifier

from ExperimentRecord import get_project_path
from TorchUtil import get_data_labels_from_dataset

RAND_SEED = 16
validate_rate = 0.1
dataset = 'CamCAN'
# 根据数据集名称读取预设数据集，分为训练集和测试集
assert dataset == 'CamCAN' or dataset == 'DecMeg2014'
train_path = get_project_path() + '/dataset/{}_train.npz'.format(dataset)
test_path = get_project_path() + '/dataset/{}_test.npz'.format(dataset)
train_data, train_labels = get_data_labels_from_dataset(train_path)
test_data, test_labels = get_data_labels_from_dataset(test_path)

train_data, test_data = train_data.reshape(len(train_labels), -1), test_data.reshape(len(test_labels), -1)

# 训练集进一步划分训练集和验证集
# train_data, validate_data, train_labels, validate_labels \
#     = train_test_split(train_data, train_labels, test_size=validate_rate, random_state=RAND_SEED, shuffle=True,
#                        stratify=train_labels)

time_start = time.perf_counter()

# clf = tree.DecisionTreeClassifier(max_depth=6)
# clf = clf.fit(train_data, train_labels)

clf = DecisionListClassifier(random_state=RAND_SEED)
# clf = ClassificationTree(max_depth=10, random_state=RAND_SEED)
clf.fit(train_data, train_labels)

time_end = time.perf_counter()  # 记录结束时间
run_time = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(run_time)


def predict_tree(model, data, labels):
    predict_labels = model.predict(data)
    correct = 0
    for i in range(len(labels)):
        if predict_labels[i] == labels[i]:
            correct += 1
    test_accuracy = 100. * correct / len(labels)
    return test_accuracy


print(predict_tree(clf, train_data, train_labels))
print(predict_tree(clf, test_data, test_labels))

# from sklearn.tree import export_text
# r = export_text(clf)
# print(r)
