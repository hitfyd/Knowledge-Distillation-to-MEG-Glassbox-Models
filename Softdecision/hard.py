"""Training and evaluating a soft decision tree on the MNIST dataset."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torchvision import datasets, transforms

from Classifier import init_global_network_parameters, LFCNN
from ExperimentRecord import get_project_path
from SoftDecisionTree import SDT
from TorchUtil import get_data_labels_from_dataset, get_data_loader, set_device, restore_baseline_checkpoint, predict, \
    setup_seed

# def onehot_coding(target, device, output_dim):
#     """Convert the class labels into one-hot encoded vectors."""
#     target_onehot = torch.FloatTensor(target.size()[0], output_dim).to(device)
#     target_onehot.data.zero_()
#     target_onehot.scatter_(1, target.view(-1, 1), 1.0)
#     return target_onehot


if __name__ == "__main__":
    # Load data
    dataset = 'CamCAN'

    train_path = get_project_path() + '/dataset/{}_train.npz'.format(dataset)
    test_path = get_project_path() + '/dataset/{}_test.npz'.format(dataset)
    train_data, train_labels = get_data_labels_from_dataset(train_path)
    test_data, test_labels = get_data_labels_from_dataset(test_path)

    # 固定随机数种子
    setup_seed(1)
    # 设置运算硬件
    set_device(-1)

    # 初始化模型
    init_global_network_parameters(channels=204, points=100, classes=2)
    model = LFCNN()
    baseline_checkpoint = 20220616160458  # DecMeg2014：20220616192753     CamCAN：20220616160458
    restore_baseline_checkpoint(model, get_project_path() + '/checkpoint/Models_Train/', dataset,
                                baseline_checkpoint)
    transfer_labels, _ = predict(model, tensor(train_data))
    # T = 1
    # for i in range(len(transfer_labels)):
    #     exp_sum = np.exp(transfer_labels[i][0] / T) + np.exp(transfer_labels[i][1] / T)
    #     transfer_labels[i] = [np.exp(transfer_labels[i][0] / T) / exp_sum, np.exp(transfer_labels[i][1] / T) / exp_sum]

    # Parameters
    input_dim = 204 * 100  # the number of input dimensions
    output_dim = 2  # the number of outputs (i.e., # classes on MNIST)
    depth = 10  # tree depth
    lamda = 1e-3  # coefficient of the regularization term
    lr = 1e-3  # learning rate
    weight_decaly = 5e-4  # weight decay
    batch_size = 256  # batch size
    epochs = 50  # the number of training epochs
    log_interval = 100  # the number of batches to wait before printing logs
    use_cuda = False  # whether to use GPU
    soft = True

    # Model and Optimizer
    tree = SDT(input_dim, output_dim, depth, lamda, use_cuda)

    optimizer = torch.optim.Adam(tree.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decaly)
    if soft:
        train_loader = get_data_loader(train_data, transfer_labels, batch_size)
    else:
        train_loader = get_data_loader(train_data, train_labels, batch_size)
    test_loader = get_data_loader(test_data, test_labels, batch_size)

    # Utils
    best_testing_acc = 0.0
    testing_acc_list = []
    training_loss_list = []
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if use_cuda else "cpu")

    correct = 0.
    for batch_idx, (data, target) in enumerate(test_loader):
        batch_size = data.size()[0]
        data, target = data.to(device), target.to(device)

        output = F.softmax(model.forward(data), dim=1)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1).data).sum()

    accuracy = 100.0 * float(correct) / len(test_loader.dataset)

    msg = (
        "\nTesting Accuracy: {}/{} ({:.3f}%) |"
    )
    print(
        msg.format(
            correct,
            len(test_loader.dataset),
            accuracy,
        )
    )

    for epoch in range(epochs):

        # Training
        tree.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            batch_size = data.size()[0]
            data, target = data.to(device), target.to(device)
            # target_onehot = onehot_coding(target, device, output_dim)

            output, penalty = tree.forward(data, is_training_data=True)

            if soft:
                loss = criterion(output, target)
            else:
                loss = criterion(output, target.view(-1))
            loss += penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training status
            if batch_idx % log_interval == 0:
                pred = output.data.max(1)[1]
                if soft:
                    correct = pred.eq(target.data.max(1)[1]).sum()
                else:
                    correct = pred.eq(target.view(-1).data).sum()

                msg = (
                    "Epoch: {:02d} | Batch: {:03d} | Loss: {:.5f} |"
                    " Correct: {:03d}/{:03d}"
                )
                print(msg.format(epoch, batch_idx, loss, correct, batch_size))
                training_loss_list.append(loss.cpu().data.numpy())

        # Evaluating
        tree.eval()
        correct = 0.

        for batch_idx, (data, target) in enumerate(test_loader):
            batch_size = data.size()[0]
            data, target = data.to(device), target.to(device)

            output = F.softmax(tree.forward(data), dim=1)

            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()

        accuracy = 100.0 * float(correct) / len(test_loader.dataset)

        if accuracy > best_testing_acc:
            best_testing_acc = accuracy

        msg = (
            "\nEpoch: {:02d} | Testing Accuracy: {}/{} ({:.3f}%) |"
            " Historical Best: {:.3f}%\n"
        )
        print(
            msg.format(
                epoch, correct,
                len(test_loader.dataset),
                accuracy,
                best_testing_acc
            )
        )
        testing_acc_list.append(accuracy)