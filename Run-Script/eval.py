import argparse
from glob import glob
from statistics import mean, pstdev

import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from distilllib.distillers import Vanilla
from distilllib.models import model_dict
from distilllib.engine.utils import load_checkpoint, get_data_loader_from_dataset


def evaluate(val_loader, distiller):
    distiller.eval()
    all_num, tp, fn, fp, tn = 0, 0, 0, 0, 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(val_loader):
            data = data.float()
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = distiller(data=data)
            _, pred_labels = torch.max(output.data, 1)
            all_num += len(target)
            for i in range(len(target)):
                if target[i] == pred_labels[i]:
                    if target[i] == 1:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if target[i] == 1:
                        fn += 1
                    else:
                        fp += 1
    print('all: {}\ttp: {}\tfp: {}\tfn: {}\ttn:{}'.format(all_num, tp, fp, fn, tn))
    accuracy = 1.0 * (tp + tn) / all_num
    precision = 1.0 * tp / (tp + fp)
    recall = 1.0 * tp / (tp + fn)
    F1 = (2 * precision * recall) / (precision + recall)
    return accuracy, precision, recall, F1, tp, fn, fp, tn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="CamCAN_lfcnn", choices=
    ["CamCAN_lfcnn", "CamCAN_varcnn", "CamCAN_hgrn", "DecMeg2014_lfcnn", "DecMeg2014_varcnn", "DecMeg2014_hgrn", "sdt"])
    parser.add_argument("-d", "--dataset", type=str, default="CamCAN", choices=["CamCAN", "DecMeg2014"])
    parser.add_argument("-bs", "--batch-size", type=int, default=1024)
    parser.add_argument("-cp", "--pretrain-checkpoint", type=str, default="")
    args = parser.parse_args()

    model_name, dataset, batch_size, pretrain_checkpoint = \
        args.model, args.dataset, args.batch_size, args.pretrain_checkpoint
    val_loader = get_data_loader_from_dataset('../dataset/{}_test.npz'.format(dataset), batch_size)
    net, pretrain_model_path = model_dict[model_name]
    if model_name == "sdt":
        assert pretrain_checkpoint != ""
        pretrain_model_path = pretrain_checkpoint
    if dataset == "CamCAN":
        channels, points, num_classes = 204, 100, 2
    elif dataset == "DecMeg2014":
        channels, points, num_classes = 204, 250, 2
    else:
        raise Exception
    model = net(channels=channels, points=points, num_classes=num_classes)
    test_acc_l, test_f1_l = [], []
    for pretrain_model_path_i in glob(pretrain_model_path + 'student_best_*'):
        model.load_state_dict(load_checkpoint(pretrain_model_path_i))
        model_distiller = Vanilla(model)
        model_distiller = model_distiller.cuda()
        model_distiller = torch.nn.DataParallel(model_distiller)
        test_acc, precision, recall, test_f1, _, _, _, _ = evaluate(val_loader, model_distiller)
        print(test_acc, precision, recall, test_f1)
        test_acc_l.append(test_acc)
        test_f1_l.append(test_f1)
    print("best_acc(mean±std)\t{:.4f} ± {:.4f}\t{}".format(mean(test_acc_l), pstdev(test_acc_l), test_acc_l))
    print("best_f1(mean±std)\t{:.2f} ± {:.2f}\t{}".format(mean(test_f1_l), pstdev(test_f1_l), test_f1_l))
