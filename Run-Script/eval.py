import argparse
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from distilllib.distillers import Vanilla
from distilllib.models import model_dict
from distilllib.engine.utils import load_checkpoint, validate, get_data_loader_from_dataset
from distilllib.engine.cfg import CFG as cfg

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
    model.load_state_dict(load_checkpoint(pretrain_model_path))
    model = Vanilla(model)
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    test_acc, test_loss = validate(val_loader, model)
