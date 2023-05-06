import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from Util.TorchUtil import get_data_loader_from_dataset

cudnn.benchmark = True

from distilllib.models import CamCAN_model_dict
from distilllib.distillers import distiller_dict
from distilllib.engine.utils import load_checkpoint, log_msg
from distilllib.engine.cfg import CFG as cfg
from distilllib.engine.cfg import show_cfg
from distilllib.engine import trainer_dict


def main(cfg, resume, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    train_loader = get_data_loader_from_dataset('../dataset/CamCAN_train.npz', cfg.SOLVER.BATCH_SIZE)
    val_loader = get_data_loader_from_dataset('../dataset/CamCAN_test.npz', cfg.DATASET.TEST.BATCH_SIZE)
    num_classes = 2
    # train_loader, val_loader, num_classes = get_dataset(cfg)

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "CamCAN":
            model_student = CamCAN_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        else:
            model_student = CamCAN_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = CamCAN_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = CamCAN_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            net, pretrain_model_path = CamCAN_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            if cfg.DATASET.TYPE == "CamCAN":
                model_teacher.load_state_dict(load_checkpoint(pretrain_model_path))
            else:
                model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = CamCAN_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )

        distiller = distiller_dict[cfg.DISTILLER.TYPE](
            model_student, model_teacher, cfg
        )
    distiller = torch.nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, args.resume, args.opts)
