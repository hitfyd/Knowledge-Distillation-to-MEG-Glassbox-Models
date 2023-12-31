import os
import sys
import argparse
from statistics import mean, pstdev

import torch

from distilllib.distillers import distiller_dict
from distilllib.engine import trainer_dict
from distilllib.engine.cfg import CFG as cfg
from distilllib.engine.cfg import show_cfg
from distilllib.engine.utils import load_checkpoint, log_msg, get_data_loader_from_dataset, setup_seed
from distilllib.models import model_dict


def main(cfg, resume, opts):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS
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

    best_acc_l = []
    for repetition_id in range(cfg.EXPERIMENT.REPETITION_NUM):
        # set the random number seed
        setup_seed(cfg.EXPERIMENT.SEED+repetition_id)

        # init dataloader & models
        train_loader = get_data_loader_from_dataset('../dataset/{}_train.npz'.format(cfg.DATASET.TYPE),
                                                    cfg.SOLVER.BATCH_SIZE)
        val_loader = get_data_loader_from_dataset('../dataset/{}_test.npz'.format(cfg.DATASET.TYPE),
                                                  cfg.DATASET.TEST.BATCH_SIZE)

        # vanilla
        if cfg.DISTILLER.TYPE == "NONE":
            model_student = model_dict[cfg.DISTILLER.STUDENT][0](
                channels=cfg.DATASET.CHANNELS, points=cfg.DATASET.POINTS, num_classes=cfg.DATASET.NUM_CLASSES)
            distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
        # distillation
        else:
            print(log_msg("Loading teacher model", "INFO"))
            net, pretrain_model_path = model_dict[cfg.DISTILLER.TEACHER]
            assert (pretrain_model_path is not None), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(
                channels=cfg.DATASET.CHANNELS, points=cfg.DATASET.POINTS, num_classes=cfg.DATASET.NUM_CLASSES)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path))

            model_student = model_dict[cfg.DISTILLER.STUDENT][0](
                channels=cfg.DATASET.CHANNELS, points=cfg.DATASET.POINTS, num_classes=cfg.DATASET.NUM_CLASSES)

            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )
        distiller = torch.nn.DataParallel(distiller.cuda())

        # train
        trainer = trainer_dict[cfg.SOLVER.TRAINER](experiment_name, distiller, train_loader, val_loader, cfg)
        best_acc = trainer.train(resume=resume, repetition_id=repetition_id)
        best_acc_l.append(float(best_acc))
    print(log_msg("best_acc(mean±std)\t{:.2f} ± {:.2f}\t{}".format(mean(best_acc_l), pstdev(best_acc_l), best_acc_l), "INFO"))
    with open(os.path.join(trainer.log_path, "worklog.txt"), "a") as writer:
        writer.write("CONFIG:\n{}".format(cfg.dump()))
        writer.write(os.linesep + "-" * 25 + os.linesep)
        writer.write("best_acc(mean±std)\t{:.2f} ± {:.2f}\t{}".format(mean(best_acc_l), pstdev(best_acc_l), best_acc_l))
        writer.write(os.linesep + "-" * 25 + os.linesep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # Debug模式下不上传wandb
    isDebug = True if sys.gettrace() else False
    cfg.LOG.WANDB = cfg.LOG.WANDB and not isDebug
    cfg.freeze()

    main(cfg, args.resume, args.opts)
