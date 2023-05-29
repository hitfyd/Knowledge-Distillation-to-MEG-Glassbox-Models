import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import getpass
from .utils import (
    AverageMeter,
    accuracy,
    validate,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
)


class BaseTrainer(object):
    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg):
        self.cfg = cfg
        self.distiller = distiller
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = self.init_optimizer(cfg)
        self.best_acc = -1

        username = getpass.getuser()
        # init loggers
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        # self.tf_writer = SummaryWriter(os.path.join(self.log_path, "train.events"))

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif cfg.SOLVER.TYPE == "Adam":
            optimizer = optim.Adam(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def log(self, epoch, log_dict):
        if self.cfg.LOG.WANDB:
            import wandb

            # wandb.log({"current lr": lr})
            wandb.log(log_dict)
        if log_dict["test_acc"] > self.best_acc:
            self.best_acc = log_dict["test_acc"]
            if self.cfg.LOG.WANDB:
                wandb.run.summary["best_acc"] = self.best_acc
        # worklog.txt
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            lines = [
                "epoch: {}\t".format(epoch),
            ]
            for k, v in log_dict.items():
                lines.append("{}: {:.4f}\t".format(k, v))
            lines.append(os.linesep)
            writer.writelines(lines)

    # 增加L1正则化
    def __l1_regularization__(self, l1_penalty=3e-4):
        regularization_loss = 0
        for param in self.distiller.module.get_learnable_parameters():
            regularization_loss += torch.sum(abs(param))  # torch.norm(param, p=1)
        return l1_penalty * regularization_loss

    def train(self, resume=False, repetition_id=0):
        epoch = 0
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS:
            self.train_epoch(epoch, repetition_id=repetition_id)
            epoch += 1
        print(log_msg("repetition_id:{} Best accuracy:{}".format(repetition_id, self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("repetition_id:{}\tbest_acc:{:.4f}".format(repetition_id, float(self.best_acc)))
            writer.write(os.linesep + "-" * 25 + os.linesep)
        return self.best_acc

    def train_epoch(self, epoch, repetition_id=0):
        # lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        lr = self.cfg.SOLVER.LR
        train_meters = {
            "training_time": AverageMeter(),
            # "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter))

        # train loops
        self.distiller.train()
        for idx, (data, target) in enumerate(self.train_loader):
            # if idx >= 10:   # 临时截断，缩短训练时间
            #     break
            msg = self.train_iter(data, target, epoch, train_meters, idx)
            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()
        pbar.close()

        # validate
        test_acc, test_loss = validate(self.val_loader, self.distiller)

        # log
        log_dict = OrderedDict(
            {
                "train_acc": train_meters["top1"].avg,
                "train_loss": train_meters["losses"].avg,
                "test_acc": test_acc,
                "test_loss": test_loss,
            }
        )
        self.log(epoch, log_dict)
        # saving checkpoint
        state = {
            "epoch": epoch,
            "model": self.distiller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }
        student_state = self.distiller.module.student.state_dict()
        # save_checkpoint(state, os.path.join(self.log_path, "latest"))
        # save_checkpoint(
        #     student_state, os.path.join(self.log_path, "student_latest")
        # )
        # if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
        #     save_checkpoint(
        #         state, os.path.join(self.log_path, "epoch_{}".format(epoch))
        #     )
        #     save_checkpoint(
        #         student_state, os.path.join(self.log_path, "student_{}".format(epoch)),
        #     )
        # update the best
        if test_acc >= self.best_acc:
            save_checkpoint(state, os.path.join(self.log_path, "state_best_{}".format(repetition_id)))
            save_checkpoint(
                student_state, os.path.join(self.log_path, "student_best_{}".format(repetition_id))
            )

    def train_iter(self, data, target, epoch, train_meters, data_itx: int = 0):  # data_itx参数只在FAKD中使用
        self.optimizer.zero_grad()
        train_start_time = time.time()

        # train_meters["data_time"].update(time.time() - train_start_time)
        data = data.float()
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(data=data, target=target, epoch=epoch, data_itx=data_itx)

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])    # + self.__l1_regularization__()
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = data.size(0)
        acc1, _ = accuracy(preds, target, topk=(1, 2))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        # print info
        msg = "Epoch:{}|Time(train):{:.2f}|Loss:{:.2f}|Top-1:{:.2f}".format(
            epoch,
            # train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
        )
        return msg
