from yacs.config import CfgNode as CN
from .utils import log_msg


def show_cfg(cfg):
    dump_cfg = CN()
    dump_cfg.EXPERIMENT = cfg.EXPERIMENT
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.DISTILLER = cfg.DISTILLER
    dump_cfg.SOLVER = cfg.SOLVER
    dump_cfg.LOG = cfg.LOG
    if cfg.DISTILLER.TYPE in cfg:
        dump_cfg.update({cfg.DISTILLER.TYPE: cfg.get(cfg.DISTILLER.TYPE)})
    print(log_msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO"))


CFG = CN()

# Experiment
CFG.EXPERIMENT = CN()
CFG.EXPERIMENT.PROJECT = "distill"
CFG.EXPERIMENT.NAME = ""
CFG.EXPERIMENT.TAG = "default"
CFG.EXPERIMENT.SEED = 0

# Dataset
CFG.DATASET = CN()
CFG.DATASET.TYPE = "CamCAN"
CFG.DATASET.CHANNELS = 204
CFG.DATASET.POINTS = 100
CFG.DATASET.NUM_CLASSES = 2
CFG.DATASET.NUM_WORKERS = 2
CFG.DATASET.TEST = CN()
CFG.DATASET.TEST.BATCH_SIZE = 1024

# Distiller
CFG.DISTILLER = CN()
CFG.DISTILLER.TYPE = "NONE"  # Vanilla as default
CFG.DISTILLER.TEACHER = "lfcnn"
CFG.DISTILLER.STUDENT = "std"

# Solver
CFG.SOLVER = CN()
CFG.SOLVER.TRAINER = "base"
CFG.SOLVER.BATCH_SIZE = 1024
CFG.SOLVER.EPOCHS = 100
CFG.SOLVER.LR = 0.003
CFG.SOLVER.LR_DECAY_STAGES = [150]
CFG.SOLVER.LR_DECAY_RATE = 0.1
CFG.SOLVER.WEIGHT_DECAY = 0.0005
CFG.SOLVER.MOMENTUM = 0.9
CFG.SOLVER.TYPE = "SGD"

# Log
CFG.LOG = CN()
CFG.LOG.SAVE_CHECKPOINT_FREQ = 20
CFG.LOG.PREFIX = "./output"
CFG.LOG.WANDB = True

# Distillation Methods

# KD CFG
CFG.KD = CN()
CFG.KD.TEMPERATURE = 4
CFG.KD.LOSS = CN()
CFG.KD.LOSS.CE_WEIGHT = 0.1
CFG.KD.LOSS.KD_WEIGHT = 0.9

# ESKD CFG
CFG.ESKD = CN()
CFG.ESKD.TEMPERATURE = 4
CFG.ESKD.STOP_EPOCH = 75
CFG.ESKD.LOSS = CN()
CFG.ESKD.LOSS.CE_WEIGHT = 0.1
CFG.ESKD.LOSS.KD_WEIGHT = 0.9

# GKD CFG
CFG.GKD = CN()
CFG.GKD.TEMPERATURE = 4
CFG.GKD.LOSS = CN()
CFG.GKD.LOSS.CE_WEIGHT = 0.1
CFG.GKD.LOSS.KD_WEIGHT = 0.9

# MSEKD CFG
CFG.MSEKD = CN()
CFG.MSEKD.LOSS = CN()
CFG.MSEKD.LOSS.CE_WEIGHT = 0.1
CFG.MSEKD.LOSS.KD_WEIGHT = 0.9

# DKD(Decoupled Knowledge Distillation) CFG
CFG.DKD = CN()
CFG.DKD.CE_WEIGHT = 1.0
CFG.DKD.ALPHA = 1.0
CFG.DKD.BETA = 8.0
CFG.DKD.T = 4.0
CFG.DKD.WARMUP = 20

# CLKD CFG
CFG.CLKD = CN()
CFG.CLKD.LOSS = CN()
CFG.CLKD.LOSS.CE_WEIGHT = 0.1
CFG.CLKD.LOSS.KD_WEIGHT = 0.5
CFG.CLKD.LOSS.CLA_COEFFICIENT = 1
CFG.CLKD.LOSS.CC_WEIGHT = 0.4

# FAKD CFG
CFG.FAKD = CN()
CFG.FAKD.LOSS = CN()
CFG.FAKD.LOSS.CE_WEIGHT = 1
CFG.FAKD.LOSS.KD_WEIGHT = 100   # HGRN:1;其他100
