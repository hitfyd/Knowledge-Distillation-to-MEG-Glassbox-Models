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
CFG.EXPERIMENT.SEED = 0  # Random number seed, which is beneficial to the repeatability of the experiment.
CFG.EXPERIMENT.GPU_IDS = "1"    # List of GPUs used
CFG.EXPERIMENT.REPETITION_NUM = 5   # Number of repetition times

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
CFG.DISTILLER.STUDENT = "sdt"

# Solver
CFG.SOLVER = CN()
CFG.SOLVER.TRAINER = "base"
CFG.SOLVER.BATCH_SIZE = 1024   # Grid search
CFG.SOLVER.EPOCHS = 100
CFG.SOLVER.LR = 0.003
# CFG.SOLVER.LR_DECAY_STAGES = [150, 180, 210]
# CFG.SOLVER.LR_DECAY_RATE = 0.1
CFG.SOLVER.WEIGHT_DECAY = 0.0005
CFG.SOLVER.MOMENTUM = 0.9
CFG.SOLVER.TYPE = "SGD"

# Log
CFG.LOG = CN()
# CFG.LOG.SAVE_CHECKPOINT_FREQ = 20
CFG.LOG.PREFIX = "./output"
CFG.LOG.WANDB = False

# Distillation Methods

# KD CFG
CFG.KD = CN()
CFG.KD.TEMPERATURE = 4  # Grid search
CFG.KD.LOSS = CN()
CFG.KD.LOSS.CE_WEIGHT = 0.1
CFG.KD.LOSS.KD_WEIGHT = 0.9

# ESKD CFG
CFG.ESKD = CN()
CFG.ESKD.TEMPERATURE = 4    # Grid search
CFG.ESKD.STOP_EPOCH = 75   # Grid search
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
CFG.DKD.CE_WEIGHT = 1
CFG.DKD.ALPHA = 1
CFG.DKD.BETA = 8
CFG.DKD.T = 4
CFG.DKD.WARMUP = 20

# # CLKD CFG
# CFG.CLKD = CN()
# CFG.CLKD.LOSS = CN()
# CFG.CLKD.LOSS.CE_WEIGHT = 0.1
# CFG.CLKD.LOSS.KD_WEIGHT = 0.5
# CFG.CLKD.LOSS.CLA_COEFFICIENT = 1
# CFG.CLKD.LOSS.CC_WEIGHT = 0.4

# SCFAKD CFG
CFG.SCFAKD = CN()
CFG.SCFAKD.LOSS = CN()
CFG.SCFAKD.LOSS.CE_WEIGHT = 1     # +KD时为0.1，+DKD时为1，FAKD时为1
CFG.SCFAKD.LOSS.FA_WEIGHT = 1000
CFG.SCFAKD.WITH_KD = False
CFG.SCFAKD.TEMPERATURE = 4
CFG.SCFAKD.LOSS.KD_WEIGHT = 0     # +KD时为0.9

# ShapleyFAKD CFG
CFG.ShapleyFAKD = CN()
CFG.ShapleyFAKD.M = 1
CFG.ShapleyFAKD.LOSS = CN()
CFG.ShapleyFAKD.LOSS.CE_WEIGHT = 1     # +KD时为0.1，+DKD时为1，FAKD时为1
CFG.ShapleyFAKD.LOSS.FA_WEIGHT = 1000
CFG.ShapleyFAKD.WITH_KD = False
CFG.ShapleyFAKD.TEMPERATURE = 4
CFG.ShapleyFAKD.LOSS.KD_WEIGHT = 0     # +KD时为0.9
