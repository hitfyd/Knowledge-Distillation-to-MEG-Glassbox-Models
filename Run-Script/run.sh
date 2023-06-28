export PYTHONPATH=$PYTHONPATH:../

# Dataset: CamCAN; Vanilla
#python train.py --cfg ../configs/CamCAN/Vanilla/SDT.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/CamCAN/Vanilla/SDT.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/CamCAN/Vanilla/SDT.yaml  SOLVER.BATCH_SIZE 1024
#python train.py --cfg ../configs/CamCAN/Vanilla/LFCNN.yaml

# Dataset: CamCAN; Teacher: lfcnn; Student: soft decision tree
#python train.py --cfg ../configs/CamCAN/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256  KD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512  KD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 KD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256  KD.TEMPERATURE 8
#python train.py --cfg ../configs/CamCAN/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512  KD.TEMPERATURE 8
#python train.py --cfg ../configs/CamCAN/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 KD.TEMPERATURE 8
#
#python train.py --cfg ../configs/CamCAN/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 GKD.TEMPERATURE 8
#python train.py --cfg ../configs/CamCAN/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 GKD.TEMPERATURE 8
#python train.py --cfg ../configs/CamCAN/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 GKD.TEMPERATURE 8
##
#python train.py --cfg ../configs/CamCAN/MSEKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/CamCAN/MSEKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/CamCAN/MSEKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024
#
#python train.py --cfg ../configs/CamCAN/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 DKD.T 4
#python train.py --cfg ../configs/CamCAN/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 DKD.T 4
#python train.py --cfg ../configs/CamCAN/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 DKD.T 4
#python train.py --cfg ../configs/CamCAN/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 DKD.T 8
#python train.py --cfg ../configs/CamCAN/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 DKD.T 8
#python train.py --cfg ../configs/CamCAN/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 DKD.T 8


# Dataset: CamCAN; Teacher: varcnn; Student: soft decision tree
#python train.py --cfg ../configs/CamCAN/KD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256  KD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/KD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512  KD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/KD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 KD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/KD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256  KD.TEMPERATURE 8
#python train.py --cfg ../configs/CamCAN/KD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512  KD.TEMPERATURE 8
#python train.py --cfg ../configs/CamCAN/KD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 KD.TEMPERATURE 8
#
#python train.py --cfg ../configs/CamCAN/GKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/GKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/GKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/GKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 GKD.TEMPERATURE 8
#python train.py --cfg ../configs/CamCAN/GKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 GKD.TEMPERATURE 8
#python train.py --cfg ../configs/CamCAN/GKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 GKD.TEMPERATURE 8
#
#python train.py --cfg ../configs/CamCAN/MSEKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/CamCAN/MSEKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/CamCAN/MSEKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024
#
#python train.py --cfg ../configs/CamCAN/DKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 DKD.T 4
#python train.py --cfg ../configs/CamCAN/DKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 DKD.T 4
#python train.py --cfg ../configs/CamCAN/DKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 DKD.T 4
#python train.py --cfg ../configs/CamCAN/DKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 DKD.T 8
#python train.py --cfg ../configs/CamCAN/DKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 DKD.T 8
#python train.py --cfg ../configs/CamCAN/DKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 DKD.T 8


# Dataset: CamCAN; Teacher: hgrn; Student: soft decision tree
#python train.py --cfg ../configs/CamCAN/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256  KD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512  KD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 1024 KD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256  KD.TEMPERATURE 8
#python train.py --cfg ../configs/CamCAN/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512  KD.TEMPERATURE 8
#python train.py --cfg ../configs/CamCAN/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 1024 KD.TEMPERATURE 8
#
#python train.py --cfg ../configs/CamCAN/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 1024 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/CamCAN/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 GKD.TEMPERATURE 8
#python train.py --cfg ../configs/CamCAN/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512 GKD.TEMPERATURE 8
#python train.py --cfg ../configs/CamCAN/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 1024 GKD.TEMPERATURE 8
#
#python train.py --cfg ../configs/CamCAN/MSEKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/CamCAN/MSEKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/CamCAN/MSEKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 1024
#
#python train.py --cfg ../configs/CamCAN/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 DKD.T 4
#python train.py --cfg ../configs/CamCAN/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512 DKD.T 4
#python train.py --cfg ../configs/CamCAN/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 1024 DKD.T 4
#python train.py --cfg ../configs/CamCAN/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 DKD.T 8
#python train.py --cfg ../configs/CamCAN/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512 DKD.T 8
#python train.py --cfg ../configs/CamCAN/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 1024 DKD.T 8


# Dataset: DecMeg2014; Vanilla
#python train.py --cfg ../configs/DecMeg2014/Vanilla/SDT.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/DecMeg2014/Vanilla/SDT.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/DecMeg2014/Vanilla/SDT.yaml  SOLVER.BATCH_SIZE 1024
#python train.py --cfg ../configs/DecMeg2014/Vanilla/LFCNN.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/DecMeg2014/Vanilla/LFCNN.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/DecMeg2014/Vanilla/LFCNN.yaml  SOLVER.BATCH_SIZE 1024
#python train.py --cfg ../configs/DecMeg2014/Vanilla/VARCNN.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/DecMeg2014/Vanilla/VARCNN.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/DecMeg2014/Vanilla/VARCNN.yaml  SOLVER.BATCH_SIZE 1024
#python train.py --cfg ../configs/DecMeg2014/Vanilla/HGRN.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/DecMeg2014/Vanilla/HGRN.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/DecMeg2014/Vanilla/HGRN.yaml  SOLVER.BATCH_SIZE 1024


# Dataset: DecMeg2014; Teacher: lfcnn; Student: soft decision tree
#python train.py --cfg ../configs/DecMeg2014/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256  KD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 KD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024  KD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256  KD.TEMPERATURE 8
#python train.py --cfg ../configs/DecMeg2014/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 KD.TEMPERATURE 8
#python train.py --cfg ../configs/DecMeg2014/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024  KD.TEMPERATURE 8
#
#python train.py --cfg ../configs/DecMeg2014/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 GKD.TEMPERATURE 8
#python train.py --cfg ../configs/DecMeg2014/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 GKD.TEMPERATURE 8
#python train.py --cfg ../configs/DecMeg2014/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 GKD.TEMPERATURE 8
#
#python train.py --cfg ../configs/DecMeg2014/MSEKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/DecMeg2014/MSEKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/DecMeg2014/MSEKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024
#
#python train.py --cfg ../configs/DecMeg2014/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 DKD.T 4
#python train.py --cfg ../configs/DecMeg2014/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 DKD.T 4
#python train.py --cfg ../configs/DecMeg2014/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 DKD.T 4
#python train.py --cfg ../configs/DecMeg2014/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 DKD.T 8
#python train.py --cfg ../configs/DecMeg2014/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 DKD.T 8
#python train.py --cfg ../configs/DecMeg2014/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 DKD.T 8


# Dataset: DecMeg2014; Teacher: varcnn; Student: soft decision tree
#python train.py --cfg ../configs/DecMeg2014/KD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256  KD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/KD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 KD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/KD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024  KD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/KD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256  KD.TEMPERATURE 8
#python train.py --cfg ../configs/DecMeg2014/KD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 KD.TEMPERATURE 8
#python train.py --cfg ../configs/DecMeg2014/KD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024  KD.TEMPERATURE 8
#
#python train.py --cfg ../configs/DecMeg2014/GKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/GKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/GKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/GKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 GKD.TEMPERATURE 8
#python train.py --cfg ../configs/DecMeg2014/GKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 GKD.TEMPERATURE 8
#python train.py --cfg ../configs/DecMeg2014/GKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 GKD.TEMPERATURE 8
#
#python train.py --cfg ../configs/DecMeg2014/MSEKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/DecMeg2014/MSEKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/DecMeg2014/MSEKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024
#
#python train.py --cfg ../configs/DecMeg2014/DKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 DKD.T 4
#python train.py --cfg ../configs/DecMeg2014/DKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 DKD.T 4
#python train.py --cfg ../configs/DecMeg2014/DKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 DKD.T 4
#python train.py --cfg ../configs/DecMeg2014/DKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 DKD.T 8
#python train.py --cfg ../configs/DecMeg2014/DKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 DKD.T 8
#python train.py --cfg ../configs/DecMeg2014/DKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 DKD.T 8


# Dataset: DecMeg2014; Teacher: hgrn; Student: soft decision tree
#python train.py --cfg ../configs/DecMeg2014/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256  KD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512 KD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 1024  KD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256  KD.TEMPERATURE 8
#python train.py --cfg ../configs/DecMeg2014/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512 KD.TEMPERATURE 8
#python train.py --cfg ../configs/DecMeg2014/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 1024  KD.TEMPERATURE 8
#
#python train.py --cfg ../configs/DecMeg2014/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 1024 GKD.TEMPERATURE 4
#python train.py --cfg ../configs/DecMeg2014/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 GKD.TEMPERATURE 8
#python train.py --cfg ../configs/DecMeg2014/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512 GKD.TEMPERATURE 8
#python train.py --cfg ../configs/DecMeg2014/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 1024 GKD.TEMPERATURE 8
##
#python train.py --cfg ../configs/DecMeg2014/MSEKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/DecMeg2014/MSEKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/DecMeg2014/MSEKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 1024
#
#python train.py --cfg ../configs/DecMeg2014/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 DKD.T 4
#python train.py --cfg ../configs/DecMeg2014/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512 DKD.T 4
#python train.py --cfg ../configs/DecMeg2014/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 1024 DKD.T 4
#python train.py --cfg ../configs/DecMeg2014/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 DKD.T 8
#python train.py --cfg ../configs/DecMeg2014/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512 DKD.T 8
#python train.py --cfg ../configs/DecMeg2014/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 1024 DKD.T 8


# FAKD
#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/CamCAN/ShapleyFAKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/CamCAN/ShapleyFAKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/CamCAN/ShapleyFAKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512
python train.py --cfg ../configs/CamCAN/ShapleyFAKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
python train.py --cfg ../configs/CamCAN/ShapleyFAKD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
python train.py --cfg ../configs/CamCAN/ShapleyFAKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256

#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD+KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/CamCAN/ShapleyFAKD+KD/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512

#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD-CE/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD-CE/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD-CE/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/CamCAN/ShapleyFAKD-CE/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/CamCAN/ShapleyFAKD-CE/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/CamCAN/ShapleyFAKD-CE/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512
python train.py --cfg ../configs/CamCAN/ShapleyFAKD-CE/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
python train.py --cfg ../configs/CamCAN/ShapleyFAKD-CE/varcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
python train.py --cfg ../configs/CamCAN/ShapleyFAKD-CE/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256

#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 64 SOLVER.LR 0.003
#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 64 SOLVER.LR 0.0003
#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 128 SOLVER.LR 0.003
#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 128 SOLVER.LR 0.0003
#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 SOLVER.LR 0.0003
#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 ShapleyFAKD.LOSS.FA_WEIGHT 10
#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 ShapleyFAKD.LOSS.FA_WEIGHT 100
#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 64 ShapleyFAKD.M 2
#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 64 ShapleyFAKD.M 4

# self-distillation
#python train.py --cfg ../configs/CamCAN/ShapleyFAKD/varcnn_varcnn.yaml
#python train.py --cfg ../configs/CamCAN/ShapleyFAKD/varcnn_hgrn.yaml  SOLVER.BATCH_SIZE 128
#python train.py --cfg ../configs/CamCAN/ShapleyFAKD/hgrn_hgrn.yaml  SOLVER.BATCH_SIZE 128
