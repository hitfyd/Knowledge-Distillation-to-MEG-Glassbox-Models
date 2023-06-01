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

#python train.py --cfg ../configs/CamCAN/SCFAKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/CamCAN/SCFAKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/CamCAN/SCFAKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024

#python train.py --cfg ../configs/CamCAN/ShapleyFAKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
#python train.py --cfg ../configs/CamCAN/ShapleyFAKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/CamCAN/ShapleyFAKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024


# Dataset: CamCAN; Teacher: varcnn; Student: soft decision tree


# Dataset: CamCAN; Teacher: hgrn; Student: soft decision tree


# Dataset: DecMeg2014; Vanilla
python train.py --cfg ../configs/DecMeg2014/Vanilla/SDT.yaml  SOLVER.BATCH_SIZE 128
python train.py --cfg ../configs/DecMeg2014/Vanilla/SDT.yaml  SOLVER.BATCH_SIZE 256
python train.py --cfg ../configs/DecMeg2014/Vanilla/SDT.yaml  SOLVER.BATCH_SIZE 512
python train.py --cfg ../configs/DecMeg2014/Vanilla/LFCNN.yaml  SOLVER.BATCH_SIZE 128
python train.py --cfg ../configs/DecMeg2014/Vanilla/LFCNN.yaml  SOLVER.BATCH_SIZE 256
python train.py --cfg ../configs/DecMeg2014/Vanilla/LFCNN.yaml  SOLVER.BATCH_SIZE 512
python train.py --cfg ../configs/DecMeg2014/Vanilla/VARCNN.yaml  SOLVER.BATCH_SIZE 128
python train.py --cfg ../configs/DecMeg2014/Vanilla/VARCNN.yaml  SOLVER.BATCH_SIZE 256
python train.py --cfg ../configs/DecMeg2014/Vanilla/VARCNN.yaml  SOLVER.BATCH_SIZE 512
#python train.py --cfg ../configs/DecMeg2014/Vanilla/HGRN.yaml

# Dataset: DecMeg2014; Teacher: lfcnn; Student: soft decision tree

# Dataset: DecMeg2014; Teacher: hgrn; Student: soft decision tree
python train.py --cfg ../configs/DecMeg2014/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 128  KD.TEMPERATURE 4
python train.py --cfg ../configs/DecMeg2014/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256  KD.TEMPERATURE 4
python train.py --cfg ../configs/DecMeg2014/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512 KD.TEMPERATURE 4
python train.py --cfg ../configs/DecMeg2014/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 128  KD.TEMPERATURE 8
python train.py --cfg ../configs/DecMeg2014/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256  KD.TEMPERATURE 8
python train.py --cfg ../configs/DecMeg2014/KD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512 KD.TEMPERATURE 8

python train.py --cfg ../configs/DecMeg2014/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 128 GKD.TEMPERATURE 4
python train.py --cfg ../configs/DecMeg2014/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 GKD.TEMPERATURE 4
python train.py --cfg ../configs/DecMeg2014/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512 GKD.TEMPERATURE 4
python train.py --cfg ../configs/DecMeg2014/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 128 GKD.TEMPERATURE 8
python train.py --cfg ../configs/DecMeg2014/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 GKD.TEMPERATURE 8
python train.py --cfg ../configs/DecMeg2014/GKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512 GKD.TEMPERATURE 8
#
python train.py --cfg ../configs/DecMeg2014/MSEKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 128
python train.py --cfg ../configs/DecMeg2014/MSEKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256
python train.py --cfg ../configs/DecMeg2014/MSEKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512

python train.py --cfg ../configs/DecMeg2014/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 128 DKD.T 4
python train.py --cfg ../configs/DecMeg2014/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 DKD.T 4
python train.py --cfg ../configs/DecMeg2014/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512 DKD.T 4
python train.py --cfg ../configs/DecMeg2014/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 128 DKD.T 8
python train.py --cfg ../configs/DecMeg2014/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 DKD.T 8
python train.py --cfg ../configs/DecMeg2014/DKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 512 DKD.T 8

#python train.py --cfg ../configs/CamCAN/SCFAKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 128

python train.py --cfg ../configs/CamCAN/ShapleyFAKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 128