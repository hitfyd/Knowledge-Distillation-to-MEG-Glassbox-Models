export PYTHONPATH=$PYTHONPATH:../

# Dataset: CamCAN; Vanilla
python train.py --cfg ../configs/CamCAN/Vanilla/SDT.yaml  SOLVER.BATCH_SIZE 256
python train.py --cfg ../configs/CamCAN/Vanilla/SDT.yaml  SOLVER.BATCH_SIZE 512
python train.py --cfg ../configs/CamCAN/Vanilla/SDT.yaml  SOLVER.BATCH_SIZE 1024
#python train.py --cfg ../configs/CamCAN/Vanilla/LFCNN.yaml

# Dataset: CamCAN; Teacher: lfcnn; Student: soft decision tree
python train.py --cfg ../configs/CamCAN/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256  KD.TEMPERATURE 4
python train.py --cfg ../configs/CamCAN/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512  KD.TEMPERATURE 4
python train.py --cfg ../configs/CamCAN/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 KD.TEMPERATURE 4
python train.py --cfg ../configs/CamCAN/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256  KD.TEMPERATURE 8
python train.py --cfg ../configs/CamCAN/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512  KD.TEMPERATURE 8
python train.py --cfg ../configs/CamCAN/KD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 KD.TEMPERATURE 8

python train.py --cfg ../configs/CamCAN/ESKD/lfcnn_sdt.yaml SOLVER.BATCH_SIZE 256 ESKD.TEMPERATURE 4  ESKD.STOP_EPOCH 50
python train.py --cfg ../configs/CamCAN/ESKD/lfcnn_sdt.yaml SOLVER.BATCH_SIZE 512 ESKD.TEMPERATURE 4  ESKD.STOP_EPOCH 50
python train.py --cfg ../configs/CamCAN/ESKD/lfcnn_sdt.yaml SOLVER.BATCH_SIZE 1024 ESKD.TEMPERATURE 4  ESKD.STOP_EPOCH 50
python train.py --cfg ../configs/CamCAN/ESKD/lfcnn_sdt.yaml SOLVER.BATCH_SIZE 256 ESKD.TEMPERATURE 8  ESKD.STOP_EPOCH 50
python train.py --cfg ../configs/CamCAN/ESKD/lfcnn_sdt.yaml SOLVER.BATCH_SIZE 512 ESKD.TEMPERATURE 8  ESKD.STOP_EPOCH 50
python train.py --cfg ../configs/CamCAN/ESKD/lfcnn_sdt.yaml SOLVER.BATCH_SIZE 1024 ESKD.TEMPERATURE 8  ESKD.STOP_EPOCH 50
python train.py --cfg ../configs/CamCAN/ESKD/lfcnn_sdt.yaml SOLVER.BATCH_SIZE 256 ESKD.TEMPERATURE 4  ESKD.STOP_EPOCH 75
python train.py --cfg ../configs/CamCAN/ESKD/lfcnn_sdt.yaml SOLVER.BATCH_SIZE 512 ESKD.TEMPERATURE 4  ESKD.STOP_EPOCH 75
python train.py --cfg ../configs/CamCAN/ESKD/lfcnn_sdt.yaml SOLVER.BATCH_SIZE 1024 ESKD.TEMPERATURE 4  ESKD.STOP_EPOCH 75
python train.py --cfg ../configs/CamCAN/ESKD/lfcnn_sdt.yaml SOLVER.BATCH_SIZE 256 ESKD.TEMPERATURE 8  ESKD.STOP_EPOCH 75
python train.py --cfg ../configs/CamCAN/ESKD/lfcnn_sdt.yaml SOLVER.BATCH_SIZE 512 ESKD.TEMPERATURE 8  ESKD.STOP_EPOCH 75
python train.py --cfg ../configs/CamCAN/ESKD/lfcnn_sdt.yaml SOLVER.BATCH_SIZE 1024 ESKD.TEMPERATURE 8  ESKD.STOP_EPOCH 75

python train.py --cfg ../configs/CamCAN/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 GKD.TEMPERATURE 4
python train.py --cfg ../configs/CamCAN/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 GKD.TEMPERATURE 4
python train.py --cfg ../configs/CamCAN/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 GKD.TEMPERATURE 4
python train.py --cfg ../configs/CamCAN/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 GKD.TEMPERATURE 8
python train.py --cfg ../configs/CamCAN/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 GKD.TEMPERATURE 8
python train.py --cfg ../configs/CamCAN/GKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 GKD.TEMPERATURE 8

python train.py --cfg ../configs/CamCAN/MSEKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
python train.py --cfg ../configs/CamCAN/MSEKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512
python train.py --cfg ../configs/CamCAN/MSEKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024

python train.py --cfg ../configs/CamCAN/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 DKD.T 4
python train.py --cfg ../configs/CamCAN/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 DKD.T 4
python train.py --cfg ../configs/CamCAN/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 DKD.T 4
python train.py --cfg ../configs/CamCAN/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256 DKD.T 8
python train.py --cfg ../configs/CamCAN/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512 DKD.T 8
python train.py --cfg ../configs/CamCAN/DKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 1024 DKD.T 8

python train.py --cfg ../configs/CamCAN/SCFAKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
python train.py --cfg ../configs/CamCAN/SCFAKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512

python train.py --cfg ../configs/CamCAN/ShapleyFAKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 256
python train.py --cfg ../configs/CamCAN/ShapleyFAKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 512

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
#python train.py --cfg ../configs/DecMeg2014/KD/lfcnn_sdt.yaml
#python train.py --cfg ../configs/DecMeg2014/ESKD/lfcnn_sdt.yaml
#python train.py --cfg ../configs/DecMeg2014/GKD/lfcnn_sdt.yaml
#python train.py --cfg ../configs/DecMeg2014/MSEKD/lfcnn_sdt.yaml
#python train.py --cfg ../configs/DecMeg2014/DKD/lfcnn_sdt.yaml
#python train.py --cfg ../configs/DecMeg2014/CLKD/lfcnn_sdt.yaml

# Dataset: DecMeg2014; Teacher: hgrn; Student: soft decision tree
#python train.py --cfg ../configs/DecMeg2014/KD/hgrn_sdt.yaml
#python train.py --cfg ../configs/DecMeg2014/ESKD/hgrn_sdt.yaml
#python train.py --cfg ../configs/DecMeg2014/GKD/hgrn_sdt.yaml
#python train.py --cfg ../configs/DecMeg2014/MSEKD/hgrn_sdt.yaml
#python train.py --cfg ../configs/DecMeg2014/DKD/hgrn_sdt.yaml
#python train.py --cfg ../configs/DecMeg2014/CLKD/hgrn_sdt.yaml
#python train.py --cfg ../configs/DecMeg2014/SCFAKD/hgrn_sdt.yaml
#python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/hgrn_sdt.yaml