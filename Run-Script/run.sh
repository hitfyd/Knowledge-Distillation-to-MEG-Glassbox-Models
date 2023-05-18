export PYTHONPATH=$PYTHONPATH:../

# Dataset: CamCAN; Vanilla
#python train.py --cfg ../configs/CamCAN/Vanilla/SDT.yaml
#python train.py --cfg ../configs/CamCAN/Vanilla/LFCNN.yaml

# Dataset: CamCAN; Teacher: lfcnn; Student: soft decision tree
#python train.py --cfg ../configs/CamCAN/KD/lfcnn_sdt.yaml
#python train.py --cfg ../configs/CamCAN/ESKD/lfcnn_sdt.yaml
#python train.py --cfg ../configs/CamCAN/GKD/lfcnn_sdt.yaml
#python train.py --cfg ../configs/CamCAN/MSEKD/lfcnn_sdt.yaml
#python train.py --cfg ../configs/CamCAN/DKD/lfcnn_sdt.yaml
#python train.py --cfg ../configs/CamCAN/CLKD/lfcnn_sdt.yaml
python train.py --cfg ../configs/CamCAN/FAKD/lfcnn_sdt.yaml

# Dataset: DecMeg2014; Vanilla
#python train.py --cfg ../configs/DecMeg2014/Vanilla/vanilla.yaml

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
python train.py --cfg ../configs/DecMeg2014/FAKD/hgrn_sdt.yaml