export PYTHONPATH=$PYTHONPATH:../

# Dataset: CamCAN; Vanilla
#python train.py --cfg ../configs/CamCAN/Vanilla/vanilla.yaml

# Dataset: CamCAN; Teacher: lfcnn; Student: soft decision tree
#python train.py --cfg ../configs/CamCAN/KD/lfcnn_std.yaml
#python train.py --cfg ../configs/CamCAN/ESKD/lfcnn_std.yaml
#python train.py --cfg ../configs/CamCAN/GKD/lfcnn_std.yaml
#python train.py --cfg ../configs/CamCAN/MSEKD/lfcnn_std.yaml
#python train.py --cfg ../configs/CamCAN/DKD/lfcnn_std.yaml
#python train.py --cfg ../configs/CamCAN/CLKD/lfcnn_std.yaml
python train.py --cfg ../configs/CamCAN/FAKD/lfcnn_std.yaml

# Dataset: DecMeg2014; Vanilla
#python train.py --cfg ../configs/DecMeg2014/Vanilla/vanilla.yaml

# Dataset: DecMeg2014; Teacher: lfcnn; Student: soft decision tree
#python train.py --cfg ../configs/DecMeg2014/KD/lfcnn_std.yaml
#python train.py --cfg ../configs/DecMeg2014/ESKD/lfcnn_std.yaml
#python train.py --cfg ../configs/DecMeg2014/GKD/lfcnn_std.yaml
#python train.py --cfg ../configs/DecMeg2014/MSEKD/lfcnn_std.yaml
#python train.py --cfg ../configs/DecMeg2014/DKD/lfcnn_std.yaml
#python train.py --cfg ../configs/DecMeg2014/CLKD/lfcnn_std.yaml

# Dataset: DecMeg2014; Teacher: hgrn; Student: soft decision tree
#python train.py --cfg ../configs/DecMeg2014/KD/hgrn_std.yaml
#python train.py --cfg ../configs/DecMeg2014/ESKD/hgrn_std.yaml
#python train.py --cfg ../configs/DecMeg2014/GKD/hgrn_std.yaml
#python train.py --cfg ../configs/DecMeg2014/MSEKD/hgrn_std.yaml
#python train.py --cfg ../configs/DecMeg2014/DKD/hgrn_std.yaml
#python train.py --cfg ../configs/DecMeg2014/CLKD/hgrn_std.yaml
python train.py --cfg ../configs/DecMeg2014/FAKD/hgrn_std.yaml