export PYTHONPATH=$PYTHONPATH:../

# Dataset: BCIIV2a; Vanilla
#python train.py --cfg ../configs/BCIIV2a/Vanilla/EEGNetv4.yaml SOLVER.LR 0.003
#python train.py --cfg ../configs/BCIIV2a/Vanilla/EEGNetv4.yaml SOLVER.LR 0.001
#python train.py --cfg ../configs/BCIIV2a/Vanilla/EEGNetv4.yaml SOLVER.LR 0.0003
#python train.py --cfg ../configs/BCIIV2a/Vanilla/EEGNetv4.yaml SOLVER.BATCH_SIZE 32
#python train.py --cfg ../configs/BCIIV2a/Vanilla/EEGNetv4.yaml SOLVER.BATCH_SIZE 128

python train.py --cfg ../configs/BCIIV2a/Vanilla/EEGNetv1.yaml SOLVER.LR 0.003
python train.py --cfg ../configs/BCIIV2a/Vanilla/EEGNetv1.yaml SOLVER.LR 0.001
python train.py --cfg ../configs/BCIIV2a/Vanilla/EEGNetv1.yaml SOLVER.LR 0.0003
python train.py --cfg ../configs/BCIIV2a/Vanilla/EEGNetv1.yaml SOLVER.BATCH_SIZE 32
python train.py --cfg ../configs/BCIIV2a/Vanilla/EEGNetv1.yaml SOLVER.BATCH_SIZE 128

# Dataset: CamCAN; Teacher: EEGNetv4; Student: soft decision tree
python train.py --cfg ../configs/BCIIV2a/KD/eegnetv4_eegnetv1.yaml  SOLVER.BATCH_SIZE 64  KD.TEMPERATURE 4

python train.py --cfg ../configs/BCIIV2a/GKD/eegnetv4_eegnetv1.yaml  SOLVER.BATCH_SIZE 64 GKD.TEMPERATURE 4

python train.py --cfg ../configs/BCIIV2a/MSEKD/eegnetv4_eegnetv1.yaml  SOLVER.BATCH_SIZE 64

python train.py --cfg ../configs/BCIIV2a/DKD/eegnetv4_eegnetv1.yaml  SOLVER.BATCH_SIZE 64 DKD.T 4

python train.py --cfg ../configs/BCIIV2a/ShapleyFAKD/eegnetv4_eegnetv1.yaml  SOLVER.BATCH_SIZE 64
