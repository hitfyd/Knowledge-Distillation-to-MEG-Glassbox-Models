export PYTHONPATH=$PYTHONPATH:../

# Dataset: BCIIV2a; Vanilla
python train.py --cfg ../configs/BCIIV2a/Vanilla/EEGNetv4.yaml

python train.py --cfg ../configs/BCIIV2a/Vanilla/EEGNetv1.yaml

# Dataset: BCIIV2a; Teacher: EEGNetv4; Student: EEGNetv1
python train.py --cfg ../configs/BCIIV2a/KD/eegnetv4_eegnetv1.yaml

python train.py --cfg ../configs/BCIIV2a/GKD/eegnetv4_eegnetv1.yaml

python train.py --cfg ../configs/BCIIV2a/MSEKD/eegnetv4_eegnetv1.yaml

python train.py --cfg ../configs/BCIIV2a/DKD/eegnetv4_eegnetv1.yaml

python train.py --cfg ../configs/BCIIV2a/ShapleyFAKD/eegnetv4_eegnetv1.yaml
