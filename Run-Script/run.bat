set PYTHONPATH=%PYTHONPATH%;../
python train.py --cfg ../configs/CamCAN/Vanilla/vanilla.yaml
python train.py --cfg ../configs/CamCAN/KD/lfcnn_std.yaml
python train.py --cfg ../configs/CamCAN/GKD/lfcnn_std.yaml
python train.py --cfg ../configs/CamCAN/MSEKD/lfcnn_std.yaml
python train.py --cfg ../configs/CamCAN/DKD/lfcnn_std.yaml
python train.py --cfg ../configs/CamCAN/CLKD/lfcnn_std.yaml