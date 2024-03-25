# Knowledge-Distillation-to-MEG-Glassbox-Models

## Published Research

 "Magnetoencephalography Decoding Transfer Approach: From Deep Learning Models to Intrinsically Interpretable Models," in *IEEE Journal of Biomedical and Health Informatics*, https://doi.org/10.1109/JBHI.2024.3365051.

## Experimental Preparation

### Experimental Environment

In the conde environment, do the following:

```
conda create -n <environment name> python=3.11
conda activate <environment name>
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install mne
pip install yacs
pip install wandb
pip install scikit-learn
pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp311-cp311-manylinux2014_x86_64.whl
```

Major dependency packages:

```
python==3.11.3
torch==2.0.1+cu117
mne==1.3.1
yacs==0.1.8
wandb==0.15.2
scikit-learn==1.2.2
ray==3.0.0.dev0
```

### Dataset Preprocessing

The CamCAN dataset can be downloaded from the Cambridge Centre for Ageing Neuroscience website at \url{https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/}.
The DecMeg2014 dataset is available at \url{https://www.kaggle.com/c/decoding-the-human-brain/data}.
The pre-processed training and test set are provided in \url{https://drive.google.com/drive/folders/1d1xHb9bYQzoCcvlZ7mrFJvm570-MISvu}.

CamCAN Preprocessing Script:
```angular2html
cd ./Dataset-Preprocessing-Script
python CamCAN2npz.py
python CamCAN2Dataset.py
```

DecMeg2014 Preprocessing Script:
```angular2html
cd ./Dataset-Preprocessing-Script
python DecMeg2Dataset.py
```

Generating the topographic map location information of the gradient sensors:
```angular2html
cd ./Dataset-Preprocessing-Script
python CreateGradChannelsInfo.py
```

### Baseline Performance of the Pre-trained Teacher Models

| Model\Dataset | CamCAN |                   | DecMeg2014 |                   |
| ----------- | ------ | ----------------- | ---------- | ----------------- |
|             | Loss   | Top-1 Accuracy(%) | Loss       | Top-1 Accuracy(%) |
| LFCNN       | 0.1167 | 95.6131           | 0.5895     | 81.6498           |
| VARCNN      | 0.1214 | 95.6640           | 0.6250     | 79.2929           |
| HGRN        | 0.1286 | 95.1897           | 0.5574     | 80.4714           |

## Experimental Running

### Evaluation on the CamCAN dataset

```angular2html
cd ./Run-Script
export PYTHONPATH=$PYTHONPATH:../

# for instance, our FAKD approach.
python train.py --cfg ../configs/CamCAN/ShapleyFAKD/varcnn_sdt.yaml

# you can also change settings at command line
python train.py --cfg ../configs/CamCAN/ShapleyFAKD/lfcnn_sdt.yaml  SOLVER.BATCH_SIZE 128 ShapleyFAKD.M 2
```

### Evaluation on the DecMeg2014 dataset

```angular2html
cd ./Run-Script
export PYTHONPATH=$PYTHONPATH:../

# for instance, our FAKD approach.
python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/hgrn_sdt.yaml

# you can also change settings at command line
python train.py --cfg ../configs/DecMeg2014/ShapleyFAKD/hgrn_sdt.yaml  SOLVER.BATCH_SIZE 256 ShapleyFAKD.LOSS.FA_WEIGHT 100
```

### Validation of Feature Attribution Map Knowledge Transfer

```angular2html
cd ./Run-Script
python attribution.py
```

## Acknowledgement

1. https://github.com/megvii-research/mdistiller
2. https://github.com/xuyxu/Soft-Decision-Tree
