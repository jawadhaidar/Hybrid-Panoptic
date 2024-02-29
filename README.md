# Hybrid-Panoptic
# Hybrid-Panoptic
## PROJECT DESCRIPTION

## HOW TO INSTALL
### ODISE 
```bash
conda create -n odise python=3.9
conda activate odise
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://@github.com/NVlabs/ODISE.gitcd ODISE
```
### ClosedInstanceSegmentation
```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
git clone https://github.com/open-mmlab/mmdetection.git
mim install mmdet
```
## Heuristic
git clone 

## INFERENCE 
```bash
cd ~/
```
## command based
## docker based
## TRAIN
### preprare dataset
### train
