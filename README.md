# Hybrid-Panoptic
# Hybrid-Panoptic
## PROJECT DESCRIPTION


## ODISE 
### HOW TO INSTALL
```bash
conda create -n odise python=3.9
conda activate odise
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://@github.com/NVlabs/ODISE.gitcd ODISE
cd ~/ODISE
wget https://github.com/NVlabs/ODISE/releases/download/v1.0.0/odise_label_coco_50e-b67d2efc.pth

```
### Test ODISE
```bash
conda activate odise
cd ~/ODISE
python demo/demo.py --input demo/examples/coco.jpg --output demo/coco_pred.jpg 
```
## ClosedInstanceSegmentation (CIS)
### HOW TO INSTALL
```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
git clone https://github.com/open-mmlab/mmdetection.git
mim install mmdet
cd ~/mmdetection
#download base model
wget https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth 
```

### Test CIS
```bash
cd ~/mmdetection
#create folder for configs
mkdir myconfigs

```
## Heuristic
### HOW TO INSTALL
```bash
mkdir ~/HybridPan
git clone https://github.com/jawadhaidar/Hybrid-Panoptic.git
```
## INFERENCE 
```bash
cd ~/HybridPan
bash multi_runner.sh
```
## command based
## docker based
## TRAIN
### preprare dataset
### Configuration
### train
