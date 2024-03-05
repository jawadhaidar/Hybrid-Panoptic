# Hybrid-Panoptic
## PROJECT DESCRIPTION
In this project, we propose a simple Hybrid panoptic method designed to cover a large number of classes (2000+) without compromising on accuracy.
### Cons of open-Vocab
Open-vocab panoptic segmentation techniques often face the challenge of significantly increasing the number of classes. While this approach offers greater flexibility, it tends to achieve lower precision on novel classes. Moreover, continually adding new classes to such models can result in a computational burden, as it requires retraining these massive models (with over 1000M parameters).
### Cons of Closed-Vocab
On the other hand, Closed Vocab methods excel in achieving high accuracy across all annotated classes. However, they are inherently limited to the initially annotated classes. Expanding the scope of such models by adding new classes typically necessitates manual annotation, which can be labor-intensive and time-consuming.
### Pros of Hybrid-Panoptic
### Method Schematic
<img width="700" alt="heuristic" src="https://github.com/jawadhaidar/Hybrid-Panoptic/assets/74460048/30e82e68-265a-4744-8207-3875cee0fbcd">

### Results

<img width="700" alt="result" src="https://github.com/jawadhaidar/Hybrid-Panoptic/assets/74460048/1409bd02-a460-4ccc-99b3-5d4ad3638a4f">

## ODISE 
### HOW TO INSTALL
```bash
conda create -n combine python=3.9
conda activate combine
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://@github.com/NVlabs/ODISE.git
pip install pillow==9.5.0
python -m pip uninstall numpy
python -m pip install numpy==1.23.1

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
conda activate combine
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
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
```
```bash
#add this to a file and run
from mmdet.apis import init_detector, inference_detector
config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/demo.jpg')
```
## Heuristic
### HOW TO INSTALL
```bash
git clone -b fast https://github.com/jawadhaidar/Hybrid-Panoptic.git
```
## INFERENCE 
## command based
```bash
#download finetuned model
cd ~/HybridPan/models
wget --content-disposition "https://drive.usercontent.google.com/download?id=1HW-V50SboP0kEsTh6c3h3g8bGffjBifL&export=download&confirm=t&uuid=368ec624-1afc-4d8c-b6af-dc8e96b3f070"
cd ~/HybridPan
```
```bash
#run on image
python image_runner.py --image_path /home/examplepath.jpg
```
```bash
#run on video
python image_runner.py --video_path /home/examplepath.mp4
```

## docker based

## TRAIN
### preprare dataset
### Configuration
### train
