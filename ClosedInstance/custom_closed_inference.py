from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet.registry import VISUALIZERS
import cv2
import os 
import re
import argparse
import torch
from mmdet.registry import VISUALIZERS
import os 

# Create argument parser
parser = argparse.ArgumentParser(description='Process an image.')
# Add argument for the image path
parser.add_argument('--image_path', type=str, help='Path to the input image')
# Parse the arguments
args = parser.parse_args()
# Call the main function with the provided image path
path=args.image_path

#load
home=os.path.expanduser("~/")
config_file = home + 'HybridPan/configs/idealconfig.py'
checkpoint_file = home + 'HybridPan/models/epoch_5.pth'
# checkpoint_file = '/home/aub/mmdetection/work_dirs/idealworks_training_no_neg/epoch_30.pth'

model = init_detector(config_file, checkpoint_file, device='cuda')  # or device='cuda:0'

result=inference_detector(model, path)
torch.save(result.pred_instances.masks, home + "HybridPan/TempSave/closed_masks.pt")
torch.save(result.pred_instances.scores, home + "HybridPan/TempSave/closed_scores.pt")
torch.save(result.pred_instances.labels, home + "HybridPan/TempSave/closed_labels.pt")

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

img = mmcv.imread(path)
visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    show=True
)

print(result)
