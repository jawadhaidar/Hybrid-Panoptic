#!/bin/bash

# Activate the first conda environment
source activate odise1
# # Run the script for the first model
image_path="/home/aub/mmdetection/data/coco/val2017/000000186345.jpg" #806
# cd /home/aub/ODISE
python /home/aub/codes/HybridPan/ODISE/custom_demo.py --image_path $image_path
# # Deactivate the first conda environment
conda deactivate

# Activate the second conda environment
source activate openmmlab
# Run the script for the second model
python /home/aub/codes/HybridPan/ClosedInstance/custom_closed_inference.py --image_path $image_path
conda deactivate

# Activate  back the first conda environment
source activate odise1
#run heuristic
python /home/aub/codes/HybridPan/heuristic_applied.py --image_path $image_path
# Deactivate the second conda environment
conda deactivate

