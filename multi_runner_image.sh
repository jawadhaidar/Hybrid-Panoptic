#!/bin/bash

# Activate the first conda environment
source activate combine
#load odise model
# python /home/aub/HybridPan/ODISE/load_save_odise.py
# # Run the script for the first model
image_path="/home/aub/mmdetection/data_ideal/idealworks/val/rgb_0821.png" #"/home/aub/ODISE/demo/examples/ideal1.png" #"/home/aub/HybridPan/rgb_0001.png" #806
# cd /home/aub/ODISE
python /home/aub/HybridPan/ODISE/custom_demo.py --image_path $image_path
# # Deactivate the first conda environment
conda deactivate

# Activate the second conda environment
source activate combine
# Run the script for the second model
python /home/aub/HybridPan/ClosedInstance/custom_closed_inference.py --image_path $image_path
conda deactivate

# # Activate  back the first conda environment
source activate combine
# #run heuristic
python /home/aub/HybridPan/heuristic_applied.py  --image_path $image_path
# # Deactivate the second conda environment
conda deactivate

