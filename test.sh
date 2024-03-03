# Activate the second conda environment
image_path="/home/aub/mmdetection/data/coco/val2017/000000186345.jpg" #806
source activate openmmlab
# Run the script for the second model
python /home/aub/HybridPan/ClosedInstance/custom_closed_inference.py --image_path $image_path