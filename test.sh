# Activate the second conda environment
image_path="/home/aub/ODISE/demo/examples/ideal1.png" #806
source activate openmmlab
# Run the script for the second model
python /home/aub/HybridPan/ClosedInstance/custom_closed_inference.py --image_path $image_path