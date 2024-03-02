from heuristic import*
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color
import argparse
import os
# Create argument parser
parser = argparse.ArgumentParser(description='Process an image.')
# Add argument for the image path
parser.add_argument('--image_path', type=str, help='Path to the input image')
# Parse the arguments
args = parser.parse_args()
# Call the main function with the provided image path
path=args.image_path
#get needed variables
home= os.path.expanduser("~/")
panoptic_results=torch.load(home + "/HybridPan/TempSave/odise_prediction.pt")
panoptic_masks=panoptic_results[0].cpu()
pred_pan_cls=panoptic_results[1] #dict
# print(pred_pan_cls)
maskrcnn_masks=torch.load( home + "/HybridPan/TempSave/closed_masks.pt").cpu()
pred_maskrcnn_cls=torch.load(home + "/HybridPan/TempSave/closed_labels.pt").cpu()
maskrcnn_scores=torch.load(home + "/HybridPan/TempSave/closed_scores.pt").cpu()
# print(maskrcnn_scores)
# print(pred_maskrcnn_cls)
h=HeuristicMimic()

replaced_panoptic_masks, new_cls = h.replace_masks(panoptic_masks,pred_pan_cls, maskrcnn_masks,pred_maskrcnn_cls,maskrcnn_scores)

print(replaced_panoptic_masks.shape)
print(type(replaced_panoptic_masks))
print(new_cls)
plt.imshow(panoptic_masks,cmap=plt.get_cmap('tab20')) #'tab20''gist_rainbow'
plt.show()
# v=VisualizePan()
# v.visualize_mask(replaced_panoptic_masks,new_cls)
image = Image.open(path)
visualizer = Visualizer(image, h.demo_metadata, instance_mode=ColorMode.IMAGE)

vis_output = visualizer.draw_panoptic_seg(
    torch.from_numpy(replaced_panoptic_masks), new_cls
)
vis_output=Image.fromarray(vis_output.get_image())
#add a write option

#plot
plt.imshow(vis_output) #'tab20''gist_rainbow'
plt.show()