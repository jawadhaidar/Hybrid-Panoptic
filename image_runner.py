import cv2
from PIL import Image
from demo_helper import*
import matplotlib.pyplot as plt
import cv2 as cv
import argparse
import json
import os 
from mmdet.apis import init_detector, inference_detector
from heuristic import*
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color
import argparse
import os
import cv2

def inference(image, vocab, label_list):

    demo_classes, demo_metadata = build_demo_classes_and_metadata(vocab, label_list)
    with ExitStack() as stack:
        inference_model = OpenPanopticInference(
            model=model,
            labels=demo_classes,
            metadata=demo_metadata,
            semantic_on=False, #was False
            instance_on=False,#was False
            panoptic_on=True,
        )
        stack.enter_context(inference_context(inference_model))
        stack.enter_context(torch.no_grad())

        demo = VisualizationDemo(inference_model, demo_metadata, aug)
        predictions, visualized_output = demo.run_on_image(np.array(image))
        return predictions,Image.fromarray(visualized_output.get_image())
    
# Create argument parser
parser = argparse.ArgumentParser(description='Process an image.')
# Add argument for the video path
parser.add_argument('--image_path', type=str, help='Path to the input image')
# Parse the arguments
args = parser.parse_args()
# Call the main function with the provided image path
path=args.image_path

#import odise models
cfg = model_zoo.get_config("Panoptic/odise_label_coco_50e.py", trained=True) # home+"ODISE/configs/Panoptic/odise_label_coco_50e.py"

cfg.model.overlap_threshold = 0
seed_all_rng(42)

dataset_cfg = cfg.dataloader.test
wrapper_cfg = cfg.dataloader.wrapper

aug = instantiate(dataset_cfg.mapper).augmentations

model = instantiate_odise(cfg.model)
model.to(cfg.train.device)
ODISECheckpointer(model).load(cfg.train.init_checkpoint)
# print("finished loading model")

#load closed model
home=os.path.expanduser("~/")
config_file = home + 'HybridPan/configs/idealconfig.py'
checkpoint_file = home + 'HybridPan/models/epoch_5.pth'
# checkpoint_file = '/home/aub/mmdetection/work_dirs/idealworks_training_no_neg/epoch_30.pth'
modelclosed = init_detector(config_file, checkpoint_file, device='cuda')  # or device='cuda:0'

input_image = Image.open(path)
#do odise infernece 

vocab = "racks;palletracks;boxes;boxespallet;pallet;railing;iwhub;dolly;stillage;forklift;charger;iw;forklift_with_forks;forkliftforklift_with_forks;forklift_with_forksforks;mark turntable"
label_list = ["COCO", "ADE", "LVIS"]
predictions,result_img=inference(input_image, vocab, label_list)
# print(predictions)
odise_res=predictions['panoptic_seg']
panoptic_masks=odise_res[0].cpu()
pred_pan_cls=odise_res[1] #dict

#do closed inference 
result=inference_detector(modelclosed, path) #TODO: does this affect given it is rgb
maskrcnn_masks=result.pred_instances.masks.cpu()
maskrcnn_scores=result.pred_instances.scores.cpu()
pred_maskrcnn_cls=result.pred_instances.labels.cpu()

#heuristic
h=HeuristicMimic()
replaced_panoptic_masks, new_cls = h.replace_masks(panoptic_masks,pred_pan_cls, maskrcnn_masks,pred_maskrcnn_cls,maskrcnn_scores)

#visualize and save

visualizer = Visualizer(input_image, h.demo_metadata, instance_mode=ColorMode.IMAGE)

vis_output = visualizer.draw_panoptic_seg(
    torch.from_numpy(replaced_panoptic_masks), new_cls
)
vis_output=Image.fromarray(vis_output.get_image())

vis_output.save(os.path.join(home+"HybridPan/outputs/example.jpg"))


#plot
# plt.imshow(vis_output) #'tab20''gist_rainbow'
# plt.show()




    

