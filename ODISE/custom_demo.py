from demo_helper import*
import matplotlib.pyplot as plt
import cv2 as cv
import argparse
import json

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

if  __name__=="__main__":

    # Create argument parser
    parser = argparse.ArgumentParser(description='Process an image.')
    # Add argument for the image path
    parser.add_argument('--image_path', type=str, help='Path to the input image')
    # Parse the arguments
    args = parser.parse_args()
    # Call the main function with the provided image path
    path=args.image_path
    # print(path)
    
    #load model
    home=os.path.expanduser("~/") #this assumes you cloned odise at home path
    cfg = model_zoo.get_config(home+"/ODISE/configs/Panoptic/odise_label_coco_50e.py", trained=True)

    cfg.model.overlap_threshold = 0
    seed_all_rng(42)

    dataset_cfg = cfg.dataloader.test
    wrapper_cfg = cfg.dataloader.wrapper

    aug = instantiate(dataset_cfg.mapper).augmentations

    model = instantiate_odise(cfg.model)
    model.to(cfg.train.device)
    ODISECheckpointer(model).load(cfg.train.init_checkpoint)
    # print("finished loading model")

    #get image
    # requests.get("http://images.cocodataset.org/val2017/000000467848.jpg"
    # input_image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000467848.jpg", stream=True).raw)

    input_image = Image.open(path)

    #Add additional classes 
    vocab = "racks;palletracks;boxes;boxespallet;pallet;railing;iwhub;dolly;stillage;forklift;charger;iw;forklift_with_forks;forkliftforklift_with_forks;forklift_with_forksforks;mark turntable"

    label_list = ["COCO", "ADE", "LVIS"]
    predictions,result_img=inference(input_image, vocab, label_list)
    # print(predictions)
    #save prediction
    torch.save(predictions['panoptic_seg'], home + '/HybridPan/TempSave/odise_prediction.pt')
    
    # plt.imshow(predictions['panoptic_seg'][0].cpu(),cmap=plt.get_cmap('tab20')) #'tab20''gist_rainbow'
    # plt.show()
    plt.imshow(result_img)
    plt.axis('off')  # Turn off axis labels
    plt.show()