import numpy as np
from demo_helper import*
import torch

class Heuristic:

    def __init__(self):
        vocab=""
        label_list = ["COCO", "ADE", "LVIS"]
        self.demo_classes, self.demo_metadata = build_demo_classes_and_metadata(vocab, label_list)
        self.stuff_classes=self.demo_metadata.stuff_classes
        self.things_classes=self.demo_metadata.thing_classes
        self.closed_names=['racks','pallet,racks','boxes','boxes,pallet','pallet','railing','iwhub','dolly','stillage',
        'forklift','charger','iw','forklift_with_forks',
        'forklift,forklift_with_forks','forklift_with_forks,forks','mark turntable']


    def categoryId2Name(self,predicted_pan):
        '''description:
            it takes a category id and returns the class name 
        '''
        for i in range(len(predicted_pan)):
            if  not isinstance(predicted_pan[i]["category_id"], str):
                #get the category name 
                if predicted_pan[i]["isthing"]==True:
                    predicted_pan[i]["category_id"]=self.things_classes[predicted_pan[i]["category_id"]]
                else:
                    predicted_pan[i]["category_id"]=self.stuff_classes[predicted_pan[i]["category_id"]]
        return predicted_pan

    def categoryId2NameClosed(self,predicted_closed):
        predicted_closed_names=[]

        for i in predicted_closed:
            predicted_closed_names.append(self.closed_names[i])
        return predicted_closed_names


    def replace_masks(self,panoptic_masks,pred_pan_cls, maskrcnn_masks,pred_maskrcnn_cls,maskrcnn_scores, iou_threshold=0.9, score_threshold=0.5):
        '''
        description:
        
        '''
        replaced_panoptic_masks = np.copy(panoptic_masks)
        #change from ids to classes names
        pred_maskrcnn_cls=self.categoryId2NameClosed(pred_maskrcnn_cls)
        print(f'pred_maskrcnn_cls {pred_maskrcnn_cls}')
        # Select maskrcnn masks with scores above threshold
        print(maskrcnn_scores > score_threshold)
        selected_maskrcnn_masks = maskrcnn_masks[maskrcnn_scores > score_threshold]
   
        pred_maskrcnn_cls=[ name for count,name in enumerate(pred_maskrcnn_cls) if maskrcnn_scores[count] > score_threshold ]
        print(pred_maskrcnn_cls)
        # Build IOU matrix between selected maskrcnn masks and panoptic masks
        iou_matrix = np.zeros((selected_maskrcnn_masks.shape[0], panoptic_masks.max() + 1))
        for i, maskrcnn_mask in enumerate(selected_maskrcnn_masks):
            for j in range(1, panoptic_masks.max() + 1):  # Exclude background
                intersection = np.logical_and(maskrcnn_mask, panoptic_masks == j)
                union = np.logical_or(maskrcnn_mask, panoptic_masks == j)
                iou = intersection.sum() / union.sum()
                iou_matrix[i, j] = iou
        
        # Replace masks in panoptic segmentation
        for i in range(selected_maskrcnn_masks.shape[0]):
            best_iou = np.max(iou_matrix[i])
            if best_iou > iou_threshold:
                best_j = np.argmax(iou_matrix[i])
                replaced_panoptic_masks[panoptic_masks == best_j] = 0  # Remove old mask
                replaced_panoptic_masks[maskrcnn_masks[i]] = best_j  # Add new mask
                pred_pan_cls[best_j]["category_id"]=pred_maskrcnn_cls[i] 
                #TODO: you should also replace the other keys values (isthing,area)
            else:
                max_id = len(pred_pan_cls) #panoptic_masks.max() + 1 #add a new id
                replaced_panoptic_masks[maskrcnn_masks[i]] = max_id  # Add new mask
                pred_pan_cls.append({'id': max_id, 'isthing': True, 'category_id': pred_maskrcnn_cls[i], 'area': 200})

        #change category id to name
        pred_pan_cls=self.categoryId2Name(pred_pan_cls)
        return replaced_panoptic_masks,pred_pan_cls
    
class HeuristicMimic:
    def __init__(self):
        vocab="racks;palletracks;boxes;boxespallet;pallet;railing;iwhub;dolly;stillage;forklift;charger;iw;forklift_with_forks;forkliftforklift_with_forks;forklift_with_forksforks;mark turntable"
        label_list = ["COCO", "ADE", "LVIS"]
        self.demo_classes, self.demo_metadata = build_demo_classes_and_metadata(vocab, label_list)
        self.closed_names=['racks','palletracks','boxes','boxespallet','pallet','railing','iwhub','dolly','stillage',
        'forklift','charger','iw','forklift_with_forks',
        'forkliftforklift_with_forks','forklift_with_forksforks','mark turntable']
        self.closed_colors=[random_color(rgb=True, maximum=1) for _ in range(len(self.closed_names))]
        self.closedIds=[self.demo_metadata.thing_classes.index(name) for name in self.closed_names]
        print(self.closedIds)
        
        
        #add to original new classes and colours
        # self.demo_metadata.thing_classes=self.demo_metadata.thing_classes + self.closed_names #not allowed
        # self.demo_metadata.thing_colors=self.demo_metadata.thing_colors + self.closed_colors


        self.stuff_classes=self.demo_metadata.stuff_classes
        self.things_classes=self.demo_metadata.thing_classes



    def replace_masks(self,panoptic_masks,pred_pan_cls, maskrcnn_masks,pred_maskrcnn_cls,maskrcnn_scores, iou_threshold=0.7, score_threshold=0.5):
        '''
        description:
        
        '''
        replaced_panoptic_masks = np.copy(panoptic_masks)

        # Select maskrcnn masks with scores above threshold
        print(maskrcnn_scores > score_threshold)
        selected_maskrcnn_masks = maskrcnn_masks[maskrcnn_scores > score_threshold]
        # pred_maskrcnn_cls=[ self.closedIds[id] for count,id in enumerate(pred_maskrcnn_cls) if maskrcnn_scores[count] > score_threshold ] #TODO: starts at zero?

  
        # Build IOU matrix between selected maskrcnn masks and panoptic masks
        iou_matrix = np.zeros((selected_maskrcnn_masks.shape[0], panoptic_masks.max() + 1))
        for i, maskrcnn_mask in enumerate(selected_maskrcnn_masks):
            for j in range(1, panoptic_masks.max() + 1):  # Exclude background
                intersection = np.logical_and(maskrcnn_mask, panoptic_masks == j)
                union = np.logical_or(maskrcnn_mask, panoptic_masks == j)
                iou = intersection.sum() / union.sum()
                iou_matrix[i, j] = iou
        
        # Replace masks in panoptic segmentation
        for i in range(selected_maskrcnn_masks.shape[0]):
            best_iou = np.max(iou_matrix[i])
            if best_iou > iou_threshold:
                best_j = np.argmax(iou_matrix[i])
                replaced_panoptic_masks[panoptic_masks == best_j] = 0  # Remove old mask
                replaced_panoptic_masks[maskrcnn_masks[i]] = best_j + 1  # Add new mask #TODO: check the plus one
                v=pred_pan_cls[best_j]["category_id"]
                print(f'pred_pan_cls[best_j]["category_id"] {v} iou {best_iou}')
                pred_pan_cls[best_j]["category_id"]=pred_maskrcnn_cls[i].item() 
                #TODO: you should also replace the other keys values (isthing,area)
            else:
                max_id = len(pred_pan_cls) + 1 #panoptic_masks.max() + 1 #add a new id
                replaced_panoptic_masks[maskrcnn_masks[i]] = max_id  # Add new mask
                pred_pan_cls.append({'id': max_id, 'isthing': True, 'category_id': pred_maskrcnn_cls[i].item(), 'area': 200}) #TODO: calculate true area

        #change category id to name
        # pred_pan_cls=self.categoryId2Name(pred_pan_cls)
        print("inside")
        return replaced_panoptic_masks,pred_pan_cls
    





class VisualizePan(Heuristic):
    def __init__(self):
        super().__init__()
        self.total_classes=self.stuff_classes+self.things_classes+self.closed_names
        self.unique_colors=[random_color(rgb=True, maximum=1) for _ in range(len(self.total_classes))]
        self.name2color=dict(zip(self.total_classes,self.unique_colors))

    def get_pred_names_colors(self,pred_cls):
        names_pred=[]
        colors=[]
        for pred in pred_cls:
            name=pred['category_id']
            names_pred.append(name)
            colors.append(self.name2color[name])
        return names_pred,colors

    
    def visualize_mask(self,mask,pred_cls):
        """
        Visualize a mask using specified colors.

        Parameters:
            mask (np.array): Mask array.

        Returns:
            None
        """
        names_pred,colors=self.get_pred_names_colors(pred_cls)
        # colors=self.unique_colors
        mask = mask.astype(int)

        # Create a new figure
        plt.figure(figsize=(8, 8))

        # Display the mask using the specified colors
        plt.imshow(mask, cmap=ListedColormap(colors), vmin=0, vmax=len(colors))
        plt.axis('off')

        # Annotate each region with its corresponding class label
        for i, color in enumerate(colors):
            # Find the coordinates of the center of the region
            y, x = np.where(mask == i)
            if len(y) > 0:
                center_y = np.mean(y)
                center_x = np.mean(x)

                # Add the class label as text at the center of the region
                plt.text(center_x, center_y, names_pred[i], color='black', ha='center', va='center')


        plt.show()


    
if __name__=="__main__":
    print("here")
    h=HeuristicMimic()
    print(np.argmax(np.array([1,2,3])))
    # # Example usage:
    panoptic_masks = np.array([[1, 1, 2],
                            [1, 0, 2],
                            [0, 0, 0]]) #0 background are you sure?

    pred_pan_cls=[{'id': 1, 'isthing': True, 'category_id': "dog", 'area': 0},
                {'id': 2, 'isthing': True, 'category_id': "cat", 'area': 0}]

    maskrcnn_masks = np.array([[[True, True, False],
                                [True, True, False],
                                [False, False, False]],

                                [[False, False, False],
                                [False, True, True],
                                [False, False, False]]])
    
    pred_maskrcnn_cls=torch.tensor([1,0]) #weired why this works but [] dont

    maskrcnn_scores = np.array([0.9, 0.2])
    print("hereeeeeeeeeeeeeee")
    replaced_panoptic_masks, new_cls = h.replace_masks(panoptic_masks,pred_pan_cls, maskrcnn_masks,pred_maskrcnn_cls,maskrcnn_scores)
    print("Replaced Panoptic Masks:")
    print(replaced_panoptic_masks)
    print("IDs of Replaced Masks:", new_cls)
    # v=VisualizePan()
    # v.print()

