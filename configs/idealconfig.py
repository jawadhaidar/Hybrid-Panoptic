# The new config inherits a base config to highlight the necessary modification
import os 

_base_ = [
   os.path.expanduser("~/")+ 'mmdetection/configs/common/ms-poly_3x_coco-instance.py',
  os.path.expanduser("~/")+ 'mmdetection/configs/_base_/models/mask-rcnn_r50_fpn.py'
]
data_root = os.path.expanduser("~/") + 'mmdetection/data_ideal/idealworks' #change will not affect inference

# Use the needed backbone for Mask-RCNN
# We also need to change the num_classes in head to match the dataset's annotation

model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')),
    roi_head = dict(
        bbox_head = dict(num_classes = 16), mask_head = dict(num_classes = 16)))

# Modify dataset related settings
metainfo = {
    'classes': ('racks','pallet,racks','boxes','boxes,pallet','pallet','railing','iwhub','dolly','stillage','forklift','charger','iw','forklift_with_forks','forklift,forklift_with_forks','forklift_with_forks,forks','mark turntable' ),
    'palette': [
        (255,197,25),
        (0,255,0),
        (140,25,255),
        (25,82,255),
        (255,25,197),
        (140,255,25),
        (25,255,82),
        (255,111,25),
        (226,255,25),
        (54,255,25),
        (25,255,168),
        (25,168,255),
        (54,255,25),
        (226,25,255),
        (54,25,255),
        (255,25,111)
    ]
}
a=len(metainfo['classes'])
print(f'length : {a}')

classes =  metainfo['classes'] #('stillage') 

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type = 'CocoDataset',
        data_root=data_root,
        metainfo=dict(classes = classes),
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    batch_size = 1,
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes = classes),
        ann_file='annotations/instances_val.json', 
        data_prefix=dict(img='val/'))) # change to val once we get a full dataset
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')
test_evaluator = val_evaluator

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.0001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[9, 11],
        gamma=0.1)
]

# optimizer

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', 
                   lr=0.0001,
                     momentum=0.9, 
                     weight_decay=0.00001,
                     ))


# max_epochs
train_cfg = dict(max_epochs=30)
train_cfg = dict(val_interval=31)
# log config
default_hooks = dict(logger=dict(interval=100))


# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = '/home/aub/mmdetection/work_dirs/idealworks_training_dolly_e6/epoch_6.pth'
load_from = os.path.expanduser("~/") + 'mmdetection/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth' #change will not affect inference
