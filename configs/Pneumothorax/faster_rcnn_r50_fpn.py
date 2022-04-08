_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1
            )))

# Modify dataset related settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = ('Pneumothorax',)
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        img_prefix='/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images',
        classes=classes,
        ann_file='/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/Pneumothorax_yolov5_train.json'),
    val=dict(
        img_prefix='/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images',
        classes=classes,
        ann_file='/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/Pneumothorax_yolov5_val.json'),
    test=dict(
        img_prefix='/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images',
        classes=classes,
        ann_file='/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/Pneumothorax_yolov5_val.json'))


# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[48, 66])
runner = dict(type='EpochBasedRunner', max_epochs=72)




work_dir = './work_dirs/fast_rcnn_r50_fpn_coarse_baseline'
checkpoint_config = dict(interval=50)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1),('val',1)]



# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
