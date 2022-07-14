_base_ = '../reppoints/reppoints_moment_r50_fpn_gn-neck+head_2x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    bbox_head=dict(
        num_classes=1
    ))

optimizer = dict(lr=0.01*9/16)


classes = ('Pneumothorax',)
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        img_prefix='/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images',
        classes=classes,
        ann_file='/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/training_list/Pneumothorax_yolov5_train.json'),
    val=dict(
        img_prefix='/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images',
        classes=classes,
        ann_file='/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/training_list/Pneumothorax_yolov5_val.json'),
    test=dict(
        img_prefix='/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images',
        classes=classes,
        ann_file='/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/training_list/Pneumothorax_yolov5_val.json'))


lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22, 48, 60])
runner = dict(type='EpochBasedRunner', max_epochs=72)
evaluation = dict(interval=1, save_best='bbox_mAP_50')
checkpoint_config = dict(interval=200)
work_dir = './work_dirs/reppoints_new'



# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r101_fpn_gn-neck%2Bhead_2x_coco/reppoints_moment_r101_fpn_gn-neck%2Bhead_2x_coco_20200329-4fbc7310.pth'
