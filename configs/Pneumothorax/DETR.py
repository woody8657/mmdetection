_base_ = '../detr/detr_r50_8x2_150e_coco.py'
model = dict(
    bbox_head=dict(
        num_classes=1
    )
)

classes = ('Pneumothorax',)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        img_prefix='/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/images',
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


optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

work_dir = './work_dirs/DETR_pre_new_L1_0.01'
checkpoint_config = dict(interval=100)
runner = dict(type='EpochBasedRunner', max_epochs=600)
workflow = [('train', 1),('val',1)]

load_from = '/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/detr/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'