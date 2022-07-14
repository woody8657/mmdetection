_base_ = '../deformable_detr/deformable_detr_r50_16x2_50e_coco.py'

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


optimizer = dict(
    type='AdamW',
    lr=2e-4*6/32,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

workflow = [('train', 1)]

work_dir = './work_dirs/deformable_DETR_refine'

checkpoint_config = dict(interval=50000)
evaluation = dict(interval=1, save_best='bbox_mAP_50')

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth'
