_base_ = '../yolox/yolox_s_8x8_300e_coco.py'
img_scale = (640,640)
model = dict(
    type='YOLOX',
    input_size= img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=1, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

dataset_type = 'COCODataset'

classes = ('Pneumothorax',)
train_dataset = dict(
    dataset=dict(
        ann_file='/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/Pneumothorax_train.json',
        img_prefix='/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/images/',
        classes=classes
    )
)

data = dict(
    samples_per_gpu = 6,
    workers_per_gpu=0,
    train=train_dataset,
    val=dict(
        ann_file='/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/Pneumothorax_train.json',
        img_prefix='/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/images/',
        classes=classes
    ),
    test=dict(
        ann_file='/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/Pneumothorax_val.json',
        img_prefix='/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/images/',
        classes=classes
    )
)

# runner = dict(type='EpochBasedRunner', max_epochs=300)

work_dir = './work_dirs/yolox'
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=10)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1),('val',1)]