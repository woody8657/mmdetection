_base_ = '../tood/tood_r101_fpn_mstrain_2x_coco.py'
model = dict(
    bbox_head=dict(
        num_classes=1
    )
)

classes = ('Pneumothorax',)
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        img_prefix='/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/images',
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



work_dir = './work_dirs/DETR_others'
checkpoint_config = dict(interval=200)

workflow = [('train', 1),('val',1)]
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# runner = dict(type='EpochBasedRunner', max_epochs=20)
load_from = '/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/weights/tood_r101_fpn_mstrain_2x_coco_20211210_144232-a18f53c8.pth'
