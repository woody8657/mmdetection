_base_ = '../deformable_detr/deformable_detr_r50_16x2_50e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=1
    )
)

classes = ('Pneumothorax',)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
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


workflow = [('train', 1),('val',1)]