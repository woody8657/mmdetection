_base_ = [
    '../faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
]

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1
            )))

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

optimizer = dict(type='SGD', lr=0.02*9/8, momentum=0.9, weight_decay=0.0001)
work_dir = './work_dirs/fast_rcnn'
workflow = [('train', 1),('val',1)]


# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/weights/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
