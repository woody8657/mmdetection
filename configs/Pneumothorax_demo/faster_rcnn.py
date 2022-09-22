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
        img_prefix='/mmdetection/data/',
        classes=classes,
        ann_file='/mmdetection/configs/Pneumothorax_demo/training_list/Pneumothorax_train_positive.json'),
    val=dict(
        img_prefix='/mmdetection/data/',
        classes=classes,
        ann_file='/mmdetection/configs/Pneumothorax_demo/training_list/Pneumothorax_val_positive.json'),
    test=dict(
        img_prefix='/mmdetection/data/',
        classes=classes,
        ann_file='/mmdetection/configs/Pneumothorax_demo/training_list/Pneumothorax_val_positive.json'))

checkpoint_config = dict(interval=24)
optimizer = dict(type='SGD', lr=0.02*3/16, momentum=0.9, weight_decay=0.0001)
evaluation = dict(interval=1, save_best='bbox_mAP_50')

work_dir = './work_dirs/demo_faster_rcnn'

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
