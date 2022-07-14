_base_ = '../tood/tood_r101_fpn_mstrain_2x_coco.py'
# model settings
model = dict(
    bbox_head=dict(num_classes=1))

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


optimizer = dict(type='SGD', lr=0.01*3/4, momentum=0.9, weight_decay=0.0001)

work_dir = './work_dirs/coco+RSNA/tood_RSNA'

checkpoint_config = dict(interval=24)
evaluation = dict(interval=1, save_best='bbox_mAP_50')
workflow = [('train', 1)]

load_from = '/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/pretrained_weights/tood_RSNA.pth'
