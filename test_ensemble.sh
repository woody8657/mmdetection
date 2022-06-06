python ./tools/test_ensemble.py \
    --configs /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/faster_rcnn_r50_fpn.py /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/DETR.py \
    --checkpoints /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/work_dirs/fast_rcnn_r50_fpn_coarse_baseline/latest.pth /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/work_dirs/DETR_coarse_baseline/epoch_150.pth \
    --eval bbox