python ./tools/test.py \
    /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/vfnet.py \
    /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/work_dirs/coco+RSNA/vfnet_coco/best_bbox_mAP_50_epoch_18.pth \
    --eval-options jsonfile_prefix=./vfnet_output \
    --eval bbox \
    