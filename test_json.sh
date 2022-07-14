python ./tools/test.py \
    /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/vfnet.py \
    /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/work_dirs/vfnet_e24/best_bbox_mAP_50_epoch_17.pth \
    --format-only \
    --eval-options 'jsonfile_prefix=./vfnet_output' \
