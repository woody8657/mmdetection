CUDA_VISIBLE_DEVICES=3 python ../tools/test.py \
    /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/tood.py \
    /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/work_dirs/tood_pneumothorax/epoch_24.pth \
    --format-only \
    --eval-options 'jsonfile_prefix=./tood_output' \

