python ensemble.py \
    --gt /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/training_list/Pneumothorax_yolov5_val.json \
    --jsons /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/tood_output.bbox.json /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/vfnet_output.bbox.json \
    --method nmw     \
    --output-name ensemble.json

python cocoeval.py \
    --gt /home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/training_list/Pneumothorax_yolov5_val.json \
    --json ensemble.json