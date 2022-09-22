docker run -it \
    --gpus all \
    --runtime=nvidia \
    --shm-size=128g \
    -v $(pwd)/configs/_base_/default_runtime.py:/mmdetection/configs/_base_/default_runtime.py \
    -v $(pwd)/mmdet/apis/train.py:/mmdetection/mmdet/apis/train.py \
    -v $(pwd)/configs/Pneumothorax_demo:/mmdetection/configs/Pneumothorax_demo \
    -v /data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/images/:/mmdetection/data \
    -v $(pwd)/work_dirs:/mmdetection/work_dirs \
    mmdetection

    

   
    
