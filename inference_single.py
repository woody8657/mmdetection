from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = 'configs/Pneumothorax/fast_rcnn_r50_fpn.py'
checkpoint_file = 'work_dirs/fast_rcnn_r50_fpn/epoch_100.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

PID = 'P214260000299.png'
# test a single image and show the results
img = '/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images/' + PID # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file=PID)
