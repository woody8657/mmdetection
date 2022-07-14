import argparse
import json
from tqdm import tqdm
from ensemble_boxes import *
import numpy as np
import time


def json2dict(path):
    with open(path, mode='r') as file:
        data = json.load(file)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', help='ground truth path')
    parser.add_argument('--jsons', nargs='+', type=str)
    parser.add_argument('--method', default='nms', help='ensemble algorithm to fuse bboxs')
    parser.add_argument('--output-name', default='output.json', help='output file name')
    opt = parser.parse_args()
    
    # hyperparameters
    iou_thr = 0.35
    skip_box_thr = 0.0001
    sigma = 0.1
    weights = [1,3]

    gt = json2dict(opt.gt)

    model_output_list = []
    print(f"loading {len(opt.jsons)} models...")
    for i in range(len(opt.jsons)):
        model_output_list.append(json2dict(opt.jsons[i]))
    
    # Since the 'ensemble_boxes' requires the cooredinates been normalized, we need the image size
    id_size_LUT = {}
    for i in range(len(gt['images'])):
        id_size_LUT[gt['images'][i]['id']] = [gt['images'][i]['width'], gt['images'][i]['height']]

    
    output_ensemble = []
    for id in tqdm(id_size_LUT.keys()):
        boxes_list = []
        scores_list = []
        labels_list = []
    
        for output in model_output_list:
            boxes = []
            scores = []
            labels = []
            for box in output:
                if box['image_id'] != id:
                    continue
               
                xywh = box['bbox']
                xyxy = [xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]]
                xyxy_norm = [xyxy[0]/id_size_LUT[id][0], xyxy[1]/id_size_LUT[id][1], xyxy[2]/id_size_LUT[id][0], xyxy[3]/id_size_LUT[id][1]]
                boxes.append(xyxy_norm)
                scores.append(box['score'])
                labels.append(box['category_id'])
            boxes = np.array(boxes)
            boxes[boxes>1]=1
            boxes[boxes<0]=0
            boxes = boxes.tolist()
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
        boxes_list = [boxes for boxes in boxes_list if boxes != []]
        scores_list = [scores for scores in scores_list if scores != []]
        labels_list = [labels for labels in labels_list if labels != []]
        if opt.method == 'nms':
            boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
        elif opt.method == 'soft_nms':
            boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
        elif opt.method == 'nmw':
            boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
        elif opt.method == 'wbf':
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
        else:
            raise NameError(f'the algorithm name {opt.method} is not supported')
        
        
        for xyxy_norm, score, label in zip(boxes, scores, labels):
            xyxy = [xyxy_norm[0]*id_size_LUT[id][0], xyxy_norm[1]*id_size_LUT[id][1], xyxy_norm[2]*id_size_LUT[id][0], xyxy_norm[3]*id_size_LUT[id][1]]
            xywh = [xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]]
            xywh = [coordinate.tolist() for coordinate in xywh]
            output_ensemble.append({"image_id": int(id), "bbox": xywh, "score": float(score), "category_id": int(label)})


    with open(opt.output_name, mode='w') as file:
        json.dump(output_ensemble, file)
     
        
 