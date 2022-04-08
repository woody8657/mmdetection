import json
import glob
import pandas as pd
import skimage
import skimage.measure
import numpy as np

def grid_bbox2mask(tmp):
    mask = np.zeros((tmp['size'][1], tmp['size'][0]))
    for j in range(len(tmp['shapes'])):
        x_a, y_a = tmp['shapes'][j]['a']
        x_b, y_b = tmp['shapes'][j]['b']
        x, y, w, h = int(min(x_a, x_b)), int(min(y_a, y_b)), int(max(x_a, x_b)-min(x_a, x_b)), int(max(y_a, y_b)-min(y_a, y_b))
        mask[y:y+h,x:x+w] = 255
    
    return mask

def mask2bbox(mask):
    mask[mask<125]=0
    mask[mask>=125]=1
    mask = skimage.measure.label(mask,connectivity=1)
    xyxy_bbox = np.empty((0,4), int)
    
    for region in skimage.measure.regionprops(mask):
        minr, minc, maxr, maxc = region.bbox
        xyxy_bbox = np.append(xyxy_bbox, np.expand_dims(np.array([minc, minr, maxc, maxr]), axis=0), axis=0)

    xywh_bbox = []
    for i in range(xyxy_bbox.shape[0]):
        (x1,y1) = map(int,(xyxy_bbox[i,0], xyxy_bbox[i,1]))
        (x2,y2) = map(int,(xyxy_bbox[i,2], xyxy_bbox[i,3]))
        xywh_bbox.append([x1, y1, x2-x1, y2-y1])
    return mask, xywh_bbox


def main():
    path_train = '/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/train_list_coarse_review.txt'
    path_val = '/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/valid_list_coarse_review.txt'
    with open(path_train) as f:
        train_list = f.readlines()
        train_list = [train_list[i][-18:-5] for i in range(len(train_list))]
    with open(path_val) as f:
        val_list = f.readlines()
        val_list = [val_list[i][-18:-5] for i in range(len(val_list))]

    ann_path = glob.glob(r"/home/u/woody8657/data/C426_Pneumothorax_grid/annotation_review/json/*.json")
    
    img_list_train = []
    ann_list_train  = []
    img_list_val = []
    ann_list_val  = []
    cat_list = []

    box_id_train = 1
    box_id_val = 1
    image_id_train = 0
    image_id_val = 0
    avoid_repeat_list = []

   
    for count, i in enumerate(ann_path):
        print(f'processing {count} / {len(ann_path)} json...')
        # images
        tmp = json.load(open(i))
        mask = grid_bbox2mask(tmp)
        _,xywh_bbox = mask2bbox(mask)
        if tmp['accessionNumber'] in avoid_repeat_list:
            continue
        avoid_repeat_list.append(tmp['accessionNumber'])
        df = pd.read_csv("/home/u/woody8657/tmp/C426_G1_01_RADNCLREPORT.csv", encoding= 'unicode_escape')
        idx = df[df['ACCESSNO2']==tmp['accessionNumber']].index.values[0]
        patient_id = df.iloc[idx,0]
        file_name = patient_id + ".png"
        if patient_id in train_list:
            img_list_train.append({'file_name': file_name, 'height': tmp['size'][1], 'width': tmp['size'][0], 'id': image_id_train} )
            for j in range(len(xywh_bbox)):
                x, y, w, h = xywh_bbox[j][0], xywh_bbox[j][1], xywh_bbox[j][2], xywh_bbox[j][3]
                print(x,y,w,h)
                ann_list_train.append({'bbox': [x, y, w, h], 'area': w*h, 'score': 1.0, 'category_id': 1,  'id': box_id_train , 'image_id':  image_id_train, 'iscrowd': 0})
                box_id_train  = box_id_train  + 1
            image_id_train = image_id_train + 1
        elif patient_id in val_list:
            img_list_val.append({'file_name': file_name, 'height': tmp['size'][1], 'width': tmp['size'][0], 'id': image_id_val} )
            for j in range(len(xywh_bbox)):
                x, y, w, h = xywh_bbox[j][0], xywh_bbox[j][1], xywh_bbox[j][2], xywh_bbox[j][3]
                print(x,y,w,h)
                ann_list_val.append({'bbox': [x, y, w, h], 'area': w*h, 'score': 1.0, 'category_id': 1,  'id': box_id_val , 'image_id':  image_id_val, 'iscrowd': 0})
                box_id_val = box_id_val  + 1
            image_id_val = image_id_val + 1
        else:
            pass
            
    cat_list = [{'id':1, 'name': "Pneumothorax"}]
    json_dict_train = {'images': img_list_train, 'annotations': ann_list_train, 'categories': cat_list}
    json_dict_val = {'images': img_list_val, 'annotations': ann_list_val, 'categories': cat_list}

    with open("/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/Pneumothorax_yolov5_train.json", "w") as outfile:
        json.dump(json_dict_train, outfile)
    with open("/home/u/woody8657/projs/Pneumothorax-detection/mmdetection/configs/Pneumothorax/Pneumothorax_yolov5_val.json", "w") as outfile:
        json.dump(json_dict_val, outfile)
    
if __name__ == '__main__':
    main()