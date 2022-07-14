import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', help='ground truth json')
    parser.add_argument('--json', help='prediction json')
    opt = parser.parse_args()
    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here


    cocoGt=COCO(opt.gt)

    cocoDt=cocoGt.loadRes(opt.json)

    imgIds=sorted(cocoGt.getImgIds())
    imgId = imgIds

    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()