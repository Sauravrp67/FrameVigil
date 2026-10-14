from __future__ import annotations
from typing import Dict, Set
from collections import defaultdict
import contextlib,io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BDD_TO_COCO: Dict[str, int] = {
    "pedestrian":   0,   # COCO: person
    "rider":        0,   # COCO: person
    "car":          2,   # COCO: car
    "truck":        7,   # COCO: truck
    "bus":          5,   # COCO: bus
    "bicycle":      1,   # COCO: bicycle
    "motorcycle":   3,   # COCO: motorcycle
    "train":        6,   # COCO: train
    # Skipped: "other vehicle", "other person", "trailer"
}

DEFAULT_EVAL_CLASSES: Set[str] = {
    "pedestrian", "rider", "car", "truck", "bus", "bicycle", "motorcycle",
}

IDS=[0,1,2,3,5,7]

def avg(x):
    if (x >= 0).any():
        return float(x[x >= 0].mean())
    else:
        return None

def ground_truth(source):
    fs=source.frames()
    anns=[]
    raw={}
    lookup={
        (f['sequence'],f['frame_index']):f 
        for f in fs
    }

    for v in sorted(
        {
            f['sequence'] for f in fs
        }
    ):
        for r in source.labels(v):
            f=lookup[v,r['frameIndex']]
            rr=[]

            for a in r.get('labels',[]):
                if a['category'] not in DEFAULT_EVAL_CLASSES or 'box2d' not in a:
                    continue

                b=a['box2d']
                xy=[b[k] 
                    for k in ['x1','y1','x2','y2']]
                
                w=xy[2]-xy[0]
                h=xy[3]-xy[1]
                
                if min(w,h)<=0:
                    continue

                cid=BDD_TO_COCO[a['category']]
                
                rr.append(dict(box_xyxy=xy,
                               class_id=cid,
                               track_id=str(a['id']),
                               source_category=a['category']))
                anns.append(dict(id=len(anns)+1,
                                 image_id=f['image_id'],
                                 category_id=cid,
                                 bbox=[*xy[:2],w,h],\
                                 area=w*h,
                                 iscrowd=0,
                                 track_id=str(a['id'])))
            
            raw[f['image_id']]=rr

    gt=COCO()
    gt.dataset=dict(info={},
                    images=[dict(id=f['image_id'],
                                 width=f['width'],
                                 height=f['height']) 
                                 for f in fs],
                    annotations=anns,
                    categories=[dict(id=i,
                                     name=str(i)) 
                                for i in IDS])
    
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()

    return gt,raw

def category(d):
    if 'class_id' in d:
        return d['class_id']

    class_ids = {
        'person': 0,
        'bicycle': 1,
        'car': 2,
        'motorcycle': 3,
        'bus': 5,
        'truck': 7,
    }

    return class_ids[d['class_name']]

def coco(gt,preds,img_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        if preds:
            dt=gt.loadRes(preds)
        else:
            dt=COCO();dt.dataset=dict(images=gt.dataset['images'],annotations=[],categories=gt.dataset['categories']);dt.createIndex()
        
        e=COCOeval(gt,dt,'bbox')
        e.params.imgIds=list(img_ids)
        e.params.maxDets=[1,10,300]
        e.evaluate()
        e.accumulate()

    p=e.eval['precision'][:,:,:,0,-1]
    
    m=dict(
        AP50_95=avg(p),
        AP50=avg(p[0]),
        AP75=avg(p[5]),
        AR300=avg(e.eval['recall'][:,:,0,-1]),
        AP_small=avg(e.eval['precision'][:,:,:,1,-1]),
        emitted_boxes=len(preds))
    
    for ti,tag in [(0,'50'),(5,'75')]:
        tp=fp=ngt=0

        for x in e.evalImgs:
            if x is None or x['aRng']!=e.params.areaRng[0]:
                continue

            valid=~x['dtIgnore'][ti]
            tp+=int(((x['dtMatches'][ti]>0)&valid).sum())
            fp+=int(((x['dtMatches'][ti]==0)&valid).sum())
            ngt+=int((~x['gtIgnore'].astype(bool)).sum())

        m.update(
            {
                f'TP{tag}':tp,
                f'FP{tag}':fp,
                f'FN{tag}':ngt-tp,
                f'precision{tag}':tp/max(1,tp+fp),
                f'recall{tag}':tp/max(1,ngt)
            }
        )

    return m,e

def predrows(ds,id):
    return [dict(
        image_id=id,
        category_id=category(d),
        score=d['conf'],
        bbox=[d['box_xyxy'][0],
              d['box_xyxy'][1],
              d['box_xyxy'][2]-d['box_xyxy'][0],
              d['box_xyxy'][3]-d['box_xyxy'][1]]) 
        for d in ds]



def localization_curve(evaluator):
    """AP at each IoU threshold, all object sizes, maxDets=300."""
    precision = evaluator.eval['precision'][:, :, :, 0, -1]

    return [
        {'iou': float(threshold), 'AP_percent': as_percent(avg(precision[index]))}
        for index, threshold in enumerate(evaluator.params.iouThrs)
    ]


def as_percent(value):
    return None if value is None else 100 * value


def frame_errors(evaluator, threshold=0.75):
    """Count COCO TP/FP/FN per image at one IoU, retaining its ignore policy."""
    threshold_index = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, threshold))[0])
    counts = defaultdict(lambda: {'TP75': 0, 'FP75': 0, 'FN75': 0})

    for result in evaluator.evalImgs:
        if result is None or result['aRng'] != evaluator.params.areaRng[0]:
            continue

        valid = ~result['dtIgnore'][threshold_index]
        matched = result['dtMatches'][threshold_index] > 0
        true_positive = int((matched & valid).sum())
        ground_truth_count = int((~result['gtIgnore'].astype(bool)).sum())
        row = counts[result['image_id']]
        row['TP75'] += true_positive
        row['FP75'] += int((~matched & valid).sum())
        row['FN75'] += ground_truth_count - true_positive

    return counts
