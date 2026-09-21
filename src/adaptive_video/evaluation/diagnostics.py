import numpy as np
from .coco import category

def iou(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)

    boxes_a = a[:, None, :]  
    boxes_b = b[None, :, :] 

    intersection_top_left = np.maximum(
        boxes_a[..., :2],
        boxes_b[..., :2],
    )
    intersection_bottom_right = np.minimum(
        boxes_a[..., 2:],
        boxes_b[..., 2:],
    )

    intersection_dimensions = np.maximum(
        0,
        intersection_bottom_right - intersection_top_left,
    )
    intersection_area = intersection_dimensions.prod(axis=2)

    dimensions_a = np.maximum(0, a[:, 2:] - a[:, :2])
    dimensions_b = np.maximum(0, b[:, 2:] - b[:, :2])

    areas_a = dimensions_a.prod(axis=1)
    areas_b = dimensions_b.prod(axis=1)

    union_area = (
        areas_a[:, None] + areas_b[None, :] - intersection_area
    )

    return intersection_area / np.maximum(union_area, 1e-12)

def greedy(ds, gs, threshold):
    """Match detections to same-class ground truth in confidence order.

    Returns detection-to-ground-truth indices, the set of used ground-truth
    indices, and the pairwise IoU matrix. Each ground-truth box is used once.
    """
    detection_boxes = [detection['box_xyxy'] for detection in ds]
    ground_truth_boxes = [ground_truth['box_xyxy'] for ground_truth in gs]
    overlaps = iou(detection_boxes, ground_truth_boxes)

    used = set()
    matched = {}

    detection_order = sorted(
        range(len(ds)),
        key=lambda index: -ds[index]['conf'],
    )

    for detection_index in detection_order:
        candidates = [
            ground_truth_index
            for ground_truth_index, ground_truth in enumerate(gs)
            if ground_truth_index not in used
            and category(ds[detection_index]) == ground_truth['class_id']
        ]
        if not candidates:
            continue

        best_match = max(
            candidates,
            key=lambda index: overlaps[detection_index, index],
        )
        if overlaps[detection_index, best_match] >= threshold:
            used.add(best_match)
            matched[detection_index] = best_match

    return matched, used, overlaps

