"""Two-pass assignment with explicit eligibility and acceptance gates."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def overlap_ratio(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second

    horizontal = max(0.0, min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2))
    vertical = max(0.0, min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2))
    
    intersection = horizontal * vertical
    union = aw * ah + bw * bh - intersection

    if union > 0:
        return intersection / union 
    else:
        return 0.0
    

def _assign(costs, row_ids, column_ids, accept):
    rows, columns = linear_sum_assignment(costs)
    return [(row_ids[r], column_ids[c]) 
            for r, c in zip(rows, columns) 
            if accept(costs[r, c])]


def associate(objects, observations, minimum_overlap, distance_limit):
    if not objects or not observations:
        return [], []
    boxes = np.array([item['box'] 
                    for item in observations])
    available_columns = list(range(len(boxes)))
    recent = [i for i, item in enumerate(objects) if item.frames_since_update <= 2]
    pairs = []
    
    if recent:
        costs = np.array([[1.0 - overlap_ratio(objects[i].get_box_cxcywh(), box)
                           for box in boxes] for i in recent], dtype=np.float64)
        pairs.extend(_assign(costs, recent, available_columns, lambda cost: 1.0 - cost >= minimum_overlap))
    
    used_rows = {i for i, _ in pairs}
    used_columns = {j for _, j in pairs}
    remaining_rows = [i for i in range(len(objects)) if i not in used_rows]
    remaining_columns = [j for j in available_columns if j not in used_columns]
    
    if remaining_rows and remaining_columns:
        costs = np.full((len(remaining_rows), len(remaining_columns)), 1e9, dtype=np.float64)
        for row, i in enumerate(remaining_rows):
            for column, j in enumerate(remaining_columns):
                distance = objects[i].mahalanobis_sq(boxes[j])
                if distance < distance_limit:
                    costs[row, column] = distance
        pairs.extend(_assign(costs, remaining_rows, remaining_columns, lambda cost: cost < distance_limit))
    
    return [i for i, _ in pairs], [j for _, j in pairs]
