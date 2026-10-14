"""Shared frame-decision fields and baseline detector output conversion."""

ACTION_KEYS = [
    'frame_idx',
    'ran_inference',
    'ran_erd',
    'erd_result',
    'num_rescued',
    'num_tracks',
    'num_confirmed',
]


def always_rows(result):
    """Keep the six evaluated classes; the baseline detector has no track IDs."""
    rows = []

    for detection in result[0].boxes.data.detach().cpu().numpy():
        class_id = int(detection[5])

        if class_id in {0, 1, 2, 3, 5, 7}:
            rows.append({
                'box_xyxy': detection[:4].tolist(),
                'conf': float(detection[4]),
                'class_id': class_id,
                'class_name': result[0].names[class_id],
            })

    return rows
