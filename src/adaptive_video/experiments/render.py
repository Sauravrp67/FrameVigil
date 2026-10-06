"""Draw completed prediction outputs on a selected video; never run inference."""

import colorsys
import hashlib
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np

from ..dataset import read, save, sha
from ..evaluation.dataset import EvaluationDataset
from ..project import ROOT


HEADER_HEIGHT = 90


def track_color(identity):
    """Give an actual track ID a stable color, including string-valued IDs."""
    digest = hashlib.sha256(str(identity).encode()).digest()
    hue = int.from_bytes(digest[:4], 'big') / 2**32
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.7, 1.0)

    return tuple(int(channel * 255) for channel in (blue, green, red))


def draw_text(image, message, position, color=(240, 240, 240)):
    """Draw readable text with a dark outline."""
    cv2.putText(image, message, position, cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, message, position, cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA)


def load_predictions(paths, video, frame_count):
    """Load complete trial files for one sequence, with one file per variant."""
    records = []
    variants = set()

    for path in paths:
        record = read(path)
        variant = record['variant']

        if record['video'] != video:
            raise ValueError(f'Prediction video differs from selected video: {path}')

        if variant in variants:
            raise ValueError(f'Provide only one trial per variant: {variant}')

        if record.get('status', 'PASS') != 'PASS':
            raise ValueError(f'Incomplete or failed trial: {path}')

        if len(record['outputs']) != frame_count:
            raise ValueError(f'Predictions must cover the complete selected video: {path}')

        if record.get('frames', frame_count) != frame_count:
            raise ValueError(f'Recorded frame count differs from the dataset: {path}')

        trace = record.get('trace')

        if trace is not None:
            if len(trace) != frame_count:
                raise ValueError(f'Incomplete decision trace: {path}')

            for index, action in enumerate(trace):
                if action.get('frame_idx') != index or type(action.get('ran_inference')) is not bool:
                    raise ValueError(f'Invalid decision trace at frame {index}: {path}')

        variants.add(variant)
        records.append(record)

    if not records:
        raise ValueError('Provide at least one prediction file')

    return records


def render_panel(image, detections, variant, frame_index, action):
    """Annotate a copy; retain every emitted box without a new score cutoff."""
    height, width = image.shape[:2]
    panel = np.full((height + HEADER_HEIGHT, width, 3), 22, dtype=np.uint8)
    panel[HEADER_HEIGHT:] = image
    status = 'decision unavailable'

    if action is not None:
        status = 'detector ran' if action['ran_inference'] else 'detector skipped'

    draw_text(panel, variant, (12, 24))
    draw_text(panel, f'Frame {frame_index} | {status}', (12, 49))
    draw_text(panel, 'Saved predictions | no additional score cutoff', (12, 74))

    for detection in detections:
        box = np.array(detection['box_xyxy'], dtype=float, copy=True)
        confidence = float(detection['conf'])

        if box.shape != (4,) or not np.isfinite(box).all() or not math.isfinite(confidence):
            raise ValueError('Prediction contains invalid coordinates or confidence')

        identity = detection.get('track_id')
        color = track_color(identity) if identity is not None else (80, 220, 255)
        class_name = detection.get('class_name', str(detection.get('class_id', '?')))
        label = f'{class_name} {confidence:.3f}'

        if identity is not None:
            label = f'ID {identity} | {label}'

        # Clip only the drawing coordinates, never the saved prediction values.
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
        box[[1, 3]] += HEADER_HEIGHT
        top_left = tuple(np.rint(box[:2] * 16).astype(int))
        bottom_right = tuple(np.rint(box[2:] * 16).astype(int))
        cv2.rectangle(panel, top_left, bottom_right, color, 2, cv2.LINE_AA, shift=4)
        position = (int(box[0]), max(HEADER_HEIGHT + 18, int(box[1]) - 5))
        draw_text(panel, label, position, color)

    return panel


class VideoWriter:
    """Stream BGR frames to FFmpeg and verify the finished file with ffprobe."""

    def __init__(self, path, width, height, fps):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.frame_count = 0
        self.process = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
            '-f', 'rawvideo', '-pixel_format', 'bgr24',
            '-video_size', f'{width}x{height}', '-framerate', str(fps),
            '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '18', '-threads', '2', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(self.path),
        ], stdin=subprocess.PIPE)

    def write(self, frame):
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError('Unexpected video frame shape or dtype')

        self.process.stdin.write(frame.tobytes())
        self.frame_count += 1

    def abort(self):
        self.process.terminate()
        self.process.communicate()

    def close(self):
        self.process.stdin.close()

        if self.process.wait() != 0:
            raise RuntimeError('FFmpeg failed; partial output preserved')

        result = subprocess.check_output([
            'ffprobe', '-v', 'error', '-count_frames', '-show_streams',
            '-show_format', '-of', 'json', str(self.path),
        ])
        metadata = json.loads(result)
        stream = metadata['streams'][0]

        if (int(stream['nb_read_frames']) != self.frame_count
                or int(stream['width']) != self.width
                or int(stream['height']) != self.height):
            raise ValueError('Encoded video does not match the requested frames')

        return {
            'frames': self.frame_count,
            'width': self.width,
            'height': self.height,
            'frame_rate': stream['avg_frame_rate'],
            'duration_s': float(metadata['format']['duration']),
        }


def run(dataset_manifest, prediction_paths, video, output, fps=5.0):
    """Render one or more variants side by side from an explicit dataset manifest."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Playback FPS must be positive and finite')

    source = EvaluationDataset(dataset_manifest)
    frames = source.frames(video)
    prediction_paths = [Path(path) for path in prediction_paths]
    records = load_predictions(prediction_paths, video, len(frames))
    output = Path(output).resolve()

    for protected in (ROOT / 'data',):
        if output == protected or protected in output.parents:
            raise ValueError('Dataset inputs are read-only')

    output.mkdir(parents=True, exist_ok=False)
    columns = min(2, len(records))
    rows = math.ceil(len(records) / columns)
    width = frames[0]['width'] * columns
    height = (frames[0]['height'] + HEADER_HEIGHT) * rows
    partial_path = output / 'comparison.partial.mp4'
    input_hashes = {str(path): sha(path) for path in prediction_paths}
    image_hashes = {}
    writer = VideoWriter(partial_path, width, height, fps)

    try:
        for index, frame in enumerate(frames):
            path = source.root / frame['path']
            checksum = sha(path)

            if checksum != source.manifest['image_sha256'][frame['path']]:
                raise ValueError(f'Image changed after dataset preparation: {path}')

            image_hashes[frame['path']] = checksum
            image = source.image(frame)
            panels = []

            for record in records:
                trace = record.get('trace')
                action = None if trace is None else trace[index]
                panels.append(render_panel(
                    image, record['outputs'][index], record['variant'], index, action,
                ))

            while len(panels) < rows * columns:
                panels.append(np.full_like(panels[0], 22))

            panel_rows = [np.hstack(panels[start:start + columns])
                          for start in range(0, len(panels), columns)]
            comparison = np.vstack(panel_rows)
            writer.write(comparison)

            if index == 0 and not cv2.imwrite(str(output / 'preview.png'), comparison):
                raise RuntimeError('Could not write preview image')

        encoded = writer.close()

    except BaseException:
        if writer.process.poll() is None:
            writer.abort()
        raise

    video_path = output / 'comparison.mp4'
    partial_path.rename(video_path)
    manifest = {
        'video': video,
        'variants': [record['variant'] for record in records],
        'playback_fps': fps,
        'dataset_manifest_sha256': sha(dataset_manifest),
        'prediction_sha256': input_hashes,
        'image_sha256': image_hashes,
        'output': {**encoded, 'path': video_path.name, 'sha256': sha(video_path)},
        'scope': 'Visualization of saved outputs; no inference, ground truth or score filtering.',
        'identity_policy': 'Only saved track IDs are displayed; IDs are never inferred.',
    }
    save(output / 'manifest.json', manifest)

    return manifest
