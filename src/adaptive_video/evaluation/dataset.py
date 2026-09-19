"""Explicit, immutable evaluation populations independent of saved splits."""

from pathlib import Path

import yaml

from .. import dataset as serialization
from ..project import ROOT


class EvaluationDataset:
    """Use a generated frame inventory, retaining stable per-frame identities."""

    def __init__(self, manifest, root=ROOT):
        self.root = Path(root)
        self.manifest = serialization.read(manifest)
        self._frames = self.manifest['frames']
        self.videos = self.manifest['videos']
        identities = [frame['image_id'] for frame in self._frames]

        if len(identities) != len(set(identities)):
            raise ValueError('Duplicate image identity in dataset manifest')

        self._by_video = {video: [] for video in self.videos}

        for frame in self._frames:
            self._by_video[frame['sequence']].append(frame)

        for video, frames in self._by_video.items():
            indices = [frame['frame_index'] for frame in frames]

            if not indices or indices != sorted(set(indices)):
                raise ValueError(f'Invalid frame order: {video}')

        if len(self.videos) != len(set(self.videos)):
            raise ValueError('Duplicate video in dataset manifest')

    def frames(self, video=None):
        if video is None:
            return self._frames

        return self._by_video[video]

    def labels(self, video):
        if video not in self._by_video:
            raise ValueError(f'Video outside selected dataset: {video}')

        path = self.root / self.manifest['labels'][video]

        if serialization.sha(path) != self.manifest['label_sha256'][video]:
            raise ValueError(f'Annotation changed after dataset preparation: {video}')

        return serialization.read(path)

    def image(self, frame):
        import cv2
        image = cv2.imread(str(self.root / frame['path']))

        if image is None or image.shape[:2] != (frame['height'], frame['width']):
            raise ValueError(f'Invalid image: {frame["path"]}')

        return image


def prepare_dataset_manifest(destination, root=ROOT, expected_videos=200,
                             expected_frames=39973, config=None):
    """Index all validation annotations; hash inputs without changing old manifests."""

    root = Path(root).resolve()
    destination = Path(destination)

    if destination.exists():
        raise FileExistsError(destination)

    if config is None:
        images_root = root / 'data/bdd100k/images/track/val'
        labels_root = root / 'data/bdd100k/labels/box_track_20/val'
    else:
        with Path(config).open() as config_file:
            settings = yaml.safe_load(config_file)

        dataset_settings = settings['dataset']
        images_root = root / dataset_settings['images']
        labels_root = root / dataset_settings['labels']

    videos = sorted(path.name for path in images_root.iterdir() if path.is_dir())

    if len(videos) != expected_videos:
        raise ValueError(f'Expected {expected_videos} videos; found {len(videos)}')

    if {path.stem for path in labels_root.glob('*.json')} != set(videos):
        raise ValueError('Image and annotation video inventories differ')

    frames, labels, label_hashes, image_hashes = [], {}, {}, {}

    for video in videos:
        label_path = labels_root / f'{video}.json'
        labels[video] = str(label_path.relative_to(root))
        label_hashes[video] = serialization.sha(label_path)
        annotations = sorted(serialization.read(label_path), key=lambda row: row['frameIndex'])
        indices = [row['frameIndex'] for row in annotations]
        names = [row['name'] for row in annotations]

        if len(indices) != len(set(indices)) or len(names) != len(set(names)):
            raise ValueError(f'Duplicate annotation frame: {video}')

        if set(names) != {path.name for path in (images_root / video).glob('*.jpg')}:
            raise ValueError(f'Image and annotation frames differ: {video}')

        for row in annotations:
            path = images_root / video / row['name']
            relative = str(path.relative_to(root))
            image_hashes[relative] = serialization.sha(path)
            frames.append(dict(image_id=len(frames) + 1, sequence=video,
                               frame_index=row['frameIndex'], path=relative,
                               width=1280, height=720))

    if len(frames) != expected_frames:
        raise ValueError(f'Expected {expected_frames} frames; found {len(frames)}')

    manifest = dict(schema_version=1, population='full_validation', videos=videos,
                    frames=frames, labels=labels, label_sha256=label_hashes,
                    image_sha256=image_hashes)
    serialization.save(destination, manifest)

    return manifest
