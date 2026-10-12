"""Separate experiment preparation, telemetry validation and live measurement."""

from pathlib import Path
import random

import yaml

from ..evaluation.dataset import EvaluationDataset
from ..detectors.configuration import inference_settings
from ..project import ROOT, bootstrap, read, save
from . import guard, power, power_check, runner
from .clocks import locked_clocks
from .isolated import IsolatedPower
from ..variants.registry import VARIANTS



def experiment_path(path):
    """Resolve a working directory while protecting dataset inputs."""
    directory = Path(path).resolve()

    if directory == ROOT / 'data' or ROOT / 'data' in directory.parents:
        raise ValueError('Dataset inputs are read-only')

    return directory


def prepare(output, dataset_manifest, repeats=None, config=None):
    """Create a new experiment directory and schedule; never execute a model."""
    directory = experiment_path(output)

    if directory.exists():
        raise FileExistsError('Choose a new experiment directory')

    (directory / 'manifests').mkdir(parents=True)
    (directory / 'results').mkdir()

    prepare_full_protocol(directory, dataset_manifest, repeats, config=config)

    print('Prepared', directory, '; inspect manifests/freeze.json before measurement')


def checked_experiment(path):
    """Resolve the experiment directory and initialize package paths."""
    directory = experiment_path(path)
    bootstrap()

    return directory


def validate_power(experiment):
    """Record fresh telemetry checks and require them to pass before a campaign."""
    directory = checked_experiment(experiment)
    
    guard.check_competing_benchmark()
    guard.device_snapshot()
    
    destination = directory / 'results/power_validation.json'

    if destination.exists():
        raise FileExistsError('Preserve existing power validation')

    power.validate(destination, sampler_factory=IsolatedPower)
    power_check.main(directory)

    if read(directory / 'results/power_check.json')['status'] != 'PASS':
        raise RuntimeError('Telemetry validation failed')


def measure(experiment, power_backend='none', lock_clocks=False):
    """Select the prepared dataset and launch the live trial runner."""
    directory = checked_experiment(experiment)

    if lock_clocks:
        guard.check_competing_benchmark()

        with locked_clocks(directory / 'results'):
            runner.main(directory, power_backend=power_backend)
    else:
        runner.main(directory, power_backend=power_backend)


def select_videos(videos, selection):
    """Select alphabetically sorted videos using integer start/end bounds."""
    if not isinstance(selection, dict):
        raise ValueError('sequence must contain integer start and end fields')

    start = selection.get('start', 0)
    end = selection.get('end', len(videos))

    if type(start) is not int or type(end) is not int:
        raise ValueError('sequence.start and sequence.end must be integers')

    selected = sorted(videos)[start:end]

    if not selected:
        raise ValueError(f'Video selection {selection!r} is empty')

    return selected


def prepare_full_protocol(out, dataset_manifest, repeats=None, config=None):
    """Save selected variants, complete videos and randomized trial order."""
    configuration = {}

    if config is not None:
        with Path(config).open() as config_file:
            configuration = yaml.safe_load(config_file) or {}

    settings = configuration.get('experiment', {})
    dataset_settings = configuration.get('dataset', {})

    variants = settings.get('variants', list(VARIANTS) + ['baseline'])

    if not isinstance(variants, list) or not variants:
        raise ValueError('experiment.variants must be a nonempty list')

    available = set(VARIANTS) | {'baseline'}

    for variant in variants:
        if not isinstance(variant, str) or variant not in available:
            raise ValueError(f'Unknown variant: {variant!r}; available: {sorted(available)}')

    if len(variants) != len(set(variants)):
        raise ValueError('Experiment variants must not contain duplicates')

    if repeats is None:
        repeats = settings.get('repeats', 3)

    if type(repeats) is not int or repeats < 1:
        raise ValueError('Repeats must be a positive integer')

    source = EvaluationDataset(dataset_manifest)
    if 'videos' in settings or 'sequence' in settings:
        raise ValueError('Put sequence.start and sequence.end under dataset in YAML')

    selection = dataset_settings.get('sequence', {'start': 0, 'end': len(source.videos)})
    videos = select_videos(source.videos, selection)
    selected_names = set(videos)

    # Retain original image identities and coordinates; filter only by video.
    frames = [frame for frame in source.frames() if frame['sequence'] in selected_names]
    image_paths = {frame['path'] for frame in frames}
    selected_manifest = {
        **source.manifest,
        'population': 'selected_videos',
        'videos': videos,
        'frames': frames,
        'labels': {video: source.manifest['labels'][video] for video in videos},
        'label_sha256': {video: source.manifest['label_sha256'][video] for video in videos},
        'image_sha256': {
            path: checksum
            for path, checksum in source.manifest['image_sha256'].items()
            if path in image_paths
        },
    }

    schedule = []
    random_generator = random.Random(20261114)

    for repeat in range(repeats):
        for video in videos:
            order = variants.copy()
            random_generator.shuffle(order)

            for variant in order:
                schedule.append({
                    'repeat': repeat,
                    'variant': variant,
                    'video': video,
                })

    save(out / 'manifests/dataset.json', selected_manifest)
    protocol = {
        'schema_version': 3,
        'inference': inference_settings(configuration.get('inference')),
        'population': 'selected_videos',
        'arms': variants,
        'videos': videos,
        'sequence': selection,
        'video_order': 'alphabetical',
        'repeats': repeats,
        'schedule': schedule,
        'dataset_manifest': 'manifests/dataset.json',
    }
    save(out / 'manifests/freeze.json', protocol)
