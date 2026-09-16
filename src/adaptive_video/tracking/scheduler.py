"""Video orchestration: advance, decide, observe, reconcile, publish.

Filtering, assignment, lifecycle and output are separate operations. BoxTrack types
are injected per controller; selecting a variant never changes module globals.
Method attribution is documented in licenses/ALBIREO-MIT.txt.
"""

from ..detectors.empty_scene import load_scene_classifier, classify_scene
from .association import associate, overlap_ratio
from .kalman import BoxTrack
from .parameters import (
    BOUNDARY_MARGIN,
    IOU_THRESHOLD,
    MAHAL_THRESHOLD,
    MAX_LOST_FRAMES,
    MAX_RESCUE_DEFAULT,
    MAX_SKIP,
    MIN_INFER_EVERY,
    RESCUE_CONF_DEFAULT,
    SIGMA_MEAS_PX,
    UNCERTAINTY_THRES,
)


class AdaptiveDetectionTracker:
    """Track objects and schedule detector refreshes from motion uncertainty and age."""

    def __init__(
        self,
        uncertainty_threshold=UNCERTAINTY_THRES,
        max_skip=MAX_SKIP,
        min_infer_every=MIN_INFER_EVERY,
        iou_threshold=IOU_THRESHOLD,
        mahal_threshold=MAHAL_THRESHOLD,
        frame_width=1920,
        frame_height=1080,
        sigma_meas_px=SIGMA_MEAS_PX,
        max_lost_frames=MAX_LOST_FRAMES,
        boundary_margin=BOUNDARY_MARGIN,
        class_filter=None,
        yolo_conf=0.25,
        rescue_conf=RESCUE_CONF_DEFAULT,
        max_rescue=MAX_RESCUE_DEFAULT,
        track_type=BoxTrack,
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.max_skip = max_skip
        self.min_infer_every = min_infer_every
        self.iou_threshold = iou_threshold
        self.mahal_threshold = mahal_threshold
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_lost_frames = max_lost_frames
        self.boundary_margin = boundary_margin
        self.class_filter = class_filter
        self.yolo_conf = yolo_conf
        self.rescue_conf = rescue_conf
        self.max_rescue = max_rescue
        self.track_type = track_type
        self.tracks = []
        self.frame_idx = 0
        self._imgsz = tuple(
            ((size + 31) // 32) * 32 for size in (frame_height, frame_width)
        )
        self._sigma_meas_norm = sigma_meas_px / max(self._imgsz)

        self._erd_device = 'cpu'
        self._erd_model = load_scene_classifier(self._erd_device)

    def _advance(self):
        """Predict each track and remove tracks outside the allowed image margin."""
        for track in self.tracks:
            track.predict()

            if track.is_out_of_frame(self.boundary_margin):
                track.state = 'lost'

        self.tracks = [track for track in self.tracks if track.state != 'lost']

    def _decision(self, image):
        """Decide whether empty-scene screening, uncertainty or age requires detection."""
        if not self.tracks:
            occupied = classify_scene(self._erd_model, image, self._erd_device) == 1

            return occupied, True, 'non-empty' if occupied else 'empty'

        uncertain = any(
            track.is_uncertain(self.uncertainty_threshold) for track in self.tracks
        )
        stale = any(track.needs_forced_infer(self.max_skip) for track in self.tracks)
        periodic = self.frame_idx % self.min_infer_every == 0

        return bool(uncertain or stale or periodic), False, None

    def _observations(self, result):
        """Convert accepted detector boxes into measurements for association."""
        boxes = result.boxes

        if boxes is None:
            return []

        observations = []

        for index in range(len(boxes)):
            category = int(boxes.cls[index].item())

            if self.class_filter is None or category in self.class_filter:
                observations.append(
                    dict(
                        box=boxes.xywh[index].cpu().numpy(),
                        class_id=category,
                        class_name=result.names[category],
                        conf=float(boxes.conf[index].item()),
                    )
                )

        return observations

    def _match(self, active, observations):
        """Assign detector measurements to existing predicted tracks."""
        return associate(active, observations, self.iou_threshold, self.mahal_threshold)

    def _reconcile(self, observations):
        """Correct matches, rescue eligible misses and create tracks for new objects."""
        active = list(self.tracks)
        matched_objects, matched_observations = self._match(active, observations)

        for object_index, observation_index in zip(
            matched_objects, matched_observations
        ):
            observation = observations[observation_index]
            track = active[object_index]
            track.update(
                observation['box'],
                observation['class_id'],
                observation['class_name'],
                self.frame_idx,
                conf=observation['conf'],
            )
            track.consecutive_rescues = 0

        matched_track_indices = set(matched_objects)
        rescued = 0

        for index, track in enumerate(active):
            if index in matched_track_indices:
                continue

            can_rescue = (
                track.state == 'confirmed' and track.last_conf >= self.rescue_conf
            )
            can_rescue = can_rescue and track.consecutive_rescues < self.max_rescue

            if can_rescue:
                track.consecutive_rescues += 1
                track.frames_since_infer = 0
                rescued += 1
            else:
                track.state = 'lost'

        matched_detection_indices = set(matched_observations)

        for index, observation in enumerate(observations):
            if index not in matched_detection_indices:
                self.tracks.append(
                    self.track_type(
                        observation['box'],
                        observation['class_id'],
                        observation['class_name'],
                        self.frame_idx,
                        frame_width=self.frame_width,
                        frame_height=self.frame_height,
                        sigma_meas=self._sigma_meas_norm,
                        max_lost_frames=self.max_lost_frames,
                        conf=observation['conf'],
                    )
                )

        self.tracks = [track for track in self.tracks if track.state != 'lost']

        return rescued

    def step(self, frame, model, device):
        """Advance tracks, optionally detect, then publish the current frame outputs."""
        self._advance()
        run_detector, ran_scene_classifier, scene_result = self._decision(frame)
        rescued = 0

        if run_detector:
            prediction = model.predict(
                frame, imgsz=self._imgsz, conf=self.yolo_conf, verbose=False
            )
            rescued = self._reconcile(self._observations(prediction[0]))

        confirmed = [track for track in self.tracks if track.state == 'confirmed']
        published = [
            dict(
                track_id=track.track_id,
                class_name=track.class_name,
                box_xyxy=track.get_box_xyxy().tolist(),
                conf=track.get_confidence(self.uncertainty_threshold),
                state=track.state,
            )
            for track in confirmed
        ]
        result = dict(
            frame_idx=self.frame_idx,
            ran_inference=run_detector,
            ran_erd=ran_scene_classifier,
            erd_result=scene_result,
            detections=published,
            num_tracks=len(self.tracks),
            num_confirmed=len(confirmed),
            num_rescued=rescued,
        )
        self.frame_idx += 1

        return result
