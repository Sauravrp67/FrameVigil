"""Normalized box estimation and track lifecycle.

The probability operations are functions; BoxTrack owns the evolving numerical
state and lifecycle counters. See licenses/ALBIREO-MIT.txt for method provenance.
"""
import numpy as np
from scipy.linalg import solve, LinAlgError
from .parameters import *


def transition_matrix():
    matrix = np.eye(10, dtype=np.float64)
    
    for observed in range(4):
        matrix[observed, observed + 4] = 1.0
    
    for position, velocity, acceleration in ((0, 4, 8), (1, 5, 9)):
        matrix[position, acceleration] = 0.5
        matrix[velocity, acceleration] = 1.0
    
    return matrix


def observation_vector(box, horizontal_scale, vertical_scale):
    # Scalar division preserves the frozen NumPy promotion convention.
    scale = (horizontal_scale, vertical_scale) * 2
    return np.array(
        [value / unit for value, unit in zip(box, scale)], 
        dtype=np.float64)


def advance_distribution(mean, covariance, transition, noise):
    x_pred = transition @ mean
    P_pred = transition @ covariance @ transition.T + noise
    
    return x_pred , P_pred


def innovation_terms(mean, covariance, observation, projection, noise):
    difference = observation - projection @ mean
    projected_covariance = projection @ covariance @ projection.T + noise
    
    return difference, projected_covariance


def correct_distribution(mean, covariance, observation, projection, noise):
    difference, projected = innovation_terms(mean, covariance, observation, projection, noise)
    gain = covariance @ projection.T @ np.linalg.inv(projected)
    
    corrected_mean = mean + gain @ difference
    corrected_covariance = (np.eye(len(mean)) - gain @ projection) @ covariance
    
    return corrected_mean, corrected_covariance


class BoxTrack:
    """One object's geometry, motion, class, confidence and lifetime counters.

    x uses [cx,cy,w,h,vx,vy,vw,vh,ax,ay]; x and P are float64. Serialized
    field names remain the serialized experiment contract, not a dependency
    on a previous implementation. IDs are allocated by the concrete class.
    """
    _id_counter = 0

    def __init__(self, 
                 box_cxcywh_px, 
                 class_id, 
                 class_name, 
                 frame_idx,
                 frame_width=1920, 
                 frame_height=1080, 
                 sigma_p=SIGMA_P,
                 sigma_s=SIGMA_S, 
                 sigma_v=SIGMA_V, 
                 sigma_a=SIGMA_A,
                 sigma_meas=SIGMA_MEAS_PX / 640, 
                 max_lost_frames=MAX_LOST_FRAMES,
                 conf=1.0):
        concrete = type(self)
        self.track_id = concrete._id_counter
        concrete._id_counter += 1
        
        self.frame_width, self.frame_height = frame_width, frame_height
        self._sx, self._sy = float(frame_width), float(frame_height)
        
        self.x = np.r_[observation_vector(box_cxcywh_px, self._sx, self._sy), np.zeros(6)]
        self.F = transition_matrix()
        self.H = np.eye(4, 10, dtype=np.float64)
        deviations = [sigma_p] * 2 + [sigma_s] * 2 + [sigma_v] * 4 + [sigma_a] * 2
        self.Q = np.diag([value ** 2 for value in deviations]).astype(np.float64)
        self.R = np.diag([sigma_meas ** 2] * 4).astype(np.float64)
        initial = [1e-2] * 2 + [5e-3] * 6 + [1e-3] * 2
        self.P = np.diag([value ** 2 for value in initial]).astype(np.float64)
        self.class_id, self.class_name = class_id, class_name
        self.last_conf, self.last_seen_frame = conf, frame_idx
        self.max_lost_frames = max_lost_frames
        self.state, self.hits = 'tentative', 1
        self.age = self.frames_since_update = self.frames_since_infer = 0
        self.consecutive_rescues = 0

    def _bound_size(self):
        self.x[2] = max(self.x[2], 1.0 / self._sx)
        self.x[3] = max(self.x[3], 1.0 / self._sy)

    def predict(self):
        self.x, self.P = advance_distribution(self.x, self.P, self.F, self.Q)
        self.age += 1
        
        self.frames_since_update += 1
        self.frames_since_infer += 1
        
        self._bound_size()
        
        return self.get_box_cxcywh()

    def update(self, box_cxcywh_px, class_id, class_name, frame_idx, conf=1.0):
        measured = observation_vector(box_cxcywh_px, self._sx, self._sy)
        
        self.x, self.P = correct_distribution(self.x, self.P, measured, self.H, self.R)
        self._bound_size()
        self.hits += 1
        
        self.frames_since_update = self.frames_since_infer = 0
        self.last_seen_frame, self.last_conf = frame_idx, conf
        self.class_id, self.class_name = class_id, class_name
        
        if self.state == 'tentative' and self.hits >= MIN_HITS_CONFIRM:
            self.state = 'confirmed'

    def mark_missed(self):
        self.frames_since_update += 1
        expires = self.state == 'confirmed' and self.frames_since_update > self.max_lost_frames
        
        if self.state == 'tentative' or expires:
            self.state = 'lost'

    def is_uncertain(self, threshold=UNCERTAINTY_THRES):
        return float(np.trace(self.P[:4, :4])) > threshold

    def needs_forced_infer(self, max_skip=MAX_SKIP):
        return self.frames_since_infer >= max_skip

    def is_out_of_frame(self, margin=BOUNDARY_MARGIN):
        return any(value < -margin or value > 1.0 + margin for value in self.x[:2])

    def get_box_cxcywh(self):
        units = (self._sx, self._sy) * 2
        return np.array([value * unit for value, unit in zip(self.x[:4], units)], dtype=np.float64)

    def get_box_xyxy(self):
        cx, cy, width, height = self.get_box_cxcywh()
        edges = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
        limits = (self.frame_width - 1, self.frame_height - 1) * 2
        
        return np.array([max(0, min(int(edge), limit)) for edge, limit in zip(edges, limits)], dtype=np.int32)

    def mahalanobis_sq(self, box_cxcywh_px):
        measured = observation_vector(box_cxcywh_px, self._sx, self._sy)
        difference, projected = innovation_terms(self.x, self.P, measured, self.H, self.R)
        
        try:
            weighted = solve(projected, difference, assume_a='pos')
        except LinAlgError:
            return 1e9
        return float(difference @ weighted)

    def get_confidence(self, uncertainty_threshold=UNCERTAINTY_THRES):
        fraction = np.trace(self.P[:4, :4]) / uncertainty_threshold
        return self.last_conf * max(0.1, 1.0 - fraction)

    @classmethod
    def reset_id_counter(cls):
        cls._id_counter = 0
