"""Compute fresh VPI hardware flow between consecutive images.

acquire() returns float32 [dx, dy] displacements in pixels.
"""

import time

import numpy as np


class HardwareFlow:
    """Calculate optical flow between image pairs using NVIDIA VPI's OFA backend."""

    def __init__(self):
        # VPI is required only for hardware execution. Keeping this import here
        # keeps package imports usable on hosts where NVIDIA VPI is not installed.
        import vpi

        self.vpi = vpi
        self.stream = vpi.Stream()
        self.last_index = None
        self.previous = None
        self.timings = []

    def convert(self, im):
        """Prepare a BGR image as the grayscale pyramid expected by OFA."""
        vpi = self.vpi
        image = vpi.asimage(im, vpi.Format.BGR8)
        grayscale = image.convert(vpi.Format.Y8_ER, backend=vpi.Backend.CUDA)
        pyramid = grayscale.gaussian_pyramid(3, backend=vpi.Backend.CUDA)
        prepared_pyramid = pyramid.convert(vpi.Format.Y8_ER_BL, backend=vpi.Backend.VIC)

        return prepared_pyramid

    def acquire(self, previous, current, index):
        """Compute a flow field, reusing the prior pyramid for consecutive pairs.

        previous and current are BGR images. index identifies the current frame.
        Stored timings separate image preparation, synchronized OFA execution,
        and CPU readback. Final conversion to float pixels is outside that timer.
        """
        vpi = self.vpi
        start = time.perf_counter()
        reuse_previous = self.last_index == index - 1 and self.previous is not None

        with self.stream:
            if reuse_previous:
                previous_pyramid = self.previous
            else:
                previous_pyramid = self.convert(previous)

            current_pyramid = self.convert(current)

        self.stream.sync()
        conversion_finished = time.perf_counter()

        with self.stream, vpi.Backend.OFA:
            motion_vectors = vpi.optflow_dense(
                previous_pyramid,
                current_pyramid,
                quality=vpi.OptFlowQuality.MEDIUM,
                gridsize=4,
            )

        self.stream.sync()
        flow_finished = time.perf_counter()

        with motion_vectors.rlock_cpu() as data:
            raw_flow = np.array(data, dtype=np.int16)

        readback_finished = time.perf_counter()
        self.previous = current_pyramid
        self.last_index = index

        self.timings.append({
            'index': index,
            'reused_previous_pyramid': reuse_previous,
            'conversion_ms': (conversion_finished - start) * 1000,
            'ofa_sync_ms': (flow_finished - conversion_finished) * 1000,
            'readback_ms': (readback_finished - flow_finished) * 1000,
            'total_ms': (readback_finished - start) * 1000,
            'raw_flow_bytes': raw_flow.nbytes,
        })

        # VPI's signed fixed-point vectors store 32 units per pixel.
        return raw_flow.astype(np.float32) / 32
