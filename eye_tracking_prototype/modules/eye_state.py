"""Per-frame fixation and saccade classification for screen-space gaze."""

import math
from collections import deque


class EyeStateClassifier:
    """Classify valid smoothed gaze samples using dispersion and jump distance."""

    FIXATION = "Fixation"
    SACCADE = "Saccade"
    UNCLASSIFIED = "Unclassified"

    def __init__(self, fixation_radius_px=15.0, fixation_frames=6, saccade_threshold_px=80.0):
        self.fixation_radius_px = float(fixation_radius_px)
        self.fixation_frames = int(fixation_frames)
        self.saccade_threshold_px = float(saccade_threshold_px)
        self._points = deque(maxlen=self.fixation_frames)
        self._previous_point = None

    def reset(self):
        """Discard state after an invalid gaze sample or a tracking restart."""
        self._points.clear()
        self._previous_point = None

    def update(self, gaze_x, gaze_y, valid):
        """Return the eye state for one frame.

        A saccade is evaluated before fixation. Fixation dispersion is the
        maximum Euclidean distance of the latest samples from their centroid.
        """
        if not valid:
            self.reset()
            return self.UNCLASSIFIED

        point = (float(gaze_x), float(gaze_y))
        if self._previous_point is not None:
            jump_distance = math.dist(point, self._previous_point)
            if jump_distance >= self.saccade_threshold_px:
                self._points.clear()
                self._points.append(point)
                self._previous_point = point
                return self.SACCADE

        self._points.append(point)
        self._previous_point = point
        if len(self._points) < self.fixation_frames:
            return self.UNCLASSIFIED

        centroid_x = sum(x for x, _ in self._points) / len(self._points)
        centroid_y = sum(y for _, y in self._points) / len(self._points)
        max_radius = max(
            math.hypot(x - centroid_x, y - centroid_y)
            for x, y in self._points
        )
        return self.FIXATION if max_radius <= self.fixation_radius_px else self.UNCLASSIFIED
