import numpy as np
from collections import deque

class MetricsEngine:
    def __init__(self, cfg):
        self.dwell_threshold = cfg.dwell_threshold / 1000.0
        self.transition_window_sec = cfg.transition_window_sec
        self.iris_baseline_frames = cfg.iris_baseline_frames
        
        self.current_section = None
        self.section_enter_time = 0
        self.dwell_time_ms = 0
        
        self.nrevisit_counts = {}
        self.transition_history = deque()
        
        self.iris_history = []
        self.iris_baseline = None
        self.iris_delta = 0.0
        
        self.session_sections_summary = {}

    def update(self, timestamp_ms, section, iris_size):
        ts_sec = timestamp_ms / 1000.0
        
        if self.iris_baseline is None:
            if iris_size > 0:
                self.iris_history.append(iris_size)
                if len(self.iris_history) >= self.iris_baseline_frames:
                    self.iris_baseline = np.mean(self.iris_history)
                    self.iris_delta = 0.0
        else:
            if iris_size > 0:
                self.iris_delta = iris_size - self.iris_baseline
            
        if section != self.current_section:
            if self.current_section is not None:
                if self.current_section not in self.session_sections_summary:
                    self.session_sections_summary[self.current_section] = {
                        "total_dwell_ms": 0, "visit_count": 0, "nrevisit_count": 0, "max_continuous_dwell_ms": 0
                    }
                ss = self.session_sections_summary[self.current_section]
                ss["total_dwell_ms"] += self.dwell_time_ms
                ss["visit_count"] += 1
                ss["nrevisit_count"] = self.nrevisit_counts.get(self.current_section, 0)
                if self.dwell_time_ms > ss["max_continuous_dwell_ms"]:
                    ss["max_continuous_dwell_ms"] = self.dwell_time_ms
            
            if section is not None:
                self.transition_history.append((ts_sec, section))
                if section in self.nrevisit_counts:
                    self.nrevisit_counts[section] += 1
                else:
                    self.nrevisit_counts[section] = 0
                
            self.current_section = section
            self.section_enter_time = ts_sec
            self.dwell_time_ms = 0
        else:
            if section is not None:
                self.dwell_time_ms = int((ts_sec - self.section_enter_time) * 1000)
                
        while self.transition_history and (ts_sec - self.transition_history[0][0]) > self.transition_window_sec:
            self.transition_history.popleft()
            
    def get_transition_rate(self):
        return len(self.transition_history) / self.transition_window_sec

    def get_nrevisit(self, section):
        return self.nrevisit_counts.get(section, 0)

    def finalize(self):
        """Commit the active visit before writing the session summary."""
        if self.current_section is None:
            return
        ss = self.session_sections_summary.setdefault(self.current_section, {
            "total_dwell_ms": 0, "visit_count": 0, "nrevisit_count": 0,
            "max_continuous_dwell_ms": 0
        })
        ss["total_dwell_ms"] += self.dwell_time_ms
        ss["visit_count"] += 1
        ss["nrevisit_count"] = self.nrevisit_counts.get(self.current_section, 0)
        ss["max_continuous_dwell_ms"] = max(ss["max_continuous_dwell_ms"], self.dwell_time_ms)
        self.current_section = None
