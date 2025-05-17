"""
qa_flag_engine.py

Rule-based QA flag and Affect Index computation for contact-center emotion analytics.
Loads config from YAML and processes a stream of per-turn emotion outputs.

Usage:
    from qa_flag_engine import QAFlagEngine
    engine = QAFlagEngine("path/to/qa_flags_config.yaml")
    for turn in call_turns:
        result = engine.process_turn(turn)
        # result: dict with affect index and alerts
"""

import yaml
from collections import deque

class QAFlagEngine:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.agent_emotions = self.config["desired_agent_emotions"]
        self.negative_customer_emotions = self.config["negative_customer_emotions"]
        self.thresh = self.config["thresholds"]
        self.duration_sec = self.config.get("duration_sec", 3)
        self.escalation_window_sec = self.config.get("escalation_window_sec", 10)
        # Rolling windows for burst/escalation
        self.customer_angry_window = deque()
        self.customer_time_window = deque()
        self.agent_calm_window = deque()
        self.agent_time_window = deque()

    def process_turn(self, turn):
        """
        turn: dict with keys:
            - timestamp (float or str)
            - speaker ("agent" or "customer")
            - faceEmotion: dict of emotion:prob
            - voiceTone: dict of emotion:prob
            - fusionScore: float (optional)
        Returns:
            dict with affect index and alerts
        """
        alerts = []
        ts = self._parse_time(turn["timestamp"])
        speaker = turn["speaker"]
        # Affect index: sum fusion probs for desired emotions
        affect_index = 0.0
        if speaker == "agent":
            affect_index = sum(
                turn["faceEmotion"].get(e, 0.0) for e in self.agent_emotions
            )
            # Low engagement: neutral + sad > threshold
            neutral = turn["faceEmotion"].get("neutral", 0.0)
            sad = turn["faceEmotion"].get("sad", 0.0)
            if neutral + sad > self.thresh["low_engagement"]:
                alerts.append("LOW_AGENT_ENGAGEMENT")
            # Track for escalation
            self.agent_calm_window.append((ts, turn["faceEmotion"].get("calm", 0.0)))
            self.agent_time_window.append(ts)
            self._trim_window(self.agent_calm_window, self.agent_time_window, self.escalation_window_sec)
        elif speaker == "customer":
            affect_index = sum(
                turn["faceEmotion"].get(e, 0.0) for e in self.negative_customer_emotions
            )
            # Negative burst: angry > threshold
            angry = turn["faceEmotion"].get("anger", 0.0)
            self.customer_angry_window.append((ts, angry))
            self.customer_time_window.append(ts)
            self._trim_window(self.customer_angry_window, self.customer_time_window, self.duration_sec)
            if self._rolling_avg(self.customer_angry_window) > self.thresh["negative_burst"]:
                alerts.append("NEGATIVE_BURST")
            # Track for escalation
            self._trim_window(self.customer_angry_window, self.customer_time_window, self.escalation_window_sec)
        # Escalation risk: rolling avg customer_angry - agent_calm > delta
        if (
            len(self.customer_angry_window) > 0
            and len(self.agent_calm_window) > 0
            and abs(self.customer_time_window[-1] - self.agent_time_window[-1]) < self.escalation_window_sec
        ):
            escalation = (
                self._rolling_avg(self.customer_angry_window)
                - self._rolling_avg(self.agent_calm_window)
            )
            if escalation > self.thresh["escalation_delta"]:
                alerts.append("ESCALATION_RISK")
        return {
            "affectIndex": affect_index,
            "alerts": alerts,
        }

    def _parse_time(self, ts):
        # Accepts "HH:MM:SS.xx" or float seconds
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str):
            parts = ts.split(":")
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            else:
                return float(ts)
        return 0.0

    def _rolling_avg(self, window):
        if not window:
            return 0.0
        return sum(val for _, val in window) / len(window)

    def _trim_window(self, value_window, time_window, max_sec):
        # Remove entries older than max_sec from the latest timestamp
        if not time_window:
            return
        latest = time_window[-1]
        while time_window and latest - time_window[0] > max_sec:
            time_window.popleft()
            value_window.popleft()
