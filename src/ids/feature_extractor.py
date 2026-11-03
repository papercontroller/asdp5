"""
Feature Extractor Module for MQTT Network Traffic.
Extracts statistical and temporal features from bidirectional network packet flows.
"""

from typing import List, Dict, Any
import numpy as np


class FlowFeatureExtractor:
    """
    Extracts numerical features from a sequence of raw network packet records.
    Features follow the MQTT-IoT-IDS2020 and CICIoT2023 flow standards.
    """

    FEATURE_NAMES = [
        "packet_count",
        "total_bytes",
        "mean_packet_size",
        "std_packet_size",
        "min_packet_size",
        "max_packet_size",
        "duration_seconds",
        "packet_rate",
        "byte_rate",
        "mean_inter_arrival_time",
        "std_inter_arrival_time",
        "tcp_syn_count",
        "tcp_fin_count",
        "tcp_rst_count",
        "tcp_ack_count",
        "mqtt_publish_count",
        "mqtt_connect_count",
        "mqtt_ping_count",
        "forward_backward_ratio",
        "payload_entropy"
    ]

    def __init__(self, window_size: int = 50):
        self.window_size = window_size

    def calculate_entropy(self, data: bytes) -> float:
        """Calculates Shannon entropy of payload bytes."""
        if not data:
            return 0.0
        counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probs = counts[counts > 0] / len(data)
        return float(-np.sum(probs * np.log2(probs)))

    def extract_features(self, packets: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extracts a feature vector of length 20 from a list of packet dicts.
        Each packet dict has:
          - 'timestamp' (float)
          - 'size' (int)
          - 'direction' (str: 'forward' or 'backward')
          - 'flags' (list of str: ['SYN', 'ACK', ...])
          - 'mqtt_type' (str: 'PUBLISH', 'CONNECT', 'PINGREQ', or None)
          - 'payload' (bytes, optional)
        """
        if not packets:
            return np.zeros(len(self.FEATURE_NAMES), dtype=np.float32)

        packet_count = len(packets)
        sizes = np.array([p.get("size", 0) for p in packets], dtype=np.float32)
        total_bytes = float(np.sum(sizes))
        mean_size = float(np.mean(sizes))
        std_size = float(np.std(sizes)) if packet_count > 1 else 0.0
        min_size = float(np.min(sizes))
        max_size = float(np.max(sizes))

        timestamps = np.array([p.get("timestamp", 0.0) for p in packets], dtype=np.float64)
        timestamps = np.sort(timestamps)
        duration = float(timestamps[-1] - timestamps[0]) if packet_count > 1 else 0.001
        duration = max(duration, 0.0001)

        packet_rate = float(packet_count / duration)
        byte_rate = float(total_bytes / duration)

        if packet_count > 1:
            iats = np.diff(timestamps)
            mean_iat = float(np.mean(iats))
            std_iat = float(np.std(iats))
        else:
            mean_iat = 0.0
            std_iat = 0.0

        # TCP Flag counts
        syn_count = sum(1 for p in packets if "SYN" in p.get("flags", []))
        fin_count = sum(1 for p in packets if "FIN" in p.get("flags", []))
        rst_count = sum(1 for p in packets if "RST" in p.get("flags", []))
        ack_count = sum(1 for p in packets if "ACK" in p.get("flags", []))

        # MQTT Message type counts
        pub_count = sum(1 for p in packets if p.get("mqtt_type") == "PUBLISH")
        conn_count = sum(1 for p in packets if p.get("mqtt_type") == "CONNECT")
        ping_count = sum(1 for p in packets if p.get("mqtt_type") == "PINGREQ")

        # Directional ratio
        forward_pkts = sum(1 for p in packets if p.get("direction") == "forward")
        backward_pkts = sum(1 for p in packets if p.get("direction") == "backward")
        fwd_bwd_ratio = float(forward_pkts / max(backward_pkts, 1))

        # Entropy of concatenated payloads
        raw_payloads = [p.get("payload", b"") for p in packets if isinstance(p.get("payload"), (bytes, bytearray))]
        all_payloads = b"".join(raw_payloads)
        entropy = self.calculate_entropy(all_payloads)

        vector = [
            float(packet_count),
            total_bytes,
            mean_size,
            std_size,
            min_size,
            max_size,
            duration,
            packet_rate,
            byte_rate,
            mean_iat,
            std_iat,
            float(syn_count),
            float(fin_count),
            float(rst_count),
            float(ack_count),
            float(pub_count),
            float(conn_count),
            float(ping_count),
            fwd_bwd_ratio,
            entropy
        ]
        return np.array(vector, dtype=np.float32)
