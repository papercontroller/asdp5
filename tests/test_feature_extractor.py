"""
Unit tests for FlowFeatureExtractor.
"""

import pytest
import numpy as np
from src.ids.feature_extractor import FlowFeatureExtractor


@pytest.fixture
def extractor():
    return FlowFeatureExtractor()


def test_empty_packets(extractor):
    feats = extractor.extract_features([])
    assert isinstance(feats, np.ndarray)
    assert len(feats) == len(FlowFeatureExtractor.FEATURE_NAMES)
    assert np.all(feats == 0.0)


def test_single_packet_flow(extractor):
    packets = [
        {
            "timestamp": 100.0,
            "size": 64,
            "direction": "forward",
            "flags": ["SYN"],
            "mqtt_type": "CONNECT",
            "payload": b"mqtt_init"
        }
    ]
    feats = extractor.extract_features(packets)
    assert feats[0] == 1.0  # packet count
    assert feats[1] == 64.0  # total bytes
    assert feats[2] == 64.0  # mean size
    assert feats[3] == 0.0  # std size (single packet)
    assert feats[11] == 1.0  # syn count
    assert feats[16] == 1.0  # connect count


def test_multi_packet_flow_statistics(extractor):
    packets = [
        {
            "timestamp": 1.0, "size": 100, "direction": "forward",
            "flags": ["ACK"], "mqtt_type": "PUBLISH", "payload": b"a" * 10
        },
        {
            "timestamp": 2.0, "size": 200, "direction": "forward",
            "flags": ["ACK"], "mqtt_type": "PUBLISH", "payload": b"b" * 10
        },
        {
            "timestamp": 3.0, "size": 300, "direction": "backward",
            "flags": ["ACK"], "mqtt_type": "PUBLISH", "payload": b"c" * 10
        },
    ]
    feats = extractor.extract_features(packets)
    assert feats[0] == 3.0  # packet count
    assert feats[1] == 600.0  # total bytes
    assert feats[2] == 200.0  # mean size
    assert feats[4] == 100.0  # min
    assert feats[5] == 300.0  # max
    assert feats[6] == 2.0  # duration (3.0 - 1.0)
    assert feats[9] == 1.0  # mean IAT
    assert feats[15] == 3.0  # mqtt publish count
    assert feats[18] == 2.0  # fwd/bwd ratio (2 forward / 1 backward)


def test_payload_entropy(extractor):
    # Homogeneous data -> zero entropy
    zero_entropy = extractor.calculate_entropy(b"AAAAAAA")
    assert zero_entropy == 0.0

    # Random byte string -> higher entropy
    random_bytes = bytes([i % 256 for i in range(256)])
    high_entropy = extractor.calculate_entropy(random_bytes)
    assert high_entropy > 7.0
