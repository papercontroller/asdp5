"""
Integration tests for the end-to-end Campus IoT IDS pipeline.
Tests raw telemetry -> feature extraction -> model prediction -> alert manager.
"""

import time
import pytest
from src.ids.feature_extractor import FlowFeatureExtractor
from src.ids.detector import CampusIoTDetector
from src.ids.alert_manager import AlertManager


@pytest.fixture(scope="module")
def pipeline():
    detector = CampusIoTDetector(n_estimators=25, random_state=42)
    x, y = detector.generate_synthetic_training_data(n_samples_per_class=50)
    detector.fit(x, y)
    extractor = FlowFeatureExtractor()
    alert_mgr = AlertManager()
    return {"extractor": extractor, "detector": detector, "alert_mgr": alert_mgr}


def test_end_to_end_normal_stream(pipeline):
    extractor = pipeline["extractor"]
    detector = pipeline["detector"]
    alert_mgr = pipeline["alert_mgr"]

    # Generate 50 periodic benign telemetry packets
    now = time.time()
    benign_packets = [
        {
            "timestamp": now + (i * 0.5),
            "size": 80,
            "direction": "forward",
            "flags": ["ACK"],
            "mqtt_type": "PUBLISH",
            "payload": b'{"temp": 21.5, "humidity": 45}'
        }
        for i in range(50)
    ]

    feats = extractor.extract_features(benign_packets)
    assert len(feats) == 20

    pred = detector.predict(feats)
    assert pred["predicted_class"] == 0
    assert pred["attack_type"] == "Normal"
    assert not pred["is_anomaly"]

    alert = alert_mgr.process_prediction(pred, device_id="campus-room-301-sensor")
    assert alert is None


def test_end_to_end_attack_stream(pipeline):
    extractor = pipeline["extractor"]
    detector = pipeline["detector"]
    alert_mgr = pipeline["alert_mgr"]

    # Generate DoS Publish Flood pattern (high rate, zero IAT, hundreds of packets)
    now = time.time()
    flood_packets = [
        {
            "timestamp": now + (i * 0.001),
            "size": 52,
            "direction": "forward",
            "flags": ["ACK"],
            "mqtt_type": "PUBLISH",
            "payload": b'A' * 20
        }
        for i in range(250)
    ]

    feats = extractor.extract_features(flood_packets)
    pred = detector.predict(feats)
    assert pred["is_anomaly"] is True
    assert pred["attack_type"] == "DoS_Publish_Flood"

    alert = alert_mgr.process_prediction(pred, device_id="campus-lab-gateway", source_ip="172.16.5.99")
    assert alert is not None
    assert alert["severity"] == "CRITICAL"
    assert alert["source_ip"] == "172.16.5.99"
