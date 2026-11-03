"""
Unit tests for AlertManager.
"""

import pytest
from src.ids.alert_manager import AlertManager


@pytest.fixture
def manager():
    return AlertManager(suppression_window_seconds=1.0)


def test_normal_traffic_no_alert(manager):
    pred = {
        "predicted_class": 0,
        "attack_type": "Normal",
        "confidence": 0.98,
        "attack_probability": 0.02,
        "is_anomaly": False
    }
    alert = manager.process_prediction(pred)
    assert alert is None
    assert len(manager.history) == 0


def test_attack_alert_creation(manager):
    pred = {
        "predicted_class": 1,
        "attack_type": "DoS_Publish_Flood",
        "confidence": 0.95,
        "attack_probability": 0.95,
        "is_anomaly": True
    }
    alert = manager.process_prediction(pred, device_id="temp-sensor-01", source_ip="192.168.1.100")
    assert alert is not None
    assert alert["severity"] == "CRITICAL"
    assert alert["attack_type"] == "DoS_Publish_Flood"
    assert alert["device_id"] == "temp-sensor-01"
    assert "Throttle" in alert["action_recommended"]
    assert len(manager.history) == 1


def test_alert_suppression_deduplication(manager):
    pred = {
        "predicted_class": 2,
        "attack_type": "Port_Scan",
        "confidence": 0.92,
        "attack_probability": 0.92,
        "is_anomaly": True
    }
    # First alert generated
    a1 = manager.process_prediction(pred, device_id="dev-1", source_ip="10.0.0.5")
    assert a1 is not None

    # Immediate second call with same source and attack -> suppressed
    a2 = manager.process_prediction(pred, device_id="dev-1", source_ip="10.0.0.5")
    assert a2 is None
    assert len(manager.history) == 1

    # Call with different device -> allowed
    a3 = manager.process_prediction(pred, device_id="dev-2", source_ip="10.0.0.6")
    assert a3 is not None
    assert len(manager.history) == 2
