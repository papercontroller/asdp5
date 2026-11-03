"""
Unit tests for CampusIoTDetector.
"""

import os
import pytest
import numpy as np
from src.ids.detector import CampusIoTDetector


@pytest.fixture
def trained_detector():
    det = CampusIoTDetector(n_estimators=20, random_state=42)
    x, y = det.generate_synthetic_training_data(n_samples_per_class=40)
    det.fit(x, y)
    return det


def test_initialization():
    det = CampusIoTDetector()
    assert not det.is_trained
    assert det.n_estimators == 50


def test_predict_unfitted_raises_runtime_error():
    det = CampusIoTDetector()
    with pytest.raises(RuntimeError):
        det.predict(np.zeros(20))


def test_training_and_prediction(trained_detector):
    assert trained_detector.is_trained

    # Predict normal sample
    normal_vector = np.array(
        [50, 4000, 80, 15, 40, 150, 25.0, 2.0, 160.0, 0.5, 0.05, 1, 1, 0, 48, 45, 1, 4, 1.2, 4.2],
        dtype=np.float32
    )
    res = trained_detector.predict(normal_vector)
    assert "predicted_class" in res
    assert "attack_type" in res
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["predicted_class"] == 0
    assert res["attack_type"] == "Normal"
    assert not res["is_anomaly"]


def test_dos_detection(trained_detector):
    # Predict DoS flood vector
    dos_vector = np.array(
        [500, 25000, 50, 5, 30, 70, 2.0, 250.0, 12500.0, 0.004, 0.001, 2, 0, 0, 498, 490, 1, 0, 10.0, 2.8],
        dtype=np.float32
    )
    res = trained_detector.predict(dos_vector)
    assert res["predicted_class"] == 1
    assert res["attack_type"] == "DoS_Publish_Flood"
    assert res["is_anomaly"]
    assert res["attack_probability"] > 0.7


def test_save_and_load(trained_detector, tmp_path):
    model_file = str(tmp_path / "test_model.joblib")
    trained_detector.save(model_file)
    assert os.path.exists(model_file)

    loaded = CampusIoTDetector.load(model_file)
    assert loaded.is_trained

    sample = np.zeros(20)
    res_orig = trained_detector.predict(sample)
    res_load = loaded.predict(sample)
    assert res_orig["predicted_class"] == res_load["predicted_class"]
