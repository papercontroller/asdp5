"""
Intrusion Detector Engine.
Implements supervised attack classification and anomaly scoring for campus IoT networks.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib


class CampusIoTDetector:
    """
    Intrusion detection model optimized for low-latency CPU execution.
    Categorizes flows into:
      0: Normal
      1: MQTT Publish Flood / DoS
      2: Network / Port Scan
      3: Brute Force Authentication
      4: Malformed Packet Attack
    """

    CLASS_LABELS = {
        0: "Normal",
        1: "DoS_Publish_Flood",
        2: "Port_Scan",
        3: "Brute_Force",
        4: "Malformed_Packet"
    }

    def __init__(self, n_estimators: int = 50, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=12,
            min_samples_split=4,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_trained = False

    def generate_synthetic_training_data(self, n_samples_per_class: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates realistic statistical flow samples reflecting MQTT-IoT-IDS2020 distributions.
        Used for unit tests, pipeline CI verification, and baseline initialization.
        """
        rng = np.random.default_rng(self.random_state)
        x_list = []
        y_list = []

        # 0: Normal IoT telemetry (periodic, moderate size, low syn, regular iat)
        normal_feats = rng.normal(
            loc=[50, 4000, 80, 15, 40, 150, 25.0, 2.0, 160.0, 0.5, 0.05, 1, 1, 0, 48, 45, 1, 4, 1.2, 4.2],
            scale=[5, 400, 8, 3, 5, 20, 2.0, 0.3, 20.0, 0.05, 0.01, 0.5, 0.5, 0.1, 5, 4, 0.2, 0.5, 0.2, 0.3],
            size=(n_samples_per_class, 20)
        )
        x_list.append(np.clip(normal_feats, 0, None))
        y_list.append(np.zeros(n_samples_per_class, dtype=int))

        # 1: DoS Publish Flood (massive packet count, high rate, tiny iat, huge publish count)
        dos_feats = rng.normal(
            loc=[500, 25000, 50, 5, 30, 70, 2.0, 250.0, 12500.0, 0.004, 0.001, 2, 0, 0, 498, 490, 1, 0, 10.0, 2.8],
            scale=[50, 2500, 5, 1, 3, 8, 0.5, 25.0, 1250.0, 0.001, 0.0003, 1, 0.1, 0.1, 50, 50, 0.2, 0.1, 2.0, 0.4],
            size=(n_samples_per_class, 20)
        )
        x_list.append(np.clip(dos_feats, 0, None))
        y_list.append(np.ones(n_samples_per_class, dtype=int))

        # 2: Port Scan (many SYN packets, very few bytes, short bursts, zero publish)
        scan_feats = rng.normal(
            loc=[80, 3200, 40, 2, 40, 44, 4.0, 20.0, 800.0, 0.05, 0.02, 75, 5, 10, 5, 0, 0, 0, 0.8, 1.5],
            scale=[10, 400, 3, 0.5, 2, 4, 0.8, 4.0, 160.0, 0.01, 0.005, 8, 2, 3, 2, 0.1, 0.1, 0.1, 0.2, 0.3],
            size=(n_samples_per_class, 20)
        )
        x_list.append(np.clip(scan_feats, 0, None))
        y_list.append(np.full(n_samples_per_class, 2, dtype=int))

        # 3: Brute Force (rapid CONNECT messages, failed auth, repetitive bursts)
        brute_feats = rng.normal(
            loc=[120, 9600, 80, 10, 60, 110, 10.0, 12.0, 960.0, 0.08, 0.02, 40, 35, 15, 60, 5, 45, 0, 1.0, 5.1],
            scale=[15, 1200, 8, 2, 5, 12, 1.5, 2.0, 150.0, 0.01, 0.005, 6, 5, 3, 8, 2, 6, 0.1, 0.2, 0.4],
            size=(n_samples_per_class, 20)
        )
        x_list.append(np.clip(brute_feats, 0, None))
        y_list.append(np.full(n_samples_per_class, 3, dtype=int))

        # 4: Malformed Packet (high payload entropy, irregular sizes, RST flags)
        malform_feats = rng.normal(
            loc=[30, 8500, 283, 120, 20, 800, 15.0, 2.0, 566.0, 0.5, 0.2, 4, 3, 12, 20, 5, 2, 1, 0.5, 7.8],
            scale=[5, 1200, 30, 25, 5, 100, 2.0, 0.4, 100.0, 0.1, 0.05, 1, 1, 3, 4, 2, 1, 0.5, 0.2, 0.2],
            size=(n_samples_per_class, 20)
        )
        x_list.append(np.clip(malform_feats, 0, None))
        y_list.append(np.full(n_samples_per_class, 4, dtype=int))

        x_all = np.vstack(x_list)
        y_all = np.concatenate(y_list)
        return x_all, y_all

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CampusIoTDetector":
        """Trains the internal scaler and Random Forest model."""
        x_scaled = self.scaler.fit_transform(x)
        self.model.fit(x_scaled, y)
        self.is_trained = True
        return self

    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Runs inference on a single 1D feature vector or 2D matrix.
        Returns predicted label, class name, anomaly probability, and confidence.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call fit() before predict().")

        if feature_vector.ndim == 1:
            x_input = feature_vector.reshape(1, -1)
        else:
            x_input = feature_vector

        x_scaled = self.scaler.transform(x_input)
        preds = self.model.predict(x_scaled)
        probs = self.model.predict_proba(x_scaled)

        pred_class = int(preds[0])
        confidence = float(np.max(probs[0]))

        classes = list(self.model.classes_)
        if 0 in classes:
            normal_idx = classes.index(0)
            attack_prob = float(1.0 - probs[0][normal_idx])
        else:
            attack_prob = 1.0

        return {
            "predicted_class": pred_class,
            "attack_type": self.CLASS_LABELS.get(pred_class, "Unknown"),
            "confidence": round(confidence, 4),
            "attack_probability": round(attack_prob, 4),
            "is_anomaly": bool(pred_class != 0 and attack_prob >= 0.5)
        }

    def save(self, filepath: str) -> None:
        """Serializes model and scaler to disk."""
        joblib.dump({"scaler": self.scaler, "model": self.model, "is_trained": self.is_trained}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CampusIoTDetector":
        """Loads serialized model and scaler from disk."""
        data = joblib.load(filepath)
        detector = cls()
        detector.scaler = data["scaler"]
        detector.model = data["model"]
        detector.is_trained = data.get("is_trained", True)
        return detector
