"""
Command Line Interface for Campus IoT IDS.
Used for model training, benchmarking, and real-time packet stream inspection.
"""

import argparse
import os
import time

from src.ids.detector import CampusIoTDetector
from src.ids.feature_extractor import FlowFeatureExtractor
from src.ids.alert_manager import AlertManager


def main():
    parser = argparse.ArgumentParser(description="Campus IoT IDS Engine CLI")
    parser.add_argument("--train", action="store_true", help="Train baseline model and save to disk")
    parser.add_argument("--output", type=str, default="campus_ids_model.joblib", help="Output model file")
    parser.add_argument("--test-flow", action="store_true", help="Simulate a test packet flow and predict")
    parser.add_argument("--attack-type", type=str, default="DoS_Publish_Flood", help="Simulated attack type")

    args = parser.parse_args()

    detector = CampusIoTDetector()

    if args.train or not os.path.exists(args.output):
        print("[*] Generating synthetic training data for 5 traffic classes...")
        x, y = detector.generate_synthetic_training_data(n_samples_per_class=300)
        print(f"[*] Training Random Forest model on {len(x)} flow samples...")
        detector.fit(x, y)
        detector.save(args.output)
        print(f"[+] Model saved successfully to {args.output}")

    if args.test_flow:
        detector = CampusIoTDetector.load(args.output)
        extractor = FlowFeatureExtractor()
        manager = AlertManager()

        print(f"[*] Simulating traffic flow ({args.attack_type})...")
        packets = []
        now = time.time()

        if args.attack_type == "DoS_Publish_Flood":
            for i in range(100):
                packets.append({
                    "timestamp": now + (i * 0.002),
                    "size": 54,
                    "direction": "forward",
                    "flags": ["ACK"],
                    "mqtt_type": "PUBLISH",
                    "payload": b"flood payload data 123"
                })
        else:
            for i in range(20):
                packets.append({
                    "timestamp": now + (i * 0.5),
                    "size": 80,
                    "direction": "forward",
                    "flags": ["ACK"],
                    "mqtt_type": "PUBLISH",
                    "payload": b"normal temperature reading"
                })

        feats = extractor.extract_features(packets)
        pred = detector.predict(feats)
        print(f"[+] Prediction result: {pred}")
        alert = manager.process_prediction(pred, device_id="campus-sensor-04", source_ip="192.168.10.45")
        if alert:
            print(f"[!] Security Alert Triggered: {alert['severity']} - {alert['attack_type']}")
            print(f"    Action: {alert['action_recommended']}")
        else:
            print("[-] Traffic classified as Normal or suppressed.")


if __name__ == "__main__":
    main()
