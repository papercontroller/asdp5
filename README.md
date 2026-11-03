# Campus IoT Intrusion Detection System (CI/CD with GitHub Actions)

[![CI/CD Pipeline](https://github.com/sanzharmaulkan/campus-iot-ids/actions/workflows/ci.yml/badge.svg)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-14%20passed-brightgreen.svg)](tests/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

**Course:** Applied Software Development Project (ASDP) — Assignment 5  
**Student:** Maulkan Sanzhar (CSE-2501M, Astana IT University)  
**Topic:** *Intelligent Attack Detection System for Campus IoT Infrastructure*

---

## 1. Overview

This repository contains a lightweight, real-time Intrusion Detection System (IDS) designed for MQTT-based campus IoT networks. The system intercepts MQTT packet streams, extracts 20 flow features, and classifies network behavior using a Random Forest machine learning model on CPU hardware.

The CI/CD pipeline is fully automated using **GitHub Actions** and **Docker**. Every code push automatically triggers code linting, unit testing, Docker image building, and container smoke testing.

## 2. Project Structure

```text
asdp-5/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD workflow
├── src/
│   └── ids/
│       ├── __init__.py            # Package entrypoint
│       ├── feature_extractor.py   # Flow feature extractor (20 features)
│       ├── detector.py            # Supervised ML model & inference engine
│       ├── alert_manager.py       # Alert generation & duplicate suppression
│       └── cli.py                 # Command line runner & traffic simulator
├── tests/
│   ├── conftest.py                # Pytest configuration & path resolution
│   ├── test_feature_extractor.py  # Unit tests for flow feature math
│   ├── test_detector.py           # Unit tests for ML classification & persistence
│   ├── test_alert_manager.py      # Unit tests for alert rules & deduplication
│   └── test_pipeline_integration.py # End-to-end integration tests
├── Dockerfile                     # Multi-stage production container
├── requirements.txt               # Python package dependencies
├── pytest.ini                     # Pytest settings
├── .gitignore                     # Git ignore rules
└── README.md                      # Documentation
```

## 3. Quick Start (Local Run)

### Setup & Run Tests Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run linter
python -m flake8 src/ tests/ --max-line-length=120

# Run all 14 unit and integration tests
python -m pytest -v tests/
```

### Run with Docker
```bash
# Build Docker image
docker build -t campus-iot-ids:latest .

# Run tests inside container
docker run --rm campus-iot-ids:latest pytest -v tests/

# Run CLI simulated intrusion inspection
docker run --rm campus-iot-ids:latest python -m src.ids.cli --test-flow --attack-type DoS_Publish_Flood
```

## 4. GitHub Actions CI/CD Pipeline

The automated workflow is configured in `.github/workflows/ci.yml`. On every push to `main`:
1. **Checkout Source Code** — clones repository via `actions/checkout@v4`.
2. **Setup Python 3.11** — sets up runtime with pip caching.
3. **Install Dependencies** — installs requirements from `requirements.txt`.
4. **Code Linting** — verifies PEP8 formatting via `flake8`.
5. **Pytest Suite** — runs 14 tests with code coverage and exports JUnit XML.
6. **Docker Build** — builds the container image via Docker Buildx.
7. **Smoke Test** — executes verification tests inside the newly built container.
