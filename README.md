# CSA Preddiction Engine
[![PyPI version](https://img.shields.io/pypi/v/csa-prediction-engine.svg)](https://pypi.org/project/csa-prediction-engine/)
[![Python Version](https://img.shields.io/badge/python-%20v3.11-blue)](https://github.com/CambridgeSportsAnalytics/prediction_engine)
[![CodeFactor](https://www.codefactor.io/repository/github/cambridgesportsanalytics/csa_prediction_engine/badge)](https://www.codefactor.io/repository/github/cambridgesportsanalytics/csa_prediction_engine)
[![Total Lines of Code](https://tokei.rs/b1/github/CambridgeSportsAnalytics/csa_prediction_engine?category=code)](https://github.com/CambridgeSportsAnalytics/csa_prediction_engine)

**CSA Prediction Engine** is the official Python package for interacting with the Cambridge Sports Analytics (CSA) Prediction Engine API. It enables users to run relevance-based predictions and manage job workflows with ease, whether for single models or concurrent tasks.


## 🚀 Key Features

- **Single Task Predictions**: Support for predictions with one dependent variable and one set of circumstances.
- **Multi-y Predictions**: Perform predictions with multiple dependent variables and a single set of circumstances.
- **Multi-theta Predictions**: Perform predictions with one dependent variable and multiple sets of circumstances.
- **Relevance-Based Grid Predictions**: Generate optimal predictions by evaluating all thresholds and variable combinations.
- **Grid Singularity Predictions**: Analyze grid predictions to find the singular optimal solution.
- **MaxFit Predictions**: Find the best-fit model based on adjusted relevance.


## 🧱 Package Structure

The package is structured as follows:

```bash
csa_prediction_engine/
├── api_client.py                # Main module for user interactions
├── bin/                         # Internal modules for task management
│   ├── _workers.py              # Executes various single prediction models
│   └── single_tasks.py          # Handles single task predictions
├── helpers/                     # Helper modules for internal operations
│   ├── _auth_manager.py         # Manages authentication for API access
│   ├── _details_handler.py      # Manages retrieval and storage of model details
│   ├── _payload_handler.py      # Manages API payload construction and processing
│   ├── _postmaster.py           # Manages internal communications
│   └── _router.py               # Routes tasks based on input configurations
├── parallel/                    # Modules for parallel processing
│   ├── _dispatchers.py          # Dispatches tasks to workers
│   ├── _workers.py              # Executes parallel tasks
│   └── _threaded_predictions.py # Handles predictions in a multi-threaded environment
└── __init__.py                  # Package initialization
```


## 📦 Installation

Install from PyPI:

```bash
pip install csa_prediction_engine
```
Requires Python 3.11.

## 📘 Documentation & Examples

For OpenAPI specs, quickstart examples, and dev tutorials, visit:
👉 [prediction_engine](https://github.com/CambridgeSportsAnalytics/prediction_engine)

## 🤝 Contributing

Bug reports and feature requests are welcome. Reach out to our team 📧 support@csanalytics.io

## ⚖️ License

(c) 2023 - 2025 Cambridge Sports Analytics, LLC. All rights reserved.
