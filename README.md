# Cambridge Sports Analytics Relevance Engine

Welcome to the **CSA Prediction Engine** package. This Python library provides a suite of tools and functions for performing relevance-based predictions using the Cambridge Sports Analytics (CSA) API. The package is designed to facilitate single and multi-task predictions, allowing for flexible model evaluation and experimentation.

## Package Structure

The package is structured as follows:

```
csa_prediction_engine/
    ├── api_client.py                # Main module for user interactions
    ├── bin/                         # Internal modules for task management
    │   ├── _workers_.py             # Executes various single prediction models
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

## Key Features

- **Single Task Predictions**: Support for predictions with one dependent variable and one set of circumstances.
- **Multi-y Predictions**: Perform predictions with multiple dependent variables and a single set of circumstances.
- **Multi-theta Predictions**: Perform predictions with one dependent variable and multiple sets of circumstances.
- **Relevance-Based Grid Predictions**: Generate optimal predictions by evaluating all thresholds and variable combinations.
- **MaxFit Predictions**: Find the best-fit model based on adjusted relevance.
- **Grid Singularity Predictions**: Analyze grid predictions to find the singular optimal solution.

## Installation

To install the CSAnalytics Prediction Engine package, use the following command:

```bash
pip install csa_prediction_engine
```

For more advanced usage and examples, refer to the documentation (link to detailed docs if available).

## Contributing

We welcome contributions to the CSA Relevance Engine package. If you find a bug or have a feature request, please reach out to the CSA support team: support@csanalytics.io

## License

(c) 2023 - 2024 Cambridge Sports Analytics, LLC. All rights reserved.