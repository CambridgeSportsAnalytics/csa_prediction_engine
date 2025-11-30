"""
CSA Relevance Engine: Single Task Prediction Workers

This module provides functionality for executing single task predictions
using the Cambridge Sports Analytics API. A single task is defined as a 
prediction where a single dependent variable (y) is evaluated with a 
single set of circumstances (theta).

Supported Functions:
--------------------
1. `predict_psr`: Performs a standard single task relevance-based prediction.
2. `predict_maxfit`: Executes a single task prediction optimized for maximum fit.
3. `predict_grid`: Calculates a composite prediction based on a grid evaluation.
4. `predict_grid_singularity`: Identifies the singularity of grid evaluations.

Binary versions of all functions are also available for categorical outcomes.

Usage:
------
These functions send prediction jobs to the server, either waiting for results 
synchronously (default) or returning a job ID and code for later polling.

(c) 2023 - 2025 Cambridge Sports Analytics, LLC. All rights reserved.
support@csanalytics.io
"""

# Local imports
from ..helpers import _postmaster

# Local application/library-specific imports
from csa_common_lib.classes.prediction_options import (
    PredictionOptions,
    MaxFitOptions,
    GridOptions
)


def _execute_prediction(
    y, X, theta, Options, 
    post_function, 
    poll_results: bool = False,
    is_binary: bool = False
):
    """
    Internal helper function to execute a prediction job.
    
    Handles the common logic of posting a job and optionally polling for results.
    
    Parameters
    ----------
    y : ndarray [N-by-1]
        Column vector of the dependent variable.
    X : ndarray [N-by-K]
        Matrix of independent variables.
    theta : [1-by-K]
        Row vector of circumstances.
    Options : Union[PredictionOptions, MaxFitOptions, GridOptions]
        Options object containing parameters for the prediction.
    post_function : Callable
        Postmaster function to call for posting the job.
    poll_results : bool, optional
        If True, wait for server to return results. If False, return job id and code.
    is_binary : bool, optional
        Whether to use the binary version of the function, by default False.
        
    Returns
    -------
    Union[Tuple[ndarray, dict], Tuple[int, str]]
        Either (yhat, yhat_details) if poll_results is True,
        or (job_id, job_code) if poll_results is False.
    """
    # Post job to server
    job_id, job_code = post_function(y=y, X=X, theta=theta, Options=Options, is_binary=is_binary)
    
    # Get results from server if requested
    if poll_results:
        yhat, yhat_details = _postmaster._get_results_worker(job_id, job_code)
        return yhat, yhat_details
    else:
        return job_id, job_code


def predict_psr(y, X, theta, Options: PredictionOptions, poll_results: bool = False, is_binary: bool = False):
    """ 
    Runs and evaluates a prediction using the relevance-based model. 
    
    Parameters
    ----------
    y : ndarray [N-by-1]
        Column vector of the dependent variable.
    X : ndarray [N-by-K]
        Matrix of independent variables.
    theta : [1-by-K]
        Row vector of circumstances.
    Options : PredictionOptions
        Options object that contains the necessary key-value parameters
        for predictions.
    poll_results : bool, optional
        If True, wait for server to return results, computational time
        may vary. If False, the server will return job id and job code.
    is_binary : bool, optional
        Whether to use the binary version of the function, by default False.

    Returns
    -------
    Union[Tuple[ndarray, dict], Tuple[int, str]]
        Either (yhat, yhat_details) if poll_results is True,
        or (job_id, job_code) if poll_results is False.
    """
    return _execute_prediction(
        y=y, X=X, theta=theta, Options=Options,
        post_function=_postmaster._post_predict_inputs,
        poll_results=poll_results,
        is_binary=is_binary
    )
    
    
def predict_maxfit(y, X, theta, Options: MaxFitOptions, poll_results: bool = False, is_binary: bool = False):
    """ 
    Runs and evaluates a prediction using the relevance-based model 
    and solves for maximum (adjusted) fit.

    Parameters
    ----------
    y : ndarray [N-by-1]
        Column vector of the dependent variable.
    X : ndarray [N-by-K]
        Matrix of independent variables.
    theta : [1-by-K]
        Row vector of circumstances.
    Options : MaxFitOptions
        MaxFitOptions object that contains the necessary key-value parameters
        for maxfit predictions.
    poll_results : bool, optional
        If True, wait for server to return results, computational time
        may vary. If False, the server will return job id and job code.
    is_binary : bool, optional
        Whether to use the binary version of the function, by default False.

    Returns
    -------
    Union[Tuple[ndarray, dict], Tuple[int, str]]
        Either (yhat, yhat_details) if poll_results is True,
        or (job_id, job_code) if poll_results is False.
    """
    return _execute_prediction(
        y=y, X=X, theta=theta, Options=Options,
        post_function=_postmaster._post_maxfit_inputs,
        poll_results=poll_results,
        is_binary=is_binary
    )
    
    
def predict_grid(y, X, theta, Options: GridOptions, poll_results: bool = False, is_binary: bool = False):
    """ 
    Runs and evaluates a grid prediction using the relevance-based
    model and weights each grid cell solution by its adjusted-fit to 
    solve for a composite prediction outcome. 
    
    Parameters
    ----------
    y : ndarray [N-by-1]
        Column vector of the dependent variable.
    X : ndarray [N-by-K]
        Matrix of independent variables.
    theta : [1-by-K]
        Row vector of circumstances.
    Options : GridOptions
        GridOptions object that contains the necessary key-value parameters
        for grid predictions.
    poll_results : bool, optional
        If True, wait for server to return results, computational time
        may vary. If False, the server will return job id and job code.
    is_binary : bool, optional
        Whether to use the binary version of the function, by default False.

    Returns
    -------
    Union[Tuple[ndarray, dict], Tuple[int, str]]
        Either (yhat, yhat_details) if poll_results is True,
        or (job_id, job_code) if poll_results is False.
    """
    return _execute_prediction(
        y=y, X=X, theta=theta, Options=Options,
        post_function=_postmaster._post_grid_inputs,
        poll_results=poll_results,
        is_binary=is_binary
    )
    
    
def predict_grid_singularity(y, X, theta, Options: GridOptions, poll_results: bool = False, is_binary: bool = False):
    """ 
    Runs and evaluates a grid singularity prediction using the 
    relevance-based model and solves for maximum adjusted fit with 
    optimal variable selection. 
    
    Parameters
    ----------
    y : ndarray [N-by-1]
        Column vector of the dependent variable.
    X : ndarray [N-by-K]
        Matrix of independent variables.
    theta : [1-by-K]
        Row vector of circumstances.
    Options : GridOptions
        GridOptions object that contains the necessary key-value parameters
        for grid predictions.
    poll_results : bool, optional
        If True, wait for server to return results, computational time
        may vary. If False, the server will return job id and job code.
    is_binary : bool, optional
        Whether to use the binary version of the function, by default False.

    Returns
    -------
    Union[Tuple[ndarray, dict], Tuple[int, str]]
        Either (yhat, yhat_details) if poll_results is True,
        or (job_id, job_code) if poll_results is False.
    """
    return _execute_prediction(
        y=y, X=X, theta=theta, Options=Options,
        post_function=_postmaster._post_grid_singularity_inputs,
        poll_results=poll_results,
        is_binary=is_binary
    )


# Binary function wrappers for categorical outcomes
def predict_psr_binary(y, X, theta, Options: PredictionOptions, poll_results: bool = False):
    """Binary version of predict_psr for categorical outcomes."""
    return predict_psr(y=y, X=X, theta=theta, Options=Options, poll_results=poll_results, is_binary=True)


def predict_maxfit_binary(y, X, theta, Options: MaxFitOptions, poll_results: bool = False):
    """Binary version of predict_maxfit for categorical outcomes."""
    return predict_maxfit(y=y, X=X, theta=theta, Options=Options, poll_results=poll_results, is_binary=True)


def predict_grid_binary(y, X, theta, Options: GridOptions, poll_results: bool = False):
    """Binary version of predict_grid for categorical outcomes."""
    return predict_grid(y=y, X=X, theta=theta, Options=Options, poll_results=poll_results, is_binary=True)


def predict_grid_singularity_binary(y, X, theta, Options: GridOptions, poll_results: bool = False):
    """Binary version of predict_grid_singularity for categorical outcomes."""
    return predict_grid_singularity(y=y, X=X, theta=theta, Options=Options, poll_results=poll_results, is_binary=True)