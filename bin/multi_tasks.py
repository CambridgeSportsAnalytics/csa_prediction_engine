"""
CSA Relevance Engine: Multi-Task Prediction Module

This module provides functionality for executing batch prediction tasks
using the Cambridge Sports Analytics API. Multi-task predictions involve 
either multiple dependent variables (multi-y) or multiple sets of 
circumstances (multi-theta).

Supported Prediction Types:
---------------------------
1. **Multi-y Prediction**: 
   - Executes predictions for multiple dependent variables simultaneously with a single set of circumstances.
2. **Multi-theta Prediction**: 
   - Executes predictions for a single dependent variable with multiple sets of circumstances.

Supported Functions:
--------------------
1. `predict`: Standard multi-task relevance-based prediction.
2. `predict_maxfit`: Multi-task prediction optimized for maximum fit.
3. `predict_grid`: Composite prediction from a grid evaluation.
4. `predict_grid_singularity`: Identifies the best singular solution among grid evaluations.

Usage:
------
These functions send prediction jobs to the server, either waiting 
for results synchronously (default) or returning a job ID and code 
for later polling.

(c) 2023 - 2024 Cambridge Sports Analytics, LLC. All rights reserved.
support@csanalytics.io
"""

# Local imports
from ..helpers import _postmaster  # Postmaster utility functions for managing task notifications

# CSA Common Library imports
from csa_common_lib.enum_types.functions import PSRFunction

from csa_common_lib.classes.prediction_options import (
    PredictionOptions,  # Base options for predictions
    MaxFitOptions,      # Options specific to MaxFit predictions
    GridOptions         # Options specific to Grid predictions
)

# Parallel prediction imports
from ..parallel.threaded_predictions import (
    run_multi_y,        # Execute batch prediction tasks for multiple y-variables
    run_multi_theta     # Execute batch prediction tasks for multiple circumstances (theta)
)


def predict(y_matrix, X, theta, Options:PredictionOptions, poll_results:bool=False):
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
        for grid predictions.
    poll_results : boolean, optional (default=True)
        If true, wait for server to return results, computational time
        may vary. If false, the server will return job id and job code.

    Returns
    -------
    yhat : ndarray
        Prediction outcome(s).
    yhat_details : dict
        Model details accesible via key-value pairs.
    """
    
    yhat, yhat_details = run_multi_y(PSRFunction.PREDICT, y_matrix, X, theta, Options)

    # Return results object
    return yhat, yhat_details

    
    
def predict_maxfit(y, X, theta, Options:MaxFitOptions, poll_results:bool=False):
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
        for grid predictions.
    poll_results : boolean, optional (default=True)
        If true, wait for server to return results, computational time
        may vary. If false, the server will return job id and job code.

    Returns
    -------
    yhat : ndarray
        Prediction outcome(s).
    yhat_details : dict
        Model details accesible via key-value pairs.
    """
    
    # Send maxfit partial sample regression job to postmaster
    job_id, job_code = _postmaster._post_maxfit_inputs(
        y=y, X=X, theta=theta, Options=Options)
    
    # Get results from server
    if poll_results:
        yhat, yhat_details = _postmaster._get_results_worker(job_id, job_code)
    
        # Return results object
        return yhat, yhat_details
    
    else:
        return job_id, job_code
    
    
def predict_grid(y, X, theta, Options:GridOptions, poll_results=True):
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
    poll_results : boolean, optional (default=True)
        If true, wait for server to return results, computational time
        may vary. If false, the server will return job id and job code.

    Returns
    -------
    Returns yhat and yhat_details by default, if poll_results is False,
    then this function returns job id and job code.
    
    yhat : ndarray
        Prediction outcome(s).
    yhat_details : dict
        Model details accesible via key-value pairs.
    """
    
    # Send grid prediction job to postmaster
    job_id, job_code = _postmaster._post_grid_inputs(
        y, X, theta, Options=Options)
    
    # Get results from server
    if poll_results:
        yhat, yhat_details = _postmaster._get_results(job_id, job_code)
    
        # Return results object
        return yhat, yhat_details
    
    else:
        return job_id, job_code
    
    
def predict_grid_singularity(y, X, theta, Options:GridOptions, poll_results=True):
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
    poll_results : boolean, optional (default=True)
        If true, wait for server to return results, computational time
        may vary. If false, the server will return job id and job code.

    Returns
    -------
    Returns yhat and yhat_details by default, if poll_results is False,
    then this function returns job id and job code.
    
    yhat : ndarray
        Prediction outcome(s).
    yhat_details : dict
        Model details accesible via key-value pairs.
    """
    
    # Send grid prediction job to postmaster
    job_id, job_code = _postmaster._post_grid_inputs(
        y, X, theta, Options=Options)
    
    # Get results from server
    if poll_results:
        yhat, yhat_details = _postmaster._get_results(job_id, job_code)
    
        # Return results object
        return yhat, yhat_details
    
    else:
        return job_id, job_code