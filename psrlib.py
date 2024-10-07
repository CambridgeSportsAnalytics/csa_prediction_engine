"""
CSAnalytics PSR Library

This is a Python library that interacts with Cambridge Sports Analytics'
API online. Functions defined correspond to API end-points and helps
manage input aggregation, upload, and retrieval of results.

(c) 2023 - 2024 Cambridge Sports Analytics, LLC. All rights reserved.
support@csanalytics.io

"""

from . import _postmaster as csa_postmaster

from .helpers import _payload_handler as _messenger
from .helpers._payload_handler import _get_ckt_max_limit as ckt_max_variables


from csa_common_lib.toolbox.enum_types.enum_functions import PSRFunction



# function to post job
def predict(y, X, theta, threshold=None, is_threshold_percent:bool=True, 
            most_eval:bool=True, eval_type:str="relevance", 
            cov_inv=None, poll_results:bool=False):
    """ Runs and evaluates a prediction using the partial sample regression 
    model. Returns yhat and model details. If threshold=0 for 
    is_threshold_percent=True, weights converge to full sample regression.
    

    Parameters
    ----------
    y : ndarray [N-by-1]
        Column vector of the dependent variable.
    X : ndarray [N-by-K]
        Matrix of independent variables.
    theta : [1-by-K]
        Row vector of circumstances.
    threshold : float or ndarray[1-by-T], optional (default=None)
        Threshold to evaluate relevant observations.
        If threshold=None, the model will evaluate across thresholds
        from [0, 0.90) in 0.10 increments.
    is_threshold_percent : bool, optional (default=True)
        Specify whether threshold is in percentage (decimal) units.
    most_eval : bool, optional (default=True)
        Specify the direction of threshold evluation.
        True:  relevance score > threshold
        False: relevance score < threshold
    eval_type : str, optional (default="relevance")
        Specify the censor signal type, relevance or similarity.
    cov_inv : ndarray [K-by-K], optional (default=None)
        Inverse covariance matrix, specify for speed.
    poll_results : boolean, optional (default=True)
        If true, wait for server to return results, computational time
        may vary. If false, the server will return job id and job code.

    Returns
    -------
    yhat : ndarray [1-by-T]
        Prediction outcome.
    yhat_details : dict
        Model details accesible via key-value pairs.
    """
    
    # Send linear regression job to postmaster
    job_id, job_code = csa_postmaster._post_predict_inputs(
        y=y, X=X, theta=theta, threshold=threshold,
        is_threshold_percent=is_threshold_percent,
        most_eval=most_eval, eval_type=eval_type,
        cov_inv=cov_inv
    )
    
    # Get results from server
    if poll_results:
        yhat, yhat_details = csa_postmaster._get_results(job_id, job_code)
    
        # Return results object
        return yhat, yhat_details
    
    else:
        return job_id, job_code
    
    
def predict_maxfit(y, X, theta, threshold_range=(0,1), stepsize=0.20, 
                   most_eval:bool=True, eval_type:str="relevance", 
                   cov_inv=None, poll_results:bool=False):
    """ Runs and evaluates a prediction using the partial sample regression 
    model and solves for maximum adjusted fit. 
    Returns yhat and model details.
    

    Parameters
    ----------
    y : ndarray [N-by-1]
        Column vector of the dependent variable.
    X : ndarray [N-by-K]
        Matrix of independent variables.
    theta : [1-by-K]
        Row vector of circumstances.
    threshold_range : tuple or ndarray
        Min/max range for evaluating maxfit threshold, by default (0,1)
    stepsize : float, optional (default=0.20)
        Stepsize to evaluate range of thresholds to solve for max fit.
        Decreasing stepsize will increase the granularity of the search.
    most_eval : bool, optional (default=True)
        Specify the direction of threshold evluation.
        True:  relevance score > threshold
        False: relevance score < threshold
    eval_type : str, optional (default="relevance")
        Specify the censor signal type, relevance or similarity.
    cov_inv : ndarray [K-by-K], optional (default=None)
        Inverse covariance matrix, specify for speed.
    poll_results : boolean, optional (default=True)
        If true, wait for server to return results, computational time
        may vary. If false, the server will return job id and job code.

    Returns
    -------
    yhat : ndarray [1-by-T]
        Prediction outcome.
    yhat_details : dict
        Model details accesible via key-value pairs.
    """
    
    # Send maxfit partial sample regression job to postmaster
    job_id, job_code = csa_postmaster._post_maxfit_inputs(
        y=y, X=X, theta=theta, most_eval=most_eval, 
        eval_type=eval_type, cov_inv=cov_inv)
    
    # Get results from server
    if poll_results:
        yhat, yhat_details = csa_postmaster._get_results(job_id, job_code)
    
        # Return results object
        return yhat, yhat_details
    
    else:
        return job_id, job_code
    
    
def predict_grid(y, X, theta, threshold_range=(0,1), stepsize=0.20, 
                 most_eval=True, eval_type="all", k:int=None,
                 return_grid=False, poll_results=True):
    """ Runs and evaluates a prediction using the partial sample regression 
    model and solves for maximum adjusted fit with optimal variable selection. 
    This is also known as the CKT optimal partial sample regression.
    Returns yhat and model details.
    

    Parameters
    ----------
    y : ndarray [N-by-1]
        Column vector of the dependent variable.
    X : ndarray [N-by-K]
        Matrix of independent variables.
    theta : [1-by-K]
        Row vector of circumstances.
    threshold_range : tuple or ndarray
        Min/max range for evaluating maxfit threshold, by default (0,1)
    stepsize : float, optional (default=0.20)
        Stepsize to evaluate range of thresholds to solve for max fit.
        Decreasing stepsize will increase the granularity of the search.
    most_eval : bool, optional (default=True)
        Specify the direction of threshold evluation.
        True:  relevance score > threshold
        False: relevance score < threshold
    eval_type : str, optional (default="all")
        Specify the censor signal type, relevance, similarity, or all.
    k : int, optional (default=None)
        Lower bound for the number of variables to include, by default None.
        None would be equivalent to an unconstraint optimization.
    return_grid : boolean, optional (default=False)
        Returns grid of adjusted_fits, yhats, and weights via output_details
        for all the combinations evaluated.
    poll_results : boolean, optional (default=True)
        If true, wait for server to return results, computational time
        may vary. If false, the server will return job id and job code.

    Returns
    -------
    Returns yhat and yhat_details by default, if poll_results is False,
    then this function returns job id and job code.
    
    yhat : ndarray [1-by-T]
        Prediction outcome.
    yhat_details : dict
        Model details accesible via key-value pairs.
    """
    
    # Send CKT optimal partial sample regression job to postmaster
    job_id, job_code = csa_postmaster._post_ckt_inputs(
        y, X, theta, threshold_range, stepsize, most_eval, 
        eval_type, k, return_grid
    )
    
    # Get results from server
    if poll_results:
        yhat, yhat_details = csa_postmaster._get_results(job_id, job_code)
    
        # Return results object
        return yhat, yhat_details
    
    else:
        return job_id, job_code