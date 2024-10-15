"""
CSA Relevance Engine: Payload Handler Module

This module provides functions for managing and handling the payloads 
required for interacting with the Cambridge Sports Analytics (CSA) API. 
It handles tasks such as constructing and deconstructing JSON payloads, 
posting jobs to the server, and retrieving results.

Functions Overview
------------------
- post_job(function_type: PSRFunction, **varargin): 
    Posts a job with specified function type and input parameters to 
    the CSA server. Handles API key retrieval, input validation, 
    payload construction, and server communication.

- get_results(job_id: int, job_code: str): 
    Retrieves results for a specified job from the CSA server using 
    job ID and job code.

- _construct(**data): 
    Constructs a JSON payload from the provided keyword arguments, 
    ensuring correct data serialization.

- _deconstruct(json_payload: str): 
    Deconstructs a JSON payload string into a Python dictionary.

- poll_for_results(job_id: int, job_code: str, interval=10, timeout=900): 
    Polls the server for job results at regular intervals until results 
    are available or a timeout occurs.

- _indeterminant_progress_thread(status_msg: str = None): 
    A helper function that runs in a separate thread to display 
    indeterminate progress status while waiting for a task to complete.

Global Variables
----------------
- _stop_notify_progress: bool
    A flag to control the stopping of the progress notification thread.

Usage
-----
This module is intended to be used within the CSAnalytics library to 
manage server interactions for predictive modeling tasks. To initiate 
a prediction task, use `post_job()` with the appropriate function type 
and input parameters. To retrieve the results, use `get_results()` with 
the job ID and job code returned from `post_job()`.

(c) 2023 - 2024 Cambridge Sports Analytics, LLC. All rights reserved.
support@csanalytics.io
"""

# Standard library imports
import time  # Time-related functions
import json  # JSON encoding and decoding
import threading  # Support for multi-threaded programming

# Third-party library imports
import numpy as np  # Numerical operations and array manipulations
import requests  # HTTP requests to the CSA API
from http import HTTPStatus  # Enum for standard HTTP status codes

# Local application/library-specific imports
from ._auth_manager import _get_apikeys  # Retrieve API keys for authentication
from csa_common_lib.toolbox._validate import validate_inputs  # Input validation
from csa_common_lib.toolbox import _notifier  # Sending notifications
from csa_common_lib.enum_types.functions import PSRFunction  # Enum for function types


# Global variables
_stop_notify_progress = False


def post_job(function_type:PSRFunction, **varargin):
    """Post job for PSR with corresponding inputs to CSA server.

    Parameters
    ----------
    function_type : PSRFunction
        PSR model function to invoke. See enumeration.
    **varargin
        Input variables.

    Returns
    -------
    dict
        Response object from server.
    int
        Job ID.
    str
        Job code.
    """    
    
    # Retrieve API key
    # Retrieve end-user API Access ID and Access Key.
    api_key, access_id = _get_apikeys()
    
    # Validate inputs arguments
    validate_inputs(False, function_type, **varargin)
    
    # Concatenate into an inputs dictionary
    inputs = {
            'access_id': access_id,
        }
    
    # Add the rest of the arguments
    inputs.update(varargin)
    
    # Construct payload (unpack the inputs dictionary when passing to fn)
    _notifier.task_update(f'Aggregating inputs for API ({function_type})')
    payload = _construct(**inputs)
    _notifier.task_update(is_done=True)
    
    # API end-point dictionary    
    api_resource = {
        int(PSRFunction.PSR): 'https://api.csanalytics.io/v2/prediction-engine/psr',
        int(PSRFunction.MAXFIT): 'https://api.csanalytics.io/v2/prediction-engine/maxfit',
        int(PSRFunction.GRID): 'https://api.csanalytics.io/v2/prediction-engine/grid',
        int(PSRFunction.GRID_SINGULARITY): 'https://api.csanalytics.io/v2/prediction-engine/grid-singularity',
        int(PSRFunction.RELEVANCE): 'https://api.csanalytics.io/v2/engine/relevance',
        int(PSRFunction.SIMILARITY): 'https://api.csanalytics.io/v2/engine/similarity',
        int(PSRFunction.INFORMATIVENESS): 'https://api.csanalytics.io/v2/engine/informativeness',
        int(PSRFunction.FIT): 'https://api.csanalytics.io/v2/engine/fit',
        int(PSRFunction.ADJUSTED_FIT): 'https://api.csanalytics.io/v2/engine/adjusted-fit',
        int(PSRFunction.ASYMMETRY):'https://api.csanalytics.io/v2/engine/asymmetry'
    }
    
    # Set the API end-point
    url = api_resource[int(function_type)]
    
    # Configure the header object
    header_obj = {
        'Content-Type': 'application/json',
        'Connection': 'keep-alive',
        'x-api-key': api_key
    }
    
    if _notifier.is_notifier_enabled():
        # Use the global variable _stop_notify_progress
        global _stop_notify_progress
        
        # Start indeterminant progress display
        _notify_thread = threading.Thread(
            target=_indeterminant_progress_thread,
            args=('Sending payload to server',)
            )
        _notify_thread.start()
    
    # Make POST request (set timeout to 300 seconds, payload can be large)
    _max_attempts = 3
    _wait_time = 5
    
    for attempt in range(_max_attempts):
        try:
            response = requests.post(url, data=payload, headers=header_obj, timeout=300)
            if response.status_code == HTTPStatus.OK:
                # Successful response, break out of loop
                break
        except requests.exceptions.RequestException as e:
            print(f"helpers:_payload_handler:post_job:Attempt {attempt+1} failed: {e}")
            continue
        
        # exponential backoff
        time.sleep(_wait_time ** attempt)
        
    
    if _notifier.is_notifier_enabled():
        # Finished expensive task, update status for user
        _stop_notify_progress = True
        _notify_thread.join()
        
    # Initialize output
    payload = _deconstruct(response.text)
    job_id = None
    job_code = None
    
    # Convert into a Python dict
    if response.status_code == HTTPStatus.OK:
        # The response will have a job id, so that we can use to retrieve the results
        job_id = payload.get('job_id')
        job_code = payload.get('job_code')
    else:
        # ERROR
        print(f"helpers:_payload_handler:post_job:{response.reason}:{response.text}")

    # Return the response
    return payload, job_id, job_code

def get_quota(quota_type:str, api_key:str):
    
    quota_endpoints = {
        'quota':'https://api.csanalytics.io/v2/prediction-engine/quota',
        'used':'https://api.csanalytics.io/v2/prediction-engine/quota/used',
        'remaining':'https://api.csanalytics.io/v2/prediction-engine/quota/remaining',
        'summary':'https://api.csanalytics.io/v2/prediction-engine/quota/summary'
    }

    # Configure the header object
    header_obj = {
        'Content-Type': 'application/json',
        'Connection': 'keep-alive',
        'x-api-key': api_key
    }

    try:
        url = quota_endpoints[quota_type]
    except Exception as e:
        raise Exception(f"User passed quota type does not exist. Please select quota, user, remaining or summary. Error {e}")
    
    response = requests.get(url=url, headers=header_obj)

    return response.json()

def get_results(job_id:int, job_code:str):
    """Retrieve results for a given job_id from server

    Parameters
    ----------
    job_id : int
        Job ID on server /database.
    job_code : str
        Job code on server / database.
    

    Returns
    -------
    json
        Response object.
    dict
        Result dictionary.
    """    
    
    if job_id is not None:
    
        # Retrieve API key
        # Retrieve end-user API Access ID and Access Key.
        api_key, _ = _get_apikeys()
    
        # URL to REST API
        url = 'https://api.csanalytics.io/v2/prediction-engine/results'
        
        header_obj = {
            'Content-Type': 'application/json',
            'Connection': 'keep-alive',
            'x-api-key': api_key
        }
        
        # Construct payload to request result
        payload = json.dumps({
            'job_id': job_id,
            'job_code': job_code
            })
        
        # Make Get request
        _max_attempts = 3
        _wait_time = 5
        
        for attempt in range(_max_attempts):
            try:
                response = requests.get(url, data=payload, headers=header_obj, timeout=300)
                if response.status_code == HTTPStatus.OK:
                    # Successful response, break out of loop
                    break
            except requests.exceptions.RequestException as e:
                print(f"helpers:_payload_handler:get_job:Attempt {attempt+1} failed for job_id{job_id}: {e}")
                continue
            
            # exponential backoff
            time.sleep(_wait_time ** attempt)
            
        # Initliaze output
        output = None
        
        # Decode output
        if response.status_code == HTTPStatus.OK:
            output = _deconstruct(response.text)
        elif response.status_code == HTTPStatus.FORBIDDEN:
            print("psr_library:get_results:Forbidden:Invalid or unspecified x-api-key.")
        else:
            print(f"psr_library:get_results:{response.reason}:{response.text}")
    
    return response, output


def _construct(**data):
    """Constructs the JSON payload given varargin data variables.

    Input
    -------
    **data
        Varargin of data types, input variables. Name of variables are
        preserved for the payload.
        
    Returns
    -------
    str
        JSON string of the payload, ready for AWS Lambda
        
    """ 
    
    # memory allocation for variables in the payload        
    variables = {}   
    
    # Enumerate through each input variable.
    # Convert any ndarrays to lists to serialize for JSON
    for name, obj in data.items():
        if isinstance(obj, np.ndarray):
            variables[name] = obj.tolist()
        else:
            variables[name] = obj
            
        
    # Convert the payload to a JSON string
    payload = json.dumps(variables, indent=4)
    
    # Return the JSON payload string        
    return payload


def _deconstruct(json_payload:str):
    """Deconstructs (decodes) JSON payload.

    Parameters
    ----------
    json_payload : str
        JSON
        
    Returns
    -------
    dict
        Output dictionary with key-value pairs.
    """
    
    payload = None
    
    # Call this for deconstructing json body from an AWS lambda post
    if json_payload is not None:
        payload = json.loads(json_payload)
    
    # Return dictionary of data
    return payload


def poll_for_results(job_id:int, job_code:str, interval=10, timeout=900):
    """Polls server for results, this can take up to 15 minutes 
    depending on the task.

    Parameters
    ----------
    job_id : int
        Job identification number. Provided by the post_job response.
    job_code : str
        Job code, secondary identifier. Provided by the post_job response.
    interval : int, optional
        Interval to poll server, by default 10 seconds.
        The minimum interval is 10 seconds.
    timeout : int, optional
        Maximum timeout, by default 900 seconds.

    Returns
    -------
    dict
        Results dictionary.

    Raises
    ------
    TimeoutError
        When maximum timeout reached.
    """    
    
    
    if _notifier.is_notifier_enabled():
        # Use the global variable _stop_notify_progress
        global _stop_notify_progress
        
        # Start indeterminant progress display
        _notify_thread = threading.Thread(
            target=_indeterminant_progress_thread,
            args=('Retrieving results from server',)
            )
        _notify_thread.start()
        
    # Ensure that the minimum interval = 10 seconds
    interval = max(interval, 10)
    
    # Initialize output
    output = None
    
    start_time = time.time()
    while True:
        # Get results using your function
        response, output = get_results(job_id, job_code)
        
        # Check if the result is ready
        if output is not None:  # Or whatever condition means the result is ready
            if output.get('statusMsg') == "PROCESSING":
                # The server is still processing results...
                pass
            else:
                if _notifier.is_notifier_enabled():
                    _stop_notify_progress = True
                    _notify_thread.join() 
                return output

        # Check if timeout has been reached
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            if _notifier.is_notifier_enabled():
                _stop_notify_progress = True
                _notify_thread.join() 
            raise TimeoutError("csanalytics: get_results: Polling timed out.")

        # Sleep for 30 seconds before checking again
        time.sleep(interval)
    
       
        
def _indeterminant_progress_thread(status_msg:str=None):
    """Thread function to call/show indeterminant progress status.
    This allows the progress to updates every second while the poller
    interacts with the server at different (longer) interval lengths.
    """    
    global _stop_notify_progress
    _notifier.hide_cursor()
    
    while not _stop_notify_progress:
        _notifier.display_processing(status_msg)
        time.sleep(0.3)
        
    # Done processing
    _notifier.display_processing(status_msg, is_done=True)
    
    # The thread has stopped, reset
    _notifier.show_cursor()
    _stop_notify_progress = False