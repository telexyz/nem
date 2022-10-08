import os
import sys
import numpy as np
import requests
import json
import inspect
import copy
import gzip
import re
import types
import pytest
import base64
import pickle
import datetime
from requests.packages.urllib3.exceptions import InsecureRequestWarning


"""
Note: This use of globals is pretty ugly, but it's unclear to me how to wrap this into 
a class while still being able to use pytest hooks, so this is the hacky solution for now.
"""


_values = []
_submission_key = ""
_errors = 0
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)



def get_server_url_protocol(key):
    """ Maintain legacy pickle protocol temporarily before phasing out. """
    server_urls = [
       {"prefix":"_", "url":"https://mugrade-online.dlsyscourse.org/_/api/", "protocol":"json"},
        {"prefix":None, "url":"https://mugrade.dlsyscourse.org/_/api/", "protocol":"pickle"}
    ]

    """ Hacky yet simple way to differentiate servers via key prefix """
    for s in server_urls:
        if s["prefix"] is not None and key.startswith(s["prefix"]):
            return s["url"], s["protocol"]
    return server_urls[-1]["url"], server_urls[-1]["protocol"]




def decode_json(data):
    """ Decode from our JSON encoded to a dictionary with e.g. np.ndarray objects."""
    if isinstance(data, dict):
        if "_encoded_type" in data:
            if data["_encoded_type"] == "np.ndarray":
                return np.array(decode_json(data["data"]))
            elif data["_encoded_type"] == "datetime":
                return datetime.fromisoformat(data["data"])
            else:
                return data
        else:
            return {k:decode_json(v) for k,v in data.items()}
    elif isinstance(data, list):
        return [decode_json(d) for d in data]
    else:
        return data



def encode_json(data):
    """ Encode a dictionary to our JSON serialization """
    if isinstance(data, dict):
        return {k:encode_json(v) for k,v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [encode_json(d) for d in data]
    elif isinstance(data, np.ndarray):
        return {"_encoded_type":"np.ndarray", "data":encode_json(data.tolist())}
    elif isinstance(data, datetime.datetime):
        return {"_encoded_type":"datetime", "data":data.isoformat()}
    elif isinstance(data, (type, np.dtype)):
        return {"_encoded_type":"type", "data":repr(data)}
    elif isinstance(data, (np.float16, np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(data)
    else:
        return data


def b64_pickle(obj):
    return base64.b64encode(pickle.dumps(obj)).decode("ASCII")


def start_submission(func_name):
    """ Begin a submisssion to the mugrade server """

    server_url, protocol = get_server_url_protocol(os.environ["MUGRADE_KEY"])
    response = requests.post(server_url + "submission",
                             params = {"user_key": os.environ["MUGRADE_KEY"],
                                       "func_name": func_name},
                             verify=False)


    if response.status_code != 200:
        raise Exception(f"Error : {response.text}")
    return response.json()["submission_key"]

def submit_test():
    """ Submit a single grader test. """
    global _values, _submission_key, _errors
    server_url, protocol = get_server_url_protocol(os.environ["MUGRADE_KEY"])
    if protocol == "json":
        response = requests.post(server_url + "submission_test",
                                 params = {"user_key": os.environ["MUGRADE_KEY"],
                                           "submission_key":_submission_key, 
                                           "test_case_index":len(_values)-1},
                                 json=encode_json(_values[-1]),
                                 verify=False)
    elif protocol == "pickle":
        response = requests.post(server_url + "submission_test",
                                 params = {"user_key": os.environ["MUGRADE_KEY"],
                                           "submission_key":_submission_key, 
                                           "test_case_index":len(_values)-1,
                                           "output":b64_pickle(_values[-1])})

    if response.status_code != 200:
        print(f"Error : {response.text}")
    elif response.json()["status"] != "Passed":
        print(f"Grader test {len(_values)} failed: {response.json()['status']}")
        _errors += 1
    else:
        print(f"Grader test {len(_values)} passed")



def publish(func_name):
    """ Publish an autograder. """
    global _values
    server_url, protocol = get_server_url_protocol(os.environ["MUGRADE_KEY"])
    if protocol == "json":
        response = requests.post(server_url + "publish_grader",
                                 params = {"user_key": os.environ["MUGRADE_KEY"],
                                           "func_name": func_name,
                                           "overwrite": True},
                                 json=encode_json(_values),
                                 verify=False)
    else:
        response = requests.post(server_url + "publish_grader",
                                 params = {"user_key": os.environ["MUGRADE_KEY"],
                                           "func_name": func_name,
                                           "target_values": b64_pickle(_values),
                                           "overwrite": True})
    if response.status_code != 200:
        print(f"Error : {response.text}")
    else:
        print(response.json()["status"])


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    ## prior to test, initialize submission
    global _values, _submission_key, _errors
    _values = []
    _errors = 0
    func_name = pyfuncitem.name[7:]
    if os.environ["MUGRADE_OP"] == "submit":
        _submission_key = start_submission(func_name)
        print(f"\nSubmitting {func_name}...")

    # run test
    output = yield


    # raise excepton if tests failed (previously keep running)
    if os.environ["MUGRADE_OP"] == "submit":
        if _errors > 0:
            pytest.fail(pytrace=False)

    # publish tests
    if os.environ["MUGRADE_OP"] == "publish":
        #print(values)
        publish(func_name)


def submit(result):
    global _values
    _values.append(result)
    if os.environ["MUGRADE_OP"] == "submit":
        submit_test()


