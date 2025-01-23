import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Import the ticker module
import datetime
import backtrader as bt
import quantstats as qs
import json
import hashlib
import math
from collections.abc import Iterable
import pickle
import itertools
import copy
from collections import OrderedDict

# Manages input parameters and splitting parameters into smaller subsets
class InputParameters():
    def __init__(self, parameters={}):
        self.parameters = parameters
        self.max_chunk_size = 1
        self.parameters_chunks = [{}]
        
        print(f'Total parameter combinations: {self.count_combinations(self.parameters)}')
    
    def count_combinations(self, parameters):
        # Prepare the values for itertools.product
        values = [
            value if isinstance(value, (list, tuple, set)) else [value]
            for value in parameters.values()
        ]
        # Use product to calculate all combinations
        return len(list(itertools.product(*values)))

    def _add_parameters_ids(self):
        for parameters in self.parameters_chunks:
            # Convert the parameters dictionary into a JSON string with sorted keys
            serialized_params = json.dumps(parameters, sort_keys=True)
    
            # Use hashlib to create a unique hash
            unique_id = hashlib.md5(serialized_params.encode()).hexdigest()
            parameters['ID'] = unique_id
             
    def create_parameters_chunks(self, max_chunk_size):
        self.max_chunk_size =  max_chunk_size
        self.parameters_chunks = self._split_parameters(self.parameters)
        self._add_parameters_ids()
        print(
            f'\nnumber of chunks: {len(self.parameters_chunks)}', 
            f'\ncombinations per chunk / total: {self.count_combinations(self.parameters_chunks[0])}/{len(self.parameters_chunks) * self.count_combinations(self.parameters_chunks[0])}')
    
    def _split_parameters(self, parameters):
    
        # Sort parameters dictionary by the length of each list value in ascending order
        sorted_parameters = OrderedDict(sorted(parameters.items(), key=lambda item: len(item[1]) if isinstance(item[1], (list, tuple, set)) else 0))
        if self.count_combinations(parameters) < self.max_chunk_size:
            return parameters
    
        for key_parameters, value_parameters in sorted_parameters.items():
            parameters_temp = {key: value for key, value in parameters.items() if key != key_parameters}
            if self.count_combinations(parameters_temp) <= self.max_chunk_size or key_parameters == list(sorted_parameters.keys())[-1]:
            
                parameters_chunks = []
                for parameter in value_parameters:
                    parameters_chunk = copy.deepcopy(parameters_temp)
                    parameters_chunk[key_parameters] = parameter
                    parameters_chunks.append(parameters_chunk)
                print(f'split on key "{key_parameters}"')
                break
    
        if self.count_combinations(parameters_chunks[0]) > self.max_chunk_size:       
            more_chunks = []
            for chunk in parameters_chunks:
                more_chunks.extend(self._split_parameters(chunk))
            parameters_chunks = more_chunks
    
        for i, chunk in enumerate(parameters_chunks):
            parameters_chunks[i] = {key: chunk[key] for key in list(parameters.keys()) if key in chunk}
              
        return parameters_chunks

    def get_chunks(self):
        return self.parameters_chunks
