import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Import the ticker module
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
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

def _format_parameters(parameters):
        parameters_formatted = {}
    
        for key, parameter in parameters.items():
            key_lower = key.lower()
    
            # Handle different types of parameter values
            if isinstance(parameter, (list, tuple, set)):
                parameters_formatted[key_lower]= parameter  # Ensure it's a list
            else:
                parameters_formatted[key_lower] = [parameter]  # Wrap single value in a list
    
        return parameters_formatted
    
# Manages input parameters and splitting parameters into smaller subsets
class BacktestInputCollection:
    def __init__(self, csv_data_path, name, start_timestamp=None, end_timestamp=None, 
                 train_bars: int = None, test_perc=0.3, backtest_class=None, strategy_parameters=None, 
                 benchmark=False, commission=True, slippage=True, walkforward_step: int = None, max_warmup=None):
        
        self.csv_data_path = csv_data_path
        self.pandas_datas = self._load_csv_datas()  # {ticker: data_ticker.csv}
        self.backtest_class = backtest_class
        self.strategy_parameters = _format_parameters(strategy_parameters or {})
        self.max_warmup = max_warmup if max_warmup is not None else 0  # Ensure it's not None
        
        earliest_start_timestamp, latest_end_timestamp = self._calc_max_period()
        self.start_timestamp = max(start_timestamp, earliest_start_timestamp) if start_timestamp else earliest_start_timestamp
        
        self.end_timestamp = latest_end_timestamp
        self.test_perc = test_perc
        self.train_window = train_bars  # Input based on bars
        self.window = self.calc_window()  # Train + test window
        self.test_window = self.calc_test_window()
        self.walkforward_step = walkforward_step if walkforward_step is not None else int(1.0 * self.test_window)
        self.benchmark = benchmark
        self.commission = commission
        self.slippage = slippage
        self.max_chunk_size = 1
        
        self.train_test_periods = None # all timestamps, not bars
        self.name = name  # e.g., 'WFO-backtest_class', 'Simple_Opt-backtest_class'
        self.backtest_inputs = []
        
        print(f'Total parameter combinations: {self._count_combinations(self.strategy_parameters)}')
    
    def _load_csv_datas(self):
        """Loads CSV data into a dictionary of DataFrames."""
        import pandas as pd
        import os
        
        data_dict = {}
        if not os.path.exists(self.csv_data_path):
            raise FileNotFoundError(f"CSV data path '{self.csv_data_path}' does not exist.")
        
        for file in os.listdir(self.csv_data_path):
            if file.endswith('.csv'):
                ticker = file.split('.')[0]
                file_path = os.path.join(self.csv_data_path, file)
                data_dict[ticker] = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        if not data_dict:
            raise ValueError("No CSV files found in the provided directory.")
        
        return data_dict
    
    def _calc_max_period(self):
        """Calculates the earliest possible start date and latest possible end date based on available data."""
        start_dates, end_dates = [], []
        for ticker, data in self.pandas_datas.items():
            if len(data) <= self.max_warmup:
                raise ValueError(f"Not enough data for ticker {ticker} given max_warmup={self.max_warmup}.")
            start_dates.append(data.index[self.max_warmup])
            end_dates.append(data.index[-1])
        return (max(start_dates) if start_dates else None, min(end_dates) if end_dates else None)
    
        
    def calc_window(self):
        train_perc = (1 - self.test_perc)
        window = int(self.train_window / train_perc)
        return window
          
    def calc_test_window(self):
        test_window = self.window - self.train_window
        return test_window

    def get_bar(self, timestamp, ticker=None):
        if not ticker:
            ticker = list(self.pandas_datas.keys())[0]
        return self.pandas_datas[ticker].index.get_loc(timestamp)
        
    def get_timestamp(self, bar, ticker=None):
        if not ticker:
            ticker = list(self.pandas_datas.keys())[0]
        return self.pandas_datas[ticker].index[bar]
        
    def calc_train_test_periods(self):
        # Select a reference DataFrame (first available ticker)
        ref_ticker = next(iter(self.pandas_datas))
        ref_df = self.pandas_datas[ref_ticker]
    
        # Convert timestamps to bar indices
        start_bar = ref_df.index.get_loc(self.start_timestamp)  
        end_bar = ref_df.index.get_loc(self.end_timestamp)
    
        # Generate train/test periods using bar indices
        periods = {}
        iteration = 0
        for start in range(start_bar, end_bar, self.walkforward_step):
            if start + self.window > end_bar:
                break  # Ensure we don't exceed the dataset
    
            train_end = start + self.train_window
            test_end = start + self.window
    
            periods[iteration] = {
                'train': (start, train_end),
                'test': (train_end, test_end)
            }
            iteration += 1
    
        # Convert bar indices to timestamps and store in self.train_test_periods as a dict with iteration keys
        self.train_test_periods = {
            iter_idx: {
                'train': (ref_df.index[period['train'][0]], ref_df.index[period['train'][1]]),
                'test': (ref_df.index[period['test'][0]], ref_df.index[period['test'][1]])
            }
            for iter_idx, period in periods.items()
        }
    
    def plot_train_test_periods(self, use_dates=True):
        train_test_periods = self.train_test_periods  # dict with iteration as key
        ref_ticker = next(iter(self.pandas_datas))
        ref_data = self.pandas_datas[ref_ticker]
    
        # Initialize plot
        fig, ax = plt.subplots(figsize=(15, 6))
    
        # Define colors for training and testing periods
        train_color = 'lightblue'
        test_color = 'lightgreen'
    
        # Set up plot properties
        ax.set_title('Train and Test Periods per Iteration', fontsize=14)
        ax.set_xlabel('Date' if use_dates else 'Bar Index', fontsize=12)
        ax.set_ylabel('Iteration', fontsize=12)
    
        iteration_idx = 0  # Y-axis tracking
    
        # Iterate over the train_test_periods dictionary sorted by iteration index
        for iter_idx in sorted(train_test_periods.keys()):
            period = train_test_periods[iter_idx]
            train_start, train_end = period['train']
            test_start, test_end = period['test']
    
            if use_dates:
                # Use timestamps directly
                num_train_start = mdates.date2num(train_start)
                num_train_end = mdates.date2num(train_end)
                num_test_start = mdates.date2num(test_start)
                num_test_end = mdates.date2num(test_end)
    
                width_train = num_train_end - num_train_start
                width_test = num_test_end - num_test_start
    
                ax.add_patch(Rectangle(
                    (num_train_start, iteration_idx),
                    width_train,
                    1,
                    color=train_color,
                    label="Train" if iteration_idx == 0 else "",
                    alpha=0.5
                ))
                ax.add_patch(Rectangle(
                    (num_test_start, iteration_idx),
                    width_test,
                    1,
                    color=test_color,
                    label="Test" if iteration_idx == 0 else "",
                    alpha=0.5
                ))
            else:
                # Convert timestamps to bar indices using get_bar()
                train_start_bar = self.get_bar(train_start, ref_ticker)
                train_end_bar = self.get_bar(train_end, ref_ticker)
                test_start_bar = self.get_bar(test_start, ref_ticker)
                test_end_bar = self.get_bar(test_end, ref_ticker)
    
                ax.add_patch(Rectangle(
                    (train_start_bar, iteration_idx),
                    train_end_bar - train_start_bar,
                    1,
                    color=train_color,
                    label="Train" if iteration_idx == 0 else "",
                    alpha=0.5
                ))
                ax.add_patch(Rectangle(
                    (test_start_bar, iteration_idx),
                    test_end_bar - test_start_bar,
                    1,
                    color=test_color,
                    label="Test" if iteration_idx == 0 else "",
                    alpha=0.5
                ))
    
            iteration_idx += 1  # Move to next iteration
    
        # Adjust X-axis formatting
        if use_dates:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=90)
            ax.set_xlim([self.start_timestamp, self.end_timestamp])
        else:
            ax.set_xlim(self.get_bar(self.start_timestamp), self.get_bar(self.end_timestamp))
    
        # Set proper Y-axis
        ax.set_ylim(-1, iteration_idx)
        ax.set_yticks(range(iteration_idx))
        ax.set_yticklabels([f"{i + 1}" for i in range(iteration_idx)])
    
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper left')
    
        plt.tight_layout()
        plt.show()
                
    def _count_combinations(self, strategy_parameters):
        values = [value for value in strategy_parameters.values()]
        return len(list(itertools.product(*values)))
                
    def _create_strategy_parameters_chunks(self, max_chunk_size=None):
        self.max_chunk_size =  max_chunk_size
        parameters_chunks = self._split_parameters(self.strategy_parameters)
            
        print(
            f'\nnumber of chunks: {len(parameters_chunks)}', 
            f'\ncombinations per chunk / total: {self._count_combinations(parameters_chunks[0])}/{len(parameters_chunks) * self._count_combinations(parameters_chunks[0])}')
        return parameters_chunks
    
    def _split_parameters(self, parameters):
    
        # Sort parameters dictionary by the length of each list value in ascending order
        sorted_parameters = OrderedDict(sorted(parameters.items(), key=lambda item: len(item[1]) if isinstance(item[1], (list, tuple, set)) else 0))
        
        if not self.max_chunk_size:
            return [parameters]  
            
        if self._count_combinations(parameters) < self.max_chunk_size:
            return [parameters]       
    
        for key_parameters, value_parameters in sorted_parameters.items():
            parameters_temp = {key: value for key, value in parameters.items() if key != key_parameters}
            if self._count_combinations(parameters_temp) <= self.max_chunk_size or key_parameters == list(sorted_parameters.keys())[-1]:
            
                parameters_chunks = []
                for parameter in value_parameters:
                    parameters_chunk = copy.deepcopy(parameters_temp)
                    parameters_chunk[key_parameters] = [parameter]
                    parameters_chunks.append(parameters_chunk)
                print(f'split on key "{key_parameters}"')
                break
    
        if self._count_combinations(parameters_chunks[0]) > self.max_chunk_size:       
            more_chunks = []
            for chunk in parameters_chunks:
                more_chunks.extend(self._split_parameters(chunk))
            parameters_chunks = more_chunks
    
        for i, chunk in enumerate(parameters_chunks):
            parameters_chunks[i] = {key: chunk[key] for key in list(parameters.keys()) if key in chunk}
              
        return parameters_chunks

    def create_backtest_inputs(self, train=True, max_chunk_size=None):
        backtest_inputs = []
        for iteration_key, periods in self.train_test_periods.items():
            if train:
                period_key = 'train'
            else:
                period_key = 'test'
            period = periods[period_key]
        
            strategy_parameters_chunks = self._create_strategy_parameters_chunks(max_chunk_size)
            for strategy_parameters_chunk in strategy_parameters_chunks:
                backtest_input = BacktestInput(
                    self.name, 
                    period, 
                    iteration_key,
                    period_key,
                    strategy_parameters_chunk, 
                    self.backtest_class,
                    self.benchmark, 
                    commission=self.commission, 
                    slippage=self.slippage
                )
                backtest_inputs.append(backtest_input)
                
        self.backtest_inputs = backtest_inputs
    
    def clear_backtest_inputs(self):
        self.backtest_inputs = []
        

class BacktestInput():
    def __init__(self, collection_name, period: tuple, iteration_key, period_key, strategy_parameters_chunk, backtest_class, benchmark=False, commission=True, slippage=True):
        self.input_collection_name = collection_name # identifies the parent input collection
        self.iteration_key = iteration_key # int
        self.period_key = period_key # train or test
        self.period = period
        self.strategy_parameters = strategy_parameters_chunk # this can be ranges of parameters (if otimisation)
        
        self.backtest_class = backtest_class
        self.benchmark = benchmark
        self.commission = commission
        self.slippage = slippage
        
        # if the instance is a copy of a allready existing backtest run:
        self._parent_path = None
        self._parent_idx = None 
    
    @property
    def idx(self):
        return self._calc_backtest_idx()
    
    def _calc_backtest_idx(self):
        # Serialize all attributes of the instance (excluding methods)
        attributes = {
            key: value for key, value in self.__dict__.items()
        }
        
        # Convert the attributes dictionary into a JSON string with sorted keys
        serialized_attributes = json.dumps(attributes, sort_keys=True, default=str)
    
        # Use hashlib to create a unique hash
        unique_id = hashlib.md5(serialized_attributes.encode()).hexdigest()
        
        return unique_id
    
    def set_strategy_parameters(self, strategy_parameters): 
        self.strategy_parameters = _format_parameters(strategy_parameters)

    def _set_parent_path(self, parent_path):
        self._parent_path = parent_path
        
    def _set_parent_idx(self, parent_idx: tuple[int, int]):
        self._parent_idx = parent_idx

    def _get_class_backtest_run(self):
        return self._class_backtest_run 

    def _get_parent_path(self):
        return self._parent_path 
    
    