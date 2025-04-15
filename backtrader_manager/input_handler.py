import os
import time
import pandas as pd
import numpy as np
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

from .utils import _format_parameters
from .backtests_handler import Backtest    


class BacktestInputCollection:
   
    """
    A factory class for creating collections of backtest input configurations used in optimization or walk-forward testing.

    This class manages the setup for a consistent set of backtest runs:
    - Aim is to ensure comparability of all backtest inputs created of an input collection
    - Shares the same time configuration (start/end timestamp, train/test window, walk-forward step).
    - Share the same commission and slippage settings (True/False)
    - Shares the same data source and backtest execution logic of the BacktestCollection they are added to
    - Differentiates by backtest classes and specific strategy parameter combinations.

    **Important:**
        - **Time-period related settings** (e.g., `start_timestamp`, `train_bars`, `test_perc`, `walkforward_step`) are fixed after the first backtest inputs
          have been created and apply uniformly to all backtest inputs.
        - To change time-period-related settings, a new `BacktestInputCollection` instance must be created.

    **Attributes**:
        - **Fixed attributes** (set or calculated at instantiation or exactly one time and immutable afterward):
            - `name`: Descriptive name for the collection (e.g., "WalkForward", "Simple_Opt").
            - `datas`: Shared dictionary of ticker -> pandas DataFrame provided by the parent `BacktestCollection`
        - **Fixed attributes** (set or calculated as soon as the first backtest inputs have been created):    
            - `start_timestamp`, `end_timestamp`: Actual start and end dates for backtesting, after accounting for warmup and available data.
            - `train_bars`: Number of bars used in the training window.
            - `test_perc`: Percentage of data used for testing in each window.
            - `walkforward_step`: Step size (in bars) between walk-forward splits.
            - `max_warmup`: Maximum number of warm-up bars used to offset the earliest start timestamp.
            - `commission`, `slippage`: Flags to indicate whether commission or slippage should be applied during backtests.

        - **Adjustable attributes** (can be modified via setter methods after instantiation):
            - `backtest_class`: The class used to run backtests.
            - `strategy_parameters`: Dictionary of strategy parameters for optimization (supports value ranges).
            - `max_chunk_size`: Defines how many runs are grouped together in one backtrader backtest when executed in chunks.

        - **Outputs** (populated later):
            - `train_test_periods`: A dictionary of train/test periods (indexed by iteration). Each entry contains 'train' and 'test' periods as tuples of
               timestamps.
            - `backtest_inputs`: A list of instantiated `BacktestInput` objects representing the different backtest configurations, based on the generated 
               train/test periods and strategy parameter combinations.
            - `id_to_param`: A dictionary mapping unique IDs to specific parameter combinations used in each backtest.

    **Parameters**:
        - `datas (dict)`: Refer to the attribute `datas` for details.
        - `name (str)`: Descriptive name for this collection of backtests (stored in `name` attribute).
        - `start_timestamp (datetime, optional)`: Start of the data window (subject to warmup offset, stored in `start_timestamp`).
        - `end_timestamp (datetime, optional)`: End of the data window (stored in `end_timestamp`).
        - `train_bars (int, optional)`: Number of bars for the training window (stored in `train_bars`).
        - `test_perc (float)`: Percentage of data used for testing in each window (default is 0.3, stored in `test_perc`).
        - `backtest_class (Backtest)`: The class used to execute individual backtests (stored in `backtest_class`).
        - `strategy_parameters (dict)`: Dictionary of strategy parameters for optimization (supports value ranges, stored in `strategy_parameters`).
        - `commission (bool)`: Whether to apply commission to backtests (stored in `commission`).
        - `slippage (bool)`: Whether to apply slippage to backtests (stored in `slippage`).
        - `walkforward_step (int, optional)`: Step size (in bars) between walk-forward splits (stored in `walkforward_step`).
        - `max_warmup (int, optional)`: Max number of warm-up bars to offset the earliest start timestamp (stored in `max_warmup`).

    **Usage**:
        This class is used to generate consistent backtest input configurations that vary by strategy parameters but share the same dataset and time slicing    
        logic. For example, run multiple optimizations or walk-forward iterations across different parameter sets, all within the same time framework.
    """



    
    def __init__(self, name, start_timestamp=None, end_timestamp=None, 
                 window: int = None, train_perc=0.7, backtest_class=None, strategy_parameters=None,
                 commission=True, slippage=True, walkforward_step: int = None, max_warmup=None):

        # === Fixed attributes: Private, some with getters ===
        self._name = name  # Descriptive name for the type of backtest sets e.g., 'WFO-backtests', 'Simple_Opt-backtests'
        self._datas = None  # is set once when instance is added to an backtest collection
        self._id_to_param = {} # maps parameter_id to startegy parameter sets
        self._alias_to_backtest_class = {} # maps the backtest class alias to the backtest object
        self._backtest_inputs_created = False # flags if backtest inputs have been created
        
        # === Attributes adjustable as long no backtest inputs have been created yet ===
        self._max_warmup = max_warmup if max_warmup is not None else 0  # Indicator max warmup bars effect the earliest possible start timestamp 
        self._train_perc = train_perc
        self._window = window  # Train + test window        
        self._walkforward_step = walkforward_step
        self._start_timestamp = start_timestamp 
        self._end_timestamp = end_timestamp 
        self._commission = commission
        self._slippage = slippage

        # === Allways adjustable attributes: Public or with setters and getters ===
        self._backtest_class = backtest_class 
        self._strategy_parameters = _format_parameters(strategy_parameters or {}) 
        self.max_chunk_size = 1

        # ===  Outputs  ===
        self.earliest_start_timestamp = None
        self.latest_end_timestamp = None
        self._train_test_periods = None # all timestamps, not bars
        self._backtest_inputs = []
                
        print(f'Total parameter combinations: {self._count_combinations(self.strategy_parameters)}')
    
    # === Puplic methods ===
    
    def calc_train_test_periods(self):
        # Select a reference DataFrame (first available ticker)
        ref_ticker = next(iter(self._datas))
        ref_df = self._datas[ref_ticker]
    
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
        self.plot_train_test_periods()
       
    def plot_train_test_periods(self, use_dates=True, ref_ticker=None):
        train_test_periods = self.train_test_periods  # dict with iteration as key
        if ref_ticker:
            ref_ticker = ref_ticker
        else:
            ref_ticker = next(iter(self._datas))
        ref_data = self._datas[ref_ticker]
        
        # Initialize plot
        fig, ax1 = plt.subplots(figsize=(15, 6))
        plt.tight_layout()
        
        # Define colors for training and testing periods
        train_color = 'lightblue'
        test_color = 'lightgreen'
        
        # Set up plot properties
        ax1.set_title('Train and Test Periods per Iteration', fontsize=14)
        ax1.set_xlabel('Date' if use_dates else 'Bar Index', fontsize=12)
        ax1.set_ylabel('Iteration', fontsize=12)
        
        iteration_idx = 0  # Y-axis tracking
        
        # Create the secondary y-axis for the 'close' prices
        ax2 = ax1.twinx()  # Twin axis for 'close' values
        ax2.set_ylabel(f'Close Price for {ref_ticker}', fontsize=12)
        
        # Plot the 'close' price on ax2
        if use_dates:
            ax2.plot(ref_data.index, ref_data['close'], color='grey', label=f'Close Price {ref_ticker}', linewidth=1)
        else:
            ax2.plot(range(len(ref_data)), ref_data['close'], color='grey', label=f'Close Price {ref_ticker}', linewidth=1)
        
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
        
                ax1.add_patch(Rectangle(
                    (num_train_start, iteration_idx),
                    width_train,
                    1,
                    color=train_color,
                    label="Train" if iteration_idx == 0 else "",
                    alpha=0.5
                ))
                ax1.add_patch(Rectangle(
                    (num_test_start, iteration_idx),
                    width_test,
                    1,
                    color=test_color,
                    label="Test" if iteration_idx == 0 else "",
                    alpha=0.5
                ))
            else:
                # Convert timestamps to bar indices using get_bar()
                train_start_bar = self._get_bar(train_start, ref_ticker)
                train_end_bar = self._get_bar(train_end, ref_ticker)
                test_start_bar = self._get_bar(test_start, ref_ticker)
                test_end_bar = self._get_bar(test_end, ref_ticker)
        
                ax1.add_patch(Rectangle(
                    (train_start_bar, iteration_idx),
                    train_end_bar - train_start_bar,
                    1,
                    color=train_color,
                    label="Train" if iteration_idx == 0 else "",
                    alpha=0.5
                ))
                ax1.add_patch(Rectangle(
                    (test_start_bar, iteration_idx),
                    test_end_bar - test_start_bar,
                    1,
                    color=test_color,
                    label="Test" if iteration_idx == 0 else "",
                    alpha=0.5
                ))
        
            iteration_idx += 1  # Move to next iteration
        
        earliest_start = self.earliest_start_timestamp        
        latest_end = self.latest_end_timestamp         
        earliest_start_bar = self._get_bar(earliest_start, ref_ticker)
        latest_end_bar = self._get_bar(latest_end, ref_ticker)
        
        # Adjust X-axis formatting
        timestamp_delta = ref_data.index.to_series().diff().min()
        offset_in_bars = int((latest_end_bar - earliest_start_bar) * 0.05)
        offset_in_timestamp_deltas = offset_in_bars * timestamp_delta
    
        if use_dates:
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax1.set_xlim([earliest_start - offset_in_timestamp_deltas, latest_end + offset_in_timestamp_deltas])
    
            # Ensure that the tick labels are rotated 90 degrees
            for label in ax1.get_xticklabels():
                label.set_rotation(90)
                label.set_horizontalalignment('right')
        else:
            earliest_start = earliest_start_bar  
            latest_end = latest_end_bar
            ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))  
            #ax1.set_xlim([earliest_start - offset_in_bars, latest_end + offset_in_bars])
        
        # Set proper Y-axis
        ax1.set_ylim(-1, iteration_idx)
        ax1.set_yticks(range(iteration_idx))
        ax1.set_yticklabels([f"{i + 1}" for i in range(iteration_idx)])
        
        # Add vertical line for earliest start and latest end
        ymin, ymax = plt.ylim()
        y_pos = ymin - (ymax - ymin) * 0.3
        plt.axvline(x=earliest_start, color='grey', linestyle='--', linewidth=1)
        plt.text(earliest_start, y_pos, 'earliest_start', rotation=90, verticalalignment='bottom', color='grey')
        plt.axvline(x=latest_end, color='grey', linestyle='--', linewidth=1)
        plt.text(latest_end, y_pos, 'latest_end', rotation=90, verticalalignment='bottom', color='grey')
    
        # Add legend
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc='upper left')
    
        plt.show()
    

    
    def create_backtest_inputs(self, backtest_class=None, strategy_parameters=None, period_keys=('train',), max_chunk_size=None):
        backtest_inputs = []
        runs = 0
        if backtest_class: 
            self.backtest_class = backtest_class
        if strategy_parameters:
            self.strategy_parameters = strategy_parameters
        
        for iteration_key, periods in self.train_test_periods.items():
            for period_key in period_keys:
                period = periods[period_key]
        
                # Select strategy parameter sets
                if period_key=='train':
                    strategy_parameters_chunks = self._create_strategy_parameters_chunks(max_chunk_size)
                if period_key=='test':
                    strategy_parameters_chunks = [self._strategy_parameters][:] # create a shallow copy
    
                # For each parameter chunk, create a BacktestInput
                while strategy_parameters_chunks:
                    chunk = strategy_parameters_chunks.pop(0)  # Pop the first chunk
                    runs += self._count_combinations(chunk)
                    backtest_input = BacktestInput(
                        self.name,
                        period,
                        iteration_key,
                        period_key,
                        chunk,
                        backtest_class=self.backtest_class,
                        commission=self.commission,
                        slippage=self.slippage
                    )
                    backtest_inputs.append(backtest_input)
        
        self._backtest_inputs += backtest_inputs
        self._backtest_inputs_created = True
        print(f'Created {len(backtest_inputs)} backtest input(s) with total of {runs} run(s)')

    def clear_backtest_inputs(self):
        self._backtest_inputs = []

    def create_cloned_inputs(self, summary_filtered, period_keys=None): 
        '''
        Takes summary_filtered as input and creates new backtest inputs with the same parameter backtest class combinations.
        The cloned inputs can be slightly altered by defining optional parameter: period_key,
        It is also possible to create copies of backtests from other input collections.
        
        '''
        
        if (summary_filtered['input_collection_name'].nunique() > 1):
            print('summary filtered must only contain backtests from the same input_collection')
            return

        backtests = {}  # backtests from summary_filtered to be copied
                
        for index, row in summary_filtered.iterrows():
            parameter_id = row['parameter_id']
            backtest_class_alias = row['backtest_class']
            if not period_keys:
                period_keys = (row['period_key'],)
                
            if backtest_class_alias not in backtests:
                backtests[backtest_class_alias] = {}
            if parameter_id not in backtests[backtest_class_alias].keys():
                backtests[backtest_class_alias][parameter_id] = []
            for period_key in period_keys:
                if period_key not in backtests[backtest_class_alias][parameter_id]:         
                    backtests[backtest_class_alias][parameter_id].append(period_key)

### future implementation: rejoin parameter sets to ranges so that also train backtests with mutltible runs are posibble
# maybe not possible, because of how backtrader combines parameter values
        
        for backtest_class_alias, param_period_keys in backtests.items():
            self.backtest_class = backtest_class_alias
            for parameter_id, period_keys in param_period_keys.items():
                self.strategy_parameters = parameter_id
                self.create_backtest_inputs(period_keys=period_keys, max_chunk_size=None)
    
    def create_copy_self(self, name):
        '''
        Returns a deepcopy of the input_collection instance with many attributs the same, except for a few that are reset to default values as also defined
        in __init__() 
        Attributes with references to external objects will still point to the same place (deepcopy doesnt change that)
        '''
        copy_input_collection = copy.deepcopy(self)
        copy_input_collection._name = name
        copy_input_collection._backtest_inputs_created = False 
        copy_input_collection._backtest_inputs = []
        copy_input_collection._strategy_parameters = {}

        return copy_input_collection

    def maximise_window(self): 
        self._window = self._get_bar(self._end_timestamp) - self._get_bar(self._start_timestamp)
           
    # === Property getters/setters ===
    @property
    def backtest_class(self):
        return self._backtest_class

    @backtest_class.setter
    def backtest_class(self, value):
        if isinstance(value, str):
            self._backtest_class = self._alias_to_backtest_class.get(value, None)
        else:
            self._backtest_class = value
            
    
    @property
    def strategy_parameters(self) -> dict:
        return self._strategy_parameters
        
    @strategy_parameters.setter
    def strategy_parameters(self, value):
        if isinstance(value, str):
            self._strategy_parameters = self._id_to_param.get(value, None)[1]
            self._backtest_class = self._id_to_param.get(value, None)[0]
        else:   
            self._strategy_parameters = _format_parameters(value)

    @property
    def name(self):
        return self._name
        
    @property 
    def datas(self):
        return self._datas
    
    @property
    def max_warmup(self):
        return self._max_warmup
    
    error_message = 'This attribute can only be set before first backtest inputs have been created'
    
    @max_warmup.setter
    def max_warmup(self, value):
        if not self._backtest_inputs_created:
            self._max_warmup = value
        else:
            print(error_message)
    
    @property
    def train_perc(self):
        return self._train_perc

    @train_perc.setter
    def train_perc(self, value):
        if not self._backtest_inputs_created:
            self._train_perc = value
        else:
            print(error_message)

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, value):
        if not self._backtest_inputs_created:
            self._window = value
        else:
            print(error_message)

    @property
    def train_window(self):
        return int(self._train_perc * self._window)

    @property
    def test_window(self):
        return self._window - self.train_window
    
    @property
    def walkforward_step(self):
        if self._walkforward_step is not None:
            return self._walkforward_step
        return int(1.0 * self.test_window)  # Default logic

    @walkforward_step.setter
    def walkforward_step(self, value):
        if not self._backtest_inputs_created:
            self._walkforward_step = value
        else:
            print(error_message)
    
    @property
    def start_timestamp(self):
        return self._start_timestamp
    
    @start_timestamp.setter
    def start_timestamp(self, value):
        if not self._backtest_inputs_created:
            self._start_timestamp = value
    
    @property
    def end_timestamp(self):
        return self._end_timestamp
    
    @end_timestamp.setter
    def end_timestamp(self, value):
        if not self._backtest_inputs_created:
            self._end_timestamp = value
        else:
            print(error_message)

    @property
    def commission(self):
        return self._commission

    @commission.setter
    def commission(self, value):
        if not self._backtest_inputs_created:
            self._commission = value
        else:
            print(error_message)

    @property
    def slippage(self):
        return self._slippage

    @slippage.setter
    def slippage(self, value):
        if not self._backtest_inputs_created:
            self._slippage = value
        else:
            print(error_message)
            
    @property
    def backtest_inputs(self):
        return self._backtest_inputs
        
    # === Private helper methods ===
    
    def _set_datas(self, datas):
        self._datas = datas
        self.earliest_start_timestamp, self.latest_end_timestamp = self._calc_max_period()
        self._start_timestamp = max(self._start_timestamp, self.earliest_start_timestamp) if self._start_timestamp else self.earliest_start_timestamp
        self._end_timestamp = min(self._end_timestamp, self.latest_end_timestamp) if self._end_timestamp else self.latest_end_timestamp
        if not self._window: # if no window size was defined:
            self.maximise_window() #  window is maximised, leading to an optimisation with one single test/train period (no wfo)
        self.calc_train_test_periods()
            
    def _calc_max_period(self):
        """Calculates the earliest possible start date and latest possible end date based on available data."""
        start_dates, end_dates = [], []
        for ticker, data in self._datas.items():
            if len(data) <= self._max_warmup:
                raise ValueError(f"Not enough data for ticker {ticker} given max_warmup={self.max_warmup}.")
            start_dates.append(data.index[self._max_warmup])
            end_dates.append(data.index[-1])
        return (max(start_dates) if start_dates else None, min(end_dates) if end_dates else None)
            
    def _get_bar(self, timestamp, ticker=None):
        if not ticker:
            ticker = list(self._datas.keys())[0]
        return self._datas[ticker].index.get_loc(timestamp)
        
    def _get_timestamp(self, bar, ticker=None):
        if not ticker:
            ticker = list(self._datas.keys())[0]
        return self._datas[ticker].index[bar]  
                
    def _count_combinations(self, strategy_parameters):
        values = [value for value in strategy_parameters.values()]
        return len(list(itertools.product(*values)))
                
    def _create_strategy_parameters_chunks(self, max_chunk_size=None):
        self.max_chunk_size =  max_chunk_size
        parameters_chunks = self._split_parameters(self._strategy_parameters)
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
                break
    
        if self._count_combinations(parameters_chunks[0]) > self.max_chunk_size:       
            more_chunks = []
            for chunk in parameters_chunks:
                more_chunks.extend(self._split_parameters(chunk))
            parameters_chunks = more_chunks
    
        for i, chunk in enumerate(parameters_chunks):
            parameters_chunks[i] = {key: chunk[key] for key in list(parameters.keys()) if key in chunk}
              
        return parameters_chunks 
    
    def _set_parameter_mapping(self, id_to_param):
        self._id_to_param = id_to_param

    def _set_backtest_class_mapping(self, alias_to_backtest_class):
        self._alias_to_backtest_class = alias_to_backtest_class


        
class BacktestInput:
    """
    Represents a single unit of backtest configuration, derived from a BacktestInputCollection.

    Each BacktestInput instance defines a unique combination of:
    - A specific train or test time period
    - A specific strategy parameter subset (chunk)
    - Meta-information used for tracking and comparison (e.g., input_collection_name, iteration key)

    Attributes:
        input_collection_name (str): Name of the parent BacktestInputCollection.
        iteration_key (int): time period iteration key.
        period_key (str): Indicates whether this input is for 'train' or 'test'.
        period (tuple): Time window for the backtest (start_timestamp, end_timestamp).
        strategy_parameters (dict): Dictionary of backtrader strategy class parameters used for this backtest.
        backtest_class (Type): The backtest class to instantiate for execution.
        commission (bool): Whether to apply commission during the backtest.
        slippage (bool): Whether to apply slippage during the backtest.
        _parent_path (str): Optional reference to the parent run's path, if reused.
        _parent_input_id (tuple[int, int]): Optional reference to the original input ID, if reused.
    """
    
    
    def __init__(self, input_collection_name, period: tuple, iteration_key, period_key, 
                 strategy_parameters_chunk, backtest_class, commission=True, slippage=True):
        self.input_collection_name = input_collection_name # identifies the parent input collection
        self.iteration_key = iteration_key # int
        self.period_key = period_key # train or test
        self.period = period
        self.strategy_parameters = strategy_parameters_chunk # this can be ranges of parameters (if otimisation)
        
        self.backtest_class = backtest_class
        self.commission = commission
        self.slippage = slippage
        
    
    @property
    def input_id(self):
        return self._calc_input_id()
    
    def _calc_input_id(self):
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

    #def _get_class_backtest_run(self):
    #    return self._class_backtest_run 

    
    