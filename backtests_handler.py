import os
import time
from .utils import wait_for_user_confirmation
from .utils import print_headline
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


class Backtest():
    def __init__(self, parameters, start_date, end_date, max_lookback_period=0):  
        print('\nInitialising new backtest instance')
        self.strategy_class = self._set_strategy_class()
        self.parameters = parameters
        self.result = None
        self.runs = None
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.max_lookback_period = max_lookback_period   
        self.buffer_start_date = self.start_date - datetime.timedelta(days=max_lookback_period)
        self.cerebro = self._instantiate_cerebro()
        self._add_strategy()
        self._add_datafeeds()
        self.setup_cerebro()       
    
    def get_id(self):
        return self.parameters.get('ID')

    def get_runs(self):
        if self.runs:
            return self.runs
        return None
    
    def get_result(self):
        if self.result:
            return self.result
        return None
    
    def get_parameters(self):
        return self.parameters
        
    def summarize(self):
        pass
    
    def count_combinations(self):
        # Prepare the values for itertools.product
        values = [
            value if isinstance(value, (list, tuple, set)) else [value]
            for value in self.parameters.values()
        ]
        # Use product to calculate all combinations
        return len(list(itertools.product(*values)))
    
    def _is_iterable(self, obj):
            if isinstance(obj, str):
                return False
            try:
                iter(obj)
                return True
            except TypeError:
                return False
                
    def _check_if_optimisation(self):
        for parameter_tuple in self.parameters.items():
            parameter_value = parameter_tuple[1]
            # Check if parameter is iterable, exclude strings, bytes, and other single-value-like objects
            if self._is_iterable(parameter_value):
                optimise = True
                break
            else:
                optimise = False
        return optimise
        
    def _set_strategy_class(self):
        strategy_class = self.set_strategy_class()
        print(f'Strategy_class set to: {strategy_class}')
        return strategy_class
        
    def set_strategy_class(self):
        pass
        
    def _add_strategy(self):
        
        if self._check_if_optimisation():
            self.cerebro.optstrategy(self.strategy_class, **{key.lower(): value for key, value in self.parameters.items()})      
            print('Added strategy to cerebro as backtest with multiple runs')
        else:
            self.cerebro.addstrategy(self.strategy_class, **{key.lower(): value for key, value in self.parameters.items()})  
            print('Added strategy to cerebro as backtest with single run')

    def _add_datafeeds(self):
        
        self.add_datafeeds()
    
    def add_datafeeds(self):
        pass
        
    def run(self, clear_result=True):
        
        cerebro = self.cerebro
        starting_value = cerebro.broker.getvalue()
        
        if self._check_if_optimisation():
            result = cerebro.run(maxcpus=1) #plot=False 
        else:
            result = cerebro.run()
    
        self.result = result
        print(f'\nStart Portfolio Value: {starting_value} \nFinal Portfolio Value: {self.cerebro.broker.get_value()}')
        self._analyse_result()
        if clear_result:
            self.result = None
            self.cerebro = None
    
    def setup_cerebro(self):
        pass
        # to be defined in subclass
        
    
    def _instantiate_cerebro(self):
        cerebro = bt.Cerebro(stdstats=True)
        print('Created cerebro instance')
        return cerebro
        
        
             

    def _is_nested_list(self, obj):
        if isinstance(obj, list) and any(isinstance(item, list) for item in obj):
            return True
        return False

    def _analyse_result(self):
        runs = []

        if self._is_nested_list(self.result):
            for run in self.result:
                runs.append(run)  
        else:
            runs = [self.result]    
    
        runs_with_metrics = []
        for i, run in enumerate(runs):
            strategy = run[0]
            runs_with_metrics.append({  
                'index': i,
                **vars(strategy.params),
                'rnorm100': strategy.analyzers.returns.get_analysis()['rnorm100'],
                'max_drawdown': strategy.analyzers.drawdown.get_analysis()['max']['drawdown'],
                'sharperatio': strategy.analyzers.sharperatio.get_analysis()['sharperatio']
                #'time_in_market': strategy.analyzers.time_in_market.get_analysis()['time_in_market']
            })
    
        self.runs = runs_with_metrics
        
class BacktestCollection():
    
    def __init__(self,
                 backtest_class=None,
                 path_new ='backtests_new.pkl', 
                 path_completed = 'backtests_completed.pkl', 
                 start_date = None, 
                 end_date = None, 
                 max_lookback_period=0):
        if not callable(backtest_class):
            raise ValueError("The backtest_class must be a callable class.")   
        self.backtest_class = backtest_class
        self.path_new = path_new 
        self.path_completed = path_completed
        self._load_backtests()
        self.start_date = start_date
        self.end_date = end_date
        self.max_lookback_period = max_lookback_period
        
                        
    def set_date(self, start_date=None, end_date=None, max_lookback_period=None):
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date
        if max_lookback_period:
            self.max_lookback_period = max_lookback_period
        print(f'start date: {self.start_date} end_date: {self.end_date} max_lookback_period: {self.max_lookback_period}')
        
    def summarize(self):
        data = [self.backtests_completed, self.backtests_new]
        summaries = []
        for dataset in data:    
            if dataset:
                runs = []
                for backtest in dataset:
                    if backtest.get_runs():
                        for run in backtest.get_runs():
                            runs.append(run)
    
                summary = pd.DataFrame(runs)
                summaries.append(summary)
            else:
                summaries.append(None)
        return summaries       
    
    def _load_pkl(self, path, default):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                file = pickle.load(f)
                print(f'\nLoaded file {path} from harddrive')
                return file
                
        return default
    
    def _save_pkl(self, backtests, path):
        with open(path, 'wb') as f:
            pickle.dump(backtests, f)
        print(f'\nSaved file {path} to harddrive')
            
        
    def _load_backtests(self):
        self.backtests_new = self._load_pkl(self.path_new, [])
        self.backtests_completed = self._load_pkl(self.path_completed, [])
       
    
    def save_backtests(self, clear_result=True):
        if clear_result:
            for backtest in self.backtests_new + self.backtests_completed:
                backtest.result = None
        
        self._save_pkl(self.backtests_new, self.path_new)
        self._save_pkl(self.backtests_completed, self.path_completed)
        
    def create_new_backtests(self, input_parameters):
        if not (self.start_date or self.end_date):
            print('Please first set start and end dates for backtests')
            return
        for i, parameters in enumerate(input_parameters.parameters_chunks):
            new = True
            
            # Check if new parameters are allready in backtests_completed
            if self.backtests_completed:
                for backtest_completed in self.backtests_completed:
                    if backtest_completed.get_id() == parameters.get('ID'):
                        new = False
                        print(f'Backtest with ID {parameters.get("ID")} allready in backtests_completed list')
                        
            # Check if new parameters are allready in backtests_new
            if self.backtests_new:
                for backtest_new in self.backtests_new:
                    if backtest_new.get_id() == parameters.get('ID'):
                        new = False
                        print(f'Backtest with ID {parameters.get("ID")} allready in backtests_new list')
                
            if new:
                backtest = self.backtest_class(parameters, self.start_date, self.end_date, self.max_lookback_period)
                self.backtests_new.append(backtest)  
                print(f'Added backtest {i+1} to backtests_new')

    def count_combinations(self):
        combinations = 0
        for backtest in self.backtests_new:
            combinations += backtest.count_combinations()
        print(f'Total number of runs(combinations) to backtest: {combinations}')
            
    def run_backtests(self):
        if len(self.backtests_new)==0:
            print('backtests_new is empty!')
            return
        
        self.count_combinations()    
        if not wait_for_user_confirmation():
            print('Backtests aborted by user')
            return
            
        for i, backtest in enumerate(self.backtests_new):
            backtest_count = len(self.backtests_new)
            if backtest.get_result():
                print(f'Skipping backttest {i+1}/{backtest_count}, result allready present')
                continue
            
            
            print_headline(f'Starting backtest {i+1}/{backtest_count} with {backtest.count_combinations()} run(s)\n')
            backtest.run()
            self.backtests_completed.append(self.backtests_new[i])
            print(f'\nFinished backtest {i+1} of {backtest_count}')   
            
            # Save backtests to harddrive
            self.save_backtests()  
            