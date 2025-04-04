import uuid
import os
import time

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Import the ticker module
import datetime
import backtrader as bt
import json
import hashlib
import math
from collections.abc import Iterable
import pickle
import itertools
import copy
from collections import OrderedDict
import math
import numbers
import quantstats_lumi as qs

from .utils import wait_for_user_confirmation, is_iterable, print_headline
from .analyzers import AnalyzerCollection

class Backtest():
    '''
    Class defines what every backtest instance looks like
    Handles default actions for cerebro:
    - sizer
    - broker
    - strategy instantiation
    - analyzers
    - observers
    All functions containing pass are defined in subclass
    Some are mandatory, some are optional
    
    
    '''
        
    def __init__(self, input_backtest, pandas_datas=None, mode='default'):  
        print_headline('Initialising Backtest', level=2)
               
        ### settings that may be variable for each new backtest instance
        self.mode = mode
        self.input_backtest = input_backtest
       
        ### fixed settings for all class instances
        
        # backtest class settings
        self.strategy_class = self._set_strategy_class()
        self.name = str(self.strategy_class.__name__).replace('.', '_')
        self.warmup_bars = self._set_warmup_bars() # for indicators with lags
        self.pandas_datas = pandas_datas
        
        # bt.cerebro standard settings
        self.analyzer_collection = AnalyzerCollection()  # Instantiate the default analyzers
        self.observers = [
            #{'parameters': (bt.observers.Value)},
            #{'parameters': (bt.observers.Trades)},
        ]
        self.standard_sizer = self._set_standard_sizer() # dict: {sizer: None, argument1: None, argument2: ...)
        
        # bt.cerebro broker settings
        self.cash = self._set_cash()
        self.commission_value = self._set_commission_value()
        self.slippage_value = self._set_slippage_value()
        self.coc = self._set_coc()
            
        ### result and metrics of backtest
        self.result = None # cerebro result object

        self.run_numbers = None
        self.meta = None
        self.metrics = None
        self.parameters = None
        
          
    # Functions to be defined in child classes(mandatory):

    def _set_strategy_class(self):
        strategy_class = self.set_strategy_class()
        return strategy_class
        
    def set_strategy_class(self):
        return None

    # Functions with standard values, may be changed in child classes(optional):
    def _set_cash(self):
        cash = self.set_cash()
        return cash

    def set_cash(self):
        return 10000        
    
    def _set_commission_value(self):
        commission_value = self.set_commission_value()
        return commission_value
    
    def set_commission_value(self):
        return 0

    def _set_slippage_value(self):
        slippage_value = self.set_slippage_value()
        return slippage_value

    def set_slippage_value(self):
        return 0
        
    def _set_coc(self):
        coc = self.set_coc()
        return coc

    def set_coc(self):
        return False
    
    def _set_warmup_bars(self):
        warmup_bars = self.set_warmup_bars()
        return warmup_bars
        
    def set_warmup_bars(self):
        warmup_bars = 0
        return warmup_bars

    def _set_standard_sizer(self):
        standard_sizer = self.set_standard_sizer()
        return standard_sizer

    def set_standard_sizer(self):
        return None
    
    # Functions that shouldn't be changed in child classes
    def get_parameter_max(self, key):
        parameter_max = max(self.input_backtest.strategy_parameters.get(key))
        return parameter_max
        
    def calc_buffer_start_date(self, ticker):
        pandas_data = self.pandas_datas[ticker]
            
        max_lag = self.warmup_bars
        position_start_date = pandas_data.index.get_loc(self.input_backtest.period[0])
        position_buffer_start_date = position_start_date - max_lag
        buffer_start_date = pandas_data.index[position_buffer_start_date]
        print(buffer_start_date)
        return buffer_start_date       
    
    def set_mode(self, mode):
        self.mode = mode
        
    def get_result(self):
        if self.result:
            return self.result
        return None
    
    def _get_idx(self):
        idx = self.input_backtest.idx
        return idx        
            
    def get_analysis(self, analyzer_name, run_number, strat_number=0):
        cerebro = self.cerebro
        runstrats = cerebro.runstrats 
        optreturn = runstrats[run_number][strat_number]                 
    
        return self.analyzer_collection.get_analysis(analyzer_name, optreturn)
    
    def count_combinations(self):
        # Prepare the values for itertools.product
        values = [
            value if isinstance(value, (list, tuple, set)) else [value]
            for value in self.input_backtest.strategy_parameters.values()
        ]
        # Use product to calculate all combinations
        return len(list(itertools.product(*values)))
                    
    def _check_if_optimisation(self):
        optimise = False
        for parameter_tuple in self.input_backtest.strategy_parameters.items():
            parameter_value = parameter_tuple[1]
            # Check if parameter is iterable, exclude strings, bytes, and other single-value-like objects
            if is_iterable(parameter_value):
                optimise = True
                break
                
        return optimise         

    def run(self):
        cerebro = bt.Cerebro(stdstats=False) 
        self.cerebro = cerebro
        self._cerebro_add_datafeeds()
        self._cerebro_add_standard_sizer()
        self._cerebro_add_strategy()
        self._cerebro_add_analyzers()        
        self._cerebro_setup_broker()
        self._cerebro_add_observers()
               
        starting_value = cerebro.broker.getvalue()
        
        if self._check_if_optimisation():
            result = cerebro.run(maxcpus=1) #plot=False 
        else:
            result = cerebro.run()
    
        self.result = result
        print(f'\nStart Portfolio Value: {starting_value} \nFinal Portfolio Value: {self.cerebro.broker.get_value()}')
        self._unpack_runs()           

    def _cerebro_add_standard_sizer(self):
        standard_sizer = self.standard_sizer
        if standard_sizer:
            self.cerebro.addsizer(standard_sizer['sizer'], **{k: v for k, v in standard_sizer.items() if k != 'sizer'})
            
    def _cerebro_add_strategy(self):
        start_date=self.input_backtest.period[0]
        
        if self._check_if_optimisation():
            self.cerebro.optstrategy(
                self.strategy_class, 
                **{key.lower(): value for key, value in self.input_backtest.strategy_parameters.items()},
                start_date=start_date
            )
            print('Strategy added as optimisation backtest')
        else:
            self.cerebro.addstrategy(
                self.strategy_class, 
                **{key.lower(): value[0] for key, value in self.input_backtest.strategy_parameters.items()}, 
                start_date=start_date
            )
            
            print('Strategy added as single backtest')
            
    def _cerebro_add_datafeeds(self):
       
        print('Adding strategy datafeeds')
        cerebro = self.cerebro
        for ticker, pandas_data in self.pandas_datas.items():
            fromdate = self.calc_buffer_start_date(ticker)
            todate = self.input_backtest.period[1]
            datafeed =  bt.feeds.PandasData(dataname=pandas_data, fromdate=fromdate, todate=todate, plot=True)     
            self.cerebro.adddata(datafeed, name=ticker)
        print(f'Added all datafeeds')   
    
    def _clear(self):
        self.result = None
        self.cerebro = None

    def _cerebro_add_analyzers(self):    
        self.analyzer_collection.add_analyzers_cerebro(self.cerebro)
        
    def _cerebro_setup_broker(self):
        cerebro = self.cerebro
        commission = self.commission_value
        slippage = self.slippage_value
        cash = self.cash
        coc = self.coc
        
        if self.input_backtest.commission:
            cerebro.broker.setcommission(commission=commission) # times 100 for % 
        if self.input_backtest.slippage:
            cerebro.broker.set_slippage_perc(slippage)

        cerebro.broker.setcash(cash)
        cerebro.broker.set_coc(coc) # set cheat on close        
                                 
    def _cerebro_add_observers(self):
        cerebro = self.cerebro

        cerebro.addobserver(bt.observers.BuySell) #Buy/Sell Markers
        cerebro.addobserver(bt.observers.DrawDown)
        #cerebro.addobserver(bt.observers.Value, plot=True, subplot=True)        
        #self.cerebro.addobserver(bt.observers.Cash)  # Cash
        #self.cerebro.addobserver(bt.observers.Trades)  # 
    
    def _is_nested_list(self, obj):
        if isinstance(obj, list) and any(isinstance(item, list) for item in obj):
            return True
        return False

    def _unpack_runs(self):
        input_backtest = self.input_backtest

        run_numbers = []
        meta = []
        metrics = []
        parameters = []
        
        runs = []
        if self._is_nested_list(self.result):
            for run in self.result:
                runs.append(run)  
        else:
            runs = [self.result]           
        
        for i, run in enumerate(runs):
            run_number_of_run = {}
            run_number_of_run['run_number'] = i
            run_numbers.append(run_number_of_run)
            
            meta_of_run = {}
            meta_of_run['id'] = self._calc_id() #unique id for the combination
            meta_of_run['input_collection_name'] = input_backtest.input_collection_name
            meta_of_run['period_key'] = input_backtest.period_key 
            meta_of_run['iteration_key'] = input_backtest.iteration_key 
            meta_of_run['period'] = input_backtest.period
            meta_of_run['backtest_class'] = input_backtest.backtest_class
            meta.append(meta_of_run)
            
            metrics_of_run = {}
            metrics_of_run.update(self.analyzer_collection.get_outputs(run[0]))
            metrics.append(metrics_of_run)

            parameters_of_run = {**vars(run[0].params)}
            parameters_of_run['commission'] = input_backtest.commission
            parameters_of_run['slippage'] = input_backtest.slippage
            parameters.append(parameters_of_run)
    
        self.meta = meta
        self.metrics = metrics 
        self.parameters = parameters
    
    def _calc_id(self):
        return 0
    
    def _deactivate_plots(self):
        cerebro = self.cerebro
    
        # Disable plotting for all data feeds
        for data in cerebro.datas:
            data.plotinfo.plot = False
    
    def _activate_plot(self, ticker=None):
        if ticker:         
            data_dict = {data._name: data for data in self.cerebro.datas}
            data = data_dict.get(ticker)            
            
        else:
            data = self.cerebro.datas[0]

        data.plotinfo.plot = True    
        # Set the selected data feed as the plot master so that observers use it
        data.plotinfo.plotmaster = data
            
        
        
    
    def _deactivate_observers(self):
        # Check if optimization mode
        cerebro = self.cerebro
        runstrats = cerebro.runstrats 
    
        # Disable observer plots
        for optreturn in runstrats:
            for strat in optreturn:  # Each `strat` is an instance of the strategy
                for observer in strat.observers:
                    observer.plotinfo.plot = False     
        
    
    def _activate_observer_plot(self, observer_class):
        cerebro = self.cerebro
        
        runstrats = cerebro.runstrats 
    
        for optreturn in runstrats:
            for strat in optreturn:  # Each `strat` is an instance of the strategy
                for observer in strat.observers:
                    if isinstance(observer, observer_class):
                        observer.plotinfo.plot = True  
                    
    def plot(self, ticker=None, start=None , end=None, style='candlestick'):
        if self._check_if_optimisation():
            print('Plottimg is disabled for optimisation backtests')
            return
            
        cerebro = self.cerebro
        
        self._deactivate_plots()
        self._activate_plot(ticker)       

        self._deactivate_observers()
        self._activate_observer_plot(bt.observers.BuySell)   
              
        if style == 'line':
            cerebro.plot(
                iplot=False, 
                style=style, 
                grid=False,            # Remove background grid
                volume=True,          # Show volume bars
                start=start or self.start_date, 
                end=end or self.end_date)
        else:    
            cerebro.plot(
                iplot=False, 
                style=style, 
                barup='green',         # Color for bullish candles
                bardown='red',        # Color for bearish candles
                barupfill=False,       # Fill up bars with color
                bardownfill=False,     # Fill down bars with color
                grid=False,            # Remove background grid
                volume=True,          # Show volume bars
                volup='#00FF0020',         # Transparent green volume bars (50% alpha)
                voldown='#FF000020',       # Transparent red volume bars (50% alpha)
                start=start or self.start_date, 
                end=end or self.end_date)
            
class BacktestCollection():
    
    def __init__(self, csv_data_path, name = 'Backtests_Temp'):
        print_headline('Initialising Backtest Collection', level=1)  
        
        ### settings that may be variable for each new instance        
        self.path = name + '.pkl'
        self.train_test_periods = None # every backtest input collection that is added must have the same train_test_periods
        
        ### fixed settings for all class instances
        self.csv_data_path = csv_data_path
        self.pandas_datas = self._load_csv_datas() # {ticker: data_ticker.csv}

        self.backtest_input_collections = []
        self.backtests_new = []
        self.backtests_completed = [None] #  0 is reserved for benchmark backtest
        self.backtests_temp = [] # only temporary backtests, not saved to disk
        self.backtests_temp_map = {}

        
        self.meta = []
        self.metrics = []
        self.parameters = []
        
        if name != 'Backtests_Temp':
            self._load()
        self.summary = None

    def _load_csv_datas(self):
        pandas_datas = {}
        csv_data_path = self.csv_data_path
        if not csv_data_path:
            return pandas_datas
        # Loop through all files in the directory
        for filename in os.listdir(csv_data_path):
            if filename.endswith(".csv"):  # Ensure only CSV files are processed
                file_path = os.path.join(csv_data_path, filename)  # Get full file path
                key_name = os.path.splitext(filename)[0]  # Remove '.csv' from filename
        
            pandas_datas[key_name] = pd.read_csv(file_path, parse_dates=['timestamp'], index_col=['timestamp'])
        
        return pandas_datas
        
    def summarize(self, parameters=False, metrics=True): 
            
        rows_list = []  # Initialize an empty list to store all rows of the summary df
        meta_list = self.meta # list of dict
        metrics_list = self.metrics # list of dict
        parameters_list = self.parameters # list of dict
               
        for i, row_meta in enumerate(meta_list):
            row = row_meta.copy()
            if metrics:
                row.update(metrics_list[i])
            if parameters:
                row.update(parameters_list[i])
            rows_list.append(row)
        
        summary = pd.DataFrame(rows_list)
        
        self.summary = summary
    
        return summary  # Return 

    def copy_input_backtest(self, idx: tuple[int, int]):
        backtest = self.backtests_completed[idx[0]]
        backtest_input =  copy.deepcopy(backtest.input_backtest)
        backtest_input.set_strategy_parameters(backtest.parameters[idx[1]]) # this makes shure only the parameters of the specific run are present
        backtest_input._set_parent_path(self.path)
        backtest_input._set_parent_idx(idx) #this tracks the class_backtest_run tuple from the original source 
        return [backtest_input]
    
    def get_backtest(self, idx: tuple[int, int]):
        if idx in self.backtests_temp_map.keys():
            idx = self.backtests_temp_map[idx]
            backtest = self.backtests_temp[idx[0]]
            return backtest, idx
        else:
            return self.backtests_completed[idx[0]], idx         
        
    def get_analysis(self, analyzer_name: str, idx: tuple[int, int], strat_number: int = 0):
        backtest, idx_backtest = self.get_backtest(idx)
        return backtest.get_analysis(analyzer_name, idx_backtest[1], strat_number)

    def get_reports(self, idx):
        backtest, idx_backtest = self.get_backtest(idx)
        benchmark, idx_benchmark = self.get_backtest((0, 0))
                   
        returns_backtest = backtest.get_analysis('time_return', idx_backtest[1])
        returns_backtest =  pd.Series(returns_backtest)
        returns_backtest.name = backtest.name 
        
        if benchmark:
            returns_benchmark = benchmark.get_analysis('time_return', 0)
            returns_benchmark =  pd.Series(returns_benchmark)
            returns_benchmark.name = benchmark.name 
        else:
            returns_benchmark = None
        
        qs.extend_pandas()
        
        path = self.path.split(".")[0]
        file_name = f'{path}_{str(idx)}.html'
        qs.reports.html(returns_backtest, returns_benchmark, output=file_name)
        
    def _load_pkl(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                file = pickle.load(f)
                return file
        print(f'\nNo file found for {path} on harddrive')        
        return None
    
    def _save_pkl(self):
        print(f"Saving object to {self.path}: {self}")
        with open(self.path, 'wb') as f:
            pickle.dump(self, f)
        print(f'\nSaved file {self.path} to harddrive')
            
        
    
    def _load(self):
        """Load the current class instance from pickle file or create a new one."""
        loaded_data = self._load_pkl(self.path)
        if loaded_data is not None:
            self.__dict__.update(loaded_data.__dict__)  # Update the current instance with the loaded data
            print(f"Loaded {self.path} backtest collection successfully")
        else:
            print(f"No saved backtest collection found, initialised new instance")

    
    def _save(self, clear=True):
        copy_self = copy.deepcopy(self)
        copy_self.backtests_tmep = []
        if clear:
            self._clear(backtests=copy_self.backtests_completed, range_to=0)
        copy_self._save_pkl()
    
    def _clear(self, backtests=None, range_to=-2):
        """Clears all backtests up to the specified range"""
        if not backtests:
            backtests = self.backtests_completed

        # Clear the selected backtests
        limit = max(0, len(backtests) + range_to)
        for i in range(limit):
            if backtests[i]:
                backtests[i]._clear()
        
    def clear_backtests_temp(self):
        self.backtests_temp = []
        self.backtests_temp_map = {}
        
    def add_backtest_input_collection(self, backtest_input_collection):
        if backtest_input_collection not in self.backtest_input_collections:
            self.backtest_input_collections.append(backtest_input_collection)
        else:
            print('the backtest_input_collection has been allready added in the past')
            
    def create_new_backtests(self, backtest_input_collection, mode='default'):
        if backtest_input_collection not in self.backtest_input_collections:
            print('First add_backtest_input_collection to backtest_collection instance')
            return
        backtest_inputs = backtest_input_collection.backtest_inputs
        backtests_new = self.backtests_new
        backtests_completed = self.backtests_completed
        
        print_headline('Creating new backtests', level=1)

        for i, backtest_input in enumerate(backtest_inputs):
            new = True
            
            if backtest_input._parent_idx:
                copy = True
            else:
                copy = False
                
            backtest_id = backtest_input.idx
            backtest_class = backtest_input.backtest_class
                        
            # Check if new parameters are allready in backtests_completed
            if backtests_completed and len(backtests_completed) > 1 and copy==False:
                for backtest_completed in backtests_completed:
                    if backtest_completed._get_idx() == backtest_id:
                        new = False
                        print(f'Backtest with id {backtest_id} allready in backtests_completed list')
                        
            # Check if new parameters are allready in backtests_new
            if backtests_new:
                for backtest_new in backtests_new:
                    if backtest_new._get_idx == backtest_id:
                        new = False
                        print(f'Backtest with ID {backtest_id} allready in backtests_new list')
            
            if new:                    
                backtest = backtest_class(backtest_input, pandas_datas=self.pandas_datas, mode=mode)
                backtests_new.append(backtest)  
                print(f'Added backtest {i+1} to backtests_new, mode: {mode}')
                backtest_input_collection.clear_backtest_inputs()

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
        
        total_backtests = len(self.backtests_new)
        current = 1
        while self.backtests_new:
            backtest = self.backtests_new.pop(0)
            print_headline(f'Starting backtest {current}/{total_backtests} with {backtest.count_combinations()} run(s)\n')
            backtest.run()
            self._add_backtest_completed(backtest)
            print(f'\nFinished backtest {current} of {total_backtests}')   
        
            # Save backtests to hard drive
            self._save()
            # Clear completed backtest objects
            self._clear()
        
            current += 1
        
        self._save()  

    

    def _add_backtest_completed(self, backtest):
        
        backtest_parent_idx = backtest.input_backtest._parent_idx
        if backtest_parent_idx:
            self.backtests_temp_map[backtest_parent_idx] = (len(self.backtests_temp), 0)
            self.backtests_temp.append(backtest)
            return          
        
        backtests_completed = self.backtests_completed
        benchmark = backtest.input_backtest.benchmark    
       
        if benchmark:
            backtest_index = 0
        else:         
            backtest_index = len(backtests_completed)
        
        for run_index, meta_dict in enumerate(backtest.meta):
            meta_dict['id']= (backtest_index, run_index)
            
        if benchmark:
            backtests_completed[0] = backtest
        else:          
            backtests_completed.append(backtest)
        
        self.meta += backtest.meta
        self.metrics += backtest.metrics
        self.parameters += backtest.parameters

    def plot_equity_curve(self, *backtest_run_tuples, strat_number=0):
        runs_list = list(backtest_run_tuples)
        plot_data = {}
        for backtest_run_tuple in runs_list:
            analysis = self.get_analysis('equity', backtest_run_tuple, strat_number)
            plot_data[backtest_run_tuple] = analysis

        plt.figure(figsize=(15, 8))
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.title('Equity curves')
        plt.grid(True)
        
        for backtest_run_tuple, analysis in plot_data.items():
            dates = analysis['dates']
            values = analysis['values']
            label = str(backtest_run_tuple)
            plt.plot(dates, values, label=label)
        
        plt.legend()    
        plt.show()
            
    def plot(self, backtest_run_tuple, ticker=None, start=None , end=None, style='candlestick'):
        self.backtests_completed[backtest_run_tuple[0]].plot(ticker, start, end, style)
        
        
            