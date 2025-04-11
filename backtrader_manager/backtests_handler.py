import uuid
import os
import time

import pandas as pd
import numpy as np
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
#from collections import OrderedDict
#import numbers
import quantstats_lumi as qs

from .utils import wait_for_user_confirmation, is_iterable, print_headline, _format_parameters
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
    All functions containing pass are to be defined in subclass
    Some are mandatory, some are optional
    '''
    class_alias = 'short_name_for_backtest_class'
    
    def __init__(self, input_backtest, datas=None, mode='default'):  
        print_headline('Initialising Backtest', level=2)

         # === Instantiated attributes: set directly from __init__ ===
        
        #self._name = 
        self._datas = datas
        self._input_backtest = input_backtest
        self._mode = mode
        
        # === Subclass-resolved attributes: to be defined by subclass methods ===
        
        # General backtest settings
        self._strategy_class = self._set_strategy_class() 
        self._warmup_bars = self._set_warmup_bars() # for indicators with lags
        
        # backtrader strategy/cerebro settings
        self._cash = self._set_cash()
        self._commission_value = self._set_commission_value()
        self._slippage_value = self._set_slippage_value()
        self._coc = self._set_coc()
        self._standard_sizer = self._set_standard_sizer() # dict: {sizer: None, argument1: None, argument2: ...)
        
        # backtrader analyzers and observers
        self._analyzer_collection = AnalyzerCollection()  # Instantiate the default analyzers
        self._observers = [
            #{'parameters': (bt.observers.Value)},
            #{'parameters': (bt.observers.Trades)},
        ]
               
        # ===  Outputs (populated later): ===  
        
        self._result = None # cerebro result object
        self._result_path = None  # Will store the path to the saved result

        self._meta = None
        self._metrics = None
        self._parameters = None
          
    @property
    def idx(self):
        if self._meta:
            idx = (self._meta[0]['idx'][0], None)
        else: 
            idx = None
        return idx

    @property
    def name(self):
        return self._name
    
    @property
    def input_backtest(self):
        return self._input_backtest
    
    @property
    def mode(self):
        return self._mode

    
    # -------------------------------------
    # Subclass hook methods (must override)
    # -------------------------------------
        
    def _set_strategy_class(self):
        strategy_class = None
        return strategy_class


    # ----------------------------------------------------------------
    # Subclass hook methods (optional override, elese standard values)
    # ---------------------------------------------------------------- 
        
    def _set_cash(self):
        cash_standard = 10000
        return cash_standard         
    
    def _set_commission_value(self):
        return 0

    def _set_slippage_value(self):
        return 0
          
    def _set_coc(self):
        return False
           
    def _set_warmup_bars(self):
        warmup_bars = 0
        return warmup_bars

    def _set_standard_sizer(self):
        return None
    
    # === Puplic methods, do not change in subclasses ===
    
    def get_parameter_max(self, key):
        parameter_max = max(self.input_backtest.strategy_parameters.get(key))
        return parameter_max

     
    # === Private helper methods, do not change in subclasses ===

    def _save_result_to_disk(self, base_path):
        """Save backtest result to disk in the 'pkl' subdirectory."""
        if self.result is not None:
            # Save the result to the 'pkl' subdirectory within the base path
            file_name = f"result_idx=({self.idx[0]},).pkl"
            self.result_path = os.path.join(base_path, 'pkl', file_name)
            os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
            
            with open(self.result_path, 'wb') as f:
                pickle.dump(self.result, f)
            
            self.result = None  # Free up memory after saving

    def _load_result_from_disk(self):
        """Load backtest result from disk."""
        if hasattr(self, 'result_path') and os.path.exists(self.result_path):
            with open(self.result_path, 'rb') as f:
                self.result = pickle.load(f)
        
    def _calc_buffer_start_date(self, ticker):
        data = self._datas[ticker]
            
        max_lag = self._warmup_bars
        position_start_date = data.index.get_loc(self.input_backtest.period[0])
        position_buffer_start_date = position_start_date - max_lag
        buffer_start_date = data.index[position_buffer_start_date]
        print(buffer_start_date)
        return buffer_start_date       
    
    def _set_mode(self, mode):
        self.mode = mode
        
    def _get_result(self):
        if self.result:
            return self.result
        return None
    
    def _get_input_id(self):
        input_id = self.input_backtest.input_id
        return input_id        
            
    def _get_analysis(self, analyzer_name, run_number, strat_number=0):
        if self._check_if_optimisation():
            result = self.result[run_number][strat_number]
        else:
            result = self.result[strat_number]
            
        return self._analyzer_collection.get_analysis(analyzer_name, result)
    
    def _count_combinations(self):
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

    def _run(self):
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
        standard_sizer = self._standard_sizer
        if standard_sizer:
            self.cerebro.addsizer(standard_sizer['sizer'], **{k: v for k, v in standard_sizer.items() if k != 'sizer'})
            
    def _cerebro_add_strategy(self):
        start_date=self.input_backtest.period[0]
        
        if self._check_if_optimisation():
            self.cerebro.optstrategy(
                self._strategy_class, 
                **{key.lower(): value for key, value in self.input_backtest.strategy_parameters.items()},
                start_date=start_date
            )
            print('Strategy added as optimisation backtest')
        else:
            self.cerebro.addstrategy(
                self._strategy_class, 
                **{key.lower(): value[0] for key, value in self.input_backtest.strategy_parameters.items()}, 
                start_date=start_date
            )
            
            print('Strategy added as single backtest')
            
    def _cerebro_add_datafeeds(self):
       
        print('Adding strategy datafeeds')
        cerebro = self.cerebro
        for ticker, data in self._datas.items():
            fromdate = self._calc_buffer_start_date(ticker)
            todate = self.input_backtest.period[1]
            datafeed =  bt.feeds.PandasData(dataname=data, fromdate=fromdate, todate=todate, plot=True)     
            self.cerebro.adddata(datafeed, name=ticker)
        print(f'Added all datafeeds')   
    
    def _clear(self, result=False):
        if result:
            self.result = None
        self.cerebro = None

    def _cerebro_add_analyzers(self):    
        self._analyzer_collection.add_analyzers_cerebro(self.cerebro)
        
    def _cerebro_setup_broker(self):
        cerebro = self.cerebro
        commission = self._commission_value
        slippage = self._slippage_value
        cash = self._cash
        coc = self._coc
        
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
            
            parameters_of_run = {**vars(run[0].params)}
            parameters.append(parameters_of_run)
            
            meta_of_run = {}
            meta_of_run['input_collection_name'] = input_backtest.input_collection_name
            meta_of_run['period_key'] = input_backtest.period_key 
            meta_of_run['iteration_key'] = input_backtest.iteration_key 
            meta_of_run['period'] = input_backtest.period
            meta_of_run['backtest_class'] = input_backtest.backtest_class.class_alias
            #meta_of_run['commission'] = input_backtest.commission
            #meta_of_run['slippage'] = input_backtest.slippage
            meta.append(meta_of_run)
            
            metrics_of_run = {}
            metrics_of_run.update(self._analyzer_collection.get_outputs(run[0]))
            metrics.append(metrics_of_run)

            
            
            self._meta = meta
            self._metrics = metrics 
            self._parameters = parameters
    
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
                    
    def _plot(self, ticker=None, start=None , end=None, style='candlestick'):
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
    
    def __init__(self, csv_datas_path, name = 'Backtests_Temp'):
        print_headline('Initialising Backtest Collection', level=1)  
        
        ### Fixed attributes (only set at instantiation or once): Private, some with getters ###
        self._name = name
        self._csv_datas_path = self._resolve_data_path(csv_datas_path) 
        self._datas = self._load_csv_datas() # {ticker: data_ticker.csv}
        self._base_dir = os.path.join(self._name)
        self._pkl_dir = os.path.join(self._base_dir, 'pkl')  # Directory to store both collection and result files
        self._reports_dir = os.path.join(self._base_dir, 'reports') # Directory to store reports        
        self._path = os.path.join(self._pkl_dir, 'collection.pkl')  # Path for the pkl collection file
        
        ### Allways adjustable attributes: Public or with setters and getters ###
        self._benchmark_class = None
        
        ###  Outputs (populated later): 
        self._backtest_input_collections = [] 
        self._backtests_new = []
        self._backtests_completed = [] #  
        self._backtests_temp = [] # only temporary backtests, not saved to disk
        self._backtests_temp_map = {}
        
        self._meta = []
        self._metrics = []
        self._parameters = []

        # track/map unique combinations in self._parameters to parameter_id
        self._current_id = {}
        self._param_to_id = {}  # param_tuple -> parameter_id
        self._id_to_param = {}  # parameter_id -> param_tuple
        
        self._alias_to_backtest_class = {}

        self._summary = None
        if name != 'Backtests_Temp':
            self._load()
            self.summarize()
        

    ### Puplic methods
    
    def add_backtest_input_collection(self, backtest_input_collection):
        if any(backtest_input_collection.name==existing_collection.name for existing_collection in self._backtest_input_collections):
            print(f'An backtest_input_collection with same name allready has been added')
            return
            
        self._backtest_input_collections.append(backtest_input_collection)
        backtest_input_collection._set_datas(self._datas)
        backtest_class = backtest_input_collection.backtest_class.class_alias
        if backtest_class not in self._id_to_param:
            self._param_to_id[backtest_class] = {}  # param_tuple -> parameter_id
            self._id_to_param[backtest_class] = {}  # parameter_id -> param_tuple
            self._current_id[backtest_class] = 1
    
        id_to_param = self._id_to_param
        backtest_input_collection._set_parameter_mapping(id_to_param)
        alias_to_backtest_class = self._alias_to_backtest_class
        backtest_input_collection._set_backtest_class_mapping(alias_to_backtest_class)
        
    def create_new_backtests(self, backtest_input_collection, mode='default'):
        if backtest_input_collection not in self._backtest_input_collections:
            print('First add_backtest_input_collection to backtest_collection instance')
            return
        backtest_inputs = backtest_input_collection.backtest_inputs
        backtests_new = self._backtests_new
        backtests_completed = self._backtests_completed
        
        print_headline('Creating new backtests', level=1)

        for i, backtest_input in enumerate(backtest_inputs):
            new = True
            
            if backtest_input._parent_input_id:
                copy = True
            else:
                copy = False
                
            backtest_input_id = backtest_input.input_id
            backtest_class = backtest_input.backtest_class
                        
            # Check if new parameters are allready in backtests_completed
            if backtests_completed and len(backtests_completed) > 1 and copy==False:
                for backtest_completed in backtests_completed:
                    if backtest_completed._get_input_id() == backtest_input_id:
                        new = False
                        print(f'Backtest with input_id {backtest_input_id} allready in backtests_completed list')
                        
            # Check if new parameters are allready in backtests_new
            if backtests_new:
                for backtest_new in backtests_new:
                    if backtest_new._get_input_id == backtest_input_id:
                        new = False
                        print(f'Backtest with input_id {backtest_input_id} allready in backtests_new list')
            
            if new:                    
                backtest = backtest_class(backtest_input, datas=self._datas, mode=mode)
                backtests_new.append(backtest)  
                print(f'Added backtest {i+1} to backtests_new, mode: {mode}')
                backtest_input_collection.clear_backtest_inputs()

    def run_backtests(self):
        if len(self._backtests_new)==0:
            print('backtests_new is empty!')
            return
        
        self._count_combinations()    
        if not wait_for_user_confirmation():
            print('Backtests aborted by user')
            return
        
        total_backtests = len(self._backtests_new)
        current = 1
        while self._backtests_new:
            backtest = self._backtests_new.pop(0)
            print_headline(f'Starting backtest {current}/{total_backtests} with {backtest._count_combinations()} run(s)\n')
            backtest._run()
            self._add_backtest_completed(backtest)
            print(f'\nFinished backtest {current} of {total_backtests}')   
        
            # Save backtests to hard drive
            self._save()
            # Clear completed backtest objects
            self._clear()
        
            current += 1
        
        self._save() 
        self.summarize()
    
    def summarize(self, parameters=False, metrics=True): 
            
        rows_list = []  # Initialize an empty list to store all rows of the summary df
        
        meta_list = self._meta # list of dict
        metrics_list = self._metrics # list of dict
        parameters_list = self._parameters # list of dict
               
        for i, row_meta in enumerate(meta_list):
            row = row_meta.copy()
            if metrics:
                row.update(metrics_list[i])
            if parameters:
                row.update(parameters_list[i])
            rows_list.append(row)
        
        self._meta_df = pd.DataFrame(meta_list)
        self._metrics_df = pd.DataFrame(metrics_list)
        self._parameters_df = pd.DataFrame(parameters_list)
        
        summary = pd.DataFrame(rows_list)
        
        self._summary = summary
     

    def summarize_filter_by(self, input_collection_name=None, period_key=None, iteration_key=None,
                       backtest_class=None, idx=None, parameter_id=None, show_parameters=False, **kwargs):
        
        self.summarize(show_parameters)
        summary = self._summary 
        
        # Apply filters dynamically
        filters = {
            "input_collection_name": input_collection_name,
            "period_key": period_key,
            "iteration_key": iteration_key,
            "backtest_class": backtest_class,
            "idx": idx,
            "parameter_id": parameter_id
        }
        filters.update(kwargs) 
        
        for col, value in filters.items():
            if value is not None:
                if isinstance(value, (list, tuple, set)):
                    summary = summary[summary[col].isin(value)]
                else:
                    summary = summary[summary[col] == value]

        return summary  # Return the filtered DataFrame

    def summarize_in_groups(self, groupy_cols=['input_collection_name', 'period_key', 'backtest_class', 'parameter_id'], 
                               sort_by=None, show_parameters=False, **filter_by_kwargs):
        summary = self.summarize_filter_by(show_parameters=show_parameters, **filter_by_kwargs)
        aggregation = {
            'max_drawdown': ['max', 'mean', 'std'],
            'cagr': ['mean', 'median', 'std'],
            'sharperatio': ['mean', 'std'],
            'time_in_market': ['mean', 'std']
        }
        if show_parameters:
            parameter_cols = list(self._parameters_df.columns)
            for col in parameter_cols:
                aggregation[col] = self._value
        
        summary_in_groups = (
            summary
            .groupby(groupy_cols)#[cols]
            .agg(aggregation)
        )
        
        if sort_by:
            summary_in_groups_sorted = (
                summary_in_groups
                .groupby(level=0, group_keys=False)  # Group by period_key without adding extra index levels
                .apply(lambda x: x.sort_values(by=sort_by, ascending=False))  # Sort within each period_key group
            )
            
            return summary_in_groups_sorted
        
        return summary_in_groups
            
    def copy_input_backtest(self, idx: tuple[int, int]):
        backtest = self._backtests_completed[idx[0]]
        backtest_input =  copy.deepcopy(backtest.input_backtest)
        backtest_input.set_strategy_parameters(backtest.parameters[idx[1]]) # this makes shure only the parameters of the specific run are present
        backtest_input._set_parent_path(self._path)
        backtest_input._set_parent_idx(idx) #this tracks the class_backtest_run tuple from the original source 
        return [backtest_input]
    
    def get_input_collection(self, input_collection_name):
        
        input_collection = next((input_collection for input_collection in self.input_collections if input_collection.name == input_collection_name), None)
        
        if input_collection:
            return input_collection
        else:
            print("input_collection not found")
    
    def get_backtest(self, idx):
        if idx in self._backtests_temp_map.keys():
            idx = self._backtests_temp_map[idx]
            backtest = self._backtests_temp[idx[0]]
            return backtest, idx
        else:
            return self._backtests_completed[idx[0]], idx 

    def get_benchmark_idx(self, idx):
        summary = self._summary
        mask = summary['idx'] == idx
        filtered_row = summary.loc[mask]
    
        if filtered_row.empty:
            return None  # Return None if no matching row is found
        
        # Get all column names except 'backtest_class' and 'idx'
        columns_to_match = ['input_collection_name', 'period_key', 'iteration_key']
        
        # Create mask for rows with same values in all columns except 'backtest_class'
        match_mask = (summary[columns_to_match] == filtered_row[columns_to_match].values[0]).all(axis=1)
        benchmark_class_mask = (summary['backtest_class'] == self.benchmark_class)
        idx_benchmark_series =summary.loc[match_mask & benchmark_class_mask]['idx']
        # Return filtered DataFrame with benchmark runs
        return idx_benchmark_series.item() if not idx_benchmark_series.empty else None
        
    def get_analysis(self, analyzer_name: str, idx: tuple[int, int], strat_number: int = 0):
        backtest, idx_backtest = self.get_backtest(idx)
        return backtest._get_analysis(analyzer_name, idx_backtest[1], strat_number)
    
    def get_report(self, summary_filtered):

        # Iterate over all indices
        for index, row in summary_filtered.iterrows():
            idx = row['idx']
            backtest, idx_backtest = self.get_backtest(idx) 
            idx_benchmark = self.get_benchmark_idx(idx)
            
            if idx_benchmark is not None:  # Only fetch benchmark if it exists
                benchmark, idx_benchmark = self.get_backtest(idx_benchmark)
            else:
                benchmark = None  # No benchmark found
            
            # Retrieve return series
            returns_backtest = pd.Series(backtest._get_analysis('time_return', idx_backtest[1]), name=backtest.class_alias)
            
            if benchmark:
                returns_benchmark = pd.Series(benchmark._get_analysis('time_return', idx_benchmark[1]), name=benchmark.class_alias)
            else:
                returns_benchmark = None
            
            # Prepare directory structure
            
            input_collection_name = row['input_collection_name']
            period_key = row['period_key']
            backtest_class = row['backtest_class']
            parameter_id = row['parameter_id']
            iteration_key = row['iteration_key']
            
            folder_path = os.path.join(self._reports_dir, input_collection_name)
            os.makedirs(folder_path, exist_ok=True)  # Create directory if it doesn't exist
            
            # File name format: period_key_backtest_class_parameter_id_iteration_key_idx.html
            file_name = f"{backtest_class}_parameter-id={parameter_id}_{period_key}_iteration={iteration_key}_idx={idx}.html"
            file_path = os.path.join(folder_path, file_name)
    
            # Generate report
            qs.extend_pandas()
            qs.reports.html(returns_backtest, returns_benchmark, output=file_path)
            
    
    def offload_all_results(self):
        """Save all backtest results to disk."""
        for backtest in self._backtests_completed:
            backtest._save_result_to_disk(self._base_dir)  # Use the base directory to store results

    def load_all_results(self):
        """Load all backtest results from disk."""
        for backtest in self._backtests_completed:
            backtest._load_result_from_disk()

    def clear_backtests_temp(self):
        self._backtests_temp = []
        self._backtests_temp_map = {} 
    
### rewrite plot methods
    def plot_equity_curve(self, *backtest_run_tuples, strat_number=0):
        runs_list = list(backtest_run_tuples)
        plot_data = {}
        for backtest_run_tuple in runs_list:
            analysis = self._get_analysis('equity', backtest_run_tuple, strat_number)
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
        self._backtests_completed[backtest_run_tuple[0]].plot(ticker, start, end, style)
        
    
    ### Property getters/setters ###
    @property
    def benchmark_class(self):
        return self._benchmark_class

    @benchmark_class.setter
    def benchmark_class(self, value):
        if not isinstance(value, str):
            print('Use class alias (string)')
        else:
            self._benchmark_class = value
    
    @property
    def name(self):
        return self._name
    
    @property
    def csv_datas_path(self):
        return self._csv_datas_path
    
    @property
    def datas(self):
        return self._datas

    @property
    def backtest_input_collections(self):
        return self._backtest_input_collections
    
    @property
    def backtests_new(self):
        return self._backtests_new

    @backtests_new.setter
    def backtests_new(self, value):
        if value==None or value==[]:
            value = []        
            self._backtests_new = value
        return print('backtests_new can only be set to None')
        
    @property
    def backtests_completed(self):
        return self._backtests_completed
    
    @property
    def backtests_temp(self):
        return self._backtests_temp
    
    @property
    def summary(self):
        return self._summary  

    ### Private helper methods ###
   
    def _load_csv_datas(self):
        """Loads CSV data into a dictionary of DataFrames."""
    
        datas_dict = {}
        if not os.path.exists(self._csv_datas_path):
            raise FileNotFoundError(f"CSV datas path '{self._csv_datas_path}' does not exist.")
        
        for file in os.listdir(self._csv_datas_path):
            if file.endswith('.csv'):
                ticker = file.split('.')[0]
                file_path = os.path.join(self._csv_datas_path, file)
                datas_dict[ticker] = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        if not datas_dict:
            raise ValueError("No CSV files found in the provided directory.")
        
        return datas_dict
    
        
    
    def _load(self):
        """Load the BacktestCollection instance from file."""
        loaded_data = self._load_pkl(self._path)
        if loaded_data:
            self.__dict__.update(loaded_data.__dict__)
            print(f"✅ Loaded collection from {self._path}")
        else:
            print(f"⚠️ No saved collection found at {self._path}, starting new.")

    def _save(self):
        """Save the BacktestCollection to file, offload results, and clear unnecessary data."""
        # Offload all results to disk first
        self.offload_all_results()
        
        # Deepcopy to avoid changing original during save
        copy_self = copy.deepcopy(self)
        copy_self._backtests_temp = []  # Ensure temp backtests are not saved
        
        # Clear backtests (for memory management)
        self._clear(backtests=copy_self._backtests_completed, range_to=0)  # Clears all cerebro instances since they aren't pickable
        
        # Save the collection to disk
        copy_self._save_pkl()

    def _load_pkl(self, path):
        """Load data from the given path. Returns the loaded data if found, or None if not found."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"No file found at {path} on harddrive.")
            return None

    def _save_pkl(self):
        """Save the collection object to disk in the 'pkl' subdirectory."""
        os.makedirs(self._pkl_dir, exist_ok=True)  # Ensure 'pkl' subdirectory exists
        print(f"Saving BacktestCollection to {self._path}")
        with open(self._path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Saved collection to {self._path}")


    def _clear(self, backtests=None, range_to=-2, result=False):
        """Deletes backtrader cerebro/result objects in all backtest instances up to the specified range, in order to save RAM"""
        if not backtests:
            backtests = self._backtests_completed

        # Clear the selected backtests
        limit = max(0, len(backtests) + range_to)
        for i in range(limit):
            if backtests[i]:
                backtests[i]._clear(result=result)
        
    def _count_combinations(self):
        combinations = 0
        for backtest in self._backtests_new:
            combinations += backtest._count_combinations()
        print(f'Total number of runs(combinations) to backtest: {combinations}')
        
    

    def _add_backtest_completed(self, backtest):
        
        backtest_parent_input_id = backtest.input_backtest._parent_input_id
        if backtest_parent_input_id:
            self._backtests_temp_map[backtest_parent_input_id] = (len(self._backtests_temp), 0)
            self._backtests_temp.append(backtest)
            return          
        
        backtests_completed = self._backtests_completed
            
        backtest_index = len(backtests_completed)

        # add column 'idx' to backtest.meta
        for run_index, meta_dict in enumerate(backtest._meta):
            meta_dict['idx']= (backtest_index, run_index)
            
        backtests_completed.append(backtest)
        
        backtest = self._add_parameter_id(backtest)
        self._meta += backtest._meta
        self._metrics += backtest._metrics
        self._parameters += backtest._parameters
        
    def _add_parameter_id(self, backtest):
        backtest_class = backtest.input_backtest.backtest_class
        backtest_class_alias = backtest_class.class_alias
    
        # Initialize dictionaries if not present
        if backtest_class not in self._param_to_id:
            self._param_to_id[backtest_class] = {}  # param_tuple -> parameter_id
            self._id_to_param[backtest_class] = {}  # parameter_id -> param_tuple
            self._current_id[backtest_class] = 1
    
        for i, parameters_of_run in enumerate(backtest._parameters):
            # Convert dictionary to a tuple (hashable) for comparison
            param_tuple = (backtest_class_alias, tuple(sorted(parameters_of_run.items())))
    
            # Check if this combination already exists
            if param_tuple in self._param_to_id:
                parameter_id = self._param_to_id[param_tuple]
            else:
                parameter_id = f'{backtest_class_alias}-{self._current_id[backtest_class]}'
                self._param_to_id[param_tuple] = parameter_id
                self._id_to_param[parameter_id] = (backtest_class, _format_parameters(parameters_of_run)) # store parameters in original dict format
                self._current_id[backtest_class] += 1  # Increment for next new set
    
            backtest._meta[i]['parameter_id'] = parameter_id  
        self._register_backtest_class(backtest_class)

        return backtest

    def _register_backtest_class(self, backtest_class):
        # Get the alias
        class_alias = backtest_class.class_alias
        self._alias_to_backtest_class[class_alias] = backtest_class

    def _get_backtest_class_by_alias(self, class_alias):
        # Look up the class by its alias
        return self._alias_to_backtest_class.get(class_alias)

    def _resolve_data_path(self, csv_datas_path):
        """ Resolves relative or absolute path to the correct directory """
        
        if os.path.isabs(csv_datas_path):
            # If it's an absolute path, return it directly
            return csv_datas_path
        
        # Otherwise, treat it as a relative path
        cwd_relative_path = os.path.abspath(os.path.join(os.getcwd(), csv_datas_path))
        if os.path.exists(cwd_relative_path):
            return cwd_relative_path
        
        # If path isn't found, raise an error
        raise FileNotFoundError(f"CSV data path '{csv_datas_path}' not found.")

    def _value(self, series):
        unique_vals = series.unique()
        if len(unique_vals) == 1:
            return unique_vals[0]
        else:
            return 'VARIES'  # or None, or raise an error
