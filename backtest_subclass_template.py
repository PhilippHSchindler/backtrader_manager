# Standard Library Imports
import copy
import datetime
import hashlib
import itertools
import json
import math
import os
import pickle
import time
from collections import OrderedDict
from collections.abc import Iterable

# Third-Party Imports
import backtrader as bt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import quantstats as qs
import yfinance as yf

import sys
sys.path.append('C:/Users/Phip-C/Offline Ablage/09 Coding/Backtrader')
import backtrader_manager as bm
from backtrader_manager.indicators import 

class BacktestTemplateStrategy(bm.Backtest):
            
    def set_strategy_class(self):
        strategy_class = TemplateStrategy
        return strategy_class

    def set_lag_indicators(self):
        lag_indicators = ('indicator_label',)
        return lag_indicators    
    
    def setup_cerebro(self):
        cerebro = self.cerebro
                
        # Adds a sizer
        cerebro.addsizer()
        
        # Set initial capital
        cerebro.broker.setcash(10000)
        cerebro.broker.set_coc(False) # set cheat on close
        
    def setup_broker(self):
        cerebro = self.cerebro
        cerebro.broker.setcommission(commission=0.0005) 
        cerebro.broker.set_slippage_perc(0.0001)

    def add_datafeeds(self):
        dataset_path = 'C:/Users/Phip-C/Offline Ablage/08 Trading/Historical data/...'
        # Load data
        data = pd.read_csv(data_path, parse_dates=['timestamp'], index_col=['timestamp'])

        datafeed =  bt.feeds.PandasData(dataname=data, plot=True)     
        self.cerebro.adddata(datafeed, name='ticker_name')


# Strategy 
class TemplateStrategy(bt.Strategy):
    params = (
        ('id', None),
        ....
    )
    
    def __init__(self, **kwargs):
        print('Initialising Strategy')
        super().__init__()
               
        # only pass the datafeeds for the strategy (not benchmark feeds)
        self.datas = kwargs.get('datas_strats', None)
        
        # intitialise dictionaries
        self.indicator_1 ={}
        ...
        
        # Keep track of pending orders and buy price/commission for each datafeed
        self.order = {}
        self.buyprice = {}
        self.sellprice ={}
        self.buycomm = {}
        self.sellcomm = {}
        
        self.bar_executed = {} # tracks the the index (or count) of the bar at which the buy order was executed.
        self.order_count = 0 # Tracks the total number of orders submitted
                
        # Order queue: All oreders to be placed are collected before oders are placed and sized
        self.order_queue = []
        
        
        # Initialize indicators for each datafeed
        print('\nInitializing Indicators')
        
        for data in self.datas:
            ticker = data._name

            self.indicator_1[data] = bt.indicators.ATR(data, period=14) 
            ....
            
        print('Finished \n')
                
    
    
    def prenext(self):
        self.next()
        
    def next(self):     
        
        for data in self.datas: 
              
            ticker = data._name      
            
            # Indicators
            
            
            # Buy/Sell conditions: 
            
            # Place all orders in order queue
            if len(self.order_queue) > 0:
                
                for data in self.order_queue:
                    self.buy(data=data)
                    self.log(f"BUY ORDER PLACED for {ticker}")
                                
                self.order_queue.clear()  # erases all entries in the list

