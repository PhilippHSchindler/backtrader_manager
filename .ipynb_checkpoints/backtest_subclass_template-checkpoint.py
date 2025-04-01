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
from backtrader_manager.indicators import AnchoredVWAP, TrendBasedOnMA

class BacktestEarningsStrategy(bm.Backtest):
            
    def set_strategy_class(self):
        strategy_class = BitcoinStrategy
        return strategy_class

    def set_lag_indicators(self):
        lag_indicators = ('MA_SHORT', 'MA_LONG')
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

    def add_benchmark_datafeeds(self):
        dataset_path = 'C:/Users/Phip-C/Offline Ablage/08 Trading/Historical data/US historical stock prices with earnings data'
        backtrader_path = os.path.join(dataset_path, 'backtrader')

        # add SPY ETF
        ticker = 'SPY'
        path = os.path.join(backtrader_path, f"{ticker}.csv")
        benchmark = pd.read_csv(path, parse_dates=True, index_col=['date'])  # Load from disk
        benchmark = benchmark[self.buffer_start_date : self.end_date]
            
    
        datafeed = bt.feeds.PandasData(dataname=benchmark, plot=True)
        self.cerebro.adddata(datafeed, name=ticker)
        
        
    def add_datafeeds(self):
        # paths to data
        dataset_path = 'C:/Users/Phip-C/Offline Ablage/08 Trading/Historical data/US historical stock prices with earnings data'
        backtrader_path = os.path.join(dataset_path, 'backtrader')
        tickers_path = os.path.join(backtrader_path, 'tickers.json')
        
        # load raw data
        with open(tickers_path, 'r') as file:
            tickers = json.load(file)
    
        min_date = []
        max_date = []
        for i, ticker in enumerate(tickers):
            path = os.path.join(backtrader_path, f'{ticker}.csv')
            data_symbol = pd.read_csv(path, parse_dates=['date'], index_col=['date'])
            data_symbol = data_symbol[self.buffer_start_date : self.end_date]
            
            # List columns to ignore
            ignore_columns = ["surprise", "release_time"]

            # Select only the columns that should be checked for NaN
            columns_to_check = [col for col in data_symbol.columns if col not in ignore_columns]

            # Check if any NaN exists in the selected columns
            if data_symbol[columns_to_check].isnull().values.any() or (data_symbol[columns_to_check] == 0).values.any():
                print(f"Skipping {ticker}: contains NaN or 0 values in OHLC or Volume.")
                continue

                
            # don't load data that is shorter than sma periods
            if len(data_symbol) < max(*self.parameters['MA_SHORT'], *self.parameters['MA_LONG']):
                print(f"Skipping {ticker}: insufficient data.")
                continue
        
            if i == 2:
                if self.mode=='test':
                    break
                else:
                    pass
                
        
            min_date.append(data_symbol.index.min())
            max_date.append(data_symbol.index.max())
            datafeed = OhlcEarningsSuprise(dataname=data_symbol, plot=True)     
            self.cerebro.adddata(datafeed, name=ticker)


# Datafeed Class¶
class OhlcEarningsSuprise(bt.feeds.PandasData):
    # Add 'surprise' to the standard data feed lines
    lines = ('surprise',)
    
    # Specify the default dataframe column for the new line
    params = (('surprise', 'surprise'),)

# Strategy 
class EarningsStrategy(bt.Strategy):
    params = (
        ('id', None),
        ('ma_short', 50),
        ('ma_long', 200),
        ('trend_dir', -1),
        ('hold_days', 2),  # Hold for specified days after entering
        ('threshold', 0),  # Threshold for earnings surprise
        ('stop_loss_multiplyer', None), # by default no stop loss 
        ('vwap_exit', True)
    )

    def log(self, txt, dt=None):
         ''' Logging function '''
         dt = dt or self.datas[0].datetime.date(0)
         print(f'{dt.isoformat()} {txt}')
    
    def __init__(self, **kwargs):
        print('Initialising Strategy')
        super().__init__()
               
        # only pass the datafeeds for the strategy (not benchmark feeds)
        self.datas = kwargs.get('datas_strats', None)
        
        # intitialise dictionaries
        self.atr_indicator ={}
        self.trend_indicator = {}
        self.earnings_indicator = {}
        self.anchored_vwap = {}
        self.adx = {}
        self.rsi = {}
        self.williams_r = {}
        
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

            self.atr_indicator[data] = bt.indicators.ATR(data, period=14)
            self.atr_indicator[data].plotinfo.plot = False
            self.earnings_indicator[data] = EarningsIndicator(data, threshold=self.p.threshold)
            self.earnings_indicator[data].plotinfo.plot = False
            self.trend_indicator[data] = TrendBasedOnMA(data, ma_short=self.p.ma_short, ma_long=self.p.ma_long)
            self.trend_indicator[data].plotinfo.plot = False
            self.anchored_vwap[data] = AnchoredVWAP(data)
            #self.williams_r[data] = bt.indicators.WilliamsR(data, period=14)
            #self.adx[data] = bt.indicators.ADX(data)
            #self.rsi[data] = bt.indicators.RSI(data)
             
            
        print('Finished \n')
                
    
    def notify_order(self, order): # is called automatically everytime the status of an order changed
        
        if order.status in [order.Submitted]:
            if order.exectype == bt.Order.Stop:
                self.log(f'STOP LOSS ORDER SUBMITTED for {order.data._name}: Price: {order.price:.2f}')
                
            else: 
                self.order_count += 1
                self.order[order.data] = []
            
            self.order[order.data].append(order)
                    
        elif order.status in [order.Accepted]:
            pass
                    
        
        elif order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY ORDER EXECUTED for %s: Price: %.2f, Value: %.2f, Comm %.2f' %
                    (order.data._name,
                     order.executed.price,
                     order.executed.value,
                     order.executed.comm))
                # Add stop_loss order after the buy order was executed
                if self.p.stop_loss_multiplyer:
                    stop_price = order.executed.price - self.p.stop_loss_multiplyer * self.atr_indicator[order.data][0]
                    self.sell(data=order.data, exectype=bt.Order.Stop, price=stop_price)
               
                self.buyprice[order.data] = order.executed.price
                self.buycomm[order.data] = order.executed.comm
                self.bar_executed[order.data] = len(self)
            
            else:  # Sell
                if order.exectype == bt.Order.Stop:
                    txt = 'STOP LOSS ORDER'
                else:
                    txt = 'SELL ORDER'
                self.log(
                    f'{txt} EXECUTED for {order.data._name}: '
                    f'Price: {order.executed.price:.2f}, '
                    f'Value: {order.executed.value:.2f}, '
                    f'Comm: {order.executed.comm:.2f}'
                )    
                self.sellprice[order.data] = order.executed.price
                self.sellcomm[order.data] = order.executed.comm
                
                # reset Vwap Indicator
                self.anchored_vwap[order.data].reset()
                
            
            self.order[order.data] = []
        
        elif order.status in [order.Margin, order.Rejected]:
            self.log(f'ORDER Canceled/Margin/Rejected for {order.data._name}')
            self.order[order.data] = []            
        
        elif order.status in [order.Canceled]:
            if order.exectype == bt.Order.Stop:
                self.log(f'STOP LOSS ORDER Canceled for {order.data._name}')
                self.order[order.data] = []
            
    def notify_trade(self, trade):
        if trade.isopen:
            self.size = trade.size # when trade is closed the size is set to 0 by backtrader
                                                
        if trade.isclosed:
            entry_price = trade.price
            exit_price = self.sellprice[trade.data]
            value = entry_price * abs(self.size)
            entry_date = bt.num2date(trade.dtopen).date()
            exit_date = bt.num2date(trade.dtclose).date()
            entry_date_pd = pd.to_datetime(entry_date)
            exit_date_pd = pd.to_datetime(exit_date)

             # Retrieve the bar where the trade was executed from the bar_executed dictionary
            entry_bar = self.bar_executed.get(trade.data, None)

            if entry_bar is not None:
                hold_bars = len(self) - entry_bar  # Calculate the number of bars the trade was held
            else:
                hold_bars = 0  # If no entry bar found, it should never happen

                        
            #self.log(f'TRADE {trade.data._name}: '
            #         f'Entry price {entry_price:.2f}, Exit price {exit_price:.2f}, '
            #         f'Entry date {entry_date}, Exit date {exit_date}, '
            #         f'Value {value:.2f}, Pnlcomm {trade.pnlcomm:.2f}, Commission {trade.commission:.2f}, '
            #         f'Hold bars {hold_bars}'                    
            #)
           
            self.size = 0

    def prenext(self):
        self.next()
        
    def next(self):     
        
        for data in self.datas: 
              
            ticker = data._name      
            
            # Indicators
            trend = np.sign(self.trend_indicator[data].trend[0]) # Trend indicator
            surprise_exceeds_threshold = self.earnings_indicator[data].lines.exceeds_threshold[0] 
            # Manually force update of vwap indicator (doesnt work otherwise)
            if self.getposition(data):
                self.anchored_vwap[data].update()
            typical_price = self.anchored_vwap[data].typical_price[0]
            avwap = self.anchored_vwap[data].avwap[0]
            
            # Buy condition: 
            if trend == self.params.trend_dir and surprise_exceeds_threshold:
                # Check if not already open position for the datafeed 
                if not self.getposition(data):  
                    # Check if an order is pending for the datafeed
                    if not self.order.get(data): 
                        self.order_queue.append(data)
                        self.anchored_vwap[data].set_anchor_bar(len(self)) # sets the anchored vwap
                        
            # Sell after hold_days if holding a position and price under vwap
            if self.getposition(data):
                should_sell = False

                # 
                hold_days = len(self) - self.bar_executed[data]                
                
                if hold_days >= self.params.hold_days:
                    if self.params.vwap_exit:
                        if data.close[0] < avwap:
                            should_sell = True
                    else:
                        should_sell = True                        
                    
                    if should_sell:
                        self.sell(data=data, exectype=bt.Order.Close)
                        self.log(f"SELL ORDER PLACED for {ticker}")
                        # Chancel open stop loss orders
                        for order in self.order[data]:
                            if order.status == bt.Order.Submitted or order.status == bt.Order.Accepted:
                                self.cancel(order)
                                
            # Place all orders in order queue
            if len(self.order_queue) > 0:
                
                for data in self.order_queue:
                    self.buy(data=data)
                    self.log(f"BUY ORDER PLACED for {ticker}")
                                
                self.order_queue.clear()  # erases all entries in the list

# Sizer pretends that we place order on the market open price 
class MultiAssetSizer(bt.Sizer):
    params = (('perc', 0.9),)
       
    def _getsizing(self, comminfo, cash, data, isbuy):
        strategy = self.strategy
        order_queue = strategy.order_queue
                
        # for BUY
        if isbuy and data in order_queue:
            available_cash = self.broker.get_cash() * self.params.perc
            num_orders = len(order_queue)
            cash_per_datafeed = available_cash / num_orders
            price = data.close[0]
            if price == 0:
                print(f"Data {data._name} has a zero price, skipping sizing.")
                return 0
            quantity = cash_per_datafeed // price

        # for SELL
        else:
            position = self.strategy.getposition(data)
            if position:  # If there's a position, return the size of the position
                quantity = abs(position.size)  # Ensure that we sell the entire position
            else:
                quantity = 0  # No position to sell
        return quantity

# Indicators


# EarningsIndicator
class EarningsIndicator(bt.Indicator):
    lines = (('exceeds_threshold'),)
    params = (('threshold', 0.0),)  # Default threshold for surprise

    def __init__(self, threshold=None):
        super().__init__()
        #self.data = data
        if threshold is not None:
            self.params.threshold = threshold          
            
    def log(self, txt, dt=None):
         ''' Logging function '''
         dt = dt or self.datas[0].datetime.date(0)
         print(f'{dt.isoformat()} {txt}')
         
    def next(self):                     
        surprise = self.data.surprise[0]
        
        # Calculate indicator output: exceeds_threshold
       
        if math.isnan(surprise) or math.isinf(surprise):  
            self.lines.exceeds_threshold[0] = 0
            return  # Exit early to avoid further checks
            
        # for not negative threshold and suprise greater threshold
        if self.params.threshold >= 0:
            if surprise >= self.params.threshold:
                self.lines.exceeds_threshold[0] = 1   
            else:
                self.lines.exceeds_threshold[0] = 0                  
        # for negative threshold and suprise smaller threshold
        elif self.params.threshold < 0:
            if surprise < self.params.threshold:
                self.lines.exceeds_threshold[0] = -1                     
            else:
                self.lines.exceeds_threshold[0] = 0                               
        else:
            self.lines.exceeds_threshold[0] = 0

