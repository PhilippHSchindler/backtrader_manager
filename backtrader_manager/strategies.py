import pandas as pd
import backtrader as bt
import datetime
import sys
sys.path.append('C:/Users/Phip-C/Offline Ablage/09 Coding/Backtrader')

# Parent class for all backtrader strategy classes
class Strategy(bt.Strategy):
   
    def __init__(self, **kwargs):
        super().__init__()
        print('Initialising Strategy')
                       
        # only pass the datafeeds for the strategy (not benchmark feeds)
        self.start_date = kwargs.get('start_date', None)
        self.start_date_reached = False
                    
        # Keep track of pending orders and buy price/commission and trade size for each datafeed
        self.order = {}
        self.buyprice = {}
        self.sellprice ={}
        self.buycomm = {}
        self.sellcomm = {}
        self.size = {} 
        
        self.bar_executed = {} # tracks the the index (or count) of the bar at which the buy order was executed.
        self.order_count = 0 # Tracks the total number of orders submitted    
        # Order queue: All oreders to be placed are collected before oders are placed and sized
        #self.order_queue = []
        
        self.initialise()
    
    def initialise(self):
        pass
    
    def prenextlogic(self):
        pass
        
    def nextlogic(self):
        pass

    
    def log(self, txt, dt=None):
         ''' Logging function '''
         dt = dt or self.datas[0].datetime.date(0)
         print(f'{dt.isoformat()} {txt}')
        
    def _check_start_date_reached(self):
         # Get the current datetime from the first data feed
        current_date = self.datas[0].datetime.datetime(0)

        # Check if we have reached the start date
        if current_date < self.start_date:
            return False
        if current_date >= self.start_date:
            if not self.start_date_reached:
                self.log('BACKTEST STARTING')  
                self.start_date_reached = True
            return True
    
    def prenext(self):
        if self._check_start_date_reached():
            input('ERROR: start_date reached before all indicators are populated.')
        else:
            self.prenextlogic()
           
    def next(self):
        if self._check_start_date_reached():
            self.nextlogic()
    
    def stop(self):
        self.log('BACKTEST ENDING')

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
           
            self.size = 0
                    
# Buy and Hold benchmark strategy

class BuyAndHoldStrategy(Strategy):
    def initialise(self):
        self.bought = False  # Track if we've already bought       
   
    def nextlogic(self):
        if not self.bought:
            self.buy()  # Buy 1 unit of the asset
            self.log(f"BUY ORDER PLACED")
            self.bought = True  # Mark that we have bought
        
       