import pandas as pd
import datetime
import backtrader as bt
import sys
import backtrader_manager as bm
from backtrader_manager import BuyAndHoldStrategy
    
# Parent class for all Bitcoin strategies backtests
class BacktestBitcoinStrategy(bm.Backtest):
        
    def _set_standard_sizer(self):
        standard_sizer = {'sizer': bt.sizers.PercentSizer, 'percents': 90}
        return standard_sizer
        
    def _set_commission_value(self):
        commission = 0.0005
        return commission

    def _set_slippage(self):
        slippage = 0.0001
        return slippage

        
########################################################
# Benchmark

class BenchmarkBitcoinStrategy(BacktestBitcoinStrategy):
    class_alias = "buy_hold"
    def _set_strategy_class(self):
            strategy_class = BuyAndHoldStrategy
            return strategy_class

    
#######################################################

# Bitcoin 01
# Backtest class

class BacktestBitcoinStrategy_01(BacktestBitcoinStrategy):
    class_alias = "strat_01"
    
    def _set_strategy_class(self):
        strategy_class = BitcoinStrategy_01
        return strategy_class

    def _set_warmup_bars(self):
        warmup_bars = max(self.get_parameter_max('macd_slow'), self.get_parameter_max('macd_fast')) + self.get_parameter_max('macd_signal')
        return warmup_bars    
        
# Strategy class
class BitcoinStrategy_01(bm.Strategy):
    params = (
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal',9)
    )
    
    def initialise(self):
        
        # intitialise indicators
        self.macd = bt.indicators.MACD(
            self.datas[0].close, 
            period_me1=self.p.macd_fast, 
            period_me2=self.p.macd_slow, 
            period_signal=self.p.macd_signal)
    
    def nextlogic(self):
        
        macd = self.macd.macd[0]        # MACD line
        signal = self.macd.signal[0]    # Signal line
        
        # Simple strategy: Buy when MACD crosses above signal line, sell when it crosses below
        if macd > signal and not self.position:
            self.buy()  # Buy when MACD crosses above Signal
            self.log(f"BUY ORDER PLACED")
        elif macd < signal and self.position:
            self.sell()  # Sell when MACD crosses below Signal

# Bitcoin 02: with atr filter
# Backtest class
class BacktestBitcoinStrategy_02(BacktestBitcoinStrategy):
    class_alias = "strat_02"
    
    def _set_strategy_class(self):
        strategy_class = BitcoinStrategy_02
        return strategy_class
        
    def _set_warmup_bars(self):
        max_macd = max(self.get_parameter_max('macd_slow'), self.get_parameter_max('macd_fast')) + self.get_parameter_max('macd_signal')
        warmup_bars = max(max_macd, self.get_parameter_max('atr_period'))
        return warmup_bars
         
class BitcoinStrategy_02(bm.Strategy):
    params = (
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal',9),
        ('atr_period', 14),
        ("atr_threshold", 0.01) 
    )
    
    def initialise(self):
        
        # intitialise indicators
        self.macd = bt.indicators.MACD(
            self.datas[0].close, 
            period_me1=self.p.macd_fast, 
            period_me2=self.p.macd_slow, 
            period_signal=self.p.macd_signal)
        self.atr = bt.indicators.ATR(period=self.params.atr_period)
        # Normalized ATR (ATR % of Close Price)
        self.atr_pct = self.atr / self.data.close
        
    
    def nextlogic(self):
        atr_pct = self.atr_pct[0]
        macd = self.macd.macd[0]        # MACD line
        signal = self.macd.signal[0]    # Signal line     
        
        # Prevent trading if ATR is too low (market is in consolidation)
        if atr_pct < self.params.atr_threshold:
            return  
            
        # Simple strategy: Buy when MACD crosses above signal line, sell when it crosses below
        if macd > signal and not self.position:
            self.buy()  # Buy when MACD crosses above Signal
            self.log(f"BUY ORDER PLACED")
        elif macd < signal and self.position:
            self.sell()  # Sell when MACD crosses below Signal

# Bitcoin 03
# Backtest class
class BacktestBitcoinStrategy_03(BacktestBitcoinStrategy):
    class_alias = "strat_03"
    
    def _set_strategy_class(self):
        strategy_class = BitcoinStrategy_03
        return strategy_class

    def _set_warmup_bars(self):
        macd_max = max(self.get_parameter_max('macd_slow'), self.get_parameter_max('macd_fast')) + self.get_parameter_max('macd_signal')
        warmup_bars = max(macd_max, self.get_parameter_max('ma_long'))
        return warmup_bars

class BitcoinStrategy_03(bm.Strategy):
    params = (
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal',9),
        ('ma_short', 50),
        ('ma_long', 200),        
    )
    
    def initialise(self):
        
        # intitialise indicators
        self.macd = bt.indicators.MACD(
            self.datas[0].close, 
            period_me1=self.p.macd_fast, 
            period_me2=self.p.macd_slow, 
            period_signal=self.p.macd_signal)
        self.ma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.ma_short)
        self.ma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.ma_long)
        
        
    def nextlogic(self):
        
        macd = self.macd.macd[0]        # MACD line
        signal = self.macd.signal[0]    # Signal line     
        ma_short = self.ma_short.sma[0]
        ma_long = self.ma_long.sma[0]

        if ma_short < ma_long:
            return
            
        # Simple strategy: Buy when MACD crosses above signal line, sell when it crosses below
        if macd > signal and not self.position:
            self.buy()  # Buy when MACD crosses above Signal
            self.log(f"BUY ORDER PLACED")
        elif macd < signal and self.position:
            self.sell()  # Sell when MACD crosses below Signal
    

# Bitcoin 04
# Backtest class
class BacktestBitcoinStrategy_04(BacktestBitcoinStrategy):
    class_alias = "strat_04"
    
    def _set_strategy_class(self):
        strategy_class = BitcoinStrategy_04
        return strategy_class

    def _set_warmup_bars(self):
        macd_max = max(self.get_parameter_max('macd_slow'), self.get_parameter_max('macd_fast')) + self.get_parameter_max('macd_signal')
        warmup_bars = max(macd_max, self.get_parameter_max('ma_long'), self.get_parameter_max('atr_period'))
        return warmup_bars
        
        return warmup_bars   

class BitcoinStrategy_04(bm.Strategy):
    params = (
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal',9),
        ('atr_period', 14),
        ("atr_threshold", 0.01),
        ('ma_short', 50),
        ('ma_long', 200),        
    )
    
    def initialise(self):
        
        # intitialise indicators
        self.macd = bt.indicators.MACD(
            self.datas[0].close, 
            period_me1=self.p.macd_fast, 
            period_me2=self.p.macd_slow, 
            period_signal=self.p.macd_signal)
        self.atr = bt.indicators.ATR(period=self.params.atr_period)
        # Normalized ATR (ATR % of Close Price)
        self.atr_pct = self.atr / self.data.close
        self.ma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.ma_short)
        self.ma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.ma_long)
        
    
    def nextlogic(self):
        atr_pct = self.atr_pct[0]
        macd = self.macd.macd[0]        # MACD line
        signal = self.macd.signal[0]    # Signal line     
        ma_short = self.ma_short.sma[0]
        ma_long = self.ma_long.sma[0]

        if ma_short < ma_long:
            return
        
        # Prevent trading if ATR is too low (market is in consolidation)
        if atr_pct < self.params.atr_threshold:
            return  
            
        # Simple strategy: Buy when MACD crosses above signal line, sell when it crosses below
        if macd > signal and not self.position:
            self.buy()  # Buy when MACD crosses above Signal
            self.log(f"BUY ORDER PLACED")
        elif macd < signal and self.position:
            self.sell()  # Sell when MACD crosses below Signal
    