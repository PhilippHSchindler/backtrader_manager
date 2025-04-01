import backtrader as bt
import numpy as np


class AnchoredVWAP(bt.Indicator):
    lines = ('avwap', 'typical_price')
    params = (('anchor_bar', None),)
    
    # Disable plotting 
    plotlines = dict(
        typical_price=dict(visible=False)
    )

    def __init__(self):
        # Set the default anchor from parameters
        self.anchor_bar = self.p.anchor_bar
        # Plot the indicator on the same panel as the price
        self.plotinfo.plotmaster = self.data
        self.plotinfo.subplot = False

    def set_anchor_bar(self, anchor_bar):
        """Change the anchor for future calculations."""
        self.anchor_bar = anchor_bar
        self.update()

    def log(self, txt, dt=None):
         dt = dt or self.datas[0].datetime.date(0)
         print(f'{dt.isoformat()} {txt}')

    def reset(self):
        self.set_anchor_bar(None)
        
    def update(self):
        
        current_abs_index = len(self.data) - 1
        # If we haven't reached the anchor yet, default to None
        if self.anchor_bar is None or current_abs_index < self.anchor_bar:  
            self.lines.avwap[0] = float('nan') #self.data.close[0]
            self.lines.typical_price[0] = float('nan')
            return

        # Convert the absolute anchor bar to a relative index
        # For example, if current_abs_index = 99 and anchor_bar = 80,
        # then anchor_relative = 80 - 99 = -19.
        anchor_relative = self.anchor_bar - current_abs_index
        total_volume = 0.0
        total_vwap = 0.0

        # Loop from the anchor (relative index) up to the current bar (0)
        # range(start, 1) will iterate from start up to and including 0.
        for offset in range(anchor_relative, 1):
            # Calculate the typical price from the bar at the given relative offset
            typical_price = (self.data.high[offset] +
                             self.data.low[offset] +
                             self.data.close[offset]) / 3.0
            
            volume = self.data.volume[offset]
            total_vwap += typical_price * volume
            total_volume += volume

        if total_volume > 0:
            self.lines.avwap[0] = total_vwap / total_volume
        else:
            self.lines.avwap[0] = float('nan')  # Avoid division by zero
       
        self.lines.typical_price[0] = typical_price
        
    def next(self):
        pass
        
class TrendBasedOnMA(bt.Indicator):
    lines = ('trend', 'sma_short', 'sma_long')  # Define an output line to indicate the trend
    params = (
        ('ma_short', 50),
        ('ma_long', 200),
    )

    def __init__(self):
        self.lines.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.ma_short)
        self.lines.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.ma_long)
        self.lines.trend = self.lines.sma_short - self.lines.sma_long # Positive: Uptrend, Negative: Downtrend

        # Ensure the indicator is plotted on the price axis
        #self.plotinfo.plotmaster = self.data
        #self.plotinfo.subplot = True  # Prevents a separate subplot
