import pandas as pd
import numpy as np
import backtrader as bt
import datetime
import math

class Analyzer():
    def __init__(self, analyzer_class=None, name=None, analyzer_arguments=None, outputs=None):
        self.analyzer_class = analyzer_class
        self.name = name
        self.analyzer_arguments = analyzer_arguments or {}
        self.analyzer_arguments['_name'] = name
        self.outputs = outputs            
    
    def add_cerebro(self, cerebro):
        cerebro.addanalyzer(self.analyzer_class, **self.analyzer_arguments)

    def get_outputs(self, run):
        analyzer = getattr(run.analyzers, self.analyzer_arguments.get('_name'))
        outputs = {}
        if self.outputs:
            for output in self.outputs:
                analysis = analyzer.get_analysis()
                output_name = '_'.join(output)
                for key in output:
                    analysis = analysis.get(key, {})
                outputs[output_name] =  analysis
        
        return outputs

    def get_analysis(self, run):
        analyzer = getattr(run.analyzers, self.analyzer_arguments.get('_name'))
        analysis = analyzer.get_analysis()
        return analysis
    
        
class AnalyzerCollection():

    def __init__(self):
        analyzers = {}
                
        analyzers['time_return'] = Analyzer(
            analyzer_class=CustomTimeReturn, 
            analyzer_arguments = None, 
            name="time_return", 
            outputs=None,
        )        
        
        analyzers['drawdown'] = Analyzer(
            analyzer_class=bt.analyzers.DrawDown, 
            analyzer_arguments=None, 
            name="drawdown", 
            outputs=(('max', 'drawdown'),),
        )
        
        analyzers['trade_logger'] = Analyzer(
            analyzer_class=TradeLogger,
            analyzer_arguments=None,
            name='trade_logger',
            outputs=None,
        )
                
        analyzers['equity'] = Analyzer(
            analyzer_class=EquityAnalyzer,
            analyzer_arguments=None,
            name='equity',
            outputs=None,
        )

        analyzers['custom_metrics'] = Analyzer(
            analyzer_class=CustomMetrics,
            analyzer_arguments=None,
            name='custom_metrics',
            outputs=(("cagr",), ('sharperatio',), ('time_in_market',))
        )
        
        self.analyzers = analyzers

    def add_analyzers_cerebro(self, cerebro):
        """Add registered analyzers to cerebro."""
        for analyzer in self.analyzers.values():
            analyzer.add_cerebro(cerebro)
    
    def get_outputs(self, run):
        """Retrieve analyzer results from a strategy instance."""
        outputs = {}
        for analyzer in self.analyzers.values():
             outputs.update(analyzer.get_outputs(run))
        return outputs
    
    def get_analysis(self, analyzer_name, run):
        analysis = self.analyzers[analyzer_name].get_analysis(run)
        return analysis
        

class TradeLogger(bt.Analyzer):
    def __init__(self):
        self.trades = []
        self.trade_size = {} # Track trade sizes per asset
        self.bar_executed = {} # tracks the the index (or count) of the bar at which the buy order was executed.

    def notify_trade(self, trade):
        ticker = trade.data._name  # Get asset name
        
        if trade.isopen:
            #self.trade_size[ticker] = trade.size
            self.bar_executed[ticker] = len(trade.data) - 1  # Store entry bar index
        
        if trade.isclosed:
            entry_price = trade.price
            exit_price = trade.data.close[0]
            entry_date = bt.num2date(trade.dtopen).date()
            exit_date = bt.num2date(trade.dtclose).date()

            #entry_bar = self.strategy.bar_executed.get(trade.data, None)
            
            entry_bar = self.bar_executed.get(ticker, None)
            exit_bar = len(trade.data) - 1  # Current bar index at exit
            
            if entry_bar is not None:
                hold_bars = exit_bar - entry_bar 
            else:
                hold_bars = 0  # This should not happen if the buy order was recorded in strategy.bar_executed
            
            self.trades.append({
                'ticker': ticker,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_bar': entry_bar,
                'exit_bar': exit_bar,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'hold_bars': hold_bars,
                'log_ret': np.log(exit_price / entry_price),
                'size': self.trade_size.get(ticker, 0),
                'pnlcomm': trade.pnlcomm  # Net PnL after commission
            })
        

    def get_analysis(self):
        """ Returns the stored trade data as a pandas DataFrame for easy export """
        return pd.DataFrame(self.trades)

class EquityAnalyzer(bt.Analyzer):
    def __init__(self, start_date=None, end_date=None):
        # Set the start and end dates for recording values
        self.start_date = start_date if start_date is not None else datetime.datetime.min
        self.end_date = end_date if end_date is not None else datetime.datetime.max
        # Lists to record the portfolio values and corresponding dates
        self.values = []
        self.dates = []

    def next(self):
        # Get the current bar's date
        current_date = self.strategy.datas[0].datetime.datetime(0)
        # Record value only if within the specified range
        if self.start_date <= current_date <= self.end_date:
            current_value = self.strategy.broker.get_value()
            self.values.append(current_value)
            self.dates.append(current_date)

    def get_analysis(self):
        # Return the recorded data after the run
        return {'values': self.values, 'dates': self.dates}


class CustomMetrics(bt.Analyzer):
    """
    Custom Backtrader Analyzer to calculate CAGR, Sharpe Ratio, and Time in Market.
    - Uses returns provided by `CustomTimeReturn`
    - Calculates trading days from recorded returns
    - Computes Time in Market based on non-zero returns
    """

    params = (
        ('risk_free_rate', 0.0),  # Default risk-free rate is 0%
        ('annualize_factor', 252),  # Assume 252 trading days per year (default)
    )

    def stop(self):
        
        """Executed at the end of the backtest to compute CAGR, Sharpe Ratio, and Time in Market."""
        end_value = self.strategy.broker.getvalue()  # Final portfolio value

        # Retrieve returns from `CustomTimeReturn Analyzer`
        daily_returns = self.strategy.analyzers.time_return.get_analysis()

        # Total trading days based on recorded returns
        self.trading_days = len(daily_returns)

        # Time in Market: Count non-zero returns
        non_zero_days = sum(1 for r in daily_returns.values() if r != 0.0)
        self.time_in_market = (non_zero_days / self.trading_days) * 100 if self.trading_days > 0 else 0

        # Get the timestamps from the returns to determine the start and end date
        timestamps = list(daily_returns.keys())  # Extract the timestamps (dates) from the dictionary
        start_date = timestamps[0]  # First timestamp (start date)
        end_date = timestamps[-1]  # Last timestamp (end date)

        # Calculate the difference in years (use calendar years based on actual start and end dates)
        delta_days = (end_date - start_date).days
        years = delta_days / 365.25  # Account for leap years

        # CAGR: Calculate the Compound Annual Growth Rate based on actual calendar years
        self.cagr = ((end_value / max(self.strategy.broker.startingcash, 1)) ** (1 / years)) - 1  # CAGR formula

        # Sharpe Ratio: Calculate Sharpe Ratio based on actual trading days
        if len(daily_returns) > 1:
            mean_return = np.mean(list(daily_returns.values()))  # Average daily return
            std_dev = np.std(list(daily_returns.values()))  # Standard deviation of returns

            if std_dev > 0:
                self.sharperatio = (mean_return - self.p.risk_free_rate) / std_dev
                # Annualize Sharpe Ratio based on actual trading days (not a fixed 252)
                trading_days_per_year = self.trading_days / years
                self.sharperatio *= np.sqrt(trading_days_per_year)  # Annualized Sharpe Ratio
                #self.sharperatio *= np.sqrt(252)

            else:
                self.sharperatio = None  # Avoid division by zero
        else:
            self.sharperatio = None  # Not enough data for Sharpe Ratio

    def get_analysis(self):
        """Returns the computed CAGR, Sharpe Ratio, and Time in Market."""
        return {
            "cagr": round(self.cagr * 100, 2),  # CAGR as a percentage
            "sharperatio": round(self.sharperatio, 2) if self.sharperatio is not None else "N/A",  # Sharpe Ratio or N/A
            "time_in_market": round(self.time_in_market, 2)  # Time in market as percentage
        }

class CustomTimeReturn(bt.Analyzer):
    """  
    Returns are calculated only if `self.strategy.start_date_reached` is True.
    
    By default, simple returns are calculated:
    """

    params = (
        ('log_returns', False),  # Default is False, so simple returns are calculated.
    )

    def __init__(self):
        self.returns = []    # List to store the returns data
        self.timestamps = [] # List to store the corresponding datetime

    def next(self):
        """
        Executed at each step of the backtest.
        Calculate the daily return and store it with the corresponding timestamp.
        Only if strategy.start_date_reached is True.
        """
        if self.strategy.start_date_reached:
            current_value = self.strategy.broker.getvalue()  # Current portfolio value
            if hasattr(self, "prev_value"):
                if self.p.log_returns:
                    # Calculate daily log return
                    daily_return = math.log(current_value / self.prev_value)
                else:
                    # Calculate daily simple return
                    daily_return = (current_value / self.prev_value) - 1
                self.returns.append(daily_return)
                # Append the timestamp (datetime.datetime)
                timestamp = bt.num2date(self.datas[0].datetime[0])
                self.timestamps.append(timestamp)
            self.prev_value = current_value  # Update previous value for next step

    def stop(self):
        # Ensure the first return is not zero to avoid potential issues
        if self.returns and self.returns[0] == 0:
            self.returns[0] = 1e-7

    def get_analysis(self):
        """
        Return the returns data in the same format as Backtrader's TimeReturn analyzer.
        Returns a dictionary with timestamps as keys and returns as values.
        """
        return dict(zip(self.timestamps, self.returns))
