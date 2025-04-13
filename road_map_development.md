# Roadmap / Future Enhancements

This document outlines planned improvements and features for the project. I am focused on making the backtesting framework more robust, flexible, and user-friendly.

## 0. Rename classes:
Shorter more memorizable names, fitting to core responsibilities of class

BacktestInputCollection: InputOutlet, InputFactory, InputGenerator
BacktestInput: BtInput, BtInput, InputBt
BacktestCollection: BacktestHub, BTHub, BacktestManager
Backtest: OK

## 0. Docstrings

## 0. Error handling

## 0. DeepCopy inputs / input collection
- Usage: run identical backtests with different commission/slipagge settings

## 1. Unit Testing
- **Implement Comprehensive Unit Tests**
  - Write unit tests for all major components (input handling, parameter grouping, datafeed processing, strategy execution, etc.).
  - Include integration tests to ensure that all parts of the framework work together correctly.

## 2. Improve Backtest Class
- **Refactor the Backtest Class**
  - Simplify and optimize the code for easier maintenance.
  - Increase modularity: isolate functionalities (e.g., strategy instantiation, broker settings, analyzers, etc.).
- **Enhance Extensibility**
  - Allow for easier subclassing and customization of strategies.
  - Improve error handling and logging.

## 3. Full Support for Multiple Datafeeds
- **Multi-Datafeed Capability**
  - Enable the framework to process and synchronize multiple datafeeds simultaneously.
  - Ensure compatibility with different tickers, time intervals, and data sources.
- **Flexible Data Handling**
  - Implement features for merging, aligning, and interpolating data from various sources if needed.

## 4. Interactive Visualizations
- **Equity Curves & Beyond**
  - Integrate interactive plotting (using libraries like Plotly, Bokeh, or Dash) to display equity curves, drawdowns, and performance metrics.
  - Allow users to zoom, pan, and select timeframes interactively.
- **Real-Time Insights**
  - Provide dashboards or UI components to display backtest results and performance summaries interactively.

## 5. Walk-Forward Optimization (WFO) – Selection of Optimal Parameter Set
Selecting the optimal parameter set for walk‐forward optimization requires not just the best average performance but also robustness to small variations. The goal is to pick a parameter set that not only performs well on average but also remains stable when nearby parameter combinations are tested.

    ### Possible Approaches:
    1. **Simple Heatmap Visualization**
       - Create heatmaps of performance metrics (e.g., Sharpe ratio) for different parameter combinations.
       - Identify regions where performance is consistently high.
    
    2. **Cluster Similar Parameter Sets**
       - Compute a distance metric (e.g., Euclidean distance) for parameter vectors.
       - Use clustering techniques (e.g., k-means or hierarchical clustering) to group similar parameter sets.
       - Analyze performance variability within each cluster to select robust parameter regions.
    
    3. **Local Sensitivity Analysis (Perturbation Testing)**
       - Start with the best performing parameter set.
       - Generate slight perturbations for each parameter.
       - Evaluate the performance of each perturbed set.
       - Choose the set that shows minimal performance variation under small changes.
    
    ---

*Feel free to contribute if you have ideas for further improvements or run into issues along the way!*