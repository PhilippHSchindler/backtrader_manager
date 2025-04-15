# Roadmap / Future Enhancements

This document outlines planned improvements and features for the project. I am focused on making the backtesting framework more robust, flexible, and user-friendly. Where possible reduce options, complexity. 

Items higher up in the list have curenntly higher priority.

## Create representable quickstart guide
- Reasses workflows
- Identify unnecessary complex workflows, methods

## GitHub

## Docstrings

## Bugs
- main._param_to_id has wrong entries
  
## Error handling

## Rename classes and methods:
Shorter more memorizable names, fitting to core responsibilities of class, methods

BacktestInputCollection: InputOutlet, InputsFactory, InputGenerator, InputsBuilder
BacktestInput: BtInput, BtInput, InputBt
BacktestCollection: BacktestHub, BTHub, BacktestManager, OptHub
Backtest: OK
  
## Re-evaluate Classes: Strategy and Backtest
- Review concept

## Testing
- **Implement Comprehensive Unit Tests**

## Full Support for Multiple Datafeeds
- **Multi-Datafeed Capability**
  - Ensure compatibility with different time intervals and data sources.
- **Flexible Data Handling**
  - Implement features for merging, aligning, and interpolating data from various sources if needed.

## Selection of Optimal Parameter Set 
**Research good metrics for parameter selection**
- sortino instead sharpe ratio
- skew / kurtosis
- calmar ratio
- turnover
- ...

**Use Composite Objective for ranking paramter sets**
- Compute multi-metric, composite score per window
- Focus on robustness, stability, not just peak performance. Smooth equity curves
- Aggregate score over all windows. Mean? Median?
- Optionally penalize instability, complexity (more likely leads to overfitting)

Aim for stable, even score across
- market regimes
- stability across periods/iterations
- folds (training vs test period), low variance
- high vs.low volatility enviroment

**Test for robustness to small paramtere variations**
Selecting the optimal parameter set requires not just the best average performance but also robustness to small variations. The goal is to pick a parameter set that not only performs well for the specific paramtere combination but also remains stable when nearby parameter combinations are tested.

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

## Interactive Visualizations
- **Equity Curves & Beyond**
  - Integrate propiate interactive plotting to display equity curves, drawdowns, and performance metrics.
  - Allow users to zoom, pan, and select timeframes interactively.

## WFO: Add simulated live strategy deployment
- We keep back a continous portion of our historic data for simulated live deployment
- For each walkforward step recalculate optimal parameter set
- Final evaluation using equity curve for the simulated live period

## Add more optmization methods
- Nested Cross-Validation
- Grid search

*Feel free to contribute if you have ideas for further improvements or run into issues along the way!*