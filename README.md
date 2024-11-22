# Backtesting Repository

This repository is a personal project for learning and experimenting with quantitative finance concepts. It includes a comprehensive backtesting framework, a few trading strategies, and portfolio optimization techniques. The primary focus is on creating a flexible and modular system that supports both intraday and daily trading strategies, statistical testing, and machine learning-based portfolio optimization.

The trading strategies and implementations in this repository are primarily developed 
for educational and demonstration purposes. They have not been live-tested and are 
likely subject to overfitting or performance degradation when trading costs, slippage, and real-world market conditions are considered. 
Please do not take these strategies as financial advice or recommendations for actual trading.

---

## Features

### **Backtesting Framework**
- **`BacktestEngine`**:  
  - Located in `simulation_engine.py`, this is the parent class for backtesting any trading strategy.  
  - Abstract and extensible, allowing easy integration of new strategies.  
  - Supports intraday and daily data with customizable trading frequencies.  
  - Provides functionality for:
    - Statistical testing, including Monte Carlo permutation tests.
    - Plotting and generating trading statistics.

- **Portfolio Optimization**:
  - Abstract class in `portfolio_optimization.py` for optimizing portfolio allocations.  
  - Supports various strategies, including:
    - Vanilla volatility targeting.
    - Mean-variance optimization.
    - Machine learning techniques such as Gaussian Mixture Models (GMM) and Hidden Markov Models (HMM) combined with mean-variance optimization.

---

### **Trading Strategies**
Three trading strategies have been implemented using `BacktestEngine` as the parent class:
1. **`intraday_ml.py`**:  
   Focuses on machine learning-driven intraday strategies.
2. **`tactical.py`**:  
   Implements tactical asset allocation strategies with a mix of rule-based and statistical approaches.
3. **`vol_carry.py`**:  
   Focuses on strategies that exploit volatility carry signals.

More detailed descriptions of these strategies will be provided soon.

---

### **Statistical Testing and Visualization**
The backtesting framework supports:
- Statistical testing to evaluate performance robustness.
- Comprehensive visualizations for:
  - Strategy returns and cumulative performance.
  - Drawdowns and other risk metrics.
  - Statistical summary tables.
- Sample visualizations can be found in the `images` directory, including:
  - Strategy performance boards (`stats_board_*`)
  - Permuted returns plots for testing strategy significance.

---

## Repository Structure


```console
backtest/
│  ├──backtester/
│  │  ├── database_handler/              
│  │  │   ├── intraday_db_handler.py       # Handles intraday data storage and retrieval.
│  │  │   ├── yf_db_handler.py             # Manages data sourced from Yahoo Finance.
│  │  ├── engine/
│  │  │   ├── functions/        
│  │  │   │   ├── performance.py                # Performance metrics and analysis.
│  │  │   │   ├── quant_stats.py                # Quantitative statistics utilities.
│  │  │   │   ├── portfolio_optimization.py     # Portfolio optimization strategies.
│  │  │   ├── simulation_engine.py         # Backtesting framework and portfolio optimization.
│  │  │   ├── configs.yml                  # Configuration file for custom settings.
│  │  │   ├── utils.py                     # Utility functions for calculations.
│  │  ├── buy_hold.py                  # Simple buy-and-hold strategy (example/benchmark).
│  │  ├── intraday_ml.py               # Intraday machine learning strategy.
│  │  ├── tactical.py                  # Tactical allocation strategy.
│  │  ├── vol_carry.py                 # Volatility carry strategy.
├── images/                          # Visualization outputs from strategies.
│   ├── stats_board_intraday_ml.png
│   ├── stats_board_tactical_strategy.png
│   ├── stats_board_vol_carry_strategy.png
│   ├── stats_board_combined.png
│   ├── permuted_returns_tactical_strategy.png
│   ├── permuted_returns_vol_carry_strategy.png
├── main.py                          # Entry point for running backtests.
├── main_ml.py                       # Entry point for running machine learning strategy
´´´
