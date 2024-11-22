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
The `BacktestEngine` class serves as the backbone of the backtesting framework in this repository. It allows for the backtesting of various trading strategies using both intraday and daily data.  
  - **Data Handling**: The engine can process historical data for a range of financial instruments at different time granularities (e.g., minute-level for intraday strategies).
  - **Portfolio Optimization**: The engine supports various portfolio optimization methods, including vanilla volatility targeting and more advanced approaches like mean-variance optimization and machine learning-based models (e.g., Gaussian Mixture Models, Hidden Markov Models).
  - **Trade Frequency**: It provides the flexibility to execute trades at any specified frequency, making it suitable for strategies ranging from high-frequency trading to daily rebalancing.
  - **Performance Metrics**: The engine integrates performance testing, including Monte Carlo permutation tests, and tracks key metrics such as Sharpe ratio, drawdowns, and other risk-adjusted performance measures.
  - **Statistical Analysis**: It includes tools for conducting statistical analysis on the strategy’s performance over different market conditions and time periods.

  The `BacktestEngine` is an abstract base class, and specific strategies like the `IntradayML` class can be created by inheriting and implementing the required methods.

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
   Based on intraday data, it uses a machine learning model built with Keras to predict next 15 minute price movements.
   The model is a neural network with a single hidden layer and a tanh activation function at the output layer, designed to forecast future returns. The strategy utilizes features like **VWAP** (Volume Weighted Average Price), **TWAP** (Time Weighted Average Price), **AVAT** (Average Volume at Time), and others, to create signals based on the deviation from historical averages and volatility. 

  **Key Features:**
  - **Data**: The strategy operates on minute-level intraday data, processing price and volume to create trading signals.
  - **Indicators**: It includes technical features 
  - **Model**: The strategy is based on a Keras-based neural network (KerasModel), trained on a combination of features such as the z-score of VWAP, TWAP, AVAT, and volatility.
  - **Training and Prediction**: The model is trained on 70% of historical data and tested on the remaining 30%.
  - **Forecasting**: The forecast is made using the model’s output, adjusting positions accordingly for each trading day based on the predicted returns.

2. **`tactical.py`**:  
   Implements tactical asset allocation strategies with a mix of rule-based and statistical approaches.
3. **`vol_carry.py`**:  
   Focuses on strategies that exploit volatility carry signals.

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
│  ├── stats_board_intraday_ml.png
│  ├── stats_board_tactical_strategy.png
│  ├── stats_board_vol_carry_strategy.png
│  ├── stats_board_combined.png
│  ├── permuted_returns_tactical_strategy.png
│  ├── permuted_returns_vol_carry_strategy.png
├── main.py                          # Entry point for running backtests.
├── main_ml.py                       # Entry point for running machine learning strategy
´´´
