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
This class serves as the backbone of the backtesting framework in this repository. It allows for the backtesting of various trading strategies using both intraday and daily data.  
  - **Data Handling**: The engine can process historical data for a range of financial instruments at different time granularities (e.g., minute-level for intraday strategies).
  - **Portfolio Optimization**: The engine supports various portfolio optimization methods, including vanilla volatility targeting and more advanced approaches like mean-variance optimization and machine learning-based models (e.g., Gaussian Mixture Models, Hidden Markov Models).
  - **Trade Frequency**: It provides the flexibility to execute trades at any specified frequency, making it suitable for strategies ranging from high-frequency trading to daily rebalancing.
  - **Performance Metrics**: The engine integrates performance testing, including Monte Carlo permutation tests, and tracks key metrics such as Sharpe ratio, drawdowns, and other risk-adjusted performance measures.
  - **Statistical Analysis**: It includes tools for conducting statistical analysis on the strategy’s performance over different market conditions and time periods.

- **Portfolio Optimization**:
  - Abstract class in `portfolio_optimization.py` for optimizing portfolio allocations.  
  - Supports various strategies, including:
    - Vanilla volatility targeting.
    - Mean-variance optimization.
    - Machine learning techniques such as Gaussian Mixture Models (GMM) and Hidden Markov Models (HMM) combined with mean-variance optimization.

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
