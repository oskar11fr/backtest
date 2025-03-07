import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pathlib import Path
from typing import Dict, Optional, Union, Tuple

plt.style.use("bmh")

def check_intraday_data(indx: pd.DatetimeIndex) -> tuple[bool, int]:
    """
    Check if the DataFrame has intraday data and calculate the average number of timestamps per day.
    
    Parameters:
    df (pd.DataFrame): A DataFrame with a datetime index.
    
    Returns:
    tuple: A tuple containing:
        - has_intraday (bool): True if intraday data is present, False otherwise.
        - avg_timestamps_per_day (float): The average number of timestamps per day.
    """
    
    trading_day_counts = indx.to_series().dt.date.value_counts()
    has_intraday = (trading_day_counts > 1).any()
    avg_timestamps_per_day = trading_day_counts.mean()
    return has_intraday, int(avg_timestamps_per_day)

def performance_measures(
    r_ser: pd.Series,
    plot: bool = False,
    path: str = "./images",
    market: Optional[Dict[str, pd.Series]] = None,
    show: bool = False,
    strat_name: str = "Strategy"
) -> Dict[str, Union[float, np.ndarray, pd.Series]]:
    """
    Computes performance measures for a given return series, with optional plotting.

    Parameters
    ----------
    r_ser : pd.Series
        The series of daily returns to analyze.
    plot : bool, optional
        Whether to generate performance plots, by default False.
    path : str, optional
        Directory path to save plots if `plot=True` and `show=False`, by default "/images".
    market : Optional[Dict[str, pd.Series]], optional
        Market benchmark return series for comparison, by default None.
    show : bool, optional
        If True, display the plot interactively; otherwise, save it, by default False.
    strat_name : str, optional
        Strategy name for labeling the plot, by default "".

    Returns
    -------
    Dict[str, Union[float, np.ndarray, pd.Series]]
        Dictionary containing various performance metrics.

    Raises
    ------
    AssertionError
        If `market` is provided but not of type `dict[str, pd.Series]`.
    """
    # Helper functions for statistical moments and performance metrics
    
    has_intraday, avg_timestamps_per_day = check_intraday_data(indx=r_ser.index)

    calc_const = avg_timestamps_per_day * 253

    moment = lambda x, k: np.mean((x - np.mean(x)) ** k)
    stdmoment = lambda x, k: moment(x, k) / moment(x, 2) ** (k / 2)
    rolling_drawdown = lambda cr, pr: cr / cr.rolling(pr, min_periods=1).max() - 1
    rolling_max_dd = lambda cr, pr: rolling_drawdown(cr, pr).rolling(pr, min_periods=1).min()
    cagr_fn = lambda cr: (cr[-1]/cr[0])**(1/len(cr))-1
    cagr_ann_fn = lambda cr: ((1+cagr_fn(cr))**(calc_const)) - 1

    # Compute cumulative and log returns
    r = r_ser.values
    cr = np.cumprod(1 + r)  # Cumulative returns
    lr = np.log(cr)         # Log returns
    cr_ser = pd.Series(cr, index=r_ser.index)

    # Performance metrics
    mdd = np.min(cr / np.maximum.accumulate(cr) - 1)  # Maximum drawdown
    sortino_ratio = np.mean(r) / np.std(r[r < 0]) * np.sqrt(calc_const)
    sharpe_ratio = np.mean(r) / np.std(r) * np.sqrt(calc_const)
    mean_ret = np.mean(r) * calc_const
    median_ret = np.median(r) * calc_const
    vol = np.std(r) * np.sqrt(calc_const)
    variance = vol ** 2
    skewness = stdmoment(r, 3)
    ex_kurtosis = stdmoment(r, 4) - 3
    
    cagr = cagr_ann_fn(cr) # Annualized CAGR

    rolling_sharpe = r_ser.rolling(calc_const, min_periods=50).mean() \
          / r_ser.rolling(calc_const, min_periods=50).std() * np.sqrt(calc_const)
    var95 = np.percentile(r, 5)
    cvar = r[r < var95].mean()
    calmar = cagr / mdd * -1

    # Prepare metrics table
    metrics = {
        "cagr": cagr,
        "sortino": sortino_ratio,
        "sharpe": sharpe_ratio,
        "calmar": calmar,
        "mean_ret": mean_ret,
        "median_ret": median_ret,
        "vol": vol,
        "var": variance,
        "max_drawdown": mdd,
        "skew": skewness,
        "ex_kurtosis": ex_kurtosis,
        "var95": var95,
        "cvar": cvar
    }

    # Plotting performance if requested
    if plot:
        fig = plt.figure(layout="tight",figsize=(16, 14))
        spec = fig.add_gridspec(5, 4)

        ax1 = fig.add_subplot(spec[0:2, 0:3])
        ax2 = fig.add_subplot(spec[2, 0:3], sharex=ax1)
        ax3 = fig.add_subplot(spec[0:2, -1])
        ax4 = fig.add_subplot(spec[2, -1])
        ax5 = fig.add_subplot(spec[3, 0:3], sharex=ax1)
        ax6 = fig.add_subplot(spec[3,-1], sharey=ax5)
        ax7 = fig.add_subplot(spec[4,0:3], sharex=ax1)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax5.xaxis.set_major_locator(plt.MaxNLocator(5))

        # Log returns plot
        idxs = r_ser.index.strftime('%Y-%m-%d') if not has_intraday else range(len(r_ser))
        ax1.plot(idxs, lr, label=strat_name)
        if market:
            assert isinstance(market, dict), "Market parameter must be a dictionary of {name: pd.Series}."
            for name, data in market.items():
                benchmark = data.loc[r_ser.index].pct_change().fillna(0)
                ax1.plot(idxs, np.log(np.cumprod(1 + benchmark)), linestyle=":", alpha=0.75, label=name)
        ax1.set_ylabel("Log Returns")
        ax1.legend()

        # Drawdowns
        ax2.plot(idxs, rolling_drawdown(cr_ser, calc_const), label="Drawdowns")
        ax2.plot(idxs, rolling_max_dd(cr_ser, calc_const), label="Max Drawdowns")
        ax2.set_ylabel("Drawdowns")
        ax2.legend()

        # Metrics table
        
        metrics_df = pd.Series(metrics).apply(lambda x: np.round(x, 3)).reset_index()
        metrics_df.columns = ["Metric", "Value"]
        ax3.axis("off")
        table = ax3.table(
            cellText=metrics_df.values,
            colLabels=metrics_df.columns,
            loc="center",
            colWidths=[0.4, 0.4],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        # Distribution of drawdowns
        ax4.hist(rolling_drawdown(cr_ser, calc_const), orientation="horizontal", bins=40)

        # Returns bar chart
        ax5.bar(idxs, r)
        ax5.set_ylabel("Returns")

        ax6.hist(r,orientation='horizontal',bins=60)

        ax7.plot(idxs, rolling_sharpe)
        ax7.set_ylabel("Sharpe")
        
        # Save or show the plot
        if not show:
            fig.savefig(f"{path}/stats_board_{strat_name}.png")
            plt.close()
        else:
            plt.show()

    return metrics

def plot_hypothesis(
    timer_tuple: Tuple[pd.DataFrame, float, list],
    picker_tuple: Tuple[pd.DataFrame, float, list],
    trader_tuple: Tuple[pd.DataFrame, float, list],
    return_samples: pd.Series,
    strat_name: str = ""
):
    """
    Plots the results of hypothesis testing for timing, picking, and trading strategies.

    Parameters
    ----------
    timer_tuple : Tuple[pd.DataFrame, float, list]
        Contains paths (cumulative returns), p-value, and distribution for the timing strategy.
    picker_tuple : Tuple[pd.DataFrame, float, list]
        Contains paths (cumulative returns), p-value, and distribution for the picking strategy.
    trader_tuple : Tuple[pd.DataFrame, float, list]
        Contains paths (cumulative returns), p-value, and distribution for the trading strategy.
    return_samples : pd.Series
        The observed return samples for the primary strategy.
    strat_name : str, optional
        Name of the strategy, used in plot labels and filenames, by default "".

    Returns
    -------
    None
        Saves the plot to a predefined directory.
    """
    # Create output directory if it doesn't exist
    path = "./images"
    # Path(os.path.abspath(os.getcwd() + path)).mkdir(parents=True, exist_ok=True)

    # Unpack tuples
    timer_paths, timer_p, timer_dist = timer_tuple
    picker_paths, picker_p, picker_dist = picker_tuple
    trader_paths, trader_p, trader_dist = trader_tuple

    # Align indices of paths with return samples
    timer_paths.index = return_samples.index
    picker_paths.index = return_samples.index
    trader_paths.index = return_samples.index

    # Initialize the figure and axes
    fig = plt.figure(layout="tight",figsize=(16, 14))
    ax = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(ax[:2, :])  # Main cumulative returns plot
    ax2 = fig.add_subplot(ax[2:, :2])  # KDE plot for distributions
    ax3 = fig.add_subplot(ax[2:, -1])  # Table for p-values

    # Plot cumulative returns
    ax1.plot((1 + timer_paths).cumprod().apply(np.log), color='red', alpha=0.3,)
    ax1.plot((1 + picker_paths).cumprod().apply(np.log), color='blue', alpha=0.3)
    ax1.plot((1 + trader_paths).cumprod().apply(np.log), color='green', alpha=0.3)
    ax1.plot((1 + return_samples).cumprod().apply(np.log), color='black', linewidth=4, label="Strategy")
    ax1.set_title("Cumulative Log Returns")
    ax1.legend()

    # Plot KDE for distributions
    dist_check = lambda dist: np.any(np.diff(dist) != 0)
    if dist_check(timer_dist): pd.Series(timer_dist).plot(color='red', ax=ax2, kind='kde', label="Timing Dist")
    if dist_check(picker_dist): pd.Series(picker_dist).plot(color='blue', ax=ax2, kind='kde', label="Picking Dist")
    if dist_check(trader_dist): pd.Series(trader_dist).plot(color='green', ax=ax2, kind='kde', label="Trading Dist")
    strategy_sharpe = np.mean(return_samples.values) / np.std(return_samples.values) * np.sqrt(253)
    ax2.axvline(strategy_sharpe, color='black', linewidth=4, label="Strategy Sharpe")
    ax2.set_title("Distributions and Strategy Sharpe")
    ax2.legend()

    # Create p-values table
    p_values = pd.DataFrame({
        "Tests": ["Permuted MC (Timing)", "Permuted MC (Picking)", "Permuted MC (Skill)"],
        "p values": [timer_p, picker_p, trader_p]
    }).round(4)
    ax3.table(
        cellText=p_values.values,
        colLabels=p_values.columns,
        loc="center",
        colWidths=[0.5, 0.3],
        colColours=["lightgrey", "lightgrey"],
        cellLoc="left",
    )
    ax3.axis('off')
    # ax3.set_title("P-Values")

    # Save plot to the specified directory
    strat_name = strat_name if strat_name else "default_strategy"
    fig.savefig(f"{path}/permuted_returns_{strat_name}.png")
    plt.close()

def plot_random_entries(caps: pd.DataFrame, sharpes: pd.Series, market: pd.Series | None = None):
    """
    Plots the performance of random entry simulations and compares them to the main portfolio.

    Parameters
    ----------
    caps : pd.DataFrame
        DataFrame containing cumulative returns for the portfolio and simulations. 
        Columns include "Portfolio" and simulated paths (e.g., "sim_1", "sim_2").
    sharpes : pd.Series
        Series containing Sharpe ratios for the portfolio and simulations.
        Index includes "Portfolio" and simulated paths.
    market : pd.Series, optional
        Market benchmark returns as a Series. If provided, market performance is plotted, by default None.

    Returns
    -------
    None
        Displays the plot directly.
    """
    # Extract portfolio data
    portfolio_sharpe = sharpes["Portfolio"]
    portfolio_caps = caps["Portfolio"]

    # Exclude portfolio data from simulations
    sim_sharpes = sharpes[[col for col in sharpes.index if col != "Portfolio"]]
    sim_caps = caps[[col for col in caps.columns if col != "Portfolio"]]

    # Initialize the figure and axes
    fig = plt.figure(constrained_layout=True, figsize=(15, 11))
    ax = fig.add_gridspec(4, 2)
    ax1 = fig.add_subplot(ax[:3, :])  # Cumulative log returns
    ax2 = fig.add_subplot(ax[3:, :])  # Sharpe ratio KDE

    # Plot cumulative log returns for simulations and portfolio
    ax1.plot((1 + sim_caps).cumprod().apply(np.log), color='blue', alpha=0.3)
    ax1.plot((1 + portfolio_caps).cumprod().apply(np.log), color='black', linewidth=4, label="Portfolio")
    ax1.set_title("Cumulative Log Returns")
    ax1.set_ylabel("Log Returns")
    ax1.legend()

    # Plot market performance if available
    if market is not None:
        idxs = portfolio_caps.index  # Ensure alignment
        ax1.plot(idxs, np.log(np.cumprod(1 + market.loc[idxs])), color='green', linestyle='--', label="Market")
        ax1.legend()

    # Plot Sharpe ratio distribution
    sim_sharpes.plot(color="blue", ax=ax2, kind="kde", label="Simulations")
    ax2.axvline(portfolio_sharpe, color='black', linewidth=4, label="Portfolio Sharpe")
    ax2.set_title("Sharpe Ratio Distribution")
    ax2.set_xlabel("Sharpe Ratio")
    ax2.legend()

    # Display the plot
    plt.show()

