import os
import time

import numpy as np
import pandas as pd
import polars as pl

from copy import deepcopy
from functools import wraps
from datetime import datetime
from typing import Optional, Union, Dict, Generator, Tuple

from .functions import quant_stats as quant_stats
from .functions.performance import (
    plot_hypothesis, 
    plot_random_entries, 
    performance_measures
)
from .functions.portfolio_strategies import(
    PositioningStrategy,
    MeanVarianceStrategy,
    VolatilityTargetingStrategy,
    MixtureModelsMeanVarianceStrategy
)
# from .database.constants import STRAT_PATH


def timeme(func):
    @wraps(func)
    def timediff(*args,**kwargs):
        a = time.time()
        result = func(*args,**kwargs)
        b = time.time()
        print(f"@timeme: {func.__name__} took {b - a} seconds")
        return result
    return timediff


class AbstractImplementationException(Exception):
    pass

class _abstract_exc(Exception):
    pass


class TradingFrequencyCalculator:
    def __init__(self, trade_frequency: Optional[str] = None, day_of_week: Optional[str] = None):
        self.trade_frequency = trade_frequency
        self.day_of_week = day_of_week
        self.trading_day_ser = pd.Series()

    def compute_frequency(self, trade_range: pd.DatetimeIndex) -> None:
        """
        Compute the trading frequency and generate a series indicating valid trading days
        for each date in the specified date range.

        Parameters:
        ----------
        trade_range : pd.DatetimeIndex
            A datetime index representing the range of dates to compute trading days for.

        Returns:
        -------
        None
            Sets self.trading_day_ser as a series with a boolean mask indicating trading days.
        """
        if self.trade_frequency is None or self.trade_frequency == "daily":
            self.trading_day_ser = pd.Series(1, index=trade_range)
            return

        # Create an empty series to hold trading day indicators
        self.trading_day_ser = pd.Series(False, index=trade_range)

        frequency_checks = {
            "weekly": self._is_weekly_trade_day,
            "monthly": self._is_month_end,
            "quarter": self._is_quarter_end
        }

        # Apply the corresponding function to compute trading days for specific frequency
        check_function = frequency_checks.get(self.trade_frequency)
        if check_function:
            self.trading_day_ser = trade_range.to_series().apply(check_function)

    def _is_weekly_trade_day(self, date: pd.Timestamp) -> bool:
        """Check if a given date falls on the specified weekly trade day."""
        day_of_week = "Friday" if self.day_of_week is None else self.day_of_week
        return date.day_name() == day_of_week

    def _is_month_end(self, date: pd.Timestamp) -> bool:
        """Check if a given date is the end of the month."""
        return pd.tseries.offsets.BMonthEnd().rollforward(date) == date

    def _is_quarter_end(self, date: pd.Timestamp) -> bool:
        """Check if a given date is the end of the quarter."""
        return pd.tseries.offsets.BQuarterEnd().rollforward(date) == date


class BacktestEngine(TradingFrequencyCalculator):
    def __init__(
            self, 
            insts: list[str], 
            dfs: dict[str, pd.DataFrame], #dict[str, pl.DataFrame], To do
            start: Optional[datetime] = None, 
            end: Optional[datetime] = None,
            date_range: Optional[pd.DatetimeIndex] = None, 
            trade_frequency: Optional[str] = None,
            day_of_week: Optional[str] = None,
            portf_strategy: PositioningStrategy = VolatilityTargetingStrategy(),
            portfolio_vol: float = 0.20,
            max_leverage: float = 2.0, 
            min_leverage: float = 0.0, 
            benchmark: Optional[str] = None
        ) -> None:
        """
        Initialize the BacktestEngine with instruments, data, and trading parameters.

        Parameters
        ----------
        insts : list[str]
            List of instrument identifiers.
        dfs : dict[str, pd.DataFrame]
            Dictionary mapping instrument identifiers to their respective data as Pandas DataFrames.
        start : datetime, optional
            Start date for the backtest. Required if `date_range` is not provided.
        end : datetime, optional
            End date for the backtest. Required if `date_range` is not provided.
        date_range : pd.DatetimeIndex, optional
            A pre-defined date range for the backtest. Overrides `start` and `end` if provided.
        trade_frequency : str, optional
            Frequency of trades, options include "daily", "weekly", "monthly", or "quarter". Default is "daily".
        day_of_week : str, optional
            determines which day of the week we trade. applicable if frequency == "weekly"
        portfolio_vol : float, optional
            Target portfolio volatility. Default is 0.20.
        max_leverage : float, optional
            Maximum allowable leverage. Default is 2.0.
        min_leverage : float, optional
            Minimum allowable leverage. Default is 0.0.
        benchmark : str, optional
            Benchmark instrument identifier for performance comparison. Default is None.

        Returns
        -------
        None
        """
        # Ensure that either date_range or start (and end) is provided
        assert (start is not None) or (date_range is not None), "Either date_range or start date must be initialized."

        super().__init__(trade_frequency, day_of_week)
        # Initialize parameters
        self.insts = insts
        self.dfs = deepcopy(dfs)
        self.datacopy = deepcopy(dfs)
        self.portfolio_vol = portfolio_vol
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.benchmark = benchmark
        self.date_range = date_range

        self.portf_strategy = portf_strategy

        # Determine date range for the backtest
        if date_range is None:
            self.date_range = pd.date_range(start=start, end=end, freq="D")


    def get_zero_filtered_stats(self, test: bool | float = False) -> dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        Filter and retrieve statistics for non-zero capital returns.

        Parameters
        ----------
        test_data : bool, optional
            If True, method generate stats for test data only.

        Returns
        -------
        dict[str, Union[pd.Series, pd.DataFrame]]
            A dictionary containing filtered data for capital returns, nominal returns,
            returns dataframe, weights, eligibilities, and leverages.
        
        Raises
        ------
        AssertionError
            If `self.portfolio_df` is not initialized.
        """
        assert self.portfolio_df is not None, "Portfolio data must be initialized before getting statistics."
        
        idxs = self.date_range
        is_intraday = (idxs.to_series().dt.date.value_counts() > 1).any()
        if isinstance(test, bool) and test:
            idxs = idxs[self.portf_strategy.TRAIN_ID:]
            if self.portf_strategy.TRAIN_ID == 0: print("No train / test size is initalized, will consider full dataframe")
        
        if isinstance(test, float):
            idxs = idxs[int(len(idxs)*test):]

        # Extract relevant columns
        nominal_ret = self.portfolio_df["nominal_ret"].loc[idxs]
        capital_ret = self.portfolio_df["capital_ret"].loc[idxs]

        # Filter for non-zero capital returns
        non_zero_idx = capital_ret.loc[capital_ret != 0].index if not is_intraday else capital_ret.index
        retdf = self.retdf.loc[non_zero_idx]
        weights = self.weights_df.shift(1).fillna(0).loc[non_zero_idx]
        eligs = self.eligiblesdf.shift(1).fillna(0).loc[non_zero_idx]
        leverages = self.leverages.shift(1).fillna(0).loc[non_zero_idx]

        return {
            "capital_ret": capital_ret.loc[non_zero_idx],
            "nominal_ret": nominal_ret.loc[non_zero_idx],
            "retdf": retdf,
            "weights": weights,
            "eligs": eligs,
            "leverages": leverages,
        }


    def get_perf_stats(
            self,
            plot: bool = False,
            compare: bool = False,
            show: bool = False,
            strat_name: Optional[str] = None,
            test: bool | float = False
        ) -> pd.Series:
        """
        Calculate and return performance statistics.

        Parameters
        ----------
        plot : bool, optional
            If True, plots performance measures. Default is False.
        compare : bool, optional
            If True, includes benchmark market data in the performance evaluation. Default is False.
        show : bool, optional
            If True, displays additional performance output. Default is False.
        strat_name : str, optional
            Name of the strategy for labeling purposes. Default is None.
        test_data : bool, optional
            If True, method generate stats for test data only.

        Returns
        -------
        pd.Series
            A series of calculated performance statistics including CAGR, Sharpe ratio,
            mean and median returns, volatility, value at risk, skewness, excess kurtosis,
            and other key metrics.
        
        Raises
        ------
        AssertionError
            If `self.portfolio_df` is not initialized or, if `market` is True, the benchmark is not set.
        """
        assert self.portfolio_df is not None, "Simulation must be run before calculating performance statistics."

        # Prepare benchmark data if market analysis is requested
        if compare:
            assert self.benchmark is not None, "A benchmark ticker must be set for market comparison."
            assert self.benchmark in self.insts, "The specified benchmark must be in the instrument list."
            market_dict = {
                f"benchmark: {self.benchmark}": self.dfs[self.benchmark]["close"]
            }
        else:
            market_dict = None

        # Calculate performance measures
        stats_dict = performance_measures(
            r_ser=self.get_zero_filtered_stats(test=test)["capital_ret"],
            plot=plot,
            market=market_dict,
            show=show,
            strat_name=strat_name
        )

        # Extract specific performance statistics
        stats = [
            "cagr", "sortino", "sharpe", "mean_ret", "median_ret",
            "vol", "var", "skew", "ex_kurtosis", "var95"
        ]
        return pd.Series({stat: stats_dict[stat] for stat in stats})

    def pre_compute(self,trade_range):
        pass
    
    def post_compute(self,trade_range):
        pass

    def compute_signal_distribution(self, eligibles, date):
        raise AbstractImplementationException("no concrete implementation for signal generation")

    def compute_meta_info(self, trade_range: pd.DatetimeIndex) -> None:
        """
        Compute meta-information necessary for backtesting, including trading days, eligibility,
        volatility, and returns for each instrument.

        Parameters
        ----------
        trade_range : pd.DatetimeIndex
            Range of dates over which the meta information will be calculated.

        Returns
        -------
        None
        """
        print("Initializing trading frequency...")
        self.compute_frequency(trade_range=trade_range)

        print("Pre-computing...")
        self.pre_compute(trade_range=trade_range)

        def is_any_one(x: np.ndarray) -> int:
            """Return 1 if any True element in the input array, otherwise return 0."""
            return int(np.any(x))

        print("Initializing meta info...")
        closes, eligibles, vols, rets, trading_days = [], [], [], [], []

        for inst in self.insts:
            df = pd.DataFrame(index=trade_range)
            inst_data = self.dfs[inst]
            inst_data = df.join(inst_data).ffill().bfill()

            # Calculate volatility and returns
            inst_vol = (inst_data["close"].pct_change().rolling(30).std())
            inst_ret = inst_data["close"].pct_change()

            inst_data["ret"] = inst_ret
            inst_data["vol"] = inst_vol.fillna(0).clip(lower=0.005)
            inst_data["trading_days"] = self.trading_day_ser.copy()
            
            sampled = inst_data["close"] != inst_data["close"].shift(1).bfill()
            eligible = sampled.rolling(5).apply(is_any_one, raw=True).fillna(0).astype(int)

            # Collect individual metrics for later DataFrame concatenation
            eligibles.append(eligible & (inst_data["close"] > 0).astype(int))
            vols.append(inst_data["vol"])
            rets.append(inst_data["ret"])
            closes.append(inst_data["close"])
            trading_days.append(inst_data["trading_days"])
            self.dfs[inst] = inst_data

        # Compile per-instrument metrics into DataFrames
        self.eligiblesdf = pd.concat(eligibles, axis=1, keys=self.insts)
        self.closedf = pd.concat(closes, axis=1, keys=self.insts)
        self.voldf = pd.concat(vols, axis=1, keys=self.insts)
        self.retdf = pd.concat(rets, axis=1, keys=self.insts)
        self.trading_days = pd.concat(trading_days, axis=1, keys=self.insts)
        self.stddevs = self.voldf.mean().values

        print("Post-computing...")
        self.post_compute(trade_range=trade_range)

        # Update forecast data based on trading day mask
        assert self.forecast_df is not None, "Forecast data must be initialized before computing meta info."
        self.forecast_df = pd.DataFrame(
            np.where(self.trading_days, self.forecast_df, np.nan),
            index=self.forecast_df.index,
            columns=self.forecast_df.columns
        )
        return

    def get_pnl_stats(
        self, 
        last_weights: np.ndarray, 
        last_units: np.ndarray, 
        prev_close: np.ndarray, 
        ret_row: np.ndarray, 
        leverages: list, 
        randomize: bool = False, 
        trading_day: bool = False, 
        rand_type: str = "gaussian"
    ) -> Tuple[float, float, float]:
        """
        Calculate daily PnL statistics.

        Parameters
        ----------
        last_weights : np.ndarray
            Portfolio weights from the previous period.
        last_units : np.ndarray
            Units held from the previous period.
        prev_close : np.ndarray
            Previous closing prices.
        ret_row : np.ndarray
            Returns for the current period.
        leverages : list
            List of leverage values.
        randomize : bool, optional
            Whether to randomize returns for simulation, by default False.
        trading_day : bool, optional
            Whether the current day is a trading day, by default False.
        rand_type : str, optional
            Type of randomization to apply ("gaussian" or "uniform"), by default "gaussian".

        Returns
        -------
        Tuple[float, float, float]
            Day PnL, nominal return, and capital return.
        """
        if randomize and trading_day:
            d = len(ret_row)
            rand_val = (
                self.stddevs * np.random.standard_normal(size=d)
                if rand_type == "gaussian" else
                np.random.uniform(low=-0.01, high=0.01, size=d)
            )
            ret_row += rand_val
        
        ret_row = np.nan_to_num(ret_row, nan=0, posinf=0, neginf=0)
        day_pnl = np.sum(last_units * prev_close * ret_row)
        nominal_ret = np.dot(last_weights, ret_row)
        capital_ret = nominal_ret * leverages[-1]
        return day_pnl, nominal_ret, capital_ret

    def get_strat_scaler(
        self, 
        target_vol: float, 
        ewmas: list[float], 
        ewstrats: list[float]
    ) -> float:
        """
        Calculate the strategy scaler based on target volatility.

        Parameters
        ----------
        target_vol : float
            Target annualized volatility.
        ewmas : list[float]
            Exponentially weighted moving average of variances.
        ewstrats : list[float]
            Exponentially weighted moving average of strategy scalers.

        Returns
        -------
        float
            Strategy scaler.
        """
        ann_realized_vol = np.sqrt(ewmas[-1] * 253)
        strat_scaler = target_vol / ann_realized_vol * ewstrats[-1]
        return strat_scaler
    
    
    def get_strat_positions(
        self,
        forecasts: np.ndarray,
        capitals: float,
        strat_scalar: float,
        vol_row: np.ndarray,
        close_row: np.ndarray,
        vol_target: float,
        idx: int,
        **kwargs
    ) -> np.ndarray:
        """
        Delegate to the strategy's `get_strat_positions` method.

        Parameters
        ----------
        forecasts : np.ndarray
            Forecasts for the portfolio.
        capitals : float
            Current capital level.
        strat_scalar : float
            Strategy scalar for position sizing.
        vol_row : np.ndarray
            Current volatility row.
        close_row : np.ndarray
            Current closing prices.
        vol_target : float
            Target volatility for the portfolio.
        idx : int
            Current index in the simulation.
        kwargs : dict
            Additional arguments for the strategy.

        Returns
        -------
        np.ndarray
            Array of calculated positions.
        """
        return self.portf_strategy.get_strat_positions(
            forecasts, capitals, strat_scalar, vol_row, close_row, vol_target, idx, **kwargs
        )

    
    def get_positions(
        self,
        forecasts: np.ndarray,
        eligibles_row: np.ndarray,
        capitals: float,
        strat_scalar: float,
        vol_row: np.ndarray,
        close_row: np.ndarray,
        units_held: Union[np.ndarray, None],
        use_vol_target: bool,
        trading_day: bool,
        idx: int,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate portfolio positions and weights.

        Parameters
        ----------
        forecasts : np.ndarray
            Strategy forecasts for the period
        eligibles_row : np.ndarray
            Eligible assets for the current period.
        capitals : float
            Current capital level.
        strat_scalar : float
            Strategy scalar for position sizing.
        vol_row : np.ndarray
            Current volatility row.
        close_row : np.ndarray
            Current closing prices.
        units_held : np.ndarray or None
            Units held from the previous period.
        use_vol_target : bool
            Whether to use volatility targeting.
        trading_day : bool
            Whether the current day is a trading day.
        idx : int
            Current index in the simulation.
        kwargs : dict
            Additional arguments for position calculation.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            Positions, weights, and nominal total value.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            forecasts = np.nan_to_num(forecasts / eligibles_row, nan=0, posinf=0, neginf=0)
            forecast_chips = np.sum(np.abs(forecasts))
            vol_target = (self.portfolio_vol / np.sqrt(253)) * capitals

            if trading_day or idx == 0:
                if use_vol_target:
                    positions = self.get_strat_positions(
                        forecasts, capitals, strat_scalar, vol_row, close_row, vol_target, idx, **(kwargs or {})
                    )
                else:
                    dollar_allocation = capitals / forecast_chips if forecast_chips != 0 else 0
                    positions = forecasts * dollar_allocation / close_row
                    positions = np.floor(np.nan_to_num(positions, nan=0, posinf=0, neginf=0))
            else:
                positions = units_held

            nominal_tot = np.linalg.norm(positions * close_row, ord=1)
            weights = np.nan_to_num(positions * close_row / nominal_tot, nan=0, posinf=0, neginf=0)
            return positions, weights, nominal_tot

    def zip_data_generator(self) -> Generator[Dict[str, Union[int, np.ndarray]], None, None]:
        """
        Generate zipped data for simulation.

        Yields
        ------
        Dict[str, Union[int, np.ndarray]]
            Dictionary containing data for each simulation step.
        """
        for portfolio_i, (ret_i, ret_row), (close_i, close_row), \
            (eligibles_i, eligibles_row), (trading_day_i, trading_day), \
            (vol_i, vol_row) in zip(
                range(len(self.retdf)),
                self.retdf.iterrows(),
                self.closedf.iterrows(),
                self.eligiblesdf.iterrows(),
                self.trading_day_ser.items(),
                self.voldf.iterrows()
        ):
            yield {
                "portfolio_i": portfolio_i,
                "ret_i": ret_i,
                "ret_row": ret_row.values,
                "close_row": close_row.values,
                "eligibles_row": eligibles_row.values,
                "trading_day": trading_day,
                "vol_row": vol_row.values,
            }

    @timeme
    def run_simulation(
            self,
            start_cap: float = 100000.0,
            use_vol_target: bool = True,
            randomize_entry: bool = False,
            rand_type: str = "gaussian"
        ) -> pd.DataFrame:
        """
        Run a portfolio simulation over the provided date range.

        Parameters
        ----------
        start_cap : float, optional
            Initial capital for the portfolio, by default 100000.0.
        use_vol_target : bool, optional
            Whether to use a volatility target for position sizing, by default True.
        randomize_entry : bool, optional
            Whether to randomize returns for entry timing, by default False.
        rand_type : str, optional
            Type of randomization ("gaussian" or "uniform"), by default "gaussian".

        Returns
        -------
        pd.DataFrame
            DataFrame containing portfolio metrics over time.
        """
        self.compute_meta_info(trade_range=self.date_range)
        print("Running simulation...")

        # Initialize tracking variables
        units_held, weights_held = [], []
        close_prev = None
        ewmas, ewstrats = [0.01], [1.0]
        strat_scalars = []
        capitals, nominal_rets, capital_rets = [start_cap], [0.0], [0.0]
        nominals, leverages = [], []

        # Iterate over the simulation data generator
        for data in self.zip_data_generator():
            portfolio_i = data["portfolio_i"]
            ret_i = data["ret_i"]
            ret_row = data["ret_row"]
            close_row = np.nan_to_num(data["close_row"], nan=0, posinf=0, neginf=0)
            eligibles_row = data["eligibles_row"]
            trading_day = data["trading_day"]
            vol_row = data["vol_row"]
            strat_scalar = 1.0

            if portfolio_i != 0:
                strat_scalar = self.get_strat_scaler(
                    target_vol=self.portfolio_vol,
                    ewmas=ewmas,
                    ewstrats=ewstrats
                )

                day_pnl, nominal_ret, capital_ret = self.get_pnl_stats(
                    last_weights=weights_held[-1],
                    last_units=units_held[-1],
                    prev_close=close_prev,
                    ret_row=ret_row,
                    leverages=leverages,
                    randomize=randomize_entry,
                    trading_day=trading_day,
                    rand_type=rand_type
                )

                capitals.append(capitals[-1] + day_pnl)
                nominal_rets.append(nominal_ret)
                capital_rets.append(capital_ret)
                ewmas.append(0.06 * (capital_ret ** 2) + 0.94 * ewmas[-1] if capital_ret != 0 else ewmas[-1])
                ewstrats.append(0.06 * strat_scalar + 0.94 * ewstrats[-1] if capital_ret != 0 else ewstrats[-1])

            strat_scalars.append(strat_scalar)

            # Compute forecasts
            forecasts = self.compute_signal_distribution(
                eligibles_row,
                ret_i
            )
            forecasts = forecasts.values if isinstance(forecasts, pd.Series) else forecasts

            # Strategy-specific arguments
            kwargs = {}
            if isinstance(self.portf_strategy, MixtureModelsMeanVarianceStrategy):
                kwargs = {
                    "max_leverage": self.max_leverage,
                    "retdf": self.retdf,
                    "trade_frequency": self.trade_frequency
                }
            elif isinstance(self.portf_strategy, VolatilityTargetingStrategy):
                kwargs = {
                    "max_leverage": self.max_leverage,
                    "min_leverage": self.min_leverage
                }

            # Calculate positions
            positions, weights, nominal_tot = self.get_positions(
                forecasts=forecasts,
                eligibles_row=eligibles_row,
                capitals=capitals[-1],
                strat_scalar=strat_scalar,
                vol_row=vol_row,
                close_row=close_row,
                units_held=units_held[-1] if units_held else None,
                use_vol_target=use_vol_target,
                trading_day=trading_day,
                idx=portfolio_i,
                **kwargs
            )

            # Update tracking variables
            units_held.append(positions)
            weights_held.append(weights)
            nominals.append(nominal_tot)
            leverages.append(nominal_tot / capitals[-1])
            close_prev = close_row

        # Prepare result DataFrame
        units_df = pd.DataFrame(data=units_held, index=self.date_range, columns=[f"{inst} units" for inst in self.insts])
        weights_df = pd.DataFrame(data=weights_held, index=self.date_range, columns=[f"{inst} w" for inst in self.insts])
        nom_ser = pd.Series(data=nominals, index=self.date_range, name="nominal_tot")
        lev_ser = pd.Series(data=leverages, index=self.date_range, name="leverages")
        cap_ser = pd.Series(data=capitals, index=self.date_range, name="capital")
        nomret_ser = pd.Series(data=nominal_rets, index=self.date_range, name="nominal_ret")
        capret_ser = pd.Series(data=capital_rets, index=self.date_range, name="capital_ret")
        scaler_ser = pd.Series(data=strat_scalars, index=self.date_range, name="strat_scalar")

        self.portfolio_df = pd.concat([
            units_df,
            weights_df,
            lev_ser,
            scaler_ser,
            nom_ser,
            nomret_ser,
            capret_ser,
            cap_ser
        ], axis=1)

        self.units_df = units_df
        self.weights_df = weights_df
        self.leverages = lev_ser
        return self.portfolio_df
    
    def run_randomized_entry_simulation(
        self, 
        N: int = 50, 
        use_vol_target: bool = True, 
        market: bool = False, 
        rand_type: str = "gaussian"
    ) -> None:
        """
        Runs a simulation with randomized entry conditions multiple times 
        to assess portfolio performance variability.

        Parameters
        ----------
        N : int, optional
            Number of randomized simulations to run, by default 50.
        use_vol_target : bool, optional
            Whether to use volatility targeting in the simulations, by default True.
        market : bool, optional
            Whether to include market benchmark comparison, by default False.
        rand_type : str, optional
            Type of randomization ("gaussian" or "uniform"), by default "gaussian".

        Returns
        -------
        None
            Generates plots and displays Sharpe ratios for randomized entry simulations.
        """
        # Run base simulation without randomization
        portfolio_df = self.run_simulation(
            use_vol_target=use_vol_target, 
            randomize_entry=False
        )
        
        # Collect capital returns for all simulations
        caps = [portfolio_df["capital_ret"].loc[portfolio_df["capital_ret"] != 0]]
        
        # Run additional simulations with randomized entry
        for _ in range(N):
            portfolio_df = self.run_simulation(
                use_vol_target=use_vol_target, 
                randomize_entry=True,
                rand_type=rand_type
            )
            caps.append(portfolio_df["capital_ret"].loc[portfolio_df["capital_ret"] != 0])

        # Combine results into a single DataFrame
        caps = pd.concat(caps, axis=1)
        caps.columns = ["Portfolio"] + [f"sim_{i}" for i in range(N)]
        
        # Calculate Sharpe ratios for each simulation
        sharpe = lambda ser: np.mean(ser) / np.std(ser) * np.sqrt(253)
        sharpes = caps[caps != 0].apply(sharpe, axis=0)

        # Include market benchmark if specified
        market_data = None
        if market:
            if self.benchmark is None:
                raise ValueError("Benchmark ticker name is not set. Please specify a valid benchmark.")
            if self.benchmark not in self.insts:
                raise ValueError("Benchmark does not exist in the instrument list. Please verify its presence.")
            
            market_data = self.dfs[self.benchmark]["close"].pct_change().fillna(0)

        # Generate plots for the results
        plot_random_entries(
            caps, 
            sharpes, 
            market_data
        )


    @timeme
    def run_hypothesis_tests(
            self,
            num_decision_shuffles: int = 1000,
            strat_name: Optional[str] = None
        ) -> None:
        """
        Run hypothesis tests and plot results for the strategy.

        Parameters
        ----------
        num_decision_shuffles : int, optional
            Number of shuffles for hypothesis testing, by default 1000.
        strat_name : str | None, optional
            Name of the strategy, used for labeling in plots, by default None.

        Returns
        -------
        None
        """
        assert self.portfolio_df is not None, "Portfolio data must be initialized before running hypothesis tests."
        zfs = self.get_zero_filtered_stats()
        rets = zfs["capital_ret"]

        # Run hypothesis tests and retrieve test statistics
        test_dict = quant_stats.hypothesis_tests(num_decision_shuffles=num_decision_shuffles, zfs=zfs)

        # Plot hypothesis test results
        plot_hypothesis(test_dict["timer"], test_dict["picker"], test_dict["trader"], rets, strat_name)
        return
