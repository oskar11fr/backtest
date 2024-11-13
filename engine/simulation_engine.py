import os
import time

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt


from pathlib import Path
from copy import deepcopy
from typing import Optional
from functools import wraps
from datetime import datetime
from collections import defaultdict

from .functions import quant_stats as quant_stats
from .functions.performance import (
    plot_hypothesis, 
    plot_random_entries, 
    performance_measures
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
        day_of_week = self.day_of_week or "Friday"
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
            trade_frequency: str = "daily", 
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

        # Initialize parameters
        self.insts = insts
        self.dfs = deepcopy(dfs)
        self.datacopy = deepcopy(dfs)
        self.portfolio_vol = portfolio_vol
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.benchmark = benchmark

        # Initialize additional attributes
        self.alphadf = None  # Placeholder for alpha data, if applicable
        self.day_of_week = None  # Placeholder for weekly trade day specification

        # Determine date range for the backtest
        if date_range is None:
            self.date_range = pd.date_range(start=start, end=end, freq="D")
            self.trade_frequency = trade_frequency
        else:
            self.date_range = date_range
            self.trade_frequency = trade_frequency  # Retain the provided frequency setting


    def get_zero_filtered_stats(self) -> dict[str, pd.Series | pd.DataFrame]:
        """_summary_

        Returns:
            dict[str, pd.Series]: _description_
        """
        assert self.portfolio_df is not None
        nominal_ret = self.portfolio_df.nominal_ret
        capital_ret = self.portfolio_df.capital_ret
        non_zero_idx = capital_ret.loc[capital_ret != 0].index
        retdf = self.retdf.loc[non_zero_idx]
        weights = self.weights_df.shift(1).fillna(0).loc[non_zero_idx]
        eligs = self.eligiblesdf.shift(1).fillna(0).loc[non_zero_idx]
        leverages = self.leverages.shift(1).fillna(0).loc[non_zero_idx]
        return {
            "capital_ret": capital_ret.loc[non_zero_idx],
            "nominal_ret": nominal_ret.loc[non_zero_idx],
            "retdf":retdf,
            "weights":weights,
            "eligs":eligs,
            "leverages":leverages,
        }

    def get_perf_stats(
            self,
            plot: bool = False,
            market: bool = False,
            show: bool = False,
            strat_name: str = None
        ) -> pd.Series:
        """_summary_

        Args:
            plot (bool, optional): _description_. Defaults to False.
            market (bool, optional): _description_. Defaults to False.
            show (bool, optional): _description_. Defaults to False.
            strat_name (str, optional): _description_. Defaults to None.

        Returns:
            pd.Series: _description_
        """
        assert self.portfolio_df is not None, "run simulation"
        if market:
            assert self.benchmark is not None, "set ticker name as benchmark"
            assert self.benchmark in self.insts, "make sure benchmark exists in instrument list"
            market_dict = {
                "benchmark: " + self.benchmark: 
                self.dfs[self.benchmark]["close"]
            }
        
        stats_dict = performance_measures(
            r_ser=self.get_zero_filtered_stats()["capital_ret"],
            plot=plot,
            market=None if not market else market_dict,
            show=show,
            strat_name=strat_name
        )
        stats = [
            "cagr",
            "srtno",
            "sharpe",
            "mean_ret",
            "median_ret",
            "vol",
            "var",
            "skew",
            "exkurt",
            "var95"
        ]
        return pd.Series({stat: stats_dict[stat] for stat in stats})
    
    def hypothesis_tests(
            self, 
            num_decision_shuffles: int = 1000,
            zfs: dict[str, pd.Series | pd.DataFrame] | None = None
        ):
        """_summary_

        Args:
            num_decision_shuffles (int, optional): _description_. Defaults to 1000.
            zfs (dict[str, pd.Series | pd.DataFrame] | None, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        retdf, leverages, weights, eligs = zfs["retdf"], zfs["leverages"], zfs["weights"], zfs["eligs"]

        def performance_criterion(rets, leverages, weights, **kwargs):
            capital_ret = [
                lev_scalar * np.dot(weight, ret)
                for lev_scalar, weight, ret in zip(leverages.values, rets.values, weights.values)
            ]
            sharpe = np.mean(capital_ret) / np.std(capital_ret) * np.sqrt(253)
            return round(sharpe, 5), capital_ret

        def time_shuffler(rets, leverages, weights, eligs):
            nweights = quant_stats.shuffle_weights_on_eligs(weights_df=weights, eligs_df=eligs, method="time",ord=1)
            return {"rets": rets, "leverages": leverages, "weights": nweights}

        def mapping_shuffler(rets, leverages, weights, eligs):
            nweights = quant_stats.shuffle_weights_on_eligs(weights_df=weights, eligs_df=eligs, method="xs")
            return {"rets": rets, "leverages": leverages, "weights": nweights}

        def decision_shuffler(rets, leverages, weights, eligs):
            nweights = quant_stats.shuffle_weights_on_eligs(weights_df=weights, eligs_df=eligs, method="time",ord=1)
            nweights = quant_stats.shuffle_weights_on_eligs(weights_df=nweights, eligs_df=eligs, method="xs")
            return {"rets": rets, "leverages": leverages, "weights": nweights}

        timer_paths, timer_p, timer_dist = quant_stats.permutation_shuffler_test(
            criterion_function=performance_criterion,
            generator_function=time_shuffler,
            m=num_decision_shuffles, rets=retdf, leverages=leverages, weights=weights, eligs=eligs)
        
        picker_paths, picker_p, picker_dist = quant_stats.permutation_shuffler_test(
            criterion_function=performance_criterion,
            generator_function=mapping_shuffler,
            m=num_decision_shuffles, rets=retdf, leverages=leverages, weights=weights, eligs=eligs)
        
        trader_paths, trader_p, trader_dist = quant_stats.permutation_shuffler_test(
            criterion_function=performance_criterion,
            generator_function=decision_shuffler,
            m=num_decision_shuffles, rets=retdf, leverages=leverages, weights=weights, eligs=eligs)
        return {
            "timer":(timer_paths,timer_p,timer_dist),
            "picker":(picker_paths,picker_p,picker_dist),
            "trader":(trader_paths,trader_p,trader_dist),
        }
    
    @timeme
    def run_hypothesis_tests(
            self,
            num_decision_shuffles: int = 1000,
            # zfs: dict[str, pd.DataFrame] | None = None, 
            strat_name: str | None = None
        ) -> None:
        assert self.portfolio_df is not None
        zfs = self.get_zero_filtered_stats()
        rets = zfs["capital_ret"]
        test_dict = self.hypothesis_tests(num_decision_shuffles=num_decision_shuffles,zfs=zfs)
        plot_hypothesis(test_dict["timer"],test_dict["picker"],test_dict["trader"],rets,strat_name)
        return

    def pre_compute(self,trade_range):
        pass
    
    def post_compute(self,trade_range):
        pass

    def compute_signal_distribution(self, eligibles, date):
        raise AbstractImplementationException("no concrete implementation for signal generation")

    def compute_meta_info(self,trade_range):
        print("Initializing trading frequency...")
        self.compute_frequency(trade_range=trade_range)

        print("Pre - computing...")
        self.pre_compute(trade_range=trade_range)
        
        def is_any_one(x):
            return int(np.any(x))
        
        print("Initializing meta info...")
        closes, eligibles, vols, rets, trading_day, mean_rets, std_rets = [], [], [], [], [], [], []
        for inst in self.insts:
            df=pd.DataFrame(index=trade_range)
            inst_vol = (-1 + self.dfs[inst]["close"]/self.dfs[inst]["close"].shift(1)).rolling(30).std()
            self.dfs[inst] = df.join(self.dfs[inst]).ffill().bfill()
            self.dfs[inst]["ret"] = -1 + (self.dfs[inst]["close"]/self.dfs[inst]["close"].shift(1))
            self.dfs[inst]["vol"] = inst_vol
            self.dfs[inst]["vol"] = self.dfs[inst]["vol"].ffill().fillna(0)       
            self.dfs[inst]["vol"] = np.where(self.dfs[inst]["vol"] < 0.005, 0.005, self.dfs[inst]["vol"])
            self.dfs[inst]['trading_day'] = self.trading_day_ser.copy()
            sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).bfill()
            eligible = sampled.rolling(5).apply(is_any_one,raw=True).fillna(0)

            mean_rets.append(self.dfs[inst]["ret"].mean())
            std_rets.append(self.dfs[inst]["vol"].mean())
            eligibles.append(eligible.astype(int) & (self.dfs[inst]["close"] > 0).astype(int))
            vols.append(self.dfs[inst]["vol"])
            rets.append(self.dfs[inst]["ret"])
            closes.append(self.dfs[inst]["close"])
            trading_day.append(self.dfs[inst]["trading_day"])

        self.eligiblesdf = pd.concat(eligibles,axis=1)
        self.eligiblesdf.columns = self.insts
        self.closedf = pd.concat(closes,axis=1)
        self.closedf.columns = self.insts
        self.voldf = pd.concat(vols,axis=1)
        self.voldf.columns = self.insts
        self.retdf = pd.concat(rets,axis=1)
        self.retdf.columns = self.insts
        self.trading_day = pd.concat(trading_day, axis=1)
        self.trading_day.columns = self.insts
        self.mean_rets = pd.Series(mean_rets,index=self.insts)
        self.std_rets = pd.Series(std_rets,index=self.insts)

        print("Post - computing...")
        self.post_compute(trade_range=trade_range)
        assert self.forecast_df is not None
        self.forecast_df = pd.DataFrame(
            np.where(self.trading_day,self.forecast_df,np.nan),
            index=self.forecast_df.index,
            columns=self.forecast_df.columns
        )
        return
    
    # def save_strat_rets(self, strat_name):
    #     assert self.portfolio_df is not None
    #     store_path = STRAT_PATH + "/strategy_returns"
    #     Path(os.path.abspath(store_path)).mkdir(parents=True,exist_ok=True)
    #     capital_rets = self.portfolio_df["capital"].pct_change().fillna(0)
    #     save_pickle(store_path + f"/{strat_name}.obj",obj=capital_rets)
    #     return
    
    def get_pnl_stats(self, last_weights, last_units, prev_close, ret_row, leverages, randomize=False, trading_day=False, rand_type = "gaussian"):
        if randomize and trading_day:
            d = len(ret_row)
            rand_val = self.std_rets * np.random.standard_normal(size=d) if rand_type == "gaussian" else \
                np.random.uniform(low=-0.01,high=0.01,size=d)
            ret_row += rand_val
        ret_row = np.nan_to_num(ret_row,nan=0,posinf=0,neginf=0)
        day_pnl = np.sum(last_units * prev_close * ret_row)
        nominal_ret = np.dot(last_weights, ret_row)
        capital_ret = nominal_ret * leverages[-1]
        return day_pnl, nominal_ret, capital_ret   

    def get_strat_scaler(self, target_vol, ewmas, ewstrats):
        ann_realized_vol = np.sqrt(ewmas[-1] * 253)
        strat_scaler = target_vol / ann_realized_vol * ewstrats[-1]
        return strat_scaler
    
    def get_positions(
            self,
            forecasts,
            eligibles_row,
            capitals,
            trading_day,
            i,
            use_vol_target,
            strat_scalar,
            vol_row,
            close_row,
            units_held
        ):

        forecasts = forecasts / eligibles_row
        forecasts = np.nan_to_num(forecasts,nan=0,posinf=0,neginf=0)
        forecast_chips = np.sum(np.abs(forecasts))
        vol_target = (self.portfolio_vol / np.sqrt(253)) * capitals

        if trading_day or (i==0):
            if use_vol_target:
                positions = strat_scalar * \
                    forecasts / forecast_chips  \
                    * vol_target \
                    / (vol_row * close_row) if forecast_chips != 0 else np.zeros(len(self.insts))
                lev_temp = np.linalg.norm(positions * close_row, ord=1) / capitals
                with np.errstate(invalid="ignore", divide="ignore"):
                    normalized_positions = positions / lev_temp
                positions = self.max_leverage*normalized_positions if lev_temp > self.max_leverage else positions
                positions = self.min_leverage*normalized_positions if lev_temp < self.min_leverage else positions
                positions = np.floor(np.nan_to_num(positions,nan=0,posinf=0,neginf=0))
            
            else:
                dollar_allocation = capitals/forecast_chips if forecast_chips != 0 else np.zeros(len(self.insts))
                positions = forecasts*dollar_allocation / close_row
                positions = np.floor(np.nan_to_num(positions,nan=0,posinf=0,neginf=0)) # added floor function
        
        else:
            positions = units_held

        nominal_tot = np.linalg.norm(positions * close_row, ord=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            weights = positions * close_row / nominal_tot
        weights = np.nan_to_num(weights,nan=0,posinf=0,neginf=0)
        return positions, weights, nominal_tot

    @timeme
    def run_simulation(
            self,
            start_cap: float = 100000.0,
            use_vol_target: bool = True,
            randomize_entry: bool = False,
            rand_type: str = "gaussian"
        ) -> pd.DataFrame:
        self.compute_meta_info(trade_range=self.date_range)
        print("Running simulation...")
        units_held, weights_held = [],[]
        close_prev = None
        ewmas, ewstrats = [0.01], [1]
        strat_scalars = []
        capitals, nominal_rets, capital_rets = [start_cap],[0.0],[0.0]
        nominals, leverages = [],[]
        for data in self.zip_data_generator():
            portfolio_i = data["portfolio_i"]
            ret_i = data["ret_i"]
            ret_row = data["ret_row"]
            close_row = np.nan_to_num(data["close_row"],nan=0,posinf=0,neginf=0)
            eligibles_row = data["eligibles_row"]
            trading_day = data["trading_day"]
            vol_row = data["vol_row"]
            strat_scalar = 1
           
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
                ewmas.append(0.06 * (capital_ret**2) + 0.94 * ewmas[-1] if capital_ret != 0 else ewmas[-1])
                ewstrats.append(0.06 * strat_scalar + 0.94 * ewstrats[-1] if capital_ret != 0 else ewstrats[-1])

            strat_scalars.append(strat_scalar)
            forecasts = self.compute_signal_distribution(
                eligibles_row,
                ret_i
            )
            if type(forecasts) == pd.Series: forecasts = forecasts.values

            positions, weights, nominal_tot = self.get_positions(
                forecasts,
                eligibles_row,
                capitals[-1],
                trading_day,
                portfolio_i,
                use_vol_target,
                strat_scalar,
                vol_row,
                close_row,
                units_held[-1] if len(units_held) != 0 else None
            )
            units_held.append(positions)
            weights_held.append(weights)
            nominals.append(nominal_tot)
            leverages.append(nominal_tot/capitals[-1])
            close_prev = close_row
        
        units_df = pd.DataFrame(data=units_held, index=self.date_range, columns=[inst + " units" for inst in self.insts])
        weights_df = pd.DataFrame(data=weights_held, index=self.date_range, columns=[inst + " w" for inst in self.insts])
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
        ],axis=1)
        self.units_df = units_df
        self.weights_df = weights_df
        self.leverages = lev_ser
        return self.portfolio_df
    
    def run_randomized_entry_simulation(self, N: int = 50, use_vol_target: bool = True, market: bool = False, rand_type: str = "gaussian"):
        portfolio_df = self.run_simulation(use_vol_target=use_vol_target, randomize_entry=False, verbose=False)
        caps = [portfolio_df.capital_ret]
        for _ in range(N):
            portfolio_df = self.run_simulation(use_vol_target=use_vol_target, randomize_entry=True, verbose=False, rand_type=rand_type)
            caps.append(portfolio_df.capital_ret)

        caps = pd.concat(caps,axis=1)
        caps.columns = ["Portfolio"] + ["sim_" + str(i) for i in range(N)]
        sharpe = lambda ser: np.mean(ser) / np.std(ser) * np.sqrt(253)
        sharpes = caps[caps != 0].apply(sharpe, axis=0)

        if market:
            assert self.benchmark is not None, "set ticker name as benchmark"
            assert self.benchmark in self.insts, "make sure benchmark exists in instrument list"
            market = self.dfs[self.benchmark]["close"].pct_change().fillna(0)
            
        plot_random_entries(
            caps, 
            sharpes,
            market
        )
        return


    def zip_data_generator(self):
        for (portfolio_i),\
            (ret_i, ret_row), \
            (close_i, close_row), \
            (eligibles_i, eligibles_row), \
            (trading_day_i,trading_day), \
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