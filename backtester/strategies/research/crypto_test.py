from datetime import datetime
from pandas import DatetimeIndex
from pandas.core.api import DataFrame as DataFrame
from backtester import BacktestEngine
from hmmlearn.hmm import GaussianHMM
from backtester.engine.functions.portfolio_optimization import PositioningMethod, VanillaVolatilityTargeting
from backtester.engine import save_obj, load_obj


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class VolCarry(BacktestEngine):
    def __init__(
            self, 
            insts: list[str], 
            dfs: dict[str, DataFrame], 
            start: datetime | None = None, 
            end: datetime | None = None, 
            date_range: DatetimeIndex | None = None, 
            trade_frequency: str | None = None,
            day_of_week: str | None = None,
            use_portfolio_opt: bool = True,
            portf_optimization: PositioningMethod = VanillaVolatilityTargeting(), 
            portfolio_vol: float = 0.2, 
            max_leverage: float = 2, 
            min_leverage: float = 0, 
            benchmark: str | None = None,
            train_size: float = 1.,
            costs: dict[str, float] = {"slippage": 0.},
        ) -> None:
        trade_frequency = None
        super().__init__(
            insts, dfs, start, end, date_range, trade_frequency, day_of_week, use_portfolio_opt, portf_optimization, 
            portfolio_vol, max_leverage, min_leverage, benchmark, train_size, costs
        )

    def pre_compute(self,trade_range):
        def labeler(ser): return np.where(ser > 1, 1, np.where(ser < 0, -1, np.nan))

        for inst in self.insts:
            cl = self.dfs[inst]["close"]
            ave = cl.rolling(20).mean()
            lob = ave - 2.5 * cl.rolling(20).std()
            upb = ave + 2.5 * cl.rolling(20).std()

            w = (cl - lob) / (upb - lob)
            self.dfs[inst]["critera"] = w.apply(labeler)
        
        return

    def post_compute(self,trade_range):
        alpha_df = self.compute_strategy(trade_range=trade_range)
        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alpha_df))
        self.forecast_df = alpha_df
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts
    

    def compute_entry(self, trade_range):
        self.pre_compute(trade_range=trade_range)
        entry_ser_list = []
        for inst in self.insts:
            inst_df = self.dfs[inst]
            inst_df["rolling_criteria"] = inst_df["criteria"].ffill()
            inst_df["criteria"] = inst_df["criteria"].fillna(0)

            position = 0
            trade_dir = 0
            prev_rolling_state = 0
            entry_list = []
            for idx in trade_range:
                row = inst_df.loc[idx]
                state = row["critera"]
                rolling_state = row["rolling_criteria"]

                if trade_dir != 0 and state == 0:
                    position = trade_dir 

                if prev_rolling_state == 1 and state == 1: trade_dir = 1
                if prev_rolling_state == -1 and state == -1: trade_dir = -1
                else: trade_dir = 0 
                
                prev_rolling_state = rolling_state
                entry_list.append(position)
            entry_ser_list.append(pd.Series(entry_list, index=trade_range, name=inst))
        entry_df = pd.concat(entry_ser_list, axis=1)
        return entry_df