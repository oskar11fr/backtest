from backtester import BacktestEngine, property_initializer, alpha_calculator, eligibility_calculator, forecast_calculator
from backtester.engine.functions.portfolio_optimization import PositioningMethod, VanillaVolatilityTargeting
from backtester.engine.functions import quant_tools as qt
from pandas import DataFrame, DatetimeIndex
from typing import Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TestStrategy(BacktestEngine):
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
            portfolio_vol: float = 0.15, 
            max_leverage: float = 2, 
            min_leverage: float = 0, 
            benchmark: str | None = None,
            load_model: bool = False,
            costs: dict[str, float] = {"slippage": 0.}
        ) -> None:
        trade_frequency = "monthly"
        
        super().__init__(
            insts, dfs, start, end, date_range, trade_frequency, day_of_week, 
            use_portfolio_opt, portf_optimization, portfolio_vol, max_leverage, 
            min_leverage, benchmark, costs
        )
    
    @property_initializer
    def property_func(self, df: DataFrame) -> pd.DataFrame:
        df["market_cap"] = df["numberOfShares"] * df["close"]
        return df

    @alpha_calculator
    def alpha_func(self, df: DataFrame) -> pd.Series:
        df["momentum"] = qt.momentum_distance(ser=df["close"],ewm_val=0.5)
        return df["momentum"]

    @eligibility_calculator
    def eligibility_func(self, df: DataFrame) -> pd.Series:
        df["zscore_long"] = qt.zscore(
            ser=df["ret"].rolling(150).mean(), 
            wind=150, 
            expected_value=df["indx_ret"].rolling(150).mean()
        )
        return (
            #df["market"].isin(["Large Cap", "Mid Cap", "Small Cap"]) &
            (df["market_cap"] > 1000) &
            (df["zscore_long"] > 1.5) & 
            (df["earningsPerShare"] > 0) & 
            (df["earningsPerShare_rel_ch"] > 0) &
            (df["indx_mom"] > 0) 
        )
    
    @forecast_calculator
    def forecast_func(self, df: DataFrame) -> DataFrame:
        rank_df = qt.x_rank(forecast_df=df,eligibles_df=self.eligibles_df,ascending=False,num_insts=20)
        return rank_df
    
    def pre_compute(self):
        pass 
    
    def post_compute(self):
        _, group_indxs = qt.calculate_indxs(dfs=self.dfs, insts=self.insts, cat="branch")
        for inst in self.insts:
            inst_df = self.dfs[inst]
            gr = inst_df["branch"].dropna().unique()
            if len(gr) > 0:
                gr = gr[0]
                inst_df["indx_ret"] = group_indxs[gr]
                inst_df["indx_mom"] = qt.momentum_distance((1 + group_indxs[gr]).cumprod(),wind_val=150)
            else:
                inst_df["indx_ret"] = 0
                inst_df["indx_mom"] = 0
            self.dfs[inst] = inst_df
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts