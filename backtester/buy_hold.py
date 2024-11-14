from datetime import datetime
from backtester import BacktestEngine
from pandas.core.api import DataFrame as DataFrame, DatetimeIndex

import pandas as pd
import numpy as np


class BuyHold(BacktestEngine):
    def __init__(self, insts: list[str], dfs: dict[str, DataFrame], start: datetime | None = None, end: datetime | None = None, date_range: DatetimeIndex | None = None, trade_frequency: str | None = None, day_of_week: str | None = None, portfolio_vol: float = 0.2, max_leverage: float = 2, min_leverage: float = 0, benchmark: str | None = None) -> None:
        super().__init__(insts, dfs, start, end, date_range, trade_frequency, day_of_week, portfolio_vol, max_leverage, min_leverage, benchmark)
    
    def pre_compute(self,trade_range):
        return 
    
    def post_compute(self,trade_range):
        forecast_df = []
    
        for inst in self.insts:
            inst_df = self.dfs[inst]
            inst_df["vals"] = np.ones(len(inst_df))
            forecast_df.append(inst_df["vals"])

        alphadf = pd.concat(forecast_df,axis=1)
        alphadf.columns = self.insts

        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alphadf))
        masked_df = alphadf/self.eligiblesdf
        masked_df = masked_df.replace([-np.inf, np.inf], np.nan)

        rankdf = masked_df.rank(axis=1,method="average",na_option="keep",ascending=False)

        forecast_df = masked_df
        self.forecast_df = forecast_df
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts