from datetime import datetime
from pandas import DatetimeIndex
from pandas.core.api import DataFrame as DataFrame
from backtester import BacktestEngine

import numpy as np
import pandas as pd

from backtester.engine.functions.portfolio_optimization import (
    PositioningMethod,
    VanillaVolatilityTargeting,
    MixtureModelsMeanVariance
)


class Tactical(BacktestEngine):
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
            train_size: float = 0.6,
            costs: dict[str, float] = {"slippage": 0.}
        ) -> None:
        trade_frequency = "weekly"
        super().__init__(
            insts, dfs, start, end, date_range, trade_frequency, day_of_week, use_portfolio_opt, portf_optimization, 
            portfolio_vol, max_leverage, min_leverage, benchmark, train_size, costs
        )
        
    def pre_compute(self,trade_range):
        return 
    
    def post_compute(self,trade_range):
        forecast_df = []
    
        for inst in self.insts:
            inst_df = self.dfs[inst]
            inst_df["vals"] = inst_df["close"] / inst_df["close"].ewm(0.5).mean()
            inst_df["vals"] /= inst_df["vals"].ewm(0.5).std() * np.sqrt(252)
            forecast_df.append(inst_df["vals"])

        alphadf = pd.concat(forecast_df,axis=1)
        alphadf.columns = self.insts

        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alphadf))
        masked_df = alphadf/self.eligiblesdf
        masked_df = masked_df.replace([-np.inf, np.inf], np.nan)

        rankdf = masked_df.rank(axis=1,method="average",na_option="keep",ascending=True,pct=True)
        self.forecast_df = rankdf
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts