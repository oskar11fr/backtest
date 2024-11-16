from datetime import datetime
from pandas import DatetimeIndex
from pandas.core.api import DataFrame as DataFrame
from backtester import BacktestEngine
from backtester.engine.functions.portfolio_strategies import PositioningStrategy, VolatilityTargetingStrategy

import numpy as np
import pandas as pd



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
            portf_strategy: PositioningStrategy = VolatilityTargetingStrategy(), 
            portfolio_vol: float = 0.2, 
            max_leverage: float = 2, 
            min_leverage: float = 0, 
            benchmark: str | None = None
        ) -> None:
        trade_frequency = "weekly"
        super().__init__(insts, dfs, start, end, date_range, trade_frequency, day_of_week, portf_strategy, portfolio_vol, max_leverage, min_leverage, benchmark)

    def pre_compute(self,trade_range):
        return
    
    def prep_conditions(self,trade_range):

        def is_all_true(x): return np.all(x)
        def is_any_true(x): return np.any(x)

        vix, vix3m, vix6m = self.dfs['_VIX']["close"], self.dfs['_VIX3M']["close"], self.dfs['_VIX6M']["close"]
        shortend_ratio, longend_ratio  = vix3m / vix, vix6m / vix3m
        filt_shortend_ratio = shortend_ratio / shortend_ratio.rolling(30).mean()
        filt_longend_ratio = longend_ratio / longend_ratio.rolling(30).mean()

        cond_above = (filt_shortend_ratio > 1).rolling(2).apply(is_all_true, raw=True) \
           * (filt_longend_ratio > 1).rolling(2).apply(is_all_true, raw=True)
        
        short_cond = cond_above.fillna(0).astype(int)
        neutral_pos = pd.Series(np.zeros(len(trade_range)),index=trade_range)
        temp_alpha = []
        for inst in self.insts:
            if inst == "SVXY": temp_alpha.append(short_cond)
            else: temp_alpha.append(neutral_pos)
        
        alpha_df = pd.concat(temp_alpha,axis=1)
        alpha_df.columns = self.insts
        return alpha_df
    
    def post_compute(self,trade_range):
        alpha_df = self.prep_conditions(trade_range=trade_range)
        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alpha_df))
        self.forecast_df = alpha_df
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts
    

    # def calc_rolling_beta(self,trade_range,window=50):
    #     y, x = self.dfs["SPY"]["ret"].fillna(0), self.dfs["_VIX"]["close"].apply(np.log).diff().fillna(0)
    #     betas = np.full(len(y), np.nan)
    #     for i in range(window, len(y)):
    #         y_window = y[i - window:i].values
    #         x_window = x[i - window:i].values
    #         X = np.vstack([x_window, np.ones(window)]).T
    #         Y = y_window.reshape(-1, 1)
    #         beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    #         betas[i] = beta[0, 0] 
    #     return pd.Series(betas, index=trade_range, name="Rolling_Beta").fillna(0)