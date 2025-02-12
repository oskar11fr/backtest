from backtester import BacktestEngine, property_initializer, alpha_calculator, eligibility_calculator, forecast_calculator
import pandas as pd
from pandas import DataFrame, DatetimeIndex
from datetime import datetime
from typing import Optional, Union
import numpy as np
from backtester.engine.functions.portfolio_optimization import PositioningMethod, VanillaVolatilityTargeting


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
        df["ewm"] = df["close"].rolling(50).mean()
        df["ewm_distance"] = df["close"] / df["ewm"] - 1
        return df

    @alpha_calculator
    def alpha_func(self, df: DataFrame) -> pd.Series:
        return df["ewm_distance"]

    @eligibility_calculator
    def eligibility_func(self, df: DataFrame) -> pd.Series:
        # Implement your eligibility calculation here.
        return df["ewm_distance"] > 0
    
    @forecast_calculator
    def forecast_func(self, df: DataFrame) -> DataFrame:
        # Implement your forecast calculation here.
        print(df)
        df = df.rank(axis=1,method="average",na_option="keep",ascending=True,pct=True)
        input(df)
        return (df > .95).astype(int)
    
    def pre_compute(self):
        # Implement your pre-compute steps here.
        pass 
    
    def post_compute(self):
        # Implement your post-compute steps here.
        pass 

    def compute_signal_distribution(self, eligibles, date):
        # Implement your signal distribution logic here.
        forecasts = self.forecast_df.loc[date].values
        return forecasts