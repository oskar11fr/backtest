from datetime import datetime
from pandas.core.api import DataFrame as DataFrame, DatetimeIndex
from backtester import (
    BacktestEngine, 
    alpha_calculator, 
    eligibility_calculator, 
    property_initializer
)
from backtester.engine.functions.portfolio_optimization import (
    PositioningMethod, 
    VanillaVolatilityTargeting, 
    MixtureModelsMeanVariance
)

import pandas as pd
import numpy as np



class Strategy(BacktestEngine):
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
            portf_optimization: str = "vol_targ", 
            portfolio_vol: float = 0.2, 
            max_leverage: float = 2, 
            min_leverage: float = 0, 
            benchmark: str | None = None,
            load_model: bool = False,
            costs: dict[str, float] = {"slippage": 0.}
        ) -> None:
        
        super().__init__(
            insts, dfs, start, end, date_range, trade_frequency, day_of_week, use_portfolio_opt, portf_optimization, 
            portfolio_vol, max_leverage, min_leverage, benchmark, costs
        )
    
    @property_initializer
    def property_func(self, df: pd.DataFrame) -> pd.DataFrame:
        return

    @alpha_calculator
    def alpha_func(self, df: pd.DataFrame) -> pd.Series:
        return

    @eligibility_calculator
    def eligibility_func(self, df: pd.DataFrame) -> pd.Series:
        return

    def pre_compute(self):
        return 
    
    def post_compute(self):
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts