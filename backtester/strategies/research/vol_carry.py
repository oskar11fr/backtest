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
            costs: dict[str, float] = {"slippage": 0.5 / 100},
            train: bool = False
        ) -> None:
        day_of_week = "Tuesday"
        trade_frequency = "weekly"
        self.train = True#train
        super().__init__(
            insts, dfs, start, end, date_range, trade_frequency, day_of_week, use_portfolio_opt, portf_optimization, 
            portfolio_vol, max_leverage, min_leverage, benchmark, train_size, costs
        )

    def pre_compute(self,trade_range):
        vix, vix3m, vix6m = self.dfs['_VIX']["close"], self.dfs['_VIX3M']["close"], self.dfs['_VIX6M']["close"]
        shortend_ratio, longend_ratio  = vix3m / vix, vix6m / vix3m

        filt_shortend_ratio = shortend_ratio / shortend_ratio.ewm(.81).mean()
        filt_longend_ratio = longend_ratio / longend_ratio.ewm(.81).mean()

        X_df = pd.concat([filt_shortend_ratio,filt_longend_ratio],axis=1)
        X_df = pd.DataFrame(index=trade_range).join(X_df)
        X = X_df.fillna(1.).values
        return X
    
    def get_model(self, X, train: bool = False) -> GaussianHMM:
        if train:
            return self.train_hmm(X=X)
        else:
            m = load_obj("vol_carry_model")
            if m is None:
                m = self.train_hmm(X=X)
                save_obj(m,"vol_carry_model")
            return m
    
    def train_hmm(self, X) -> GaussianHMM:
        models, scores = [], []
        for s in range(10):
            model = GaussianHMM(n_components=2,covariance_type="full",algorithm="map",tol=0.00001,random_state=s)
            model.fit(X=X)
            models.append(model)
            scores.append(model.score(X=X))
            print(f'Converged: {model.monitor_.converged} --- Score: {scores[-1]}')
        model = models[np.argmax(scores)]
        print(f'The best model had a score of {max(scores)}')
        return model
    
    def predict(self, X, m, n, target_state, trade_range) -> pd.Series:
        preds = m.predict_proba(X[:n,:])[:,target_state].tolist()
        for i in range(len(trade_range)):
            if i >= n:
                preds.append(
                    m.predict_proba(X[:i,:])[:,target_state][-1]
                )
            else: pass
        
        return pd.Series(preds,index=trade_range)
        
    
    def prep_conditions(self,trade_range):
        n = int(len(trade_range)*.7)
        X = self.pre_compute(trade_range=trade_range)
        m = self.get_model(X=X[:n,:],train=self.train)

        target_state = np.unique(np.argmax(m.means_,axis=0))[0]
        preds_df = self.predict(X, m, n, target_state, trade_range)
        def any_is_one(x): return np.any(x)
        def all_is_one(x): return np.all(x)

        cond = (preds_df > .95).astype(int)#.rolling(2,min_periods=1).apply(any_is_one).astype(int)
        neutral_pos = pd.Series(np.zeros(len(trade_range)),index=trade_range)
        temp_alpha = []
        for inst in self.insts:
            if inst == "SVXY": temp_alpha.append(cond)
            elif inst == "BIL": temp_alpha.append(1-cond)
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
    