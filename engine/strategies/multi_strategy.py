import numpy as np
import pandas as pd
from engine.utils import BacktestEngine

class EquityStrat(BacktestEngine):
    def __init__(self, insts, dfs, start=None, end=None, date_range=None, trade_frequency='daily', portfolio_vol=0.2):
        super().__init__(insts, dfs, start, end, date_range, trade_frequency, portfolio_vol)

    def pre_compute(self,trade_range):
        def _calc_break_out(ser, i):
            n = [21, 63, 252]
            mean = ser.rolling(n[i]).mean()
            return (ser/mean) - 1 
        
        def is_all_one(x):
            return np.all(x)
        
        for inst in self.insts:
            inst_df = self.dfs[inst]
            inst_df["pe"] = inst_df["close"]/inst_df["eps"]
            inst_df["bv"] = inst_df["aktier"]*inst_df["close"]*inst_df["fx"]

            alpha = _calc_break_out(inst_df["close"],0)
            cond1 = (_calc_break_out(inst_df["close"],0) > 0).rolling(5).apply(is_all_one,raw=True)
            cond2 = (_calc_break_out(inst_df["close"],2) > 0).rolling(10).apply(is_all_one,raw=True)

            inst_df["cond"] = ~cond1.astype(bool) & cond2.astype(bool) & (inst_df["pe"] > 0) & (inst_df["bv"] > 1000)
            inst_df["alpha"] = alpha
            self.dfs[inst] = inst_df
        return 
    
    def post_compute(self,trade_range):
        temp_alpha = []
        temp_criterias = []
        for inst in self.insts:
            inst_df = self.dfs[inst]
            self.dfs[inst] = inst_df
            temp_alpha.append(inst_df["alpha"])
            temp_criterias.append(inst_df["cond"])

        alphadf = pd.concat(temp_alpha,axis=1)
        criteriasdf = pd.concat(temp_criterias,axis=1)
        alphadf.columns = self.insts
        criteriasdf.columns = self.insts

        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alphadf)) & criteriasdf
        masked_df = alphadf/self.eligiblesdf
        masked_df = masked_df.replace([-np.inf, np.inf], np.nan)
        num_eligibles = self.eligiblesdf.sum(axis=1)
        rankdf = masked_df.rank(axis=1,method="average",na_option="keep",ascending=False)

        numb = num_eligibles - 10
        longdf = rankdf.apply(lambda col: col > numb, axis=0, raw=True)
        forecast_df = longdf.astype(np.int32)
        self.forecast_df = forecast_df
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts

    
class MacroStrat(BacktestEngine):
    def __init__(self, insts, dfs, start=None, end=None, date_range=None, trade_frequency='daily', portfolio_vol=0.2):
        super().__init__(insts, dfs, start, end, date_range, trade_frequency, portfolio_vol)

    def pre_compute(self,trade_range):
        def _calc_break_out(ser, i):
            n = [21, 63, 252]
            mean = ser.rolling(n[i]).mean()
            return (ser/mean) - 1 
        
        def is_all_one(x):
            return np.all(x)
        
        for inst in self.insts:
            inst_df = self.dfs[inst]
            alpha = _calc_break_out(inst_df["close"],0)
            cond1 = (_calc_break_out(inst_df["close"],0) > 0).rolling(5).apply(is_all_one,raw=True)
            cond2 = (_calc_break_out(inst_df["close"],2) > 0).rolling(10).apply(is_all_one,raw=True)

            self.dfs[inst]["cond"] = ~cond1.astype(bool) & cond2.astype(bool)
            self.dfs[inst]["alpha"] = alpha
        return 
    
    def post_compute(self,trade_range):
        temp_alpha = []
        temp_criterias = []
        for inst in self.insts:
            inst_df = self.dfs[inst]
            temp_alpha.append(inst_df["alpha"])
            temp_criterias.append(inst_df["cond"])

        alphadf = pd.concat(temp_alpha,axis=1)
        alphadf.columns = self.insts
        criteriasdf = pd.concat(temp_criterias,axis=1)
        criteriasdf.columns = self.insts

        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alphadf)) & (criteriasdf)
        masked_df = alphadf/self.eligiblesdf
        masked_df = masked_df.replace([-np.inf, np.inf], np.nan)
        num_eligibles = self.eligiblesdf.sum(axis=1)
        rankdf = masked_df.rank(axis=1,method="average",na_option="keep",ascending=False)

        numb = num_eligibles - 5
        longdf = rankdf.apply(lambda col: col > numb, axis=0, raw=True)
        forecast_df = longdf.astype(np.int32)
        self.forecast_df = forecast_df
        self.alphadf = masked_df
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts
    
class CtaStrat(BacktestEngine):
    def __init__(self, insts, dfs, start=None, end=None, date_range=None, trade_frequency='daily', portfolio_vol=0.2):
        super().__init__(insts, dfs, start, end, date_range, trade_frequency, portfolio_vol)

    def pre_compute(self,trade_range):
        for inst in self.insts:
            inst_df = self.dfs[inst]
            inst_df["forecast"] = np.ones(inst_df.shape[0])
            self.dfs[inst] = inst_df
        return 
    
    def post_compute(self,trade_range):
        forecast_df = []
        for inst in self.insts:
            inst_df = self.dfs[inst]
            forecast_df.append(inst_df["forecast"])

        alphadf = pd.concat(forecast_df,axis=1)
        alphadf.columns = self.insts

        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alphadf))
        masked_df = alphadf/self.eligiblesdf
        masked_df = masked_df.replace([-np.inf, np.inf], np.nan)

        forecast_df = masked_df
        self.forecast_df = forecast_df
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts
    
