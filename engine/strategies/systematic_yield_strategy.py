import numpy as np
import pandas as pd
from engine.utils import BacktestEngine

def f(x):
    return x/np.sqrt(1+x**2)

class SystYieldAlpha(BacktestEngine):

    def __init__(self,insts,dfs,start,end,trade_frequency):
        super().__init__(insts,dfs,start,end,trade_frequency)
    
    def pre_compute(self,trade_range):        
        for inst in self.insts:
            inst_df = self.dfs[inst]
            self.dfs[inst]['div_yield'] = inst_df["div"]/inst_df["close"]
            self.dfs[inst]['bv'] = inst_df["aktier"]*inst_df["close"]*inst_df["fx"]
            self.dfs[inst]['mom'] = inst_df["close"]/inst_df["close"].rolling(253).mean() - 1
        return 
    
    def post_compute(self,trade_range):
        alphas = []
        div_yields = []
        bvs = []
        for inst in self.insts:
            self.dfs[inst]["alpha"] = f(np.maximum(self.dfs[inst]['mom'],0))
            alphas.append(self.dfs[inst]["alpha"])
            div_yields.append(self.dfs[inst]["div_yield"])
            bvs.append(self.dfs[inst]["bv"])

        alphadf = pd.concat(alphas,axis=1)
        alphadf.columns = self.insts
        div_yielddf = pd.concat(div_yields,axis=1)
        div_yielddf.columns = self.insts
        bvdf = pd.concat(bvs,axis=1)
        bvdf.columns = self.insts

        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alphadf)) & (div_yielddf > 0.05) & (bvdf > 4000)
        self.alphadf = alphadf
        masked_df = self.alphadf/self.eligiblesdf
        masked_df = masked_df.replace([-np.inf, np.inf], np.nan)
        num_eligibles = self.eligiblesdf.sum(axis=1)
        rankdf = masked_df.rank(axis=1,method="average",na_option="keep",ascending=True)
        numb = num_eligibles - 10
        longdf = rankdf.apply(lambda col: col > numb, axis=0, raw=True)
        forecast_df = longdf.astype(np.int32)
        self.forecast_df = forecast_df
        self.div_yielddf = div_yielddf
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts
