import numpy as np
import pandas as pd
from backtest_engine.utils import BacktestEngine

class VixStrategy(BacktestEngine):

    def __init__(self,insts,dfs,start,end,trade_frequency):
        super().__init__(insts,dfs,start,end,trade_frequency)
    
    def pre_compute(self,trade_range):
        vix = self.dfs['_VIX'].close
        vix3m = self.dfs['_VIX3M'].close
        vix6m = self.dfs['_VIX6M'].close
        ratio1 = ((vix3m/vix).fillna(method='ffill'))
        ratio2 = ((vix6m/vix3m).fillna(method='ffill'))
        filtered_ratio1 = ratio1/ratio1.rolling(23).mean()
        filtered_ratio2 = ratio2/ratio2.rolling(23).mean()
        del self.dfs['_VIX']
        del self.dfs['_VIX3M']
        del self.dfs['_VIX6M']
        del self.dfs['SPY']
        self.insts = list(self.dfs.keys())
        self.dfs['filtered_ratio1'] = filtered_ratio1
        self.dfs['ratio1'] = ratio1
        self.dfs['filtered_ratio2'] = filtered_ratio2
        self.dfs['ratio2'] = ratio2
        return 
    
    def post_compute(self,trade_range):
        filtered_ratio1 = self.dfs['filtered_ratio1'].copy()
        filtered_ratio2 = self.dfs['filtered_ratio2'].copy()
        temp_alpha = []
        names = []
        eows = []
        for inst in self.insts:
            if inst == 'BIL':
                self.dfs[inst]['ratio_cond'] = ((filtered_ratio1 < 1) & (filtered_ratio2 < 1)).astype(int) + \
                                                    ((filtered_ratio1 < 1) & (filtered_ratio2 > 1)).astype(int)
                names.append(inst)
                temp_alpha.append(self.dfs[inst]['ratio_cond'])
            if inst == 'SVXY':
                self.dfs[inst]['ratio_cond'] = ((filtered_ratio1 > 1) & (filtered_ratio2 > 1)).astype(int) + \
                                                    ((filtered_ratio1 < 1) & (filtered_ratio2 > 1)).astype(int)
                names.append(inst)
                temp_alpha.append(self.dfs[inst]['ratio_cond'])
            # if inst == 'SPY':
            #     self.dfs[inst]['ratio_cond'] = ((filtered_ratio1 < 1) & (filtered_ratio2 > 1)).astype(int)
            #     names.append(inst)
            #     temp_alpha.append(self.dfs[inst]['ratio_cond'])
        
        alpha_df = pd.concat(temp_alpha,axis=1)
        alpha_df.columns = names
        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alpha_df))
        self.forecast_df = alpha_df
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts