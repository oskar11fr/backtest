import numpy as np
import pandas as pd
from engine.utils import BacktestEngine

class VixStrategy(BacktestEngine):

    def __init__(self, insts, dfs, start=None, end=None, date_range=None, trade_frequency='daily', portfolio_vol=0.2):
        super().__init__(insts, dfs, start, end, date_range, trade_frequency, portfolio_vol)
    
    def pre_compute(self,trade_range):
        vix = self.dfs['_VIX'].close
        vix3m = self.dfs['_VIX3M'].close
        vix6m = self.dfs['_VIX6M'].close
        ratio1 = ((vix3m/vix).fillna(method='ffill'))
        ratio2 = ((vix6m/vix3m).fillna(method='ffill'))
        filtered_ratio1 = ratio1/ratio1.rolling(23).mean()
        filtered_ratio2 = ratio2/ratio2.rolling(23).mean()
        del self.dfs['_VIX'],self.dfs['_VIX3M'],self.dfs['_VIX6M']

        self.insts = list(self.dfs.keys())
        self.dfs['filtered_ratio1'] = filtered_ratio1
        self.dfs['ratio1'] = ratio1
        self.dfs['filtered_ratio2'] = filtered_ratio2
        self.dfs['ratio2'] = ratio2
        return 
    
    def post_compute(self,trade_range):
        def is_all_true(x):
            return np.all(x)
        def is_any_true(x):
            return np.any(x)
        
        filtered_ratio1 = self.dfs['filtered_ratio1'].copy()
        filtered_ratio2 = self.dfs['filtered_ratio2'].copy()

        cond1_above = filtered_ratio1 > 1
        cond2_above = filtered_ratio2 > 1
        cond1_below = ~cond1_above
        cond2_below = ~cond2_above

        neutral_cond = (cond1_below.rolling(2).apply(is_all_true,raw=True) * cond2_below.rolling(5).apply(is_any_true,raw=True)) + \
                    (cond1_below.rolling(2).apply(is_all_true,raw=True) * cond2_above.rolling(2).apply(is_all_true,raw=True))

        short_cond = (cond1_above.rolling(2).apply(is_all_true,raw=True) * cond2_above.rolling(5).apply(is_any_true,raw=True)) + \
                    (cond1_below.rolling(2).apply(is_all_true,raw=True)  * cond2_above.rolling(2).apply(is_all_true,raw=True))
        temp_alpha = []
        names = []
        for inst in self.insts:
            if inst == 'BIL':
                self.dfs[inst]['ratio_cond'] = neutral_cond
                names.append(inst)
                temp_alpha.append(self.dfs[inst]['ratio_cond'])
            if inst == 'SVXY':
                self.dfs[inst]['ratio_cond'] = short_cond
                names.append(inst)
                temp_alpha.append(self.dfs[inst]['ratio_cond'])
        
        alpha_df = pd.concat(temp_alpha,axis=1)
        alpha_df.columns = names
        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alpha_df))
        self.forecast_df = alpha_df
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts
