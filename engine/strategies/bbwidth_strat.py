import numpy as np
import pandas as pd
from engine.utils import BacktestEngine

class BBwidthStrat(BacktestEngine):

    def __init__(self,insts,dfs,start,end,trade_frequency):
        super().__init__(insts,dfs,start,end,trade_frequency)
    
    def pre_compute(self,trade_range):        
        for inst in self.insts:
            inst_df = self.dfs[inst]
            upperband = inst_df["close"].rolling(23).mean() + 2*inst_df["close"].rolling(23).std()
            lowerband = inst_df["close"].rolling(23).mean() - 2*inst_df["close"].rolling(23).std()
            mid_band = inst_df["close"].rolling(23).mean()
            bandw = (upperband - lowerband)/mid_band
            
            self.dfs[inst]["bandw"] = bandw<bandw.rolling(50).mean()
            self.dfs[inst]['bv'] = inst_df["aktier"]*inst_df["close"]*inst_df["fx"]
            self.dfs[inst]["pe"] = inst_df["close"]*inst_df["eps"]
            self.dfs[inst]['ave50'] = inst_df["close"].rolling(50).mean()
            self.dfs[inst]['ave150'] = inst_df["close"].rolling(150).mean()
            self.dfs[inst]['ave200'] = inst_df["close"].rolling(200).mean()
            self.dfs[inst]['ylo'] = inst_df["low"].rolling(253).min()
        return 
    
    def post_compute(self,trade_range):
        temp0,temp1,bvs,mom1,mom2,mom3,mom4,mom5,mom6,mom7 = [],[],[],[],[],[],[],[],[],[]
        for inst in self.insts:
            inst_df = self.dfs[inst]
            temp0.append(inst_df["bandw"])
            temp1.append(inst_df["pe"])
            bvs.append(inst_df["bv"])
            mom1.append(inst_df["close"]>inst_df["ave50"])
            mom2.append(inst_df["close"]>inst_df["ave150"])
            mom3.append(inst_df["close"]>inst_df["ave200"])
            mom4.append(inst_df["ave50"]>inst_df["ave150"])
            mom5.append(inst_df["ave150"]>inst_df["ave200"])
            mom6.append(inst_df["ave200"]>inst_df["ave200"].shift(23))
            mom7.append(inst_df["close"]/inst_df["ylo"] - 1 > .3)

        bbw = pd.concat(temp0,axis=1)
        bbw.columns = self.insts
        bbw = bbw.replace(np.inf, np.nan).replace(-np.inf, np.nan)
        alpha_df = pd.concat(temp1,axis=1)
        alpha_df.columns = self.insts
        alpha_df = alpha_df.replace(np.inf, np.nan).replace(-np.inf, np.nan)

        bvdf = pd.concat(bvs,axis=1)
        bvdf.columns = self.insts
        mom1 = pd.concat(mom1,axis=1)
        mom1.columns = self.insts
        mom2 = pd.concat(mom2,axis=1)
        mom2.columns = self.insts
        mom3 = pd.concat(mom3,axis=1)
        mom3.columns = self.insts
        mom4 = pd.concat(mom4,axis=1)
        mom4.columns = self.insts
        mom5 = pd.concat(mom5,axis=1)
        mom5.columns = self.insts
        mom6 = pd.concat(mom6,axis=1)
        mom6.columns = self.insts
        mom7 = pd.concat(mom7,axis=1)
        mom7.columns = self.insts

        self.eligiblesdf = self.eligiblesdf & (~pd.isna(bbw)) & (~pd.isna(alpha_df)) & (alpha_df>5) & (bbw) & (bvdf > 1000) \
                       & (mom1) & (mom2) & (mom3) & (mom4) & (mom5) & (mom6) & (mom7)
        self.alphadf = alpha_df
        masked_df = self.alphadf/self.eligiblesdf
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
