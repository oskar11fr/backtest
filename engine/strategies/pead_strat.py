import pandas as pd
import numpy as np
from backtest_engine.utils import BacktestEngine
from backtest_engine.indicators import logret,slope,tszscore
import matplotlib.pyplot as plt

class PEAD(BacktestEngine):

    def __init__(self,insts,dfs,start,end,trade_frequency):
        super().__init__(insts,dfs,start,end,trade_frequency)
    
    def pre_compute(self,trade_range):
        for inst in self.insts:
            inst_df = self.dfs[inst]
            ret = inst_df["close"].pct_change().fillna(0)
            ma_sh = inst_df["close"].rolling(5).mean()
            ma_lo = inst_df["close"].rolling(20).mean()

            report_date = inst_df["reportdate"]
            report_date_post_1 = report_date.shift(5)
            report_date_post_2 = -1 * report_date.shift(65).fillna(0)
            window = (report_date_post_1 + report_date_post_2).cumsum()
            reportdate_return = np.where(report_date,ret,0.)
            temp_rd_ret = np.where(
                window,
                np.where(report_date_post_1,ret.shift(5).fillna(0),np.NaN),
                0.
            )

            self.dfs[inst]["reportdate_post"] = report_date_post_1
            self.dfs[inst]["rd_ret_fwd"] = pd.Series(temp_rd_ret,index=inst_df.index).fillna(method="ffill").fillna(0)
            self.dfs[inst]["rd_ret"] = reportdate_return
            self.dfs[inst]["trading_window"] = window
            self.dfs[inst]["trend"] = ma_sh/ma_lo
        return 

    def post_compute(self,trade_range):        
        temp_alpha = []
        trading_window = []
        trend = []
        for inst in self.insts:
            temp_alpha.append(self.dfs[inst]["rd_ret_fwd"].copy())
            trading_window.append(self.dfs[inst]["trading_window"].copy())
            trend.append(self.dfs[inst]["trend"].copy())

        alpha_df = pd.concat(temp_alpha,axis=1)
        alpha_df.columns = self.insts
        tw_df = pd.concat(trading_window,axis=1)
        tw_df.columns = self.insts
        trend_df = pd.concat(trend,axis=1)
        trend_df.columns = self.insts

        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alpha_df)) & (tw_df == 1) & (trend_df > 1.) #& (alpha_df > 0.05)
        self.alpha_df = alpha_df
        masked_df = self.alpha_df/self.eligiblesdf
        masked_df = masked_df.replace([-np.inf, np.inf], np.nan)
        num_eligibles = self.eligiblesdf.sum(axis=1)
        rankdf = masked_df.rank(axis=1,method="average",na_option="keep",ascending=True)
        numb = num_eligibles - 10
        longdf = rankdf.apply(lambda col: col > numb, axis=0, raw=True)
        forecast_df = longdf.astype(np.int32)
        self.forecast_df = forecast_df
        return

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts