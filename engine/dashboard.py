import numpy as np
import pandas as pd
import dill as pickle
import streamlit as st
import matplotlib.pyplot as plt

from datetime import date,timedelta
from engine.database.borsdata_api import BorsdataAPI
from engine.database.constants import API_KEY, PATH, STRAT_PATH

from engine.strategies.systematic_yield_strategy import SystYieldAlpha
from engine.strategies.vix_strategy import VixStrategy
from engine.strategies.multi_strategy import EquityStrat,MacroStrat

import os
import lzma
import warnings

warnings.filterwarnings("ignore")
plt.style.use("seaborn-dark-palette")

class DataImporter:
    def __init__(self) -> None:
        self.api = BorsdataAPI(API_KEY)
        self.information = pd.read_csv(PATH+"/instrument_with_meta_data.csv")
        end_date = date.today()
        self.end_date = end_date.strftime('%Y-%m-%d')
        start_date = end_date - timedelta(565)
        self.start_date = start_date.strftime('%Y-%m-%d')
        self.date_range = pd.date_range(start_date,end_date)
        names = self.information['name'].unique().tolist()
        self.sector_map = {name:self.information[self.information['name']==name]['sector'].values[0] for name in names}
        
    
    def syst_yield_data(self,countries,fx_map,start,end):
        from engine.database.borsdata_price_database_script import priceDataBase
        from engine.database.borsdata_kpi_database_script import kpiDataBase

        from engine.utils import merge_kpi_price,concat_kpi_price
        
        tickers = []
        dfs = {}
        for country, markets in countries.items():
            for market in markets:
                db_price = priceDataBase(country, market, start, end)
                db_kpi0 = kpiDataBase(61, market, country)
                db_kpi1 = kpiDataBase(66, market, country)
                price_tickers, price_dfs = db_price.get_data()
                kpi_tickers0,kpi_dfs0 = db_kpi0.get_data()
                kpi_tickers1,kpi_dfs1 = db_kpi1.get_data()
                tickers0,dfs0 = merge_kpi_price(
                    kpi_dfs0,
                    price_dfs,
                    kpi_tickers0,
                    price_tickers,
                    'aktier',
                    fx=fx_map[country]
                )
                tickers1,dfs1 = concat_kpi_price(
                    kpi_dfs1,
                    dfs0,
                    kpi_tickers1,
                    tickers0,
                    'div'  
                )
                
                tickers += tickers1
                dfs |= dfs1
                del tickers0,dfs0,tickers1,dfs1
        return tickers,dfs
    
    def multi_strat_macro_data(self,start=None):
        def load_daily_data(ticker, start=start):
            from yfinance import download
            df = download(ticker, start=start)[['Open','High','Low','Close','Volume','Adj Close']]
            df['Ratio'] = df['Adj Close'] / df['Close']
            df['Open'] = df['Open'] * df['Ratio']
            df['High'] = df['High'] * df['Ratio']
            df['Low'] = df['Low'] * df['Ratio']
            df['Close'] = df['Close'] * df['Ratio']
            df = df.drop(['Ratio', 'Adj Close'], axis = 1)
            df.columns = ['open','high','low','close','volume']
            return df
        
        tickers = ["KC=F","GC=F","NG=F", "HG=F", "CL=F", "PA=F", "SI=F", "ZC=F", "ZW=F", "EURSEK=X", "USDSEK=X", "SPY", "ZN=F", "TLT", "QQQ"]
        tickers_new = ["KC_F","GC_F","NG_F", "HG_F", "CL_F", "PA_F", "SI_F", "ZC_F", "ZW_F", "EURSEK_X", "USDSEK_X", "SPY", "ZN_F", "TLT", "QQQ"]
        dfs = {
            ticker_new:load_daily_data(ticker=ticker) \
                for ticker,ticker_new in zip(tickers,tickers_new)
        }
        return tickers_new,dfs

    def vix_data(self,start=None):
        def load_daily_data(ticker, start=start):
            from yfinance import download
            df = download(ticker, start=start)[['Open','High','Low','Close','Volume','Adj Close']]
            df['Ratio'] = df['Adj Close'] / df['Close']
            df['Open'] = df['Open'] * df['Ratio']
            df['High'] = df['High'] * df['Ratio']
            df['Low'] = df['Low'] * df['Ratio']
            df['Close'] = df['Close'] * df['Ratio']
            df = df.drop(['Ratio', 'Adj Close'], axis = 1)
            df.columns = ['open','high','low','close','volume']
            return df
        
        tickers = ["_VIX","_VIX3M","_VIX6M","BIL","SVXY"]
        dfs = {
            ticker_new:load_daily_data(ticker=ticker) \
                for ticker,ticker_new in zip(['^VIX','^VIX3M', '^VIX6M', "BIL", "SVXY"],tickers)
        }
        return tickers,dfs
        
    def load_pickle(self,path):
        with lzma.open(path,"rb") as fp:
            file = pickle.load(fp)
        return file

    def save_pickle(self,path,obj):
        with lzma.open(path,"wb") as fp:
            pickle.dump(obj,fp)

    def remove_obj(self,path,name):
        os.remove(path+name+'.obj')
        return

    def load_data_syst_yield(self,countries,fx_map):
        name = "data_syst_yield"
        path = STRAT_PATH + "/"
        try:
            tickers,dfs = self.load_pickle(path+name+'.obj')
        except:
            tickers,dfs = self.syst_yield_data(countries,fx_map,self.start_date,self.end_date)
            self.save_pickle(path+name+'.obj',(tickers,dfs))
        return tickers,dfs

class Dashboard(DataImporter):
    def __init__(self) -> None:
        super().__init__()
        self.SystYieldAlpha = SystYieldAlpha
        self.VixStrategy = VixStrategy
        self.MS_Macro = MacroStrat
        self.fx_map = {
            'Sverige': None,
            'Norge': 'NOKSEK_X',
            'Danmark': 'DKKSEK_X'
        }
        self.countries_syst_yield = {
            'Sverige':['Large Cap', 'Mid Cap','Small Cap'],
            'Norge':['Oslo Bors', 'Oslo Expand', 'Oslo Growth'],
            'Danmark':['Large Cap', 'Mid Cap', 'Small Cap']
        }
        return

    def _calc_ms_macro(self,start_cap):
        self.multi_strat_macro_data_tuple = self.multi_strat_macro_data()
        tickers_macro,dfs_macro = self.multi_strat_macro_data_tuple
        alpha = self.MS_Macro(insts=np.unique(tickers_macro).tolist(),dfs=dfs_macro,start=self.start_date,end=self.end_date,trade_frequency='monthly')
        alpha.run_simulation(start_cap=start_cap,use_vol_target=True)
        return alpha
    
    def _calc_ms_vix(self,start_cap):
        self.vix_data_tuple = self.vix_data()
        tickers,dfs = self.vix_data_tuple
        alpha = self.VixStrategy(insts=tickers,dfs=dfs,start=self.start_date,end=self.end_date,trade_frequency="weekly")
        alpha.run_simulation(start_cap=start_cap,use_vol_target=False)
        return alpha

    def _calc_syst_yield_strategy(self,start_cap):
        self.syst_yield_data_tuple = self.load_data_syst_yield(countries=self.countries_syst_yield,fx_map=self.fx_map)
        tickers,dfs = self.syst_yield_data_tuple
        alpha = self.SystYieldAlpha(insts=np.unique(tickers).tolist(),dfs=dfs,start=self.start_date,end=self.end_date,trade_range=None,trade_frequency='monthly')
        alpha.run_simulation(start_cap=start_cap,use_vol_target=False)
        return alpha
    
    def _get_performance_stats(self,ser):
        from .performance import performance_measures
        stats_names = [
            "cagr",
            "srtno",
            "sharpe",
            "mean_ret",
            "median_ret",
            "vol",
            "var",
            "skew",
            "exkurt",
            "var95"
        ]
        r = ser.pct_change().fillna(0)
        non_zero_idx = r.loc[r != 0].index
        retdf = r.loc[non_zero_idx]
        stats_dict = performance_measures(r=retdf)
        stats_ser = pd.Series({stat_name: stat for stat_name,stat in stats_dict.items() if stat_name in stats_names})
        return stats_ser
    
    def _plot_performance(self,ser,market=None):
        stats_table = {}
        f,ax = plt.subplots(1, figsize=(10,6))
        ax.plot(ser,linewidth=2,color="black",label="portfolio")
        for i,df in market.items():
            if "os" in i:
                idx = df.dropna().index
                ax.plot(df, linewidth=0.75)
                ax.annotate(
                    text=i.removesuffix(" os"),
                    xy=(idx[-1],df.iloc[-1]),
                    xytext=(idx[-1] + pd.Timedelta(2,"W"),df.iloc[-1]*1.02)
                )
            else:
                ax.plot(df, linewidth=0.75,color="grey")
            stats_table[i] = self._get_performance_stats(df)
        stats_table["portofolio is"] = self._get_performance_stats(ser.loc[:idx[0]])
        stats_table["portofolio os"] = self._get_performance_stats(ser.loc[idx[0]:])
        ax.axvspan(xmin=idx[0],xmax=idx[-1],color="yellow",alpha=0.15,label="Out of sample")
        ax.legend()
        ax.set_yscale("log")
        return f,pd.DataFrame(stats_table)

    def calculate_portfolio(self):
        alphas = {}
        dfs = {}
        weights = np.array([0.51,0.12,1-(0.51+0.12)])
        strats = ["syst_yield","ms_vix","ms_macro"]

        is_idxs,os_idxs = [],[]

        idx_len = 9999
        for w,strat,func in zip(
            weights,
            strats,
            [self._calc_syst_yield_strategy,self._calc_ms_vix,self._calc_ms_macro]
        ):  
            start_cap = 228000 * w
            alphas[strat] = func(start_cap=start_cap)
            is_rets = self.load_pickle(f"{STRAT_PATH}/strategy_returns/{strat}.obj").loc["2016-01-01":]
            os_rets = alphas[strat].portfolio_df["capital"].pct_change().fillna(0).loc["2016-01-01":]

            os_idx = list(os_rets.index) 
            is_idx = [i for i in is_rets.index if i not in os_rets.index]
            
            is_idxs += is_idx
            os_idxs += os_idx

            rets = pd.concat([is_rets,os_rets],axis=0).sort_index(ascending=True)
            rets = (1+rets).cumprod()
            cols = [strat + " is", strat + " os"]
            dfs[cols[0]] = rets.loc[is_idx]
            dfs[cols[1]] = rets.loc[os_idx]
            idx_len = min(idx_len,len(is_idx))
        
        is_idx, os_idx = sorted(list(set(is_idxs)))[-idx_len:], sorted(list(set(os_idxs)))
        portfolio_os = pd.concat({os_col: df.loc[os_idx].sort_index(ascending=True) for os_col,df in dfs.items() if "os" in os_col},axis=1)
        portfolio_is = pd.concat({is_col: df.loc[is_idx].sort_index(ascending=True) for is_col,df in dfs.items() if "is" in is_col},axis=1)

        portfolios = portfolio_is.join(portfolio_os,how="outer")
        f,df = self._plot_performance(
                ser=portfolios.mean(axis=1),
                market=portfolios
            )
        return f,df
    
    def _show_vix_strategy(self):
        def label_mapper(ser):
            alt_cond = np.where(ser==0.,"50 % short VIX","Liquid")
            return np.where(ser==1.,"100% short VIX",alt_cond)
        
        alpha = self._calc_ms_vix(start_cap=228000*0.09)
        dfs = {
            "Short end term structure":alpha.dfs["ratio1"],
            "Long end term structure":alpha.dfs["ratio2"],
            "Short end MA":alpha.dfs["ratio1"].rolling(23).mean(),
            "Long end MA":alpha.dfs["ratio2"].rolling(23).mean(),
        }
        forecast_ser = (
            alpha.forecast_df["SVXY"].fillna(method="ffill").fillna(0.) 
            - alpha.forecast_df["BIL"].fillna(method="ffill").fillna(1.)
        )
        forecast_ser = forecast_ser.apply(label_mapper)
        dfs.update({"Trading condition":forecast_ser})
        df = pd.DataFrame(dfs)
        return df.sort_index(ascending=False)

    def _show_func(self,alpha):
        forecasts = alpha.forecast_df.fillna(method="ffill")
        ret_df,cols = [],[]
        for inst in alpha.insts:
            if not forecasts[inst].iloc[-1] == 0.:
                ret_df.append(alpha.dfs[inst]["ret"])
                cols.append(inst)
        ret_df = pd.concat(ret_df,axis=1).fillna(0.)
        ret_df = (1+ret_df).cumprod()
        ret_df.columns = cols

        f,ax = plt.subplots(1, figsize=(12,10))
        for inst,row in ret_df.T.iterrows():
            row.plot(ax=ax)
            ax.annotate(
                xy=(row.index[-1],row.iloc[-1]),
                xytext=(5,0), 
                textcoords='offset points', 
                text=inst, 
                va='center'
            )

        weights_insts = [inst + " w" for inst in alpha.insts]
        df = alpha.portfolio_df[weights_insts].tail(50).sort_index(ascending=False).T
        return f,df.sort_values(by=df.columns[0],ascending=False)

    def _app_page(self):
        st.set_page_config(layout="wide")
        st.header('Dashboard')

        button = st.button('Press to update data')
        if button:
            path = STRAT_PATH + "/"
            name = "data_syst_yield"
            self.remove_obj(path=path,name=name)
            path = STRAT_PATH + "/"
            name = "data_multi_strat_equity"
            self.remove_obj(path=path,name=name)
        
        st.markdown("## Portfolio performance")
        f,df = self.calculate_portfolio()
        st.pyplot(f)
        st.dataframe(df)

        st.markdown("## Systematic Yield strategy")
        syst_yield_alpha = self._calc_syst_yield_strategy(start_cap=280000*0.51)
        syst_yield_plot,syst_yield_df = self._show_func(syst_yield_alpha)
        cols_syst_yield = st.columns([2,1])
        cols_syst_yield[0].pyplot(syst_yield_plot)
        cols_syst_yield[1].dataframe(syst_yield_df)

        st.markdown("## Multi strategy")
        st.markdown("### VIX strategy")
        vix_df = self._show_vix_strategy()
        st.dataframe(vix_df)

        st.markdown("### Macro strategy")
        macro_alpha = self._calc_ms_macro(start_cap=280000*0.14)
        macro_plot,macro_df = self._show_func(macro_alpha)
        cols_macro = st.columns([2,1])
        cols_macro[0].pyplot(macro_plot)
        cols_macro[1].dataframe(macro_df)


"""
    def get_dfs(self,id_map):
        dfs = {}
        def _helper(idx):
            try:
                # 66 is Ordinarie utdelning
                val0 = self.api.get_kpi_data_instrument(ins_id=idx,kpi_id=66,calc_group='last',calc='latest')
                val1 = self.api.get_kpi_data_instrument(ins_id=idx,kpi_id=61,calc_group='last',calc='latest')
                val2 = self.api.get_kpi_data_instrument(ins_id=idx,kpi_id=6,calc_group='last',calc='latest')
                df0 = pd.DataFrame(
                    data=[
                        [val0.values[0][0] for _ in range(len(self.date_range))],
                        [val1.values[0][0] for _ in range(len(self.date_range))],
                        [val2.values[0][0] for _ in range(len(self.date_range))],
                        [self.sector_map[id_map[idx]] for _ in range(len(self.date_range))],
                        self.date_range
                ]).T
                df0.columns = ['div', 'aktier', 'eps', 'sector', 'date']
                prices = self.api.get_instrument_stock_prices(
                    ins_id=idx,
                    from_date=self.start_date,
                    to_date=self.end_date,
                    max_count=300
                ).sort_index(ascending=True)
                dfs[id_map[idx]] = df0 \
                    .join(prices,how='left', on='date') \
                    .fillna(method='ffill') \
                    .set_index('date')
            except:
                print(f'Import skips {id_map[idx]}')
        
        threads = [threading.Thread(target=_helper, args=(idx,)) for idx in list(id_map.keys())]
        _ = [thread.start() for thread in threads]
        _ = [thread.join() for thread in threads]
        return list(dfs.keys()),dfs

        
    def get_dfs_sets(self,countries,strat):
        tickers = []
        dfs = {}
        if strat=="Strategy":
            data_load_func = self.strategy_data

        for country,markets in countries.items():
            for market in markets:
                if market == "Index":
                    mask_filter = (self.information['country'] == country) & (self.information['market'] == market) & (self.information['name'] == "OMX Stockholm PI")
                else:
                    mask_filter = (self.information['country'] == country) & (self.information['market'] == market)
                ids = self.information[mask_filter]['ins_id'].values.tolist()

                id_map = {}
                for idx in ids:
                    id_map[idx] = self.information[self.information['ins_id']==idx]['name'].values[0]

                tickers0, dfs0 = data_load_func(id_map)
                tickers += tickers0
                dfs |= dfs0
                del tickers0
                del dfs0
        return tickers,dfs

        def strategy_data(self,id_map):
        dfs = {}
        def _helper(idx):
            try:
                prices = self.api.get_instrument_stock_prices(
                    ins_id=idx,
                    from_date=self.start_date,
                    to_date=self.end_date,
                    max_count=300
                ).sort_index(ascending=True)
                dfs[id_map[idx]] = prices
            except:
                print(f'Import skips {id_map[idx]}')

        threads = [threading.Thread(target=_helper, args=(idx,)) for idx in list(id_map.keys())]
        _ = [thread.start() for thread in threads]
        _ = [thread.join() for thread in threads]
        return list(dfs.keys()),dfs
"""
