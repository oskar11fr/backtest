import os
import lzma
import time
import asyncio

import numpy as np
import pandas as pd
import dill as pickle
import matplotlib.pyplot as plt
import engine.quant_stats as quant_stats

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from functools import wraps

from collections import defaultdict
from engine.performance import plot_hypothesis,plot_analysis
from engine.database.finance_database_script import finance_database
from engine.database.constants import STRAT_PATH

plt.style.use("seaborn-dark-palette")

def load_pickle(path: str):
    """_summary_

    Args:
        path (str): _description_

    Returns:
        _type_: _description_
    """
    with lzma.open(path,"rb") as fp:
        file = pickle.load(fp)
    return file

def save_pickle(path,obj):
    """_summary_

    Args:
        path (_type_): _description_
        obj (_type_): _description_
    """
    with lzma.open(path,"wb") as fp:
        pickle.dump(obj,fp)

def merge_kpi_price(kpis_dfs, price_dfs, kpis_tickers, price_tickers, kpi_name='kpi',fx=None):
    tickers = list(set(kpis_tickers) & set(price_tickers))
    dfs = {}
    for ticker in tickers:
        try:
            price_df = price_dfs[ticker] \
                .sort_index(ascending=False) \
                .reset_index() \
                .rename(columns={'date':'datetime'}) \
                .set_index('datetime') \
                .drop(columns="index",errors="ignore")
            price_df.index = pd.DatetimeIndex(price_df.index)

            kpi_df = kpis_dfs[ticker] \
                .reset_index() \
                .drop(columns=ticker+'_reportDate') \
                .set_index('datetime') \
                .drop(columns="index",errors="ignore")
            kpi_df.index = pd.DatetimeIndex(kpi_df.index)
            full_df = price_df.join(kpi_df,how='left',on='datetime') \
                .sort_index(ascending=True) \
                .fillna(method='ffill')
            full_df = full_df.fillna(method='bfill') if kpi_name=='aktier' else full_df.fillna(0.)
            full_df = full_df.rename(columns={ticker:kpi_name})
            
            if fx is None:
                full_df["fx"] = np.ones(full_df.shape[0])

            if fx is not None:
                fx_db = finance_database('fx_db')
                fx_df = fx_db.export_from_database(symbol=fx).loc[full_df.index[0]:].rename(columns={'close':'fx'})
                full_df = full_df.join(fx_df['fx'], on='datetime').fillna(method='ffill').fillna(0)

            dfs[ticker] = full_df
        except KeyError:
            print(f'Passed: {ticker}')
            continue
    
    return list(dfs.keys()),dfs

def concat_kpi_price(kpis_dfs, dfs0, kpis_tickers, tickers0, kpi_name='kpi'):
    tickers = list(set(kpis_tickers) & set(tickers0))
    dfs = {}
    for ticker in tickers:
        df = dfs0[ticker]
        
        kpi_df = kpis_dfs[ticker] \
            .reset_index() \
            .drop(columns=ticker+'_reportDate') \
            .set_index('datetime')
        kpi_df.index = pd.DatetimeIndex(kpi_df.index)
        
        full_df = df.join(kpi_df,how='left',on='datetime')
        full_df = full_df.fillna(method='ffill').fillna(0)
        full_df = full_df.rename(columns={ticker:kpi_name})
        dfs[ticker] = full_df
    
    return tickers,dfs

def timeme(func):
    @wraps(func)
    def timediff(*args,**kwargs):
        a = time.time()
        result = func(*args,**kwargs)
        b = time.time()
        print(f"@timeme: {func.__name__} took {b - a} seconds")
        return result
    return timediff

def get_pnl_stats(last_weights, last_units, prev_close, ret_row, leverages):
    ret_row = np.nan_to_num(ret_row,nan=0,posinf=0,neginf=0)
    day_pnl = np.sum(last_units * prev_close * ret_row)
    nominal_ret = np.dot(last_weights, ret_row)
    capital_ret = nominal_ret * leverages[-1]
    return day_pnl, nominal_ret, capital_ret   


class AbstractImplementationException(Exception):
    pass

class _abstract_exc(Exception):
    pass

class BacktestEngine():

    def __init__(self, insts, dfs, start=None, end=None, date_range=None, trade_frequency='daily', portfolio_vol=0.20):
        assert (start is not None) | (date_range is not None), "Initialize date_range or start date"
        self.insts = insts
        self.dfs = deepcopy(dfs)
        self.datacopy = deepcopy(dfs)
        self.portfolio_vol = portfolio_vol
        self.alphadf = None

        if date_range is None:
            self.date_range = pd.date_range(start=start,end=end,freq="D")
            self.trade_frequency = trade_frequency
        else:
            self.date_range = date_range
            self.trade_frequency = None

    def get_zero_filtered_stats(self):
        assert self.portfolio_df is not None
        nominal_ret = self.portfolio_df.nominal_ret
        capital_ret = self.portfolio_df.capital_ret
        non_zero_idx = capital_ret.loc[capital_ret != 0].index
        retdf = self.retdf.loc[non_zero_idx]
        weights = self.weights_df.shift(1).fillna(0).loc[non_zero_idx]
        eligs = self.eligiblesdf.shift(1).fillna(0).loc[non_zero_idx]
        leverages = self.leverages.shift(1).fillna(0).loc[non_zero_idx]
        return {
            "capital_ret": capital_ret.loc[non_zero_idx],
            "nominal_ret": nominal_ret.loc[non_zero_idx],
            "retdf":retdf,
            "weights":weights,
            "eligs":eligs,
            "leverages":leverages,
        }

    def get_perf_stats(self,capital_rets=True,plot=False,market=False,show=False):
        from engine.performance import performance_measures
        assert self.portfolio_df is not None
        if market:
            assert self.market is not None
            market = self.market
        rets = 'capital_ret' if capital_rets else 'nominal_ret'
        
        stats_dict = performance_measures(
            r=self.get_zero_filtered_stats()[rets],
            plot=plot,
            market=None if not market else market,
            show=show
        )
        stats = [
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
        temp = {}
        for stat in stats:
            temp[stat] = stats_dict[stat]
        
        return pd.Series(temp)
    
    async def hypothesis_tests(self, num_decision_shuffles=1000,zfs=None):
        retdf, leverages, weights, eligs = zfs["retdf"], zfs["leverages"], zfs["weights"], zfs["eligs"]

        def performance_criterion(rets, leverages, weights, **kwargs):
            capital_ret = [
                lev_scalar * np.dot(weight, ret)
                for lev_scalar, weight, ret in zip(leverages.values, rets.values, weights.values)
            ]
            sharpe = np.mean(capital_ret) / np.std(capital_ret) * np.sqrt(253)
            return round(sharpe, 5),capital_ret

        async def time_shuffler(rets, leverages, weights, eligs):
            nweights = quant_stats.shuffle_weights_on_eligs(weights_df=weights, eligs_df=eligs, method="time",ord=1)
            return {"rets": rets, "leverages": leverages, "weights": nweights}

        async def mapping_shuffler(rets, leverages, weights, eligs):
            nweights = quant_stats.shuffle_weights_on_eligs(weights_df=weights, eligs_df=eligs, method="xs")
            return {"rets": rets, "leverages": leverages, "weights": nweights}

        async def decision_shuffler(rets, leverages, weights, eligs):
            nweights = quant_stats.shuffle_weights_on_eligs(weights_df=weights, eligs_df=eligs, method="time",ord=1)
            nweights = quant_stats.shuffle_weights_on_eligs(weights_df=nweights, eligs_df=eligs, method="xs")
            return {"rets": rets, "leverages": leverages, "weights": nweights}

        timer_paths,timer_p,timer_dist = await quant_stats.permutation_shuffler_test(
            criterion_function=performance_criterion,
            generator_function=time_shuffler,
            m=num_decision_shuffles, rets=retdf, leverages=leverages, weights=weights, eligs=eligs)
        picker_paths,picker_p,picker_dist = await quant_stats.permutation_shuffler_test(
            criterion_function=performance_criterion,
            generator_function=mapping_shuffler,
            m=num_decision_shuffles, rets=retdf, leverages=leverages, weights=weights, eligs=eligs)
        trader_paths,trader_p,trader_dist = await quant_stats.permutation_shuffler_test(
            criterion_function=performance_criterion,
            generator_function=decision_shuffler,
            m=num_decision_shuffles, rets=retdf, leverages=leverages, weights=weights, eligs=eligs)
        return {
            "timer":(timer_paths,timer_p,timer_dist),
            "picker":(picker_paths,picker_p,picker_dist),
            "trader":(trader_paths,trader_p,trader_dist),
        }
    
    @timeme
    async def run_hypothesis_tests(self,num_decision_shuffles=1000,zfs=None,strat_name=None):
        assert self.portfolio_df is not None
        zfs = self.get_zero_filtered_stats()
        rets = zfs["capital_ret"]
        test_dict = await self.hypothesis_tests(num_decision_shuffles=num_decision_shuffles,zfs=zfs)
        plot_hypothesis(test_dict["timer"],test_dict["picker"],test_dict["trader"],rets,strat_name)
        return

    def pre_compute(self,trade_range):
        pass
    
    def compute_frequency(self,trade_range):
        if (self.trade_frequency == 'daily') or (self.trade_frequency is None):
            self.trading_day_ser = pd.Series([1 for _ in range(len(trade_range))],index=trade_range)

        if self.trade_frequency == 'weekly':
            self.trading_day_ser = pd.Series(index=trade_range)
            for date in trade_range:
                self.trading_day_ser.loc[date] = date.day_name() == 'Friday'
        
        if self.trade_frequency == 'monthly':
            self.trading_day_ser = pd.Series(index=trade_range)
            eom_fun = pd.tseries.offsets.BMonthEnd()
            for date in trade_range:
                self.trading_day_ser.loc[date] = eom_fun.rollforward(date)==date
        return

    def post_compute(self,trade_range):
        pass

    def compute_signal_distribution(self, eligibles, date):
        raise AbstractImplementationException("no concrete implementation for signal generation")

    def get_strat_scaler(self, target_vol, ewmas, ewstrats):
        ann_realized_vol = np.sqrt(ewmas[-1] * 253)
        return target_vol / ann_realized_vol * ewstrats[-1]

    def compute_meta_info(self,trade_range):
        self.pre_compute(trade_range=trade_range)
        self.compute_frequency(trade_range=trade_range)
        
        def is_any_one(x):
            return int(np.any(x))
        
        closes, eligibles, vols, rets, trading_day = [], [], [], [], []
        for inst in self.insts:
            df=pd.DataFrame(index=trade_range)
            inst_vol = (-1 + self.dfs[inst]["close"]/self.dfs[inst]["close"].shift(1)).rolling(30).std()
            self.dfs[inst] = df.join(self.dfs[inst]).fillna(method="ffill").fillna(method="bfill")
            self.dfs[inst]["ret"] = -1 + (self.dfs[inst]["close"]/self.dfs[inst]["close"].shift(1))
            self.dfs[inst]["vol"] = inst_vol
            self.dfs[inst]["vol"] = self.dfs[inst]["vol"].fillna(method="ffill").fillna(0)       
            self.dfs[inst]["vol"] = np.where(self.dfs[inst]["vol"] < 0.005, 0.005, self.dfs[inst]["vol"])
            self.dfs[inst]['trading_day'] = self.trading_day_ser.copy()
            sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).fillna(method="bfill")
            eligible = sampled.rolling(5).apply(is_any_one,raw=True).fillna(0)
            eligibles.append(eligible.astype(int) & (self.dfs[inst]["close"] > 0).astype(int))
            closes.append(self.dfs[inst]["close"])
            vols.append(self.dfs[inst]["vol"])
            rets.append(self.dfs[inst]["ret"])
            trading_day.append(self.dfs[inst]["trading_day"])
        self.eligiblesdf = pd.concat(eligibles,axis=1)
        self.eligiblesdf.columns = self.insts
        self.closedf = pd.concat(closes,axis=1)
        self.closedf.columns = self.insts
        self.voldf = pd.concat(vols,axis=1)
        self.voldf.columns = self.insts
        self.retdf = pd.concat(rets,axis=1)
        self.retdf.columns = self.insts
        self.trading_day = pd.concat(trading_day, axis=1)
        self.trading_day.columns = self.insts

        self.post_compute(trade_range=trade_range)
        assert self.forecast_df is not None
        self.forecast_df = pd.DataFrame(
            np.where(self.trading_day,self.forecast_df,np.NaN),
            index=self.forecast_df.index,
            columns=self.forecast_df.columns
        )
        return
    
    def run_alpha_analysis(self):
        date_range = self.date_range
        n_buckets = 50
        if self.alphadf is None:
            self.compute_meta_info(date_range)
        assert self.alphadf is not None,"Compute self.alphadf"
        alphadf = self.alphadf
        rets,alphas = [],[]
        for inst in self.insts:
            inst_alpha = alphadf[inst]
            inst_df = self.dfs[inst][["trading_day","close"]]
            alpha_ser = pd.Series(np.where(inst_df["trading_day"],inst_alpha,np.NaN),index=inst_df.index).dropna()
            close_ser = pd.Series(np.where(inst_df["trading_day"],inst_df["close"],np.NaN),index=inst_df.index).dropna()
            next_ret = -1 + close_ser.shift(-1)/close_ser
            rets.append(next_ret)
            alphas.append(alpha_ser)
        alphas,rets = pd.concat(alphas,axis=1),pd.concat(rets,axis=1)
        alphas.columns,rets.columns = self.insts,self.insts
        analysis_list = []
        idx = set(alphas.index).intersection(set(rets.index))
        for date in idx:
            alpha_row = alphas.loc[date].dropna()
            alpha_row = alpha_row.loc[alpha_row!=0]
            rets_row = rets.loc[date,alpha_row.index]
            count = len(alpha_row.values)
            if count != 0:
                # ranked_row = alpha_row.rank(ascending=False)
                analysis_df = pd.concat([alpha_row,rets_row.clip(-0.1,0.1)],axis=1)
                analysis_df.columns = ["alpha","rets"]
                analysis_list.append(analysis_df.reset_index(drop=True))
        analysis_dfs = pd.concat(analysis_list,axis=0)
        analysis_dfs["bins"] = pd.qcut(analysis_dfs["alpha"],n_buckets,labels=range(n_buckets))
        analysis_dfs = analysis_dfs.drop(columns="alpha")
        analysis_dfs = analysis_dfs.groupby("bins").agg("mean").dropna()
        x,y = analysis_dfs.index,analysis_dfs["rets"]
        plot_analysis(x,y)
        return
    
    def save_strat_rets(self, strat_name):
        assert self.portfolio_df is not None
        store_path = STRAT_PATH + "/strategy_returns"
        Path(os.path.abspath(store_path)).mkdir(parents=True,exist_ok=True)
        capital_rets = self.portfolio_df["capital"].pct_change().fillna(0)
        save_pickle(store_path + f"/{strat_name}.obj",obj=capital_rets)
        return
    
    @timeme
    def run_simulation(self,start_cap=100000.0,use_vol_target=True):
        self.compute_meta_info(trade_range=self.date_range)
        units_held, weights_held = [],[]
        close_prev = None
        ewmas, ewstrats = [0.01], [1]
        strat_scalars = []
        capitals, nominal_rets, capital_rets = [start_cap],[0.0],[0.0]
        nominals, leverages = [],[]
        for data in self.zip_data_generator():
            portfolio_i = data["portfolio_i"]
            ret_i = data["ret_i"]
            ret_row = data["ret_row"]
            close_row = np.nan_to_num(data["close_row"],nan=0,posinf=0,neginf=0)
            eligibles_row = data["eligibles_row"]
            trading_day = data["trading_day"]
            vol_row = data["vol_row"]
            strat_scalar = 2
           
            if portfolio_i != 0:
                strat_scalar = self.get_strat_scaler(
                    target_vol=self.portfolio_vol,
                    ewmas=ewmas,
                    ewstrats=ewstrats
                )

                day_pnl, nominal_ret, capital_ret = get_pnl_stats(
                    last_weights=weights_held[-1], 
                    last_units=units_held[-1], 
                    prev_close=close_prev, 
                    ret_row=ret_row, 
                    leverages=leverages
                )
                
                capitals.append(capitals[-1] + day_pnl)
                nominal_rets.append(nominal_ret)
                capital_rets.append(capital_ret)
                ewmas.append(0.06 * (capital_ret**2) + 0.94 * ewmas[-1] if capital_ret != 0 else ewmas[-1])
                ewstrats.append(0.06 * strat_scalar + 0.94 * ewstrats[-1] if capital_ret != 0 else ewstrats[-1])

            strat_scalars.append(strat_scalar)
            forecasts = self.compute_signal_distribution(
                eligibles_row,
                ret_i
            )
            if type(forecasts) == pd.Series: forecasts = forecasts.values
            forecasts = forecasts / eligibles_row
            forecasts = np.nan_to_num(forecasts,nan=0,posinf=0,neginf=0)
            forecast_chips = np.sum(np.abs(forecasts))
            vol_target = (self.portfolio_vol / np.sqrt(253)) \
                * capitals[-1]

            if trading_day or (portfolio_i==0):
                if use_vol_target:
                    positions = strat_scalar * \
                            forecasts / forecast_chips  \
                            * vol_target \
                            / (vol_row * close_row) if forecast_chips != 0 else np.zeros(len(self.insts))
                    positions = np.floor(np.nan_to_num(positions,nan=0,posinf=0,neginf=0))
                else:
                    dollar_allocation = capitals[-1]/forecast_chips if forecast_chips != 0 else np.zeros(len(self.insts))
                    positions = forecasts*dollar_allocation / close_row
                    positions = np.floor(np.nan_to_num(positions,nan=0,posinf=0,neginf=0)) # added floor function
            else:
                positions = units_held[-1]
            nominal_tot = np.linalg.norm(positions * close_row, ord=1)
            units_held.append(positions)
            weights = positions * close_row / nominal_tot
            weights = np.nan_to_num(weights,nan=0,posinf=0,neginf=0)
            weights_held.append(weights)

            nominals.append(nominal_tot)
            leverages.append(nominal_tot/capitals[-1])
            close_prev = close_row
        
        units_df = pd.DataFrame(data=units_held, index=self.date_range, columns=[inst + " units" for inst in self.insts])
        weights_df = pd.DataFrame(data=weights_held, index=self.date_range, columns=[inst + " w" for inst in self.insts])
        nom_ser = pd.Series(data=nominals, index=self.date_range, name="nominal_tot")
        lev_ser = pd.Series(data=leverages, index=self.date_range, name="leverages")
        cap_ser = pd.Series(data=capitals, index=self.date_range, name="capital")
        nomret_ser = pd.Series(data=nominal_rets, index=self.date_range, name="nominal_ret")
        capret_ser = pd.Series(data=capital_rets, index=self.date_range, name="capital_ret")
        scaler_ser = pd.Series(data=strat_scalars, index=self.date_range, name="strat_scalar")
        self.portfolio_df = pd.concat([
            units_df,
            weights_df,
            lev_ser,
            scaler_ser,
            nom_ser,
            nomret_ser,
            capret_ser,
            cap_ser
        ],axis=1)
        self.units_df = units_df
        self.weights_df = weights_df
        self.leverages = lev_ser
        return self.portfolio_df

    def zip_data_generator(self):
        for (portfolio_i),\
            (ret_i, ret_row), \
            (close_i, close_row), \
            (eligibles_i, eligibles_row), \
            (trading_day_i,trading_day), \
            (vol_i, vol_row) in tqdm(zip(
                range(len(self.retdf)),
                self.retdf.iterrows(),
                self.closedf.iterrows(),
                self.eligiblesdf.iterrows(),
                self.trading_day_ser.items(),
                self.voldf.iterrows()
            )):
            yield {
                "portfolio_i": portfolio_i,
                "ret_i": ret_i,
                "ret_row": ret_row.values,
                "close_row": close_row.values,
                "eligibles_row": eligibles_row.values,
                "trading_day": trading_day,
                "vol_row": vol_row.values,
            }

class Portfolio(BacktestEngine):
    
    def __init__(self,insts,dfs,start,end,stratdfs):
        super().__init__(insts,dfs,start,end)
        self.stratdfs=stratdfs

    def post_compute(self,trade_range):
        self.positions = {}
        for inst in self.insts:
            inst_weights = pd.DataFrame(index=trade_range)
            for i in range(len(self.stratdfs)):
                inst_weights[i] = self.stratdfs[i]["{} w".format(inst)]\
                    * self.stratdfs[i]["leverage"]
                inst_weights[i] = inst_weights[i].fillna(method="ffill").fillna(0.0)
            self.positions[inst] = inst_weights

    def compute_signal_distribution(self, eligibles, date):
        forecasts = defaultdict(float)
        for inst in self.insts:
            for i in range(len(self.stratdfs)):
                forecasts[inst] += self.positions[inst].at[date, i] * (1/len(self.stratdfs))
                #parity risk allocation
        return forecasts, np.sum(np.abs(list(forecasts.values())))