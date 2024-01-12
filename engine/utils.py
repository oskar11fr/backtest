import lzma
import time
import os

import numpy as np
import pandas as pd
from copy import deepcopy

from pathlib import Path
import dill as pickle
import asyncio

from functools import wraps
from collections import defaultdict

import backtest_engine.quant_stats as quant_stats
import matplotlib.pyplot as plt
plt.style.use("seaborn-dark-palette")

def load_pickle(path):
    with lzma.open(path,"rb") as fp:
        file = pickle.load(fp)
    return file

def save_pickle(path,obj):
    with lzma.open(path,"wb") as fp:
        pickle.dump(obj,fp)

def merge_kpi_price(kpis_dfs, price_dfs, kpis_tickers, price_tickers, kpi_name='kpi',fx=None):
    tickers = list(set(kpis_tickers) & set(price_tickers))
    dfs = {}
    for ticker in tickers:
        try:
            price_df = price_dfs[ticker].sort_index(ascending=False).rename(columns={'date':'datetime'}).set_index('datetime')
            price_df.index = pd.DatetimeIndex(price_df.index)
            
            kpi_df = kpis_dfs[ticker].drop(columns=ticker+'_reportDate').set_index('datetime')
            kpi_df.index = pd.DatetimeIndex(kpi_df.index)
            
            full_df = price_df.join(kpi_df,how='left',on='datetime').fillna(method='ffill')
            full_df = full_df.fillna(method='bfill') if kpi_name=='aktier' else full_df.fillna(0.)
            full_df = full_df.rename(columns={ticker:kpi_name})
            if fx is None:
                full_df["fx"] = np.ones(full_df.shape[0])
            if fx is not None:
                from backtest_engine.database.finance_database_script import finance_database
                fx_db = finance_database('fx_db')
                fx_df = fx_db.export_from_database(symbol=fx).loc[full_df.index[0]:].rename(columns={'close':'fx'})
                full_df = full_df.join(fx_df['fx'], on='datetime').fillna(method='ffill').fillna(0)
            dfs[ticker] = full_df
        except KeyError:
            print(f'Passed: {ticker}')
            continue
    
    return list(dfs.keys()),dfs

def concat_kpi_price(kpis_dfs, df0, kpis_tickers, tickers0, kpi_name='kpi'):
    tickers = list(set(kpis_tickers) & set(tickers0))
    dfs = {}
    for ticker in tickers:
        df = df0[ticker]
        
        kpi_df = kpis_dfs[ticker].drop(columns=ticker+'_reportDate').set_index('datetime')
        kpi_df.index = pd.DatetimeIndex(kpi_df.index)
        
        full_df = df.join(kpi_df,how='left',on='datetime').fillna(method='ffill').fillna(0)
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

    def __init__(self, insts, dfs, start, end, trade_frequency='daily', portfolio_vol=0.30):
        self.insts = insts
        self.dfs = deepcopy(dfs)
        self.datacopy = deepcopy(dfs)
        self.start = start 
        self.end = end
        self.portfolio_vol = portfolio_vol
        self.trade_frequency = trade_frequency
    
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

    def get_perf_stats(self,capital_rets=True,plot=False):
        from backtest_engine.performance import performance_measures
        assert self.portfolio_df is not None
        rets = 'capital_ret' if capital_rets else 'nominal_ret'
        
        df = self.portfolio_df
        stats_dict = performance_measures(r=self.get_zero_filtered_stats()[rets],plot=plot)
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
            "cagr",
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
    async def run_hypothesis_tests(self,num_decision_shuffles=1000,zfs=None):
        assert self.portfolio_df is not None
        from backtest_engine.performance import plot_hypothesis
        zfs = self.get_zero_filtered_stats()
        rets = zfs["capital_ret"]
        test_dict = await self.hypothesis_tests(num_decision_shuffles=num_decision_shuffles,zfs=zfs)
        plot_hypothesis(test_dict["timer"],test_dict["picker"],test_dict["trader"],rets)
        return

    def pre_compute(self,trade_range):
        pass
    
    def compute_frequency(self,trade_range):
        if self.trade_frequency == 'daily':
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
            trading_day.append(self.dfs[inst]['trading_day'])

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
        self.forecast_df = pd.DataFrame(np.where(self.trading_day,self.forecast_df,np.NaN),
                                        index=self.forecast_df.index,
                                        columns=self.forecast_df.columns
                                        ).fillna(method='ffill')
        return

    @timeme
    def run_simulation(self,use_vol_target=True):
        date_range = pd.date_range(start=self.start,end=self.end, freq="D")
        self.compute_meta_info(trade_range=date_range)
        units_held, weights_held = [],[]
        close_prev = None
        ewmas, ewstrats = [0.01], [1]
        strat_scalars = []
        capitals, nominal_rets, capital_rets = [100000.0],[0.0],[0.0]
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
            forecasts = forecasts/eligibles_row
            forecasts = np.nan_to_num(forecasts,nan=0,posinf=0,neginf=0)
            forecast_chips = np.sum(np.abs(forecasts))
            vol_target = (self.portfolio_vol / np.sqrt(253)) \
                * capitals[-1]

            if use_vol_target:
                positions = strat_scalar * \
                        forecasts / forecast_chips  \
                        * vol_target \
                        / (vol_row * close_row) if forecast_chips != 0 else np.zeros(len(self.insts))
            else:
                dollar_allocation = capitals[-1]/forecast_chips if forecast_chips != 0 else np.zeros(len(self.insts))
                positions = forecasts*dollar_allocation / close_row

            if trading_day or portfolio_i==0:
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
        
        units_df = pd.DataFrame(data=units_held, index=date_range, columns=[inst + " units" for inst in self.insts])
        weights_df = pd.DataFrame(data=weights_held, index=date_range, columns=[inst + " w" for inst in self.insts])
        nom_ser = pd.Series(data=nominals, index=date_range, name="nominal_tot")
        lev_ser = pd.Series(data=leverages, index=date_range, name="leverages")
        cap_ser = pd.Series(data=capitals, index=date_range, name="capital")
        nomret_ser = pd.Series(data=nominal_rets, index=date_range, name="nominal_ret")
        capret_ser = pd.Series(data=capital_rets, index=date_range, name="capital_ret")
        scaler_ser = pd.Series(data=strat_scalars, index=date_range, name="strat_scalar")
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
            (vol_i, vol_row) in zip(
                range(len(self.retdf)),
                self.retdf.iterrows(),
                self.closedf.iterrows(),
                self.eligiblesdf.iterrows(),
                self.trading_day_ser.items(),
                self.voldf.iterrows()
            ):
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

"""
def time_shuffler(self,retdf,leverages,weights,eligs,**kwargs):
        nweights = quant_stats.shuffle_weights_on_eligs(weights_df=weights, eligs_df=eligs, shuffle_type="time")
        return {"retdf": retdf, "leverages": leverages, "weights": nweights, "eligs": eligs}
    
    def picking_shuffler(self,retdf,leverages,weights,eligs,**kwargs):
        nweights = quant_stats.shuffle_weights_on_eligs(weights_df=weights, eligs_df=eligs, shuffle_type="xs")
        return {"retdf": retdf, "leverages": leverages, "weights": nweights, "eligs": eligs}
    
    def skill_shuffler1(self,retdf,leverages,weights,eligs,**kwargs):
        nweights = quant_stats.shuffle_weights_on_eligs(weights_df=weights, eligs_df=eligs, shuffle_type="time")
        nweights = quant_stats.shuffle_weights_on_eligs(weights_df=nweights, eligs_df=eligs, shuffle_type="xs")
        return {"retdf": retdf, "leverages": leverages, "weights": nweights, "eligs": eligs}
    
    def sharpe(self,retdf,leverages,weights,**kwargs):
        capital_ret = [
            lev * np.dot(weight,ret) for lev, weight, ret \
            in zip(leverages, weights.values, retdf.values)
        ]
        sharpe = np.mean(capital_ret) / np.std(capital_ret) * np.sqrt(253)
        return round(sharpe,3),capital_ret
    
    def get_hypothesis_tests(self,zfs=None,m=1000):
        import os
        from pathlib import Path

        zfs = self.get_zero_filtered_stats() if not zfs else zfs
        return_samples = zfs["capital_ret"]
        p1=quant_stats.one_sample_signed_rank_test(sample=return_samples, m0=0.0, side="greater")
        p2=quant_stats.one_sample_sign_test(sample=return_samples, m0=0.0, side="greater")

        function_list = [self.time_shuffler,self.picking_shuffler,self.skill_shuffler1]

        # def skill_shuffler2(**kwargs):
        #     machine_copy = deepcopy(self)
        #     insts = machine_copy.insts
        #     bars = [
        #         machine_copy.datacopy[inst][["open","high","low","close","volume"]]
        #         for inst in insts
        #     ]
        #     permuted_bars = quant_stats.permute_multi_bars(bars)
        #     machine_copy.datacopy.update({inst:bar for inst,bar in zip(insts,permuted_bars)})
        #     machine_copy.dfs=machine_copy.datacopy
        #     machine_copy.run_simulation()
        #     zfs=machine_copy.get_zero_filtered_stats()
        #     return {
        #         "retdf": zfs["retdf"], "leverages": zfs["leverages"], 
        #         "weights": zfs["weights"], "eligs": zfs["eligs"]
        #     }

        paths_list, pvals, stats = [], [], []

        results = [_helper(generator_function, 
                            self.sharpe, 
                            m, 
                            zfs, 
                            paths_list, 
                            pvals, 
                            stats) for generator_function in function_list]
        
        for res in results:
            perm_paths,perm_ps,perm_stats = res[0],res[1],res[2]

        path="/images"
        Path(os.path.abspath(os.getcwd()+path)).mkdir(parents=True,exist_ok=True)

        perm_paths1,perm_paths2,perm_paths3 = perm_paths[0],perm_paths[1],perm_paths[2]
        perm_paths1.index = return_samples.index
        perm_paths2.index = return_samples.index
        perm_paths3.index = return_samples.index

        fig = plt.figure(constrained_layout=True,figsize=(15,11))
        ax = fig.add_gridspec(3, 3)
        ax1 = fig.add_subplot(ax[:2, :])
        ax2 = fig.add_subplot(ax[2:, :2])
        ax3 = fig.add_subplot(ax[2:, -1])

        ax1.plot((1+perm_paths1).cumprod().apply(np.log),color='red',alpha=0.3)
        ax1.plot((1+perm_paths2).cumprod().apply(np.log),color='blue',alpha=0.3)
        ax1.plot((1+perm_paths3).cumprod().apply(np.log),color='green',alpha=0.3)
        ax1.plot((1+return_samples).cumprod().apply(np.log),color='black',linewidth=4)

        stats1,stats2,stats3 = perm_stats[0],perm_stats[1],perm_stats[2]
        pd.Series(stats1).plot(color='red',ax=ax2,kind='kde')
        pd.Series(stats2).plot(color='blue',ax=ax2,kind='kde')
        pd.Series(stats3).plot(color='green',ax=ax2,kind='kde')
        ax2.axvline(self.get_perf_stats().loc['sharpe'],color='black',linewidth=4)

        p3,p4,p5 = perm_ps[0],perm_ps[1],perm_ps[2]
        ps = pd.Series({
            "Permuted MC (Timing)": p3,
            "Permuted MC (Picking)": p4,
            "Permuted MC (Skill)": p5
            #"skill_2": p6
        }).apply(lambda x: np.round(x,4)).reset_index().rename(columns={'index':'Tests',0:'p values'})
        ax3.table(
            cellText=ps.values,colLabels=ps.keys(),loc='center',colWidths=[0.4,0.4],colColours=['grey','grey'],cellLoc='left'
        )
        ax3.axis('off')

        fig.savefig(f".{path}/permuted_returns.png")
        plt.close()
        return 
"""