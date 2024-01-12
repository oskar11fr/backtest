from engine.database.borsdata.borsdata_api import BorsdataAPI
from engine.database.borsdata.constants import API_KEY
from engine.utils import load_pickle,save_pickle


import pandas as pd
import numpy as np
import threading
import os


def get_data():
    api = BorsdataAPI(API_KEY)
    name_id_map = pd.read_csv("/Users/oskarfransson/vs_code/trading/backtest_engine/database/borsdata/instrument_with_meta_data.csv")
    markets = [
        "Large Cap",
        "Mid Cap",
        "Small Cap",
        "Oslo Bors",
        "Oslo Expand",
        "Oslo Growth"
    ]

    countries = [
        "Sverige",
        "Norge",
        "Danmark"
    ]
    idx = [i for i in name_id_map.index if name_id_map.iloc[i].market in markets]
    names = set(name_id_map.iloc[idx].name)

    dfs = dict()
    for name in names:
    # def _helper(name):
        try:
            ins_id = name_id_map[name_id_map.name == name].ins_id.values.tolist()[0]
            dates = pd.to_datetime(api.get_instrument_report(int(ins_id), 'r12').reportDate.dropna().values)
            dates_ser = pd.Series(1.,index=dates)
            
            df = api.get_instrument_stock_prices(
                ins_id=ins_id,
                from_date=dates[0]
            )
            df = df.sort_index(ascending=True)
            df = pd.concat([df,dates_ser],axis=1).fillna(0.)
            dfs[name] = df
            print(name)
        except Exception as e:
            print(f"Skipped {name} for {e}")
    # threads = [threading.Thread(target=_helper, args=(name,)) for name in names]
    # [thread.start() for thread in threads]
    # [thread.join() for thread in threads]
    return list(set(dfs.keys())),dfs

def load_data():
    try:
        tickers,dfs0 = load_pickle("/Users/oskarfransson/vs_code/trading/backtest_engine/strategies/dataset_pead.obj")
    except Exception as err:
        tickers,dfs0 = get_data()
        save_pickle("/Users/oskarfransson/vs_code/trading/backtest_engine/strategies/dataset_pead.obj", (tickers,dfs))
    dfs = dict()
    for name,df in dfs0.items(): 
        df.columns = ['high', 'low', 'close', 'open', 'volume', "reportdate"]
        dfs[name] = df.fillna(method="ffill").fillna(method="bfill")
    tickers = set(dfs.keys())
    return tickers, dfs

tickers,dfs = load_data()

from datetime import datetime
import warnings

warnings.filterwarnings("ignore",category=RuntimeWarning)

period_start = datetime(2014,1,1)
period_end = datetime(2023,12,31)

from engine.strategies.pead_strat import PEAD
alpha = PEAD(insts=tickers,dfs=dfs,start=period_start,end=period_end,trade_frequency='weekly')
alpha.run_simulation(use_vol_target=False)
alpha.get_perf_stats(plot=True)
save_pickle('strategy_obj/pead_strat.obj',alpha.portfolio_df['capital_ret'])

