import matplotlib.pyplot as plt
import numpy as np

from pprint import pprint
from datetime import datetime
from backtest_engine.utils import save_pickle

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


def get_from_db(countries,fx_map,start,end):
    from backtest_engine.database.borsdata.borsdata_price_database_script import priceDataBase
    from backtest_engine.database.borsdata.borsdata_kpi_database_script import kpiDataBase

    from backtest_engine.utils import merge_kpi_price,concat_kpi_price
    
    tickers = []
    dfs = {}
    for country, markets in countries.items():
        for market in markets:
            db_price = priceDataBase(country, market, start, end)
            db_kpi0 = kpiDataBase(61, market, country)
            db_kpi1 = kpiDataBase(66, market, country)
            price_tickers, price_dfs = db_price.export_database()
            kpi_tickers0,kpi_dfs0 = db_kpi0.export_database()
            kpi_tickers1,kpi_dfs1 = db_kpi1.export_database()
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
            del tickers0
            del dfs0
            del tickers1
            del dfs1
    return tickers,dfs
        
                                                   
def get_ticker_dfs(countries,fx_map,start,end):
    from backtest_engine.utils import load_pickle,save_pickle
    try:
        tickers, dfs = load_pickle("/Users/oskarfransson/vs_code/trading/backtest_engine/strategies/dataset_syst_yield.obj")
    except Exception as err:
        tickers,dfs = get_from_db(countries,fx_map,start,end)
        save_pickle("/Users/oskarfransson/vs_code/trading/backtest_engine/strategies/dataset_syst_yield.obj", (tickers,dfs))
    return tickers, dfs

countries = {
    'Sverige':['Large Cap', 'Mid Cap', 'Small Cap'],
    'Norge':['Oslo Bors', 'Oslo Expand', 'Oslo Growth'],
    'Danmark':['Large Cap', 'Mid Cap', 'Small Cap']
}
fx_map = {
    'Sverige': None,
    'Norge': 'NOKSEK_X',
    'Danmark': 'DKKSEK_X'
}
start = '2010-01-01'
end = '2023-12-01'

tickers, dfs = get_ticker_dfs(countries=countries,fx_map=fx_map,start=start,end=end)
from backtest_engine.strategies.systematic_yield_strategy import SystYieldAlpha

period_start = datetime(2010,1,1)
period_end = datetime(2023,12,1)
alpha = SystYieldAlpha(insts=np.unique(tickers).tolist(),dfs=dfs,start=period_start,end=period_end,trade_frequency='monthly')
alpha.portfolio_vol = 0.2
alpha.run_simulation(use_vol_target=False)
div_yields = alpha.div_yielddf
weights_df = alpha.weights_df
trading_days = alpha.trading_day_ser
alpha.portfolio_df['capital_ret'] = alpha.portfolio_df['capital_ret'] \
                    + np.sum((weights_df.values*div_yields.values),axis=1)*trading_days.astype(np.float16)*1/12
save_pickle('strategy_obj/syst_yield_strat.obj',alpha.portfolio_df['capital_ret'])
alpha.get_perf_stats(capital_rets=True,plot=True)
import quantstats as qs

qs.reports.html(alpha.portfolio_df["capital_ret"],"SPY",output="systematic_yield.html")