import matplotlib.pyplot as plt
import numpy as np

from pprint import pprint
from datetime import datetime
from backtest_engine.utils import save_pickle,load_pickle

import asyncio
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
            db_kpi1 = kpiDataBase(6, market, country)
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
                                        'eps'  
                                    )
            
            tickers += tickers1
            dfs |= dfs1
            del tickers0
            del dfs0
            del tickers1
            del dfs1
    return tickers,dfs
                                                          
def get_ticker_dfs(countries,fx_map,start,end):
    try:
        tickers, dfs = load_pickle("/Users/oskarfransson/vs_code/trading/backtest_engine/strategies/bbwidth.obj")
    except Exception as err:
        tickers,dfs = get_from_db(countries,fx_map,start,end)
        save_pickle("/Users/oskarfransson/vs_code/trading/backtest_engine/strategies/bbwidth.obj", (tickers,dfs))
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
start = '2013-01-01'
end = '2023-12-01'

tickers, dfs = get_ticker_dfs(countries=countries,fx_map=fx_map,start=start,end=end)
from backtest_engine.strategies.bbwidth_strat import BBwidthStrat

async def main():
    period_start = datetime(2013,1,1)
    period_end = datetime(2023,12,1)
    alpha = BBwidthStrat(insts=np.unique(tickers).tolist(),dfs=dfs,start=period_start,end=period_end,trade_frequency='monthly')
    alpha.run_simulation(use_vol_target=False)
    # alpha.get_perf_stats(plot=True)
    import quantstats as qs
    qs.reports.html(alpha.portfolio_df["capital_ret"],"SPY",output="images/bbwidth_pe.html")
    await alpha.run_hypothesis_tests(num_decision_shuffles=100)
    # save_pickle('strategy_obj/bbwidth_pe_strat.obj',alpha.portfolio_df['capital_ret'])

if __name__ == "__main__":
    asyncio.run(main())