import numpy as np

from datetime import datetime
from engine.database.constants import STRAT_PATH
from engine.utils import merge_kpi_price,concat_kpi_price,save_pickle,load_pickle
from engine.database.borsdata_price_database_script import priceDataBase
from engine.database.borsdata_kpi_database_script import kpiDataBase

import asyncio
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


def get_equities(countries,fx_map,start,end):
    tickers = []
    dfs = {}
    for country, markets in countries.items():
        for market in markets:
            db_price = priceDataBase(country, market, start, end)
            db_aktier = kpiDataBase(61, market, country)
            db_div = kpiDataBase(66, market, country)

            price_tickers, price_dfs = db_price.export_database()
            aktier_tickers, aktier_dfs, = db_aktier.export_database()
            div_tickers, div_dfs = db_div.export_database()

            tickers0,dfs0 = merge_kpi_price(
                kpis_dfs=aktier_dfs,
                price_dfs=price_dfs,
                kpis_tickers=aktier_tickers,
                price_tickers=price_tickers,
                kpi_name='aktier',
                fx=fx_map[country]
            )
            tickers1,dfs1 = concat_kpi_price(
                kpis_dfs=div_dfs,
                dfs0=dfs0,
                kpis_tickers=div_tickers,
                tickers0=tickers0,
                kpi_name='div'
            )
            tickers += tickers1
            dfs |= dfs1
            del tickers0,dfs0,tickers1,dfs1
    return tickers,dfs

def get_multi_asset():
    from engine.database.finance_database_script import finance_database
    db = finance_database('multi_asset_db')
    tickers,dfs = db.export_from_database()
    return tickers,dfs

def get_vix():
    from engine.database.finance_database_script import finance_database
    db = finance_database('vix_database')
    tickers,dfs = db.export_from_database()
    for ticker in tickers:
            if ticker == 'SVXY':
                dfs[ticker].close = (1+dfs[ticker].close.pct_change().clip(-0.15,np.infty)).cumprod()
    return tickers,dfs
                                                   
def get_ticker_dfs(countries,fx_map,start,end):
    try:
        tickers, dfs = load_pickle(STRAT_PATH+"/dataset_multi_strat.obj")
    except Exception as err:
        tickers,dfs = get_equities(countries,fx_map,start,end)
        for func in [get_vix,get_multi_asset]:
            temp_tickers,temp_dfs = func()
            tickers += temp_tickers
            dfs |= temp_dfs
        save_pickle(STRAT_PATH+"/dataset_multi_strat.obj", (tickers,dfs))
    return tickers, dfs
                    
async def main():
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
    end = '2023-07-01'
    tickers,dfs = get_ticker_dfs(countries,fx_map,start,end)

    from engine.strategies.multi_strategy import MacroStrat
    from engine.strategies.vix_strategy import VixStrategy
    from engine.strategies.systematic_yield_strategy import SystYieldAlpha

    period_start = datetime(2011,1,1)
    period_end = datetime(2023,12,1)

    vix_tickers = ["SVXY", "BIL", "_VIX", "_VIX3M", "_VIX6M"]
    macro_tickers = ["KC_F","GC_F","NG_F", "HG_F", "CL_F", "PA_F", "SI_F", "ZC_F", "ZW_F", "EURSEK_X", "USDSEK_X", "SPY", "ZN_F", "TLT", "QQQ"]
    yield_tickers = [ticker for ticker in tickers if ticker not in macro_tickers+vix_tickers]
    
    yield_alpha = SystYieldAlpha(
        insts=np.unique(yield_tickers).tolist(),
        dfs={ticker: dfs[ticker] for ticker in yield_tickers},
        start=period_start,
        end=period_end,
        trade_range=None,
        trade_frequency='monthly'
    )
    yield_alpha.run_simulation(use_vol_target=False,start_cap=100000)
    # yield_alpha.save_strat_rets("syst_yield")
    
    macro_alpha = MacroStrat(
        insts=np.unique(macro_tickers).tolist(),
        dfs={ticker: dfs[ticker] for ticker in macro_tickers},
        start=period_start,
        end=period_end,
        trade_frequency='monthly'
    )
    macro_alpha.run_simulation(use_vol_target=True, start_cap=62000)
    # etf_alpha.save_strat_rets("ms_macro")

    vix_alpha = VixStrategy(
        insts=vix_tickers,
        dfs={ticker: dfs[ticker] for ticker in vix_tickers},
        start=period_start,
        end=period_end,
        trade_frequency='weekly'
    )
    vix_alpha.run_simulation(use_vol_target=False,start_cap=20000)
    # vix_alpha.save_strat_rets("ms_vix")

    cap1 = yield_alpha.portfolio_df["capital"]
    cap2 = macro_alpha.portfolio_df["capital"]
    cap3 = vix_alpha.portfolio_df["capital"]
    capital = cap1 + cap2 + cap3

    from engine.performance import performance_measures
    performance_measures(
        capital.pct_change().fillna(0),
        plot=True,
        show=False,
        market={"macro":cap2,"vix":cap3, "yield":cap1}
    )

    # await yield_alpha.run_hypothesis_tests(num_decision_shuffles=50,strat_name="syst_yield")
    # await macro_alpha.run_hypothesis_tests(num_decision_shuffles=200,strat_name="macro")
    # await vix_alpha.run_hypothesis_tests(num_decision_shuffles=200,strat_name="vix")

if __name__ == "__main__":
    asyncio.run(main())