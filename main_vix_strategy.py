import asyncio
import numpy as np
from datetime import datetime
from backtest_engine.utils import save_pickle

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


def get_from_db():
    from backtest_engine.database.finance_database_script import finance_database
    db = finance_database('vix_database')
    tickers,dfs = db.export_from_database()
    return tickers,dfs
        
                                                   
def get_ticker_dfs():
    from backtest_engine.utils import load_pickle,save_pickle
    try:
        tickers, dfs = load_pickle("/Users/oskarfransson/vs_code/trading/backtest_engine/strategies/dataset_vix_strat.obj")
    except Exception as err:
        tickers,dfs = get_from_db()
        for ticker in tickers:
            if ticker == 'SVXY':
                dfs[ticker].close = (1+dfs[ticker].close.pct_change().clip(-0.15,np.infty)).cumprod()
        save_pickle("/Users/oskarfransson/vs_code/trading/backtest_engine/strategies/dataset_vix_strat.obj", (tickers,dfs))
    return tickers, dfs

tickers,dfs = get_ticker_dfs()

from backtest_engine.strategies.vix_strategy import VixStrategy

async def main():
    period_start = datetime(2011,1,1)
    period_end = datetime(2023,11,1)
    alpha = VixStrategy(insts=tickers,dfs=dfs,start=period_start,end=period_end,trade_frequency='weekly')
    alpha.run_simulation(use_vol_target=False)
    # save_pickle('vix_strat.obj',alpha.portfolio_df['capital_ret'])
    # alpha.get_perf_stats(plot=True)
    # await alpha.run_hypothesis_tests(num_decision_shuffles=300)
    import quantstats as qs
    qs.reports.html(alpha.portfolio_df["capital_ret"],"SPY",output="vix.html")

if __name__ == "__main__":
    asyncio.run(main()) 

