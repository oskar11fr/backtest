from backtester.database_handler import finance_database
from backtester.vol_carry import VolCarry
from backtester.tactical import Tactical
from backtester.buy_hold import BuyHold
from backtester.engine.utils import bundle_strategies

from datetime import datetime



def vol_carry_main():
    TICKERS = ["SVXY", "SPY", "^VIX", "^VIX3M", "^VIX6M", "BIL"]

    db = finance_database(database_name="vix_db")
    db.import_to_database(tickers=TICKERS)
    tickers, dfs = db.export_from_database()

    strategy = VolCarry(insts=tickers, dfs=dfs, start=datetime(2013,1,1), end=datetime(2024,11,12), benchmark="SPY")
    return strategy

def tactical_main():
    TICKERS = ["QQQ", "XLE", "XLU", "GLD", "TLT", "SHY", "UUP"]
    db = finance_database(database_name="etf_db")
    db.import_to_database(tickers=TICKERS)
    tickers, dfs = db.export_from_database()

    strategy = Tactical(insts=tickers, dfs=dfs, start=datetime(2006,4,1), end=datetime(2024,11,12), benchmark="QQQ", max_leverage=1.5, portfolio_vol=0.15)
    return strategy

if __name__ == "__main__":
    tactical_strat = tactical_main()
    tactical_strat.run_simulation(use_vol_target=True)
    # tactical_strat.run_hypothesis_tests(num_decision_shuffles=100,strat_name="tactical_strategy")
    tactical_strat.get_perf_stats(plot=True,show=False,strat_name="tactical_strategy",compare=True)
    exit()

    vol_carry_strat = vol_carry_main()
    vol_carry_strat.run_simulation(use_vol_target=False)
    vol_carry_strat.run_hypothesis_tests(num_decision_shuffles=100,strat_name="vol_carry_strategy")
    vol_carry_strat.get_perf_stats(plot=True,show=False,strat_name="vol_carry_strategy",compare=False)

    
    tickers, dfs = bundle_strategies({
        "tactical_strat": tactical_strat,
        "vol_carry_strat": vol_carry_strat
    })

    combined = BuyHold(insts=tickers, dfs=dfs, start=datetime(2006,4,1), end=datetime(2024,11,12), portfolio_vol=0.15, max_leverage=1.)
    combined.run_simulation(use_vol_target=True,start_cap=1000000)
    combined.get_perf_stats(plot=True,show=False,strat_name="combined")
