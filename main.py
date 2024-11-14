from engine.database_handler import finance_database
from engine.strategies.vol_carry import VolCarry

from datetime import datetime

TICKERS = ["SVXY", "SPY", "^VIX", "^VIX3M", "^VIX6M"]


db = finance_database(database_name="test_db")
db.import_to_database(tickers=TICKERS)
tickers, dfs = db.export_from_database()

strategy = VolCarry(insts=tickers, dfs=dfs, start=datetime(2013,1,1), end=datetime(2024,11,12), benchmark="SPY")
strategy.run_simulation(use_vol_target=False)
stats = strategy.get_perf_stats(plot=True, market=True, show=True)
print(stats)