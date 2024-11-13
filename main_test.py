from engine.database_handler import finance_database
from engine.strategies.test_strategy import BuyHold

from datetime import datetime


TICKERS = ["SPY","QQQ", "TLT", "GLD"]

db = finance_database(database_name="test_db")
db.import_to_database(tickers=TICKERS)
tickers, dfs = db.export_from_database()

strategy = BuyHold(insts=tickers, dfs=dfs, start=datetime(2010,1,1), end=datetime(2024,1,1))
strategy.run_simulation()
strategy.get_perf_stats(plot=True,show=True)