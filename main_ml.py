from backtester.database_handler.intraday_db_handler import IntradayDatabase
from backtester.intraday_ml import IntradayML


db = IntradayDatabase()
tickers, dfs = db.export_from_database(["SPY","neutral"], start_date="2012-01-01", end_date="2014-01-01")
strat = IntradayML(insts=tickers, dfs=dfs, date_range=dfs["SPY"].index,benchmark="SPY")

strat.run_simulation(use_vol_target=False)
strat.get_perf_stats(plot=True, compare=True, test=.5, strat_name="intraday_ml")