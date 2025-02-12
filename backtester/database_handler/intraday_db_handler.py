import sqlalchemy

import pandas as pd
import numpy as np

from sqlalchemy import text
from sqlalchemy import inspect
from backtester.engine import get_configs

PATH: str = get_configs()["PATH"]["DB_PATH"]


class IntradayDatabase():
    def __init__(self) -> None:
        self.engine_name = "intraday_db"
        self.engine = sqlalchemy.create_engine("sqlite:///"+PATH+"/files/" + self.engine_name)

    def export_from_database(self, tickers: list[str], start_date: str = "2007-04-24", end_date: str = "2021-05-01", freq: str = "1min") -> tuple[list[str], dict[str, pd.DataFrame]]:
        with self.engine.begin() as conn:
            dfs = {}
            inspector = inspect(self.engine)
            db_tickers = inspector.get_table_names()
            assert all([ticker in db_tickers for ticker in tickers]), f"Make sure all {tickers} is in DB"

            for ticker in tickers:
                query = text(
                    f"SELECT * FROM {ticker} WHERE datetime BETWEEN :start_date AND :end_date;"
                )
                params = {"start_date": start_date + " 09:30:00", "end_date": end_date + " 16:00:00"}
                df = pd.read_sql_query(query, conn, params=params) \
                    .set_index('datetime') \
                    .drop(columns='index',errors="ignore")
                
                df.index = pd.DatetimeIndex(df.index)
                df = df.between_time("9:30", "16:00")
                ohlc_dict = {'close': 'last'}
                if len(df.columns) > 1: ohlc_dict.update({'open':'first', 'high':'max', 'low':'min', "volume": "sum"})
                dfs[ticker] = df.resample(freq, origin='start').agg(ohlc_dict) if freq != "1min" else df
                
            return tickers, dfs