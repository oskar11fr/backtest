import sqlalchemy

import pandas as pd
import numpy as np

from binance import Client
from sqlalchemy import text
from sqlalchemy import inspect
from backtester.engine import get_configs
from datetime import datetime


PATH: str = get_configs()["PATH"]["DB_PATH"]

class BinanceDB():
    def __init__(self, database_name: str):
        self.engine_name = database_name
        self.engine = sqlalchemy.create_engine("sqlite:///"+PATH+"/files/" + self.engine_name)
        self.client = Client()
        self.today = datetime.today()
        self.symbols = [
            "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","XRPUSDT","DOTUSDT","DOGEUSDT"
        ]

    def _load_data(self, symbol: str, lookback: int, granularity: str = "1h") -> pd.DataFrame:
        '''
        return: list of OHLCV values 
        (Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades, Taker buy base asset volume, Taker buy quote asset volume, Ignore)
        '''
        df = pd.DataFrame(
            self.client.get_historical_klines(symbol,granularity,str(lookback) + " days ago UTC")
        )
        df.columns = ["open_time", "open", "high", "low", "close", "volume", 
                      "close_time", "quote_asset_volume", "number_of_trades", 
                      "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
        df["datetime"] = pd.to_datetime(df["close_time"], unit="ms")
        df = df.set_index("datetime")
        df = df.drop(columns=["open_time","close_time","ignore"]).astype(np.float128)
        return df.reset_index()
    
    def sql_importer(self, symbol: str) -> None:
        try:
            max_date = pd.read_sql(f'SELECT MAX(datetime) FROM {symbol}', self.engine).values[0][0]
            new_period = (self.today - pd.to_datetime(max_date)).days
            if new_period > 0:
                df = self._load_data(symbol,lookback=new_period)
                df.to_sql(symbol,self.engine, if_exists='append')
                print(str(len(df)) + ' new rows imported to DB')
            else: print("No new rows appended")

        except:
            new_data = self._load_data(symbol, 1000)
            new_data.to_sql(symbol, self.engine)
            print(f'New table created for {symbol} with {str(len(new_data))} rows')
        return
    
    def import_to_database(self, tickers: list[str]):
        for ticker in tickers:
            self.sql_importer(ticker)

    def export_from_database(self, tickers: list[str]) -> tuple[list[str], dict[str, pd.DataFrame]]:
        with self.engine.begin() as conn:
            dfs = {}
            inspector = inspect(self.engine)
            db_tickers = inspector.get_table_names()
            assert all([ticker in db_tickers for ticker in tickers]), f"Make sure all {tickers} is in DB"

            for ticker in tickers:
                query = text(f'SELECT * FROM {ticker}')
                df = pd.read_sql_query(query, conn).set_index('datetime').drop(columns='index')
                df.index = pd.DatetimeIndex(df.index) + pd.Timedelta(hours=1)
                df = df.between_time("9:00", "22:00")
                df = df[df.index.dayofweek < 5]
                dfs[ticker] = df
        return tickers, dfs
