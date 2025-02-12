import sqlalchemy

import pandas as pd
import yfinance as yf

from sqlalchemy import text, inspect
from backtester.engine import get_configs


PATH: str = get_configs()["PATH"]["DB_PATH"]

class finance_database:
    def __init__(self, database_name: str, use_db: bool = True):
        self.engine_name = database_name
        if use_db:
            self.engine = sqlalchemy.create_engine("sqlite:///"+PATH+"/files/" + self.engine_name)
    
    def load_daily_data(self, ticker: str, start: str | None = None) -> pd.DataFrame:
        df = yf.download(ticker, start=start,auto_adjust=True)[['Open','High','Low','Close','Volume']] \
            .reset_index() \
            .droplevel(axis=1,level=1)
        
        df.columns = ['datetime','open','high','low','close','volume']
        return df.dropna()
    
    def ticker_fixer(self, ticker: str) -> str:
        return ticker \
            .replace("^","_") \
            .replace("-","_") \
            .replace("=","_") \
            .replace(".","_") \
    
    def sql_importer(self, ticker):
        fixed_ticker = self.ticker_fixer(ticker)
        try:
            max_date = pd.read_sql(f'SELECT MAX(datetime) FROM {fixed_ticker}', self.engine).values[0][0]
            new_data = self.load_daily_data(ticker, start=pd.to_datetime(max_date))
            new_rows = new_data[new_data["datetime"] > max_date]
            if not new_rows.empty:
                new_rows.to_sql(fixed_ticker, self.engine, if_exists='append')
                print(str(len(new_rows)) + ' new rows imported to DB')
        except:
           new_data = self.load_daily_data(ticker)
           new_data.to_sql(fixed_ticker, self.engine)
           print(f'New table created for {fixed_ticker} with {str(len(new_data))} rows')
            
    def import_to_database(self, tickers: list[str]):
        for ticker in tickers:
            self.sql_importer(ticker)
            
    def export_from_database(self, tickers: list[str]) -> tuple[list[str], dict[str, pd.DataFrame]]:
        tickers = [self.ticker_fixer(ticker) for ticker in tickers]
        with self.engine.begin() as conn:
            dfs = {}
            inspector = inspect(self.engine)
            db_tickers = inspector.get_table_names()
            assert all([ticker in db_tickers for ticker in tickers]), f"Make sure all {tickers} is in DB"

            for ticker in tickers:
                query = text(f'SELECT * FROM {ticker}')
                df = pd.read_sql_query(query, conn).set_index('datetime').drop(columns='index')
                df.index = pd.DatetimeIndex(df.index)
                dfs[ticker] = df
            return tickers, dfs
