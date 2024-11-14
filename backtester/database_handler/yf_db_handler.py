import sqlalchemy

import pandas as pd
import yfinance as yf

from sqlalchemy import text
from sqlalchemy import inspect
from backtester import get_configs


PATH: str = get_configs()["PATH"]["DB_PATH"]

class finance_database:
    def __init__(self, database_name: str):
        self.engine_name = database_name
        self.engine = sqlalchemy.create_engine("sqlite:///"+PATH+"/files/" + self.engine_name)
        
    def load_daily_data(self, ticker: str, start: str | None = None) -> pd.DataFrame:
        df = yf.download(ticker, start=start)[['Open','High','Low','Close','Volume','Adj Close']] \
            .reset_index() \
            .droplevel(axis=1,level=1)
        
        df['Ratio'] = df['Adj Close'] / df['Close']
        df['Open'] = df['Open'] * df['Ratio']
        df['High'] = df['High'] * df['Ratio']
        df['Low'] = df['Low'] * df['Ratio']
        df['Close'] = df['Close'] * df['Ratio']
        df = df.drop(['Ratio', 'Adj Close'], axis = 1)
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
            
    def export_from_database(self) -> tuple[list[str], dict[str, pd.DataFrame]]:
        with self.engine.begin() as conn:
            dfs = {}
            inspector = inspect(self.engine)
            tickers = inspector.get_table_names()
            for ticker in tickers:
                query = text(f'SELECT * FROM {ticker}')
                df = pd.read_sql_query(query, conn).set_index('datetime').drop(columns='index')
                df.index = pd.DatetimeIndex(df.index)
                dfs[ticker] = df
            return tickers, dfs
