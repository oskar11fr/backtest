import sqlalchemy

import yfinance as yf
import pandas as pd

from sqlalchemy import text
from sqlalchemy import inspect
from .constants import PATH


class finance_database:
    def __init__(self, database_name):
        self.engine_name = database_name
        self.engine = sqlalchemy.create_engine("sqlite:///"+PATH+"/files/" + self.engine_name)
        
    def load_daily_data(self, ticker, start=None):
        df = yf.download(ticker, start=start)[['Open','High','Low','Close','Volume','Adj Close']].reset_index()
        df['Ratio'] = df['Adj Close'] / df['Close']
        df['Open'] = df['Open'] * df['Ratio']
        df['High'] = df['High'] * df['Ratio']
        df['Low'] = df['Low'] * df['Ratio']
        df['Close'] = df['Close'] * df['Ratio']
        df = df.drop(['Ratio', 'Adj Close'], axis = 1)
        df.columns = ['datetime','open','high','low','close','volume']
        
        return df.dropna()
    
    
    def sql_importer(self, symbol):
        try:
            new_symbol = symbol.replace("^","_").replace("-","_").replace("=","_")
            max_date = pd.read_sql(f'SELECT MAX(date) FROM {new_symbol}', self.engine).values[0][0]
            print(max_date)
            new_data = self.load_daily_data(symbol, start=pd.to_datetime(max_date))
            new_rows = new_data[new_data.date > max_date]
            new_rows.to_sql(new_symbol, self.engine, if_exists='append')
            print(str(len(new_rows)) + ' new rows imported to DB')
        except:
            new_symbol = symbol.replace("^","_").replace("-","_").replace("=","_")
            new_data = self.load_daily_data(symbol)
            new_data.to_sql(new_symbol, self.engine)
            print(f'New table created for {new_symbol} with {str(len(new_data))} rows')
            
    
    def import_to_database(self, symbols):
        for symbol in symbols:
            self.sql_importer(symbol)
            
    
    def export_from_database(self,symbol=None):
        with self.engine.begin() as conn:
            if not symbol:
                dfs = {}
                inspector = inspect(self.engine)
                tickers = inspector.get_table_names()
                for ticker in tickers:
                    ticker = ticker.replace("^","_").replace("-", "_").replace("=","_")
                    query = text(f'SELECT * FROM {ticker}')
                    df = pd.read_sql_query(query, conn).set_index('datetime').drop(columns='index')
                    df.index = pd.DatetimeIndex(df.index)
                    dfs[ticker] = df
                return tickers, dfs
            else:
                ticker = symbol.replace("^","_").replace("-", "_").replace("=","_")
                query = text(f'SELECT * FROM {symbol}')
                df = pd.read_sql_query(query, conn).set_index('datetime').drop(columns='index')
                df.index = pd.DatetimeIndex(df.index)
                return df
