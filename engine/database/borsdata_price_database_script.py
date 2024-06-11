import pandas as pd
import sqlalchemy
import threading

from sqlalchemy import inspect
from sqlalchemy import text

from .constants import API_KEY, PATH
from .borsdata_client import BorsdataClient
from .borsdata_api import BorsdataAPI


class priceDataBase:
    def __init__(self, country ,market, start_date, end_date):
        self.client = BorsdataClient()
        self.api = BorsdataAPI(API_KEY)
        self.country = country
        self.market = market
        self.start_date = start_date
        self.end_date = end_date

        engine_name = f'prices_{self.market}_{self.country}'
        self.engine_name = engine_name.replace(" ", "_")
        self.engine = sqlalchemy.create_engine("sqlite:///" + PATH+ "/files/" + self.engine_name)
        self.csv_file = self.client.instruments_with_meta_data()
        names_ids = self.csv_file[['name','ins_id', 'market', 'country']]
        self.filter_names_ids = names_ids[(names_ids.country == self.country) & (names_ids.market == self.market)]
        self.id_list = self.filter_names_ids.ins_id.values.tolist()
    
    def _import_data(self):
        df_list = {}
        symbols = []
        def _helper(idn):
            df0 = self.api.get_instrument_stock_prices(idn,from_date=self.start_date, to_date=self.end_date)
            symbol = self.filter_names_ids[self.filter_names_ids.ins_id == idn].name.values.tolist()[0]
            symbol = symbol.replace("&", "and")
            symbol = symbol.replace(" ", "_")
            symbol = symbol.replace(".", "")
            symbol = symbol.replace("-", "_")
            df_list[symbol] = df0.copy()
            symbols.append(symbol)
            print(symbol)
        
        threads = [threading.Thread(target=_helper, args=(idn,)) for idn in self.id_list]
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]  
        return symbols,df_list
    
    def create_database(self):
        symbols,df_list = self._import_data()
        for symbol in symbols:
            new_data = df_list[symbol]
            new_data = new_data.rename(columns={'date':'datetime'})
            new_data.to_sql(symbol, self.engine)
        return
    
    def get_data(self):
        symbols,df_list = self._import_data()
        dfs = {}
        for symbol in symbols:
            new_data = df_list[symbol]
            new_data = new_data.rename(columns={'date':'datetime'})
            dfs[symbol] = new_data
        return list(dfs.keys()),dfs
    
    def export_database(self, query_string = '*'):
        data_frames = {}
        inspector = inspect(self.engine)
        symbols = inspector.get_table_names()
        with self.engine.begin() as conn:
            if len(symbols) > 1:
                for symbol in symbols:
                    try:
                        query = text(f'SELECT {query_string} FROM {symbol}')
                        data = pd.read_sql_query(query, conn)#.set_index('date')
                        #data.index = pd.DatetimeIndex(data.index)
                        data_frames[symbol] = data
                    except sqlalchemy.exc.OperationalError:
                        print(f'OperationalError. Skips {symbol}')
                        pass
            
            else:
                query = text(f'SELECT {query_string} FROM {symbols[0]}')
                data = pd.read_sql_query(query, conn)#.set_index('date')
                #data.index = pd.DatetimeIndex(data.index)
                data_frames[symbol] = data
         
        return symbols,data_frames
