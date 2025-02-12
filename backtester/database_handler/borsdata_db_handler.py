import sqlalchemy
import yaml
import pandas as pd

from datetime import date
from sqlalchemy import text, inspect
from backtester.engine import get_configs
from .api.borsdata_client import BorsdataClient

PATH: str = get_configs()["PATH"]["DB_PATH"]
CREDENTIALS_PATH: str = get_configs()["PATH"]["CREDENTIALS_PATH"]


class borsdata_database:
    def __init__(self) -> None:
        with open(CREDENTIALS_PATH + "/credentials.yml", "r") as file:
            API_KEY = yaml.safe_load(file)["bd_api"]

        self.client = BorsdataClient(_api_key=API_KEY)

        mapper = self.client.instrument_data()
        mapper = mapper[mapper["instrument_type"].isin(["Aktie"])]

        self.insid_to_ticker = mapper.set_index("ins_id")["ticker"].to_dict()
        self.insid_to_country = mapper.set_index("ins_id")["country"].to_dict()
        self.insid_to_market = mapper.set_index("ins_id")["market"].to_dict()
        self.insid_to_sector = mapper.set_index("ins_id")["sector"].to_dict()
        self.insid_to_branch = mapper.set_index("ins_id")["branch"].to_dict()
        return
    
    def generate_info_columns(self, df: pd.DataFrame, ins_id: int) -> pd.DataFrame:
        df["ticker"] = self.insid_to_ticker[ins_id]
        df["market"] = self.insid_to_market[ins_id]
        df["market"] = self.insid_to_market[ins_id]
        df["sector"] = self.insid_to_sector[ins_id]
        df["branch"] = self.insid_to_branch[ins_id]
        return df
        
    def initialize_engine(self, country: str, data_type: str) -> sqlalchemy.Engine:
        countries, data_types = ["Sverige", "Norge", "Danmark", "Finland"], ["price", "report"]
        assert country in countries and data_type in data_types, "country and datatype must be of valid type"

        engine_name = country + "_" + data_type
        engine = sqlalchemy.create_engine("sqlite:///"+PATH+"/files/" + engine_name)
        return engine
    
    def import_to_database(self, engine: sqlalchemy.Engine, country: str, data_type: str, single: bool, *args) -> None:
        if data_type == "price": dfs = self.client.call_stock_prices(countries=[country])
        if data_type == "report": dfs = self.client.call_report_history(countries=[country])
        for ins_id, df in dfs.items():
            self.generate_info_columns(df, ins_id).to_sql(name="ins_"+str(ins_id),con=engine,if_exists="replace")
        return
        
    def get_table_names(self, engine: sqlalchemy.Engine) -> list[str]:
        inspector = inspect(engine)
        ins_ids = inspector.get_table_names()
        return ins_ids

    def load_data(self, country: str, data_type: str, update: bool = False) -> dict[str, pd.DataFrame]:
        engine = self.initialize_engine(country=country,data_type=data_type)
        ins_ids = self.get_table_names(engine=engine)
        if len(ins_ids) == 0 or update:
            self.import_to_database(engine, country, data_type, single=False)
        
        ins_ids = self.get_table_names(engine=engine)
        with engine.begin() as conn:
            dfs = {}
            for ins_id in ins_ids:
                query = text(f'SELECT * FROM {ins_id}')
                df = pd.read_sql_query(query, conn)
                df = df[~df["datetime"].isna()].set_index('datetime')
                df.index = pd.DatetimeIndex(df.index)
                dfs[self.insid_to_ticker[int(ins_id.replace("ins_",""))]] = df
            return dfs
        
