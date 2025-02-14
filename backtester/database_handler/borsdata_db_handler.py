import sqlalchemy
import yaml
import pandas as pd

from datetime import date
from sqlalchemy import text, inspect
from backtester.engine import get_configs
from .api.borsdata_client import BorsdataClient

PATH: str = get_configs()["PATH"]["DB_PATH"]
CREDENTIALS_PATH: str = get_configs()["PATH"]["CREDENTIALS_PATH"]


def price_earnings_joiner(
        price_dfs: dict[str, pd.DataFrame],
        report_dfs: dict[str, pd.DataFrame], 
        select_cols: list[str] | None = None, 
        rel_change: bool | str | list[str] = False, 
        abs_change: bool | str | list[str] = False
    ) -> dict[str, pd.DataFrame]:
    """
    'year', 'period', 'revenues', 'grossIncome', 'operatingIncome',
       'profitBeforeTax', 'profitToEquityHolders', 'earningsPerShare',
       'numberOfShares', 'dividend', 'intangibleAssets', 'tangibleAssets',
       'financialAssets', 'nonCurrentAssets', 'cashAndEquivalents',
       'currentAssets', 'totalAssets', 'totalEquity', 'nonCurrentLiabilities',
       'currentLiabilities', 'totalLiabilitiesAndEquity', 'netDebt',
       'cashFlowFromOperatingActivities', 'cashFlowFromInvestingActivities',
       'cashFlowFromFinancingActivities', 'cashFlowForTheYear', 'freeCashFlow',
       'stockPriceAverage', 'stockPriceHigh', 'stockPriceLow',
       'reportStartDate', 'reportEndDate', 'brokenFiscalYear', 'currency',
       'currencyRatio', 'netSales'
    """
    def calc_rel_change(df: pd.DataFrame, cols: str | list[str]) -> pd.DataFrame:
        df = df.copy()
        if isinstance(cols, str):
            cols = [cols]
            
        for col in cols:
            df.loc[:, col + "_rel_ch"] = df[col].pct_change().fillna(0)
        return df
    
    def calc_abs_change(df: pd.DataFrame,cols: str | list[str]) -> pd.DataFrame:
        df = df.copy()
        if isinstance(cols, str):
            cols = [cols]

        for col in cols:
            df.loc[:, col + "_abs_ch"] = df[col].diff().fillna(0)
        return df
    
    dfs = {}
    for ticker, df in price_dfs.items():
        report_df = report_dfs[ticker].drop(columns=['ticker', 'market', 'sector', 'branch']) \
            if select_cols is None else report_dfs[ticker][select_cols]
        
        if rel_change or isinstance(rel_change, str) or isinstance(rel_change, list):
            cols = select_cols if rel_change else rel_change
            report_df = calc_rel_change(report_df, cols=cols)
        if abs_change or isinstance(abs_change, str) or isinstance(abs_change, list):
            cols = select_cols if abs_change else abs_change
            report_df = calc_abs_change(report_df, cols=cols)

        df = df.join(report_df).ffill()
        dfs[ticker] = df
    return dfs



class borsdata_database:
    def __init__(self, data_params: dict[str, list[str]] | None = None) -> None:
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

        self.mapper = mapper
        self.data_params = data_params
        return
    
    def generate_info_columns(self, df: pd.DataFrame, ins_id: int) -> pd.DataFrame:
        df["ticker"] = self.insid_to_ticker[ins_id]
        df["country"] = self.insid_to_country[ins_id]
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
        
    def get_ins_names(self, engine: sqlalchemy.Engine, country: str) -> list[str]:
        decode = lambda x: int(x.replace("ins_", ""))
        inspector = inspect(engine)
        ins_ids = inspector.get_table_names()
        if self.data_params is not None:
            self.data_params["country"] = [country]
            sets = []
            for ins, row in self.mapper.set_index("ins_id").T.iterrows():
                if ins in self.data_params.keys(): 
                    sets.append(set(row[row.isin(self.data_params[ins])].index))
            intersects = set.intersection(*sets)
            ins_ids = [ins_id for ins_id in ins_ids if decode(ins_id) in intersects]
        return ins_ids

    def load_data(self, country: str, data_type: str, update: bool = False) -> dict[str, pd.DataFrame]:
        engine = self.initialize_engine(country=country,data_type=data_type)
        ins_ids = self.get_ins_names(engine=engine,country=country)
        if len(ins_ids) == 0 or update:
            self.import_to_database(engine, country, data_type, single=False)
            ins_ids = self.get_ins_names(engine=engine,country=country)
        with engine.begin() as conn:
            dfs = {}
            for ins_id in ins_ids:
                query = text(f'SELECT * FROM {ins_id}')
                df = pd.read_sql_query(query, conn)
                df = df[
                    ~(df["datetime"].isna() | df["datetime"].duplicated()) & 
                    (df["datetime"] > "1900-01-01")
                ].set_index('datetime')
                df.index = pd.DatetimeIndex(df.index)
                dfs[self.insid_to_ticker[int(ins_id.replace("ins_",""))]] = df
            return dfs