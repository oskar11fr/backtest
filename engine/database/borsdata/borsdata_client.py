import threading
import pandas as pd

from .constants import API_KEY
from .borsdata_api import BorsdataAPI


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class BorsdataClient:
    def __init__(self):
        self._borsdata_api = BorsdataAPI(API_KEY)
        self._instruments_with_meta_data = pd.DataFrame()

    def instruments_with_meta_data(self):
        """
        creating a csv and xlsx of the APIs instrument-data (including meta-data)

        """
        
        countries = self._borsdata_api.get_countries()
        branches = self._borsdata_api.get_branches()
        sectors = self._borsdata_api.get_sectors()
        markets = self._borsdata_api.get_markets()
        instruments = self._borsdata_api.get_instruments()
        # instrument type dict for conversion (https://github.<com/Borsdata-Sweden/API/wiki/Instruments)
        instrument_type_dict = {
            0: 'Aktie', 
            1: 'Pref', 
            2: 'Index', 
            3: 'Stocks2', 
            4: 'SectorIndex',
            5: 'BranschIndex', 
            8: 'SPAC', 
            13: 'Index GI'
        }

        instrument_df = pd.DataFrame()

        # loop through the whole dataframe (table) i.e. row-wise-iteration.
        
        for index, instrument in instruments.iterrows():
            ins_id = index
            name = instrument['name']
            ticker = instrument['ticker']
            isin = instrument['isin']
            instrument_type = instrument_type_dict[instrument['instrument']]

            # .loc locates the rows where the criteria (inside the brackets, []) is fulfilled
            # located rows (should be only one) get the column 'name' and return its value-array
            # take the first value in that array ([0], should be only one value)

            market = markets.loc[markets.index == instrument['marketId']]['name'].values[0]
            country = countries.loc[countries.index == instrument['countryId']]['name'].values[0]
            sector = 'N/A'
            branch = 'N/A'

            # index-typed instruments does not have a sector or branch

            if market.lower() != 'index':
                sector = sectors.loc[sectors.index == instrument['sectorId']]['name'].values[0]
                branch = branches.loc[branches.index == instrument['branchId']]['name'].values[0]

            # appending current data to dataframe, i.e. adding a row to the table.
            
            df_temp = pd.DataFrame([{
                'name': name, 
                'ins_id': ins_id, 
                'ticker': ticker, 
                'isin': isin,
                'instrument_type': instrument_type,
                'market': market, 
                'country': country, 
                'sector': sector, 
                'branch': branch
            }])
            instrument_df = pd.concat([instrument_df, df_temp], ignore_index=True)

        return instrument_df

    def history_kpi(self, kpi, market, country):
        """
        gathers and concatenates historical kpi-values for provided kpi, market and country
        :param kpi: kpi id see https://github.com/Borsdata-Sweden/API/wiki/KPI-History
        :param market: market to gather kpi-values from
        :param country: country to gather kpi-values from
        :return: pd.DataFrame of historical kpi-values
        """
        # creating api-object

        instruments = self.instruments_with_meta_data()
        filtered_instruments = instruments.loc[(instruments['market'] == market) & (instruments['country'] == country)]
        n = filtered_instruments.shape[0]
        frames = []
        self.names = []

        def _helper(instrument):
            try:
                instrument_kpi_history = self._borsdata_api.get_kpi_history(int(instrument['ins_id']), kpi, 'r12', 'mean')
            
            except KeyError:
                instrument_kpi_history = 0
                print(f'Key error on {int(instrument["ins_id"])}. Skipping iteration.')
            
            if len(instrument_kpi_history) > 0:
                try:
                    # resetting index and adding name as a column
                    instrument_kpi_history.reset_index(inplace=True)
                    instrument_kpi_history['name'] = instrument['name']
                    frames.append(instrument_kpi_history.copy())
                    self.names.append(instrument["name"])

                except ValueError:
                    print(f'Key error on {int(instrument["ins_id"])}. Skipping iteration.')
        
        threads = [threading.Thread(target=_helper, args=(instrument,)) for _,instrument in filtered_instruments.iterrows()]
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
        return frames
        
