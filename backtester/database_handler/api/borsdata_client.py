from .borsdata_api import BorsdataAPI
from datetime import date
import pandas as pd
import threading

class BorsdataClient(BorsdataAPI):
    
    def __init__(self, _api_key):
        super().__init__(_api_key)


    def instrument_data(self) -> pd.DataFrame:
        self._countries = self.get_countries()
        self._branches = self.get_branches()
        self._sectors = self.get_sectors()
        self._markets = self.get_markets()
        self._instruments = self.get_instruments()

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
        for index, instrument in self._instruments.iterrows():
            ins_id = index
            name = instrument['name']
            ticker = instrument['ticker']
            isin = instrument['isin']
            instrument_type = instrument_type_dict[instrument['instrument']]

            market = self._markets.loc[self._markets.index == instrument['marketId']]['name'].values[0]
            country = self._countries.loc[self._countries.index == instrument['countryId']]['name'].values[0]
            sector = 'N/A'
            branch = 'N/A'

            if market.lower() != 'index':
                sector = self._sectors.loc[self._sectors.index == instrument['sectorId']]['name'].values[0]
                branch = self._branches.loc[self._branches.index == instrument['branchId']]['name'].values[0]
            
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
        # instrument_df.to_csv(PATH+"/instrument_data.csv")
        return instrument_df
    
    def instrument_mapper(
            self, 
        countries: list[str] | None = None,
        markets: list[str] | None = None,
        sectors: list[str] | None = None, 
        branches: list[str] | None = None 
    ) -> dict:
        
        instrument_df = self.instrument_data()
        countries = self._countries["name"].tolist() if countries is None else countries
        markets = self._markets["name"].tolist() if markets is None else markets
        sectors = self._sectors["name"].tolist() if sectors is None else sectors
        branches = self._branches["name"].tolist() if branches is None else branches

        return instrument_df[(
            instrument_df["country"].isin(countries) & 
            instrument_df["market"].isin(markets) & 
            instrument_df["sector"].isin(sectors) &
            instrument_df["branch"].isin(branches) &
            instrument_df["instrument_type"].isin(["Aktie"])
        )].set_index("ins_id")["ticker"].to_dict()
    
    def call_stock_prices(self, 
        countries: list[str] | None = None,
        markets: list[str] | None = None,
        sectors: list[str] | None = None, 
        branches: list[str] | None = None,
        start_date: str = "2000-01-01",
        end_date: str | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        countries: ['Sverige' 'Finland' 'Danmark' 'Norge']

        markets: ['Large Cap' 'Small Cap' 'Mid Cap' 'First North' 'NGM' 'Spotlight' 'Index'
            'Prelist' 'Oslo Bors' 'Oslo Expand' 'Oslo Growth']

        sectors: ['Dagligvaror' 'Industri' 'Hälsovård' 'Informationsteknik'
            'Kraftförsörjning' 'Finans & Fastighet' 'Sällanköpsvaror' 'Material'
            'Telekommunikation' 'Energi']
            
        branches: ['Livsmedel' 'Industrimaskiner' 'Biotech' 'IT-Konsulter'
            'Industrikomponenter' 'Elektronisk Utrustning' 'Vindkraft' 'Byggmaterial'
            'Läkemedel' 'Fastighetsbolag' 'Bil & Motor' 'Nischbanker'
            'Livsmedelsbutiker' 'Installation & VVS' 'Datorer & Hårdvara'
            'Betting & Casino' 'Skogsbolag' 'Hälsoprodukter' 'Medicinsk Utrustning'
            'Kläder & Skor' 'Gruv - Industrimetaller' 'Förpackning' 'Affärskonsulter'
            'Investmentbolag' 'Detaljhandel' 'Säkerhet & Bevakning' 'Hemelektronik'
            'Hygienprodukter' 'Media & Publicering' 'Marknadsföring' 'Kommunikation'
            'Bygginredning' 'Biometri' 'Affärs- & IT-System' 'Hälsovård & Hjälpmedel'
            'Mätning & Analys' 'Kemikalier' 'Kredit & Finansiering'
            'Bostadsbyggnation' 'Elektroniska komponenter' 'Säkerhet'
            'Gruv - Prospekt & Drift' 'Förnybarenergi' 'Elektronik & Tillverkning'
            'Bredband & Telefoni' 'Byggnation & Infrastruktur' 'Möbler & Inredning'
            'Banker' 'Fritid & Sport' 'Bemanning' 'Militär & Försvar' 'Resor & Nöjen'
            'Energi & Återvinning' 'Sjöfart & Rederi' 'Olja & Gas - Exploatering'
            'Gruv - Guld & Silver' 'Betalning & E-handel' 'Kapitalförvaltning'
            'Gruv - Service' 'Gruv - Ädelstenar' 'Accessoarer' 'Gaming & Spel'
            'Bioenergi' 'Hotell & Camping' 'Tobak' 'Internettjänster' 'Jordbruk'
            'Bryggeri' 'Konsumentservice' 'Telekomtjänster' 'Stödtjänster & Service'
            'N/A' 'Tåg- & Lastbilstransport' 'Elförsörjning'
            'Olja & Gas - Försäljning' 'Försäkring' 'Apotek' 'Flygtransport'
            'Solkraft' 'Sjukhus & Vårdhem' 'Fastighet - REIT' 'Utbildning'
            'Restaurang & Café' 'Olja & Gas - Service' 'Olja & Gas - Transport'
            'Fiskodling' 'Olja & Gas - Borrning' 'Rymd- & Satellitteknik'
            'Vattenförsörjning' 'Information & Data' 'Drycker' 'Fondförvaltning']
        """
        end_date = date.today().strftime("%Y-%m-%d") if end_date is None else end_date
        instrument_map = self.instrument_mapper(countries,markets,sectors,branches)

        dfs = {}
        for ins_id in list(instrument_map.keys()):
            dfs[ins_id] = self.get_instrument_stock_prices(ins_id=ins_id, from_date=start_date,to_date=end_date,max_count=self._params["maxCount"])
        return dfs
        # def _helper(ins_id):
        #     dfs[ins_id] = self.get_instrument_stock_prices(ins_id=ins_id, from_date=start_date,to_date=end_date,max_count=self._params["maxCount"])
        
        # threads = [threading.Thread(target=_helper, args=(ins_id,)) for ins_id in list(instrument_map.keys())]
        # [thread.start() for thread in threads]
        # [thread.join() for thread in threads]
        # return dfs
    
    def call_report_history(
            self,
        countries: list[str] | None = None,
        markets: list[str] | None = None,
        sectors: list[str] | None = None, 
        branches: list[str] | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        countries: ['Sverige' 'Finland' 'Danmark' 'Norge']

        markets: ['Large Cap' 'Small Cap' 'Mid Cap' 'First North' 'NGM' 'Spotlight' 'Index'
            'Prelist' 'Oslo Bors' 'Oslo Expand' 'Oslo Growth']

        sectors: ['Dagligvaror' 'Industri' 'Hälsovård' 'Informationsteknik'
            'Kraftförsörjning' 'Finans & Fastighet' 'Sällanköpsvaror' 'Material'
            'Telekommunikation' 'Energi']
            
        branches: ['Livsmedel' 'Industrimaskiner' 'Biotech' 'IT-Konsulter'
            'Industrikomponenter' 'Elektronisk Utrustning' 'Vindkraft' 'Byggmaterial'
            'Läkemedel' 'Fastighetsbolag' 'Bil & Motor' 'Nischbanker'
            'Livsmedelsbutiker' 'Installation & VVS' 'Datorer & Hårdvara'
            'Betting & Casino' 'Skogsbolag' 'Hälsoprodukter' 'Medicinsk Utrustning'
            'Kläder & Skor' 'Gruv - Industrimetaller' 'Förpackning' 'Affärskonsulter'
            'Investmentbolag' 'Detaljhandel' 'Säkerhet & Bevakning' 'Hemelektronik'
            'Hygienprodukter' 'Media & Publicering' 'Marknadsföring' 'Kommunikation'
            'Bygginredning' 'Biometri' 'Affärs- & IT-System' 'Hälsovård & Hjälpmedel'
            'Mätning & Analys' 'Kemikalier' 'Kredit & Finansiering'
            'Bostadsbyggnation' 'Elektroniska komponenter' 'Säkerhet'
            'Gruv - Prospekt & Drift' 'Förnybarenergi' 'Elektronik & Tillverkning'
            'Bredband & Telefoni' 'Byggnation & Infrastruktur' 'Möbler & Inredning'
            'Banker' 'Fritid & Sport' 'Bemanning' 'Militär & Försvar' 'Resor & Nöjen'
            'Energi & Återvinning' 'Sjöfart & Rederi' 'Olja & Gas - Exploatering'
            'Gruv - Guld & Silver' 'Betalning & E-handel' 'Kapitalförvaltning'
            'Gruv - Service' 'Gruv - Ädelstenar' 'Accessoarer' 'Gaming & Spel'
            'Bioenergi' 'Hotell & Camping' 'Tobak' 'Internettjänster' 'Jordbruk'
            'Bryggeri' 'Konsumentservice' 'Telekomtjänster' 'Stödtjänster & Service'
            'N/A' 'Tåg- & Lastbilstransport' 'Elförsörjning'
            'Olja & Gas - Försäljning' 'Försäkring' 'Apotek' 'Flygtransport'
            'Solkraft' 'Sjukhus & Vårdhem' 'Fastighet - REIT' 'Utbildning'
            'Restaurang & Café' 'Olja & Gas - Service' 'Olja & Gas - Transport'
            'Fiskodling' 'Olja & Gas - Borrning' 'Rymd- & Satellitteknik'
            'Vattenförsörjning' 'Information & Data' 'Drycker' 'Fondförvaltning']
        """
        instrument_map = self.instrument_mapper(countries,markets,sectors,branches)

        dfs = {}
        for ins_id in list(instrument_map.keys()):
            dfs[ins_id] = self.get_instrument_report(ins_id=ins_id, max_count=self._params["maxR12QCount"], report_type="r12")
        return dfs
    
        # def _helper(ins_id):
        #     dfs[ins_id] = self.get_instrument_report(ins_id=ins_id, max_count=self._params["maxR12QCount"], report_type="r12")

        # threads = [threading.Thread(target=_helper, args=(ins_id,)) for ins_id in list(instrument_map.keys())]
        # [thread.start() for thread in threads]
        # [thread.join() for thread in threads]
        # return dfs
    
    def call_single_stock_price(self,
        ins_id: int,
        start_date: str = "2000-01-01",
        end_date: str | None = None
    ) -> pd.DataFrame:
        end_date = date.today().strftime("%Y-%m-%d") if end_date is None else end_date
        return self.get_instrument_stock_prices(ins_id=ins_id, from_date=start_date,to_date=end_date,max_count=self._params["maxCount"])
    
    def call_single_report_history(self,ins_id: int) -> pd.DataFrame:
        return self.get_instrument_report(ins_id=ins_id, max_count=self._params["maxR12QCount"], report_type="r12")
        