from datetime import datetime
from backtester import BacktestEngine
from pandas.core.api import DataFrame as DataFrame, DatetimeIndex
from backtester.engine.functions.portfolio_strategies import PositioningStrategy, VolatilityTargetingStrategy

import numpy as np
import pandas as pd
import keras as keras


class KerasModel(keras.Model):
    SEED = keras.utils.set_random_seed(11)

    def __init__(self, d: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._d = d
    
        self.__init_model__()
        self.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=0.001)
        )
        
    def __init_model__(self):
        self.input_layer = keras.layers.InputLayer(shape=(self._d,))
        self.dense_layer1 = keras.layers.Dense(units=512,activation=keras.activations.relu)
        self.dense_layer2 = keras.layers.Dense(units=64,activation=keras.activations.relu)
        self.output_layer = keras.layers.Dense(units=1,activation=keras.activations.tanh)
    
    def call(self, inputs):
        z1 = self.dense_layer1(inputs)
        z2 = self.dense_layer2(z1)
        return self.output_layer(z2)

def _create_target(inst_df: pd.DataFrame, wind=30) -> pd.Series:
    df = inst_df.copy()
    frame_cleaner = lambda ser: np.nan_to_num(ser, nan=0, posinf=0, neginf=0)
    def assign_minute_since_open(timestamp):
        trading_start = pd.Timestamp(timestamp.date()).replace(hour=9, minute=30)
        return (timestamp - trading_start).seconds // 60
    
    df["minute_since_open"] = df.index.map(assign_minute_since_open)
    df['trading_day'] = df.index.date
    # df["ret_2"] = df["ret"] ** 2
    twap = df.groupby("trading_day")["close"].cumsum() / (df["minute_since_open"] + 1)
    # vol = df.groupby("trading_day")["ret_2"].cumsum() / (df["minute_since_open"] + 1)
    zscore = (df["close"].apply(np.log) - twap.apply(np.log)) / df["ret"].rolling(390).std()
    zscore = (zscore).apply(frame_cleaner)
    zscore[zscore.at_time("09:00")] = 0
    zscore = zscore.shift(-1).fillna(0)
    return -zscore.apply(np.tanh)

def _create_timebucket(inst_df: pd.DataFrame) -> pd.Series:
    df = inst_df.copy()
    def assign_timebucket(timestamp):
        trading_start = pd.Timestamp(timestamp.date()).replace(hour=9, minute=30)
        trading_end = pd.Timestamp(timestamp.date()).replace(hour=16, minute=0)
        if timestamp < trading_start or timestamp > trading_end:
            return None
        minutes_since_open = (timestamp - trading_start).seconds // 60
        return minutes_since_open // 15 + 1
    return df.index.map(assign_timebucket)

def _vwap(inst_df: pd.DataFrame) -> pd.Series:
    df = inst_df.copy()
    df['trading_day'] = df.index.date
    df["cumulative_volume"] = df.groupby("trading_day")['volume'].cumsum()
    df["price_volume"] = df["close"] * df["volume"]
    vwap = df.groupby("trading_day")["price_volume"].cumsum() / df["cumulative_volume"]
    return vwap

def _twap(inst_df: pd.DataFrame) -> pd.Series:
    df = inst_df.copy()
    def assign_minute_since_open(timestamp):
        trading_start = pd.Timestamp(timestamp.date()).replace(hour=9, minute=30)
        return (timestamp - trading_start).seconds // 60
    df["minute_since_open"] = df.index.map(assign_minute_since_open)
    df['trading_day'] = df.index.date
    twap = df.groupby("trading_day")["close"].cumsum() / (df["minute_since_open"] + 1)
    return twap

def _avat(inst_df: pd.DataFrame, wind: int = 30) -> pd.Series:
    df = inst_df.copy()
    def assign_minute_since_open(timestamp):
        trading_start = pd.Timestamp(timestamp.date()).replace(hour=9, minute=30)
        return (timestamp - trading_start).seconds // 60
    df["minute_since_open"] = df.index.map(assign_minute_since_open)
    avat = df.groupby("minute_since_open")["volume"].rolling(wind,min_periods=0).mean()
    return avat.droplevel(0, axis=0).sort_index()

def _vol_at_time(inst_df: pd.DataFrame, wind: int = 30) -> pd.Series:
    df = inst_df.copy()
    def assign_minute_since_open(timestamp):
        trading_start = pd.Timestamp(timestamp.date()).replace(hour=9, minute=30)
        return (timestamp - trading_start).seconds // 60
    df["minute_since_open"] = df.index.map(assign_minute_since_open)
    avolat = df.groupby("minute_since_open")["ret"].rolling(wind).std().fillna(0) * np.sqrt(252 * 390)
    return avolat.droplevel(0, axis=0).sort_index()


class IntradayML(BacktestEngine):
    def __init__(
            self, 
            insts: list[str], 
            dfs: dict[str, DataFrame], 
            start: datetime | None = None, 
            end: datetime | None = None, 
            date_range: DatetimeIndex | None = None, 
            trade_frequency: str | None = None, 
            day_of_week: str | None = None, 
            portf_strategy: PositioningStrategy = VolatilityTargetingStrategy(), 
            portfolio_vol: float = 0.2, 
            max_leverage: float = 2, 
            min_leverage: float = 0, 
            benchmark: str | None = None
        ) -> None:
        super().__init__(insts, dfs, start, end, date_range, trade_frequency, day_of_week, portf_strategy, portfolio_vol, max_leverage, min_leverage, benchmark)
    
    def prep_train_test_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        inst = "SPY"
        inst_df = self.dfs[inst][["close","ret","volume"]]
        train_n = int(len(inst_df) * .7)

        inst_df["time_bucket"] = _create_timebucket(inst_df=inst_df)
        inst_df["vwap"] = _vwap(inst_df=inst_df)
        inst_df["twap"] = _twap(inst_df=inst_df)
        inst_df["avat"] = _avat(inst_df=inst_df)
        inst_df["avolat"] = _vol_at_time(inst_df=inst_df)
        inst_df["dist_vwap"] = inst_df["close"] / inst_df["vwap"] - 1
        inst_df["dist_twap"] = inst_df["close"] / inst_df["twap"] - 1
        inst_df["dist_avat"] = inst_df["volume"] / inst_df["avat"] - 1
        inst_df["vwap_to_twap"] = inst_df["vwap"] / inst_df["twap"] - 1
        inst_df["target"] = _create_target(inst_df=inst_df)
        inst_df["target_lagged"] = inst_df["target"].shift(1)
        inst_df = inst_df[["target","target_lagged","ret","time_bucket","avolat","dist_vwap","dist_twap","dist_avat","vwap_to_twap"]].fillna(0)
        
        train_df, test_df = inst_df.iloc[:train_n], inst_df.iloc[train_n:]
        return train_df, test_df, inst_df
    
    def model_train(self):
        df_train, df_test, self.df = self.prep_train_test_data()
        self.df = self.df.drop(columns=["target"])
        
        self.y_train, self.y_test = df_train["target"].values.reshape(-1,1), df_test["target"].values.reshape(-1,1)
        self.X_train, self.X_test = df_train.drop(columns=["target"]).values, df_test.drop(columns=["target"]).values

        self._d = self.X_train.shape
        self.model = KerasModel(d=self._d[1])
        self.model.fit(x=self.X_train, y=self.y_train, batch_size=32, epochs=10)
        return
    
    def model_predict(self,x):
        return self.model.predict(x=x)

    def pre_compute(self,trade_range):
        return 
    
    def post_compute(self,trade_range):
        self.model_train()
        forecast_df = []

        func = lambda ser: np.where(ser > 0, 1,np.where(ser < 0, -1, 0))
        predictions = pd.DataFrame(
            {"preds":self.model.predict(self.df.values).reshape(-1)},index=trade_range
        )

        neutral = pd.Series(np.zeros(len(trade_range)),index=trade_range)
        
        for inst in self.insts:
            if inst == "SPY": forecast_df.append(predictions["preds"]) #
            else: forecast_df.append(1-neutral.abs())
            

        alphadf = pd.concat(forecast_df,axis=1)
        alphadf.columns = self.insts

        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alphadf))
        masked_df = alphadf/self.eligiblesdf
        masked_df = masked_df.replace([-np.inf, np.inf], np.nan)

        forecast_df = masked_df
        self.forecast_df = forecast_df
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts