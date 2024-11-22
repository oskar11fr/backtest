from datetime import datetime
from backtester import BacktestEngine
from pandas.core.api import DataFrame as DataFrame, DatetimeIndex
from backtester.engine.functions.portfolio_optimization import PositioningMethod, VanillaVolatilityTargeting

import numpy as np
import pandas as pd
import keras as keras 


class KerasModel(keras.Model):
    SEED = keras.utils.set_random_seed(11)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__init_model__()
        self.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=0.001)
        )
        
    def __init_model__(self):
        self.dense = keras.layers.Dense(units=50, activation=keras.activations.relu)
        self.output_layer = keras.layers.Dense(units=1, activation=keras.activations.tanh)
    
    def call(self, inputs):
        z1 = self.dense(inputs)
        return self.output_layer(z1)

def _assign_intraday_timestamps(timestamp: datetime) -> float:
    minutes = 1
    trading_start = pd.Timestamp(timestamp.date()).replace(hour=9, minute=30)
    trading_end = pd.Timestamp(timestamp.date()).replace(hour=16, minute=0)
    if timestamp < trading_start or timestamp > trading_end:
        return None
    minutes_since_open = (timestamp - trading_start).seconds // 60
    return minutes_since_open / minutes

def _zscore_scaler(ser: pd.Series, wind: int = 100) -> pd.Series:
    frame_cleaner = lambda ser: np.nan_to_num(ser, nan=0, posinf=0, neginf=0)
    return ((ser - ser.rolling(wind).mean()) / ser.rolling(wind).std()).apply(frame_cleaner)

def _create_target(inst_df: pd.DataFrame, wind=30) -> pd.Series:
    df = inst_df.copy()
    rets = df["close"].shift(-15).apply(np.log) - df["close"].apply(np.log)
    zscore = rets / rets.rolling(100).std()
    return zscore.fillna(0).apply(np.tanh)

def _create_timebucket(inst_df: pd.DataFrame, time_constant: int = 1) -> pd.Series:
    df = inst_df.copy()
    return df.index.map(_assign_intraday_timestamps) / time_constant

def _vwap(inst_df: pd.DataFrame) -> pd.Series:
    df = inst_df.copy()
    df['trading_day'] = df.index.date
    df["cumulative_volume"] = df.groupby("trading_day")['volume'].cumsum()
    df["price_volume"] = df["close"] * df["volume"]
    vwap = df.groupby("trading_day")["price_volume"].cumsum() / df["cumulative_volume"]
    return vwap

def _twap(inst_df: pd.DataFrame, minutes: int = 1) -> pd.Series:
    df = inst_df.copy()
    df["minute_since_open"] = df.index.map(_assign_intraday_timestamps)
    df['trading_day'] = df.index.date
    twap = df.groupby("trading_day")["close"].cumsum() / (df["minute_since_open"] + minutes)
    return twap

def _avat(inst_df: pd.DataFrame, wind: int = 30) -> pd.Series:
    df = inst_df.copy()
    df["minute_since_open"] = df.index.map(_assign_intraday_timestamps)
    avat = df.groupby("minute_since_open")["volume"].rolling(wind,min_periods=0).mean()
    return avat.droplevel(0, axis=0).sort_index()

def _vol_at_time(inst_df: pd.DataFrame, wind: int = 30, minutes: int = 1) -> pd.Series:
    df = inst_df.copy()
    df["minute_since_open"] = df.index.map(_assign_intraday_timestamps)
    avolat = df.groupby("minute_since_open")["ret"].rolling(wind).std().fillna(0) * np.sqrt(252 * 390 / minutes)
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
            portf_optimization: PositioningMethod = VanillaVolatilityTargeting(), 
            portfolio_vol: float = 0.2, 
            max_leverage: float = 2, 
            min_leverage: float = 0, 
            benchmark: str | None = None
        ) -> None:
        super().__init__(insts, dfs, start, end, date_range, trade_frequency, day_of_week, portf_optimization, portfolio_vol, max_leverage, min_leverage, benchmark)
    
    def prep_train_test_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        inst = "SPY"
        inst_df = self.dfs[inst][["close","ret","volume"]]
        train_n = int(len(inst_df) * .7)

        inst_df["time_bucket"] = _create_timebucket(inst_df=inst_df, time_constant=390)
        inst_df["vwap"] = _vwap(inst_df=inst_df)
        inst_df["twap"] = _twap(inst_df=inst_df,minutes=1)
        inst_df["avat"] = _avat(inst_df=inst_df)
        inst_df["avolat"] = _vol_at_time(inst_df=inst_df,minutes=1)
        inst_df["dist_vwap"] = _zscore_scaler(inst_df["close"] / inst_df["vwap"] - 1)
        inst_df["dist_twap"] = _zscore_scaler(inst_df["close"] / inst_df["twap"] - 1)
        inst_df["dist_avat"] = _zscore_scaler(inst_df["volume"] / inst_df["avat"] - 1)
        inst_df["vwap_to_twap"] = _zscore_scaler(inst_df["vwap"] / inst_df["twap"] - 1)
        inst_df["target"] = _create_target(inst_df=inst_df)

        inst_df = inst_df[["target","ret","time_bucket","avolat","dist_vwap","dist_twap","dist_avat","vwap_to_twap"]].fillna(0)
        train_df, test_df = inst_df.iloc[:train_n], inst_df.iloc[train_n:]
        return train_df, test_df, inst_df
    

    def model_train(self):
        df_train, df_test, self.df = self.prep_train_test_data()

        y_train, y_test = df_train["target"].values, df_test["target"].values    
        self.y_train, self.y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)

        self.X_train, self.X_test = df_train.drop(columns=["target"]).values, df_test.drop(columns=["target"]).values
        self.df = self.df.drop(columns=["target"]).values

        self.model = KerasModel()
        self.model.fit(x=self.X_train, y=self.y_train, batch_size=21, epochs=10, validation_split=0.2)
        return
    
    def model_predict(self,x):
        return self.model.predict(x=x)

    def pre_compute(self,trade_range):
        return 
    
    def post_compute(self,trade_range):
        self.model_train()
        forecast_df = []

        predictions = pd.DataFrame(
            {"preds":self.model.predict(self.df).reshape(-1)},index=trade_range
        )
        neutral = pd.Series(np.zeros(len(trade_range)),index=trade_range)
        
        for inst in self.insts:
            if inst == "SPY": forecast_df.append(predictions["preds"])
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