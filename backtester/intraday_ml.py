from datetime import datetime
from backtester import BacktestEngine
from pandas.core.api import DataFrame as DataFrame, DatetimeIndex
from backtester.engine.functions.portfolio_strategies import PositioningStrategy, VolatilityTargetingStrategy
from sklearn.mixture import GaussianMixture

import numpy as np
import pandas as pd
import keras as keras


class KerasModel(keras.Model):
    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_train, self.df_test = df_train, df_test
        self.__init_data__()
        self.__init_model__()
        self.compile(
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=keras.metrics.BinaryAccuracy()
        )
        
    def __init_model__(self):
        self.input_layer = keras.layers.InputLayer(shape=self._d[1])
        self.dense_layer1 = keras.layers.Dense(shape=512,activation=keras.activations.relu)
        self.dense_layer2 = keras.layers.Dense(shape=64,activation=keras.activations.relu)
        self.output_layer = keras.layers.Dense(shape=1,activation=keras.activations.sigmoid)

    def __init_data__(self) -> None:
        self._d = self.df_train.shape
        gmm = GaussianMixture(n_components=2).fit(self.df_train["rets"].values)
        self.y_train, self.y_test = gmm.predict(self.df_train["rets"].values), gmm.predict(self.df_test["retse"].values)
        self.X_train, self.X_test = self.df_train.values, self.df_test.values
        return
    
    def call(self):
        z1 = self.dense_layer1(self.X_train)
        z2 = self.dense_layer2(z1)
        return self.output_layer(z2)



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
            portf_strategy: PositioningStrategy = PositioningStrategy(), 
            portfolio_vol: float = 0.2, 
            max_leverage: float = 2, 
            min_leverage: float = 0, 
            benchmark: str | None = None
        ) -> None:
        super().__init__(insts, dfs, start, end, date_range, trade_frequency, day_of_week, portf_strategy, portfolio_vol, max_leverage, min_leverage, benchmark)
    
    def prep_train_test_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return
    
    def model_train(self):
        df_train, df_test = self.prep_train_test_data()
        self.model = KerasModel(df_train=df_train, df_test=df_test)
        self.model.fit(x=self.model.X_train, y=self.model.y_train, batch_size=600, epocs=10)
        return
    
    def model_predict(self,x):
        return self.model.predict(x=x)

    def pre_compute(self,trade_range):
        return 
    
    def post_compute(self,trade_range):
        self.model_train()
        forecast_df = []
    
        for inst in self.insts:
            inst_df = self.dfs[inst]
            inst_df["vals"] = np.ones(len(inst_df))
            forecast_df.append(inst_df["vals"])

        alphadf = pd.concat(forecast_df,axis=1)
        alphadf.columns = self.insts

        self.eligiblesdf = self.eligiblesdf & (~pd.isna(alphadf))
        masked_df = alphadf/self.eligiblesdf
        masked_df = masked_df.replace([-np.inf, np.inf], np.nan)

        rankdf = masked_df.rank(axis=1,method="average",na_option="keep",ascending=False)

        forecast_df = masked_df
        self.forecast_df = forecast_df
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts