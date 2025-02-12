from datetime import datetime

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from backtester import BacktestEngine
from backtester.engine.functions import quant_tools as qt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain
from pandas.core.api import DataFrame as DataFrame, DatetimeIndex
from backtester.engine.functions.portfolio_optimization import PositioningMethod, VanillaVolatilityTargeting

import numpy as np
import pandas as pd
import keras as keras 
import matplotlib.pyplot as plt



class KerasModel(keras.Model):
    SEED = keras.utils.set_random_seed(11)

    def __init__(self, output_shape: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_shape = output_shape
        self.__init_model__()
        self.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=0.001)
        )
        
    def __init_model__(self):
        self.dense = keras.layers.Dense(units=111, activation=keras.activations.leaky_relu)
        self.output_layer = keras.layers.Dense(units=self.output_shape, activation=keras.activations.linear)
    
    def call(self, inputs):
        z1 = self.dense(inputs)
        return self.output_layer(z1)
    

class MultiRidgeModel(MultiOutputRegressor):
    SEED = keras.utils.set_random_seed(11)
    def __init__(self) -> None:
        estimator = Ridge(alpha=1.1, solver="auto",random_state=self.SEED)
        super().__init__(estimator, n_jobs=2)

class RidgeChain(RegressorChain):
    SEED = keras.utils.set_random_seed(11)
    def __init__(self) -> None:
        estimator = Ridge(alpha=1.1, solver="auto",random_state=self.SEED)
        super().__init__(estimator, order="random")
    


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
            use_portfolio_opt: bool = True,
            portf_optimization: PositioningMethod = VanillaVolatilityTargeting(), 
            portfolio_vol: float = 0.2, 
            max_leverage: float = 2, 
            min_leverage: float = 0, 
            benchmark: str | None = None,
            train_size: float = 1.,
            costs: dict[str, float] = {"slippage": 0.1/100}
        ) -> None:
        super().__init__(
            insts, dfs, start, end, date_range, trade_frequency, day_of_week, use_portfolio_opt, portf_optimization, 
            portfolio_vol, max_leverage, min_leverage, benchmark, train_size, costs
        )
    

    def get_factor_states(self) -> dict[int, pd.DatetimeIndex]:
        rets_df = qt.calc_rets_df(insts=self.insts, dfs=self.dfs)

        train_n=int(len(rets_df * .7))
        factor_rets = qt.calc_pca_factors(rets_df=rets_df,train_n=train_n,factors_n=3)

        m = GaussianMixture(n_components=2).fit(factor_rets.values[:train_n,:])
        state = np.argsort(m.means_)[-1]
        pred_states = pd.Series(m.predict(factor_rets.values),index=factor_rets.index).apply(
            lambda ser: np.where(ser == state,1,0)
        )
        return pred_states
        # return {
        #     0: pred_states[pred_states==0].index,
        #     1: pred_states[pred_states==1].index
        # }
    
    def model_train(self):
        (X_train, y_train), (X_test, y_test), (X, y) = self.prep_train_test_data()
        self.X_train, self.X_test, self.X = X_train.values, X_test.values, X.values
        self.y_train, self.y_test, self.y = y_train.values, y_test.values, y.values
    
        model = KerasModel(output_shape=y.shape[1])
        model.fit(x=self.X_train,y=self.y_train, batch_size=20, epochs=20, validation_split=0.2)
        return model
    
    def pre_compute(self,trade_range):
        return 
    
    def post_compute(self,trade_range):
        # model = self.model_train()
        momentums = []
        for inst in self.insts:
            inst_df = self.dfs[inst]
            dist = inst_df["close"] / inst_df["close"].ewm(0.8).mean()
            dist_norm = dist / dist.ewm(0.8).std()
            momentums.append(dist_norm)
        momentum_df = pd.concat(momentums)
        momentum_df.columns = self.insts

        predictions = self.get_factor_states()# model.predict(self.X)

        # predictions = pd.DataFrame(pred_vals,index=trade_range,columns=self.insts)
        # predictions = predictions.apply(lambda x: np.maximum(x, 0)) # long only
        
        self.eligiblesdf = self.eligiblesdf & (~pd.isna(predictions)) & (predictions > 0)
        masked_df = predictions/self.eligiblesdf
        masked_df = masked_df.replace([-np.inf, np.inf], np.nan)
        momentum_df = masked_df.rank(axis=1,method="average",na_option="keep",ascending=True,pct=True)

        forecast_df = masked_df * masked_df
        self.forecast_df = forecast_df
        return 

    def compute_signal_distribution(self, eligibles, date):
        forecasts = self.forecast_df.loc[date].values
        return forecasts
    
#   def prep_train_test_data(self) -> tuple[
#         tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]
#     ]:

#     states_dict = self.get_factor_states()
#     Xs, ys = [], []
#     for inst in self.insts:

#         inst_df = self.dfs[inst][["close","volume","number_of_trades","taker_buy_base_asset_volume"]]
#         train_n = int(len(inst_df) * .7)

#         inst_df["vwap"] = qt._vwap(inst_df=inst_df,volume_col="volume")
#         inst_df["buy_vwap"] = qt._vwap(inst_df=inst_df,volume_col="taker_buy_base_asset_volume")
#         inst_df["twap"] = qt._twap(inst_df=inst_df,minutes=60)
#         inst_df["avat"] = qt._avat(inst_df=inst_df)
#         inst_df["avolat"] = qt._zscore_scaler(qt._vol_at_time(inst_df=inst_df,minutes=60))
#         inst_df["dist_vwap"] = qt._zscore_scaler(inst_df["close"] / inst_df["vwap"] - 1)
#         inst_df["dist_twap"] = qt._zscore_scaler(inst_df["close"] / inst_df["twap"] - 1)
#         inst_df["dist_avat"] = qt._zscore_scaler(inst_df["volume"] / inst_df["avat"] - 1)
#         inst_df["trade_size_ave"] = qt._zscore_scaler(qt._ave_size(inst_df=inst_df))
#         inst_df["bs_volume_bal"] = qt._zscore_scaler(qt._buy_sell_volume_balance(inst_df=inst_df))
#         inst_df["vwap_to_buy_vwap"] = qt._zscore_scaler(inst_df["buy_vwap"] / inst_df["vwap"] - 1)
#         inst_df["cumret"] = qt._cumret(inst_df=inst_df)
    
#         y = qt._create_target(inst_df=inst_df, train_n=train_n)

#         inst_df = inst_df[["dist_twap","avolat","dist_vwap","dist_avat","trade_size_ave","bs_volume_bal","vwap_to_buy_vwap","cumret"]].fillna(0)
#         pca0 = PCA(n_components=3).fit(inst_df.loc[states_dict[0]].values)
#         pca1 = PCA(n_components=3).fit(inst_df.loc[states_dict[1]].values)
#         Xs.append(pd.concat([
#                 pd.DataFrame(pca0.transform(inst_df.values),index=inst_df.index,columns=[inst+"_0_"+str(i) for i in range(3)]),
#                 pd.DataFrame(pca1.transform(inst_df.values),index=inst_df.index,columns=[inst+"_1_"+str(i) for i in range(3)])
#             ],axis=1
#         ))
#         ys.append(y)

#     X = pd.concat(Xs,axis=1)
#     y = pd.concat(ys,axis=1)
#     y.columns = self.insts

#     train_n = int(len(X) * .7)
#     X_train, X_test = X.iloc[:train_n], X.iloc[train_n:]
#     y_train, y_test = y.iloc[:train_n], y.iloc[train_n:]
#     return (X_train, y_train), (X_test, y_test), (X, y)