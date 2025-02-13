from datetime import datetime
from pandas import DataFrame

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def calculate_indxs(dfs: dict[str, DataFrame], insts: list[str], cat: str = "sector") -> tuple[list[str], dict[str, pd.Series]]:
    n_calc = {}
    group_indxs = {}
    for inst in insts:
        inst_df = dfs[inst]
        size = inst_df.shape[0]
        group = inst_df[cat].dropna().unique()
        if len(group) > 0:
            group = group[0]
            if group not in group_indxs.keys():
                n_calc[group] = 0
                group_indxs[group] = pd.Series(np.zeros(size), index=inst_df.index)
            n_calc[group] += 1
            group_indxs[group] += inst_df["ret"].fillna(0)
    groups = list(group_indxs.keys())
    for group in groups:
        ser, n = group_indxs[group], n_calc[group]
        group_indxs[group] = ser * 1 / n
    return groups, group_indxs

def x_rank(forecast_df: DataFrame, eligibles_df: DataFrame, num_insts: int = 10, ascending: bool = True) -> DataFrame:
    num_eligibles = eligibles_df.sum(axis=1)
    rankdf = forecast_df.rank(axis=1,na_option="keep",ascending=ascending)
    numb = num_eligibles - num_insts
    longdf = rankdf.apply(lambda col: col > numb, axis=0, raw=True)
    forecast_df = longdf.astype(np.int32)
    return forecast_df

def zscore(ser: pd.Series, wind: int = 50, expected_value: pd.Series | None = None) -> pd.Series:
    frame_cleaner = lambda ser: np.nan_to_num(ser, nan=0, posinf=0, neginf=0)
    if expected_value is None:
        expected_value = ser.rolling(wind).mean()
    zscore = ((ser - expected_value) / ser.rolling(wind).std()).apply(frame_cleaner)
    return zscore

def momentum_distance(ser: pd.Series, ewm_val: None | float = .5, wind_val: None | int = None) -> pd.Series:
    if ewm_val is not None: return ser / ser.ewm(ewm_val).mean() - 1
    if wind_val is not None: return ser / ser.rolling(window=wind_val).mean() - 1


"""
def calc_rets_df(insts: list[str], dfs: dict[str, pd.DataFrame] ) -> pd.DataFrame:
    frame_cleaner = lambda ser: np.nan_to_num(ser, nan=0, neginf=0, posinf=0)
    lags = 1
    rets = []
    for inst in insts:
        close_price = dfs[inst]["close"]
        log_close = close_price.apply(np.log)
        log_ret = log_close - log_close.shift(lags)
        rets.append(log_ret)
    rets_df = pd.concat(rets, axis=1).apply(frame_cleaner)
    rets_df.columns = insts
    return rets_df

def calc_pca_loadings(rets_df: pd.DataFrame, train_n: int, factors_n: int) -> pd.DataFrame:
    factors_n = factors_n if factors_n <= 5 else 5

    rets = rets_df.values
    R = rets[:train_n, :]
    # X = (rets - rets.mean())
    # covar = np.dot(X.T, X)
    # eigen_values, eigen_vectors = np.linalg.eig(covar)
    # ind = eigen_values.argsort()
    # eigen_vectors = eigen_vectors[:,ind].astype(np.float128)
    # eigen_values = eigen_values[ind].astype(np.float128)

    S, sigmas, D = np.linalg.svd(R.T)
    S = S[:factors_n,:]
    print((S.shape, R.shape))
    return pd.DataFrame((R @ S.T), index=rets_df.index, columns=["factor_" + str(i) for i in range(factors_n)])
    # factor_rets = []
    # for i in range(factors_n):
    #     factor_rets.append(
    #         np.dot(eigen_vectors[:,-(i+1)],rets_df.values.T) / np.sqrt(eigen_values[-(i+1)])
    #     )
    # factor_rets_df = pd.DataFrame(factor_rets, columns=rets_df.index, index=["factor_" + str(i) for i in range(factors_n)]).T
    # return factor_rets_df

def _assign_intraday_timestamps(timestamp: datetime) -> float:
    minutes = 60
    trading_start = pd.Timestamp(timestamp.date()).replace(hour=9, minute=0)
    trading_end = pd.Timestamp(timestamp.date()).replace(hour=22, minute=00)
    if timestamp < trading_start or timestamp > trading_end:
        return None
    minutes_since_open = (timestamp - trading_start).seconds // 60
    return minutes_since_open / minutes



def _create_target(inst_df: pd.DataFrame, train_n: int) -> pd.Series:
    df = inst_df.copy()
    rets = df["close"].shift(-2).apply(np.log) - df["close"].apply(np.log)
    zscore = ((rets - rets.rolling(50).mean()) / rets.rolling(50).std()).fillna(0.)
    return zscore.apply(np.tanh)

def _vwap(inst_df: pd.DataFrame, volume_col: str) -> pd.Series:
    df = inst_df.copy()
    df['trading_day'] = df.index.date
    df["cumulative_volume"] = df.groupby("trading_day")[volume_col].cumsum()
    df["price_volume"] = df["close"] * df[volume_col]
    vwap = df.groupby("trading_day")["price_volume"].cumsum() / df["cumulative_volume"]
    return vwap

def _twap(inst_df: pd.DataFrame, minutes: int = 1) -> pd.Series:
    df = inst_df.copy()
    df["minute_since_open"] = df.index.map(_assign_intraday_timestamps)
    df['trading_day'] = df.index.date
    twap = df.groupby("trading_day")["close"].cumsum() / (df["minute_since_open"] + minutes)
    return twap

def _cumret(inst_df: pd.DataFrame) -> pd.Series:
    df = inst_df.copy()
    df['trading_day'] = df.index.date
    df["ret"] = df["close"].apply(np.log) - df["close"].shift(1).apply(np.log)
    ret = df.groupby("trading_day")["ret"].cumsum()
    return ret

def _avat(inst_df: pd.DataFrame, wind: int = 30) -> pd.Series:
    df = inst_df.copy()
    df["minute_since_open"] = df.index.map(_assign_intraday_timestamps)
    avat = df.groupby("minute_since_open")["volume"].rolling(wind,min_periods=0).mean()
    return avat.droplevel(0, axis=0).sort_index()

def _vol_at_time(inst_df: pd.DataFrame, wind: int = 30, minutes: int = 1) -> pd.Series:
    df = inst_df.copy()
    df["ret"] = df["close"].apply(np.log) - df["close"].shift(1).apply(np.log)
    df["minute_since_open"] = df.index.map(_assign_intraday_timestamps)
    avolat = df.groupby("minute_since_open")["ret"].rolling(wind).std().fillna(0) #* np.sqrt(252 * 870 / minutes)
    return avolat.droplevel(0, axis=0).sort_index()

def _ave_size(inst_df: pd.DataFrame) -> pd.Series:
    df = inst_df.copy()
    return df["volume"] / df["number_of_trades"]

def _buy_sell_volume_balance(inst_df: pd.DataFrame) -> pd.Series:
    df = inst_df.copy()
    return df["taker_buy_base_asset_volume"] / df["volume"] - .5





def calc_weight(rets_df: pd.DataFrame, tau: float = .9) -> np.ndarray:
    T, N = rets_df.shape
    T_range = (np.arange(T) + 1)[::-1]
    W = np.diag(np.exp(-T_range/tau))
    trace = np.trace(W)
    kappa = (T + 1) / trace
    return kappa*W

def returns_reweighted(rets_df: pd.DataFrame, tau: float = .9) -> np.ndarray:
    W = calc_weight(rets_df, tau)
    R = rets_df.values
    return W @ R

def returns_eigen_factors(rets_df: pd.DataFrame, tau: float = .9, p_factors: int = 4) -> pd.DataFrame:
    WR = returns_reweighted(rets_df, tau)
    B, _, _ = np.linalg.svd(WR, full_matrices=False)
    B = B[:,:p_factors] #, np.diag(lambdas)[:p_factors], D[:,:p_factors]
    return pd.DataFrame(B, index=rets_df.index, columns=["factor" + str(i) for i in range(p_factors)])

"""