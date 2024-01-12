import numpy as np
import pandas as pd
from scipy import stats
from ta.volume import VolumeWeightedAveragePrice


def bscore(series, n):
    #bscore:bscore(y) = bollinger score over the last y days; 
    #equals to the number of stdev of closing price from the window mean
    rolling_mean = series.rolling(window=n).mean()
    rolling_std = series.rolling(window=n).std()

    #Calculate the Bollinger Score
    bscore = (series - rolling_mean)/rolling_std
    return bscore

# def expmn_series(series, n):
#     return talib.EMA

def zscore(series):
    series = np.nan_to_num(series, nan=0., posinf=0., neginf=0.)
    return (series - series.mean()) / np.std(series)

def tszscore(series):
    return (series - series.rolling(30).mean()) / series.rolling(30).std()

def obv(df, n):
    grssret_1 = df["close"] / df["close"].shift(1)
    sign = np.sign(grssret_1)
    mult = sign * df['volume']
    return mult.rolling(n).sum()

def logret(closes, n):
    grssret = closes / closes.shift(n)
    return np.log(grssret)

def vwap(high, low, close, volume, n):
    vwap = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume, window=n)
    return vwap.volume_weighted_average_price()

def slope(series):
    x = np.arange(len(series))
    log_series = np.log(np.log(series))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_series)
    annualized_slope = (np.power(np.exp(slope), 250) - 1) * 100
    return annualized_slope * (r_value ** 2)

def ts_argmax(series, window):
    return series.rolling(window).apply(np.argmax).add(1)

def ts_sum(series, window):
    return series.rolling(window).sum()

def ts_rank(series, window):
    return series.rolling(window).apply(lambda x : stats.rankdata(x)[-1])

def ts_max(series, window):
    return series.rolling(window).max()

def ts_min(series, window):
    return series.rolling(window).min()

def csscale(df):
    # Calculate the sum of absolute values for each row
    abs_sum = df.abs().sum(axis=1)
    # Cross-sectionally scale each row
    scaled_df = df.div(abs_sum, axis=0)
    
    return scaled_df

def csrank(series):
    return stats.rankdata(series, method="average", nan_policy="omit")

def kentau(df, series_name_1, series_name_2 ,window):
    # Assuming your dataframe is called 'df' and has a datetime index
    df['kendall_corr'] = pd.Series(dtype=float)
    

    for i in range(len(df) - window + 1):
        window_data = df.iloc[i:i+window]
        corr, _ = stats.kendalltau(window_data[series_name_1], window_data[series_name_2])
        index = df.index[i+window-1]
        df.loc[index, 'kendall_corr'] = corr
    return df['kendall_corr']

def ite(x,y,z): #if x then y else z.
    # Note: x must be time series of boolean data type entries.
    return x.fillna(0).astype(int)*y + (~x.astype(bool)).fillna(0).astype(int)*z

def sign(x):
    return np.sign(x)

def sum(x):
    return np.sum(x) 

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w