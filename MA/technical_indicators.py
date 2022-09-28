import pandas as pd

from MA.config import NORM_ROLLING_WINDOW


def rolling_normalization(df, window):
    normalized_data = (df - df.rolling(window=window, min_periods=1).min()
                       )/(
            df.rolling(window=window, min_periods=1).max() - df.rolling(window=window, min_periods=1).min())
    return normalized_data


# Volatility indicators
def bollinger_bands(close_price_series):
    from ta.volatility import BollingerBands

    df = pd.DataFrame()
    # Initialize Bollinger Bands Indicator
    indicator_bb = BollingerBands(close=close_price_series, window=20, window_dev=2)

    # Add Bollinger Bands features
    bb_bbh = indicator_bb.bollinger_hband()
    bb_bbl = indicator_bb.bollinger_lband()

    # normalizing bollinger bands with rolling window does not make much sense, therefore the proximity to the upper
    # band in percent of the bandwidth is added, as well as the indicators for higher and lower band.
    df['bb_hi_prox'] = (bb_bbh - close_price_series)/(bb_bbh - bb_bbl)

    # Add Bollinger Band high indicator
    df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

    # Add Bollinger Band low indicator
    df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

    return df


# Momentum indicators
def rsi(close_price_series):
    from ta.momentum import RSIIndicator

    # Initialize RSI
    indicator_rsi = RSIIndicator(close=close_price_series, window=9)

    return indicator_rsi.rsi()/100


# Trend indicators
def macd(close_price_series):
    from ta.trend import MACD
    df = pd.DataFrame()
    # initialize MACD
    indicator_macd = MACD(close=close_price_series, window_slow=26, window_fast=12, window_sign=9)

    # add MACD features
    df['macd'] = indicator_macd.macd()
    df['macd_diff'] = indicator_macd.macd_diff()
    df['macd_signal'] = indicator_macd.macd_signal()

    return df


def sma(close_price_series):
    from ta.trend import SMAIndicator
    df = pd.DataFrame()

    df['sma_12h'] = SMAIndicator(close_price_series, window=48).sma_indicator()
    df['sma_24h'] = SMAIndicator(close_price_series, window=96).sma_indicator()
    df['sma_48h'] = SMAIndicator(close_price_series, window=192).sma_indicator()

    df['12h_24h_sma_diff'] = df['sma_12h'] - df['sma_24h']
    df['24h_48h_sma_diff'] = df['sma_24h'] - df['sma_48h']
    df['12h_48h_sma_diff'] = df['sma_12h'] - df['sma_48h']

    df[['sma_12h', 'sma_24h', 'sma_48h']] = rolling_normalization(df[['sma_12h', 'sma_24h', 'sma_48h']],
                                                                  NORM_ROLLING_WINDOW)

    return df


def ema(close_price_series):
    from ta.trend import EMAIndicator
    df = pd.DataFrame()

    df['ema_12h'] = EMAIndicator(close_price_series, window=48).ema_indicator()
    df['ema_24h'] = EMAIndicator(close_price_series, window=96).ema_indicator()
    df['ema_48h'] = EMAIndicator(close_price_series, window=192).ema_indicator()

    df['12h_24h_ema_diff'] = df['ema_12h'] - df['ema_24h']
    df['24h_48h_ema_diff'] = df['ema_24h'] - df['ema_48h']
    df['12h_48h_ema_diff'] = df['ema_12h'] - df['ema_48h']

    df[['ema_12h', 'ema_24h', 'ema_48h']] = rolling_normalization(df[['ema_12h', 'ema_24h', 'ema_48h']],
                                                                  NORM_ROLLING_WINDOW)

    return df


def adx(high_price_series, low_price_series, close_price_series):
    from ta.trend import ADXIndicator
    df = pd.DataFrame()

    indicator_adx = ADXIndicator(high_price_series, low_price_series, close_price_series, window=9)

    df['adx'] = indicator_adx.adx()/100
    df['adx_neg'] = indicator_adx.adx_neg()/100
    df['adx_pos'] = indicator_adx.adx_pos()/100

    return df


# Volume indicator
def obv(close_price_series, vol_series):
    from ta.volume import OnBalanceVolumeIndicator
    indicator_obv = OnBalanceVolumeIndicator(close=close_price_series, volume=vol_series).on_balance_volume()
    indicator_obv = rolling_normalization(indicator_obv, NORM_ROLLING_WINDOW)

    return indicator_obv


def returns(close_price_series):
    df = pd.DataFrame()
    df['return_15min'] = close_price_series.pct_change(periods=1)
    df['return_1h'] = close_price_series.pct_change(periods=4)
    df['return_4h'] = close_price_series.pct_change(periods=16)
    df['return_12h'] = close_price_series.pct_change(periods=48)
    df['return_24h'] = close_price_series.pct_change(periods=96)
    df['return_48h'] = close_price_series.pct_change(periods=192)

    return df


def create_tech_indicators(df):
    bb_df = bollinger_bands(df['ClosePrice'])
    rsi_df = rsi(df['ClosePrice'])
    macd_df = macd(df['ClosePrice'])
    sma_df = sma(df['ClosePrice'])
    ema_df = ema(df['ClosePrice'])
    adx_df = adx(df['HighPrice'], df['LowPrice'], df['ClosePrice'])
    obv_df = obv(df['ClosePrice'], df['TotalVolume'])
    ret_df = returns(df['ClosePrice'])

    # add technical indicators to original dataframe
    df_indicator = pd.concat([df, bb_df, rsi_df, macd_df, sma_df, ema_df, adx_df, obv_df, ret_df], axis=1)
    indicator_list = [indicator for indicator in df_indicator.columns.tolist() if indicator not in df.columns.tolist()]

    return df_indicator, indicator_list
