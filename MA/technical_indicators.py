import pandas as pd

from MA.config import NORM_ROLLING_WINDOW


def rolling_normalization(df, window):
    normalized_data = (df - df.rolling(window=window, min_periods=1).min()
                       )/(
            df.rolling(window=window, min_periods=1).max() - df.rolling(window=window, min_periods=1).min())
    return normalized_data


# TODO: add more indicators
# Volatility indicators
def bollinger_bands(close_price_series):
    from ta.volatility import BollingerBands

    df = pd.DataFrame()
    # Initialize Bollinger Bands Indicator
    indicator_bb = BollingerBands(close=close_price_series, window=20, window_dev=2)

    # Add Bollinger Bands features
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()

    # Add Bollinger Band high indicator
    df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

    # Add Bollinger Band low indicator
    df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

    df = rolling_normalization(df, NORM_ROLLING_WINDOW)
    return df


# Momentum indicators
def rsi(close_price_series):
    from ta.momentum import RSIIndicator

    # Initialize RSI
    indicator_rsi = RSIIndicator(close=close_price_series, window=14)

    return indicator_rsi.rsi()


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

    df['sma50'] = SMAIndicator(close_price_series, window=50).sma_indicator()
    df['sma200'] = SMAIndicator(close_price_series, window=200).sma_indicator()

    df = rolling_normalization(df, NORM_ROLLING_WINDOW)

    return df


def ema(close_price_series):
    from ta.trend import EMAIndicator

    indicator_ema = EMAIndicator(close_price_series, window=14).ema_indicator()
    indicator_ema = rolling_normalization(indicator_ema, NORM_ROLLING_WINDOW)

    return indicator_ema


# TODO: Look at ADX again!
def adx(high_price_series, low_price_series, close_price_series):
    from ta.trend import ADXIndicator
    df = pd.DataFrame()

    indicator_adx = ADXIndicator(high_price_series, low_price_series, close_price_series, window=14)

    df['adx'] = indicator_adx.adx()
    df['adx_neg'] = indicator_adx.adx_neg()
    df['adx_pos'] = indicator_adx.adx_pos()

    return df


# Volume indicator
def obv(close_price_series, vol_series):
    from ta.volume import OnBalanceVolumeIndicator
    indicator_obv = OnBalanceVolumeIndicator(close=close_price_series, volume=vol_series).on_balance_volume()
    indicator_obv = rolling_normalization(indicator_obv, NORM_ROLLING_WINDOW)

    return indicator_obv


def create_tech_indicators(df):
    bb_df = bollinger_bands(df['ClosePrice'])
    rsi_df = rsi(df['ClosePrice'])
    macd_df = macd(df['ClosePrice'])
    sma_df = sma(df['ClosePrice'])
    ema_df = ema(df['ClosePrice'])
    adx_df = adx(df['HighPrice'], df['LowPrice'], df['ClosePrice'])
    obv_df = obv(df['ClosePrice'], df['TotalVolume'])

    # add technical indicators to original dataframe
    df_indicator = pd.concat([df, bb_df, rsi_df, macd_df, sma_df, ema_df, adx_df, obv_df], axis=1)
    indicator_list = [indicator for indicator in df_indicator.columns.tolist() if indicator not in df.columns.tolist()]

    return df_indicator, indicator_list
