# Bollinger Bands

def Bollinger_Bands(df, close_col='ClosePrice'):
    from ta.volatility import BollingerBands
    # Initialize Bollinger Bands Indicator
    indicator_bb = BollingerBands(close=df[close_col], window=20, window_dev=2)

    # Add Bollinger Bands features
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()

    # Add Bollinger Band high indicator
    df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

    # Add Bollinger Band low indicator
    df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

    return df

def RSI(df, close_col='ClosePrice'):
    from ta.momentum import RSIIndicator

    # Initialize RSI
    indicator_rsi = RSIIndicator(close=df[close_col], window=14)
    df['rsi'] = indicator_rsi.rsi()

    return df

def MACD(df, close_col='ClosePrice'):
    from ta.trend import MACD

    # initialize MACD
    indicator_macd = MACD(close=df[close_col], window_slow=26, window_fast=12, window_sign=9)

    # add MACD features
    df['macd'] = indicator_macd.macd()
    df['macd_diff'] = indicator_macd.macd_diff()
    df['macd_signal'] = indicator_macd.macd_signal()

    return df

def OnBalanceVolume(df, close_col='ClosePrice', vol_col='TotalVolume'):
    from ta.volume import OnBalanceVolumeIndicator
    indicator_obv = OnBalanceVolumeIndicator(close=df[close_col], volume=df[vol_col])
    df['obv'] = indicator_obv.on_balance_volume()
    return df

def create_tech_indicators(df):
    df = Bollinger_Bands(df)
    df = RSI(df)
    df = MACD(df)
    df = OnBalanceVolume(df)

    return df