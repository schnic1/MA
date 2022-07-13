import tensortrade.env.default as default
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import USD, instrument
from tensortrade.oms.services.execution.simulated import execute_order


# Exchange
def build_exchange(df, exchange_name):
    exchange = Exchange(exchange_name, service=execute_order)(
        Stream.source(list(df['ES:closeprice']), dtype='float').rename('USD-ES'),
        Stream.source(list(df['ZN:closeprice']), dtype='float').rename('USD-ZN')
    )
    return exchange


# DataFeed
def build_datafeed(df, exchange_name):
    ES_data = df.loc[:, [name.startswith('ES') for name in df.columns]]
    ZN_data = df.loc[:, [name.startswith('ZN') for name in df.columns]]

    with NameSpace(exchange_name):
        data_stream = [Stream.source(list(ES_data[c]), dtype='float').rename(c) for c in ES_data.columns]
        data_stream += [Stream.source(list(ZN_data[c]), dtype='float').rename(c) for c in ZN_data.columns]

        return DataFeed(data_stream)


def build_portfolio(df, exchange_name):
    exchange = build_exchange(df, exchange_name)
    ES = instrument.Instrument('ES', 8)
    ZN = instrument.Instrument('ZN', 8)
    return Portfolio(USD, [
        Wallet(exchange, 1000*USD),
        Wallet(exchange, 123*ES),
        Wallet(exchange, 0*ZN)
    ])


def build_environment(df, exchange_name):
    feed = build_datafeed(df, exchange_name)
    portfolio = build_portfolio(df, exchange_name)
    env = default.create(
        portfolio=portfolio,
        action_scheme='managed-risk',
        reward_scheme='simple',
        feed=feed
    )
    return env


from tensortrade.agents import A2CAgent

def build_agent(env):
    return A2CAgent(env)


