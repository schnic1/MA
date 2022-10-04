import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime

from MA.config import evaluation, EVAL_PATH, evaluation_period, env_kwargs, bm_pos, PLOT_PATH


def perf_evaluation(df):
    eval_dict = {}
    starting_value = 100000

    df['total PnL'] = float(df['PF value'][-1:]) - starting_value

    df['cum_ret'] = (df['returns'] + 1).cumprod()
    eval_dict['expected_return'] = np.mean(df['returns'])
    eval_dict['cum_return'] = float(df['cum_ret'][-1:]) - 1

    eval_dict['pf_std'] = np.std(df['returns'])

    # calculate downside risk
    eval_dict['pf_std_down'] = np.std(df['returns'][df['returns'] < 0])

    eval_dict['sharpe ratio'] = eval_dict['expected_return'] / eval_dict['pf_std']
    eval_dict['sortino_ratio'] = eval_dict['expected_return'] / eval_dict['pf_std_down']

    maximum = df['PF value'].rolling(min_periods=1, window=df.shape[0]).max()
    df['interval_drawdown'] = df['PF value']/maximum - 1
    df['max_dd'] = df['interval_drawdown'].rolling(min_periods=1, window=df.shape[0]).min()

    eval_dict['calmar_ratio'] = eval_dict['expected_return'] / abs(df['max_dd'].min())

    return [eval_dict, df]


def transform_set(df):

    es_set = df.loc[df['ticker'] == 'ES'].reset_index()
    es_set['date'] = es_set['date'].apply(pd.to_datetime)
    zn_set = df.loc[df['ticker'] == 'ZN'].reset_index()
    zn_set['date'] = zn_set['date'].apply(pd.to_datetime)

    filled_close_es = es_set['closeprice'].replace(to_replace=0, method='ffill')
    filled_close_zn = zn_set['closeprice'].replace(to_replace=0, method='ffill')

    data = {'date': es_set['date'], 'full_close_es': filled_close_es, 'full_close_zn': filled_close_zn}
    new_df = pd.DataFrame(data=data)

    new_df['diff_es'] = new_df['full_close_es'].diff().fillna(0)
    new_df['diff_zn'] = new_df['full_close_zn'].diff().fillna(0)

    new_df['PnL'] = (new_df['diff_es'] * env_kwargs['contract_size'][0] * bm_pos[0]
                     + new_df['diff_zn'] * env_kwargs['contract_size'][1] * bm_pos[1])

    return new_df


def same_period(bm_set, perf_df):

    start_date = perf_df['date'].min()
    end_date = perf_df['date'].max()

    df = bm_set[(bm_set['date'] <= end_date) & (bm_set['date'] >= start_date)]
    df = df.reset_index(drop=True)

    pf_values = [env_kwargs['initial_amount']]
    df = df.replace(df['PnL'][0],  0)

    for ind in range(1, len(df['PnL'])):
        new_pf_value = pf_values[-1] + df['PnL'][ind]
        pf_values.append(new_pf_value)

    df.insert(6, 'PF value', pf_values)

    returns = df['PF value'].pct_change().fillna(0)
    df.insert(7, 'returns', returns)

    return df

def visualize_evals(bm, pred):

    df_bm = pd.DataFrame.from_dict(bm[0], orient='index', columns=['Benchmark'])
    df_pred = pd.DataFrame.from_dict(pred[0], orient='index', columns=['Agent'])

    df = pd.concat([df_bm, df_pred], axis=1).transpose()
    df.to_csv(PLOT_PATH + 'results')

    # cumulative return plot
    from MA.plots import cumulative_return
    cumulative_return(bm, pred, PLOT_PATH)


"""    # Maximum Drawdown
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    plt.plot()
    bm[1]['interval_drawdown'].plot()
    bm[1]['max_dd'].plot()
    plt.show()

    pred[1]['interval_drawdown'].plot()
    pred[1]['max_dd'].plot()
    plt.show()"""



def run_eval(bm_set):
    prediction_set = pd.read_csv(EVAL_PATH + evaluation_period)
    prediction_set = prediction_set.iloc[:, 1:]
    print(f'loaded, {evaluation_period}')

    feature_cols = ['date', 'PF value', 'PnL', 'returns']

    prediction_set = prediction_set[feature_cols]
    # drop the first and the last rows. in the first nothing happens,
    # in the last default happens and cannot be treated as a valid action
    prediction_set = prediction_set.iloc[:-1, :].fillna(0)

    bm_set = transform_set(bm_set)
    bm_set = same_period(bm_set, prediction_set)
    bm_set = bm_set[feature_cols]

    bm = perf_evaluation(bm_set)
    pred = perf_evaluation(prediction_set)

    visualize_evals(bm, pred)





