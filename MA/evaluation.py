import pandas as pd
import numpy as np

from MA.config import EVAL_PATH, evaluation_period, env_kwargs, bm_pos, PLOT_PATH


def perf_evaluation(df):
    eval_dict = {}
    starting_value = 100000

    df['total PnL'] = float(df['PF value'][-1:]) - starting_value

    df['cum_ret'] = (df['returns'] + 1).cumprod()
    eval_dict['Mean Ret.'] = np.mean(df['returns'])
    eval_dict['tot. cum. Ret.'] = float(df['cum_ret'][-1:]) - 1

    eval_dict['std(R)'] = np.std(df['returns'])

    # calculate downside risk
    eval_dict['down std(R)'] = np.std(df['returns'][df['returns'] < 0])

    eval_dict['Sharpe'] = eval_dict['Mean Ret.'] / eval_dict['std(R)']
    eval_dict['Sortino'] = eval_dict['Mean Ret.'] / eval_dict['down std(R)']

    maximum = df['PF value'].rolling(min_periods=1, window=df.shape[0]).max()
    df['interval_drawdown'] = df['PF value']/maximum - 1
    df['max_dd'] = df['interval_drawdown'].rolling(min_periods=1, window=df.shape[0]).min()

    eval_dict['Calmar'] = eval_dict['Mean Ret.'] / abs(df['max_dd'].min())
    eval_dict['MDD'] = df['max_dd'].min()

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

    # price plots
    from MA.plots import price_plots
    price_plots(bm, PLOT_PATH)

    # time line of agents positions
    from MA.plots import positions_plot
    positions_plot(pred, PLOT_PATH)

    # maximum drawdown of benchmark and agent's strategies
    from MA.plots import max_drawdown
    max_drawdown(bm, pred, PLOT_PATH)

    # create return distributions
    from MA.plots import dist_of_return
    dist_of_return(bm, pred, PLOT_PATH)


def run_eval(bm_set):
    prediction_set = pd.read_csv(EVAL_PATH + evaluation_period)
    prediction_set = prediction_set.iloc[:, 1:]
    print(f'loaded, {evaluation_period}')

    feature_cols = ['date', 'PF value', 'PnL', 'returns', 'positions']

    prediction_set = prediction_set[feature_cols]
    # drop the first and the last rows. in the first nothing happens,
    # in the last default happens and cannot be treated as a valid action
    prediction_set = prediction_set.iloc[:-1, :].fillna(0)

    bm_set = transform_set(bm_set)
    feature_cols.remove('positions')
    feature_cols.extend(['full_close_es', 'full_close_zn'])
    bm_set = same_period(bm_set, prediction_set)
    bm_set = bm_set[feature_cols]
    bm = perf_evaluation(bm_set)
    pred = perf_evaluation(prediction_set)

    visualize_evals(bm, pred)





