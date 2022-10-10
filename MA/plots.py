import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

import ast

"""
This scrip defines all the different plots.
"""


def cumulative_return(bm, pred, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = bm[1]['date']
    y1 = bm[1]['cum_ret']
    y2 = pred[1]['cum_ret']

    plt.title('Cumulative Trade Returns')
    plt.plot(x, y1, 'b', label='Benchmark')
    plt.plot(x, y2, 'g', label='Agent')
    plt.ylabel('Cumulative Returns')

    date_form = DateFormatter("%d-%m-%y\n%H:%M")
    ax.xaxis.set_major_formatter(date_form)

    plt.legend(loc="upper left")
    plt.savefig(path + 'Cumulative Returns.pdf', format='pdf')

    plt.clf()


def price_plots(bm, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = bm[1]['date']
    y1 = bm[1]['full_close_es'] / bm[1]['full_close_es'][0]
    y2 = bm[1]['full_close_zn'] / bm[1]['full_close_zn'][0]

    plt.title('Price Development')

    ln1 = ax.plot(x, y1, 'crimson', label='ES')
    ln2 = ax.plot(x, y2, 'orange', label='ZN')
    ax.set_ylabel('Prices Change')

    date_form = DateFormatter("%d-%m-%y\n%H:%M")
    ax.xaxis.set_major_formatter(date_form)

    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc="upper left")

    plt.savefig(path + 'Prices.pdf', format='pdf')
    plt.clf()


def positions_plot(pred, path):
    hold_pos = [ast.literal_eval(elem) for elem in pred[1]['positions']]
    es_pos = [int(item[0]) for item in hold_pos]
    zn_pos = [int(item[1]) for item in hold_pos]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = pd.to_datetime(pred[1]['date'])

    date_form = DateFormatter("%d-%m-%y\n%H:%M")
    ax.xaxis.set_major_formatter(date_form)

    y1 = np.full((pred[1]['date'].shape[0],), 2)
    y2 = np.full((pred[1]['date'].shape[0],), 7)
    y3 = es_pos
    y4 = zn_pos
    plt.ylabel('Contract Positions')

    plt.title('Contract Positions')
    ln1 = ax.plot(x, y1, 'darkblue', label='Benchmark_ES', linestyle='--')
    ln2 = ax.plot(x, y2, 'lightblue', label='Benchmark_ZN', linestyle='--')
    ln3 = ax.plot(x, y3, 'darkgreen', label='Agent_ES', linestyle='-')
    ln4 = ax.plot(x, y4, 'yellowgreen', label='Agent_ZN', linestyle='-')

    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc="upper left")

    plt.savefig(path + 'Positions.pdf', format='pdf')
    plt.clf()


def max_drawdown(bm, pred, path):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6), sharey=True)
    fig.tight_layout(pad=3)
    x = bm[1]['date']

    date_form = DateFormatter("%d-%m-%y\n%H:%M")
    ax1.xaxis.set_major_formatter(date_form)
    ax2.xaxis.set_major_formatter(date_form)

    # bm max drawdown
    y1 = bm[1]['interval_drawdown']
    y2 = bm[1]['max_dd']

    ax1.plot(x, y1, 'b',  label='interval_drawdown')
    ax1.plot(x, y2, 'orange', label='max_drawdown')
    ax1.legend(loc="upper right")
    ax1.set_title('Maximum Drawdown Benchmark')

    # agent max drawdown
    y3 = pred[1]['interval_drawdown']
    y4 = pred[1]['max_dd']
    ax2.plot(x, y3, 'g', label='interval_drawdown')
    ax2.plot(x, y4, 'orange', label='max_drawdown')
    ax2.legend(loc="upper right")
    ax2.set_title('Maximum Drawdown Agent')

    plt.savefig(path + 'MaxDrawdown.pdf', format='pdf')
    plt.clf()


def dist_of_return(bm, pred, path):
    plt.subplots(figsize=(10, 6),)
    bm_returns = bm[1]['returns']
    agent_returns = pred[1]['returns']

    binwidth = 0.001
    bins = np.arange(min(bm_returns), max(bm_returns) + binwidth, binwidth)
    plt.xlabel('Returns')
    plt.ylabel('Count')
    plt.title('Return Distributions')
    plt.xlim(-0.01, 0.01)

    plt.hist(bm_returns, bins=bins, stacked=True, alpha=0.75, facecolor='b', label='Benchmark')
    plt.hist(agent_returns, bins=bins, stacked=True, alpha=0.75, facecolor='g', label='Agent')
    plt.legend(loc="upper left")

    plt.savefig(path + 'Prob_Dist.pdf', format='pdf')
    plt.clf()
