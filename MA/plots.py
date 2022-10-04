import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


def cumulative_return(bm, pred, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = bm[1]['date']
    y1 = bm[1]['cum_ret']
    y2 = pred[1]['cum_ret']

    plt.title('Cumulative Trade Returns')
    plt.plot(x, y1, 'b', label='benchmark')
    plt.plot(x, y2, 'g', label='agent')
    plt.ylabel('cumulative returns')

    date_form = DateFormatter("%d-%m-%Y\n%H:%M")
    ax.xaxis.set_major_formatter(date_form)

    plt.legend(loc="upper left")
    plt.savefig(path + 'Cumulative Returns.svg', format='svg')
    plt.clf()

