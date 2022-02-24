import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import numpy as np


def plot(data, filename, xticks=None, title='', yticks=None, ylimits=None, yscale='linear'):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.boxplot(data,
               boxprops=dict(linewidth=4, color='blue'),
               medianprops=dict(linewidth=4, color='red'),
               whiskerprops=dict(linewidth=4, color='black'),
               capprops=dict(linewidth=4),
               whis=1000)

    if xticks is not None:
        plt.xticks(list(range(1, len(xticks) + 1)), xticks)

    if yticks is not None:
        plt.yticks(yticks)
    if ylimits is not None:
        ax.set_ylim(ylimits)

    plt.yscale(yscale)

    ax.tick_params(axis='x', rotation=0, labelsize=20)
    ax.tick_params(axis='y', rotation=0, labelsize=20)

    plt.grid(True, color='lightgray', which='both', axis='y', linestyle='-')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)

    # plt.show()
    fig.savefig(filename)
    print(f'plot {filename} created')


def plot_reduction(data_dict, baseline, filename, title='', yticks=None, ylimits=None):
    # == preprocess data
    data = []
    xticks = []
    data_dict = data_dict.copy()
    n_data = data_dict.pop(baseline)
    for key in data_dict:
        xticks.append(key)
        data.append([(e2 - e1) / e2 for e1, e2 in zip(data_dict[key], n_data)])

    if ylimits is None:
        ylimits = [0.0, 1.0]

    plot(data, filename, xticks=xticks, title=title, yticks=yticks, ylimits=ylimits, yscale='linear')


def plot_gap_reduction(data_dict, baseline0, baseline1, filename, title='', yticks=None, ylimits=None):
    """closes the gap between baseline0 and baseline1 by x percent."""
    # == preprocess data
    data = []
    xticks = []
    data_dict = data_dict.copy()
    n0_data = data_dict.pop(baseline0)
    n1_data = data_dict.pop(baseline1)
    for key in data_dict:
        xticks.append(key)
        data.append([(e - e0) / (e1 - e0) for e, e0, e1 in zip(data_dict[key], n0_data, n1_data)])

    if ylimits is None:
        ylimits = [0.0, 1.0]

    plot(data, filename, xticks=xticks, title=title, yticks=yticks, ylimits=ylimits, yscale='linear')
