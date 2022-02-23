import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


def plot(data, filename, xticks=None, title='', yticks=None, ylimits=None):
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

    ax.tick_params(axis='x', rotation=0, labelsize=20)
    ax.tick_params(axis='y', rotation=0, labelsize=20)

    plt.grid(True, color='lightgray', which='both', axis='y', linestyle='-')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=7)

    # plt.show()
    fig.savefig(filename)
    print(f'plot {filename} created')
