import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.collections as mc
import matplotlib.style

import pandas as pd
import numpy as np


matplotlib.style.use('fast')


class EdgeLocator(ticker.AutoLocator):
    def __init__(self):
        super().__init__()

    def tick_values(self, vmin, vmax):
        ticks = super().tick_values(vmin, vmax)
        ticks[0] = vmin
        ticks[-1] = vmax
        return ticks


_time_locator = mdates.AutoDateLocator(minticks=4, maxticks=10)


def _best_subs(n, max_row=5):
    # TODO: to be implemented
    return n, 1


def show_signal(data, step=None, ylabels=None):
    """
    draw signal.

    Parameters
    ----------
    data : pd.DataFrame
        the signal data.
    step : int, optional
        down sampling factor, may speed up drawing.
    ylabels : str or list[str], optional
        the units of y-axis for each column.

    Returns
    -------
    ax : plt.Axes or np.ndarray
        the axes of the plot.
    """
    if step is None:
        step = 1

    if ylabels is None:
        ylabels = data.columns
    elif isinstance(ylabels, str):
        ylabels = [ylabels]

    n = len(data.columns)
    fig, ax = plt.subplots(*_best_subs(n), sharex=True)
    ax = np.atleast_1d(ax)

    for i in range(n):
        d = data.iloc[:, i]
        ax[i].plot(d.index[::step], d.iloc[::step], label=data.columns[i])
        if ylabels is not None:
            ax[i].set_ylabel(ylabels[i])
        ax[i].set_ylim(d.min(), d.max())
        ax[i].yaxis.set_major_locator(EdgeLocator())
    ax[-1].set_xlim(data.index[[0, -1]])
    ax[-1].xaxis.set_major_locator(_time_locator)
    ax[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(_time_locator))

    if ax.size == 1:
        return ax[0]
    else:
        return ax


def show_gps(latitude, longitude, speed=None):
    """
    show the gps trace with its speed.

    Parameters
    ----------
    latitude, longitude : pd.Series or np.ndarray
        the signal of latitude, longitude.
    speed : pd.Series, optional
        the speed signal.

    Returns
    -------
    ax : plt.Axes or np.ndarray
        the axes of plot.
    """
    gps = np.array([latitude, longitude]).T

    if speed is None:
        fig, ax = plt.subplots()    # cause ax will be returned at the end.
        gps_ax = ax
        gps_lc = mc.LineCollection(gps[np.newaxis, :, :])
    else:
        if latitude.size != speed.size:
            raise ValueError(f'the lengths of latitude and speed are mismatched.')
        speed = np.array([mdates.date2num(speed.index), speed]).T

        fig, ax = plt.subplots(2, 1)
        gps_ax = ax[1]
        segments = np.concatenate([gps[:-1, np.newaxis, :],
                                   gps[1:, np.newaxis, :]], axis=1)
        gps_lc = mc.LineCollection(segments, cmap='jet')
        gps_lc.set_array(speed[:, 1])

        segments = np.concatenate([speed[:-1, np.newaxis, :],
                                   speed[1:, np.newaxis, :]], axis=1)
        spd_lc = mc.LineCollection(segments, cmap='jet')
        spd_lc.set_array(speed[:, 1])

        ax[0].add_collection(spd_lc)
        ax[0].autoscale()
        ax[0].set_xlim(speed[0, 0], speed[-1, 0])
        ax[0].set_ylim(ymin=0)

        ax[0].xaxis.set_major_locator(_time_locator)
        ax[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(_time_locator))
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Speed [km/h]')

    gps_ax.add_collection(gps_lc)
    gps_ax.autoscale()

    gps_ax.plot(gps[0, 0], gps[0, 1], marker='o', markersize=8, color='g')
    gps_ax.plot(gps[-1, 0], gps[-1, 1], marker='^', markersize=8, color='r')
    gps_ax.set_xlabel('Latitude')
    gps_ax.set_ylabel('Longitude')

    return ax


def _plot_curves(data, curves, detail, ax):
    """
    plot curves bound in ax. if curves is None, only adjust x-axis.

    Parameters
    ----------
    data: pd.DataFrame
        same as show_curves.
    curevs: pd.DataFrame, optional
        same as show_curves.
    detail: int
        further information atteched with bound.
        0: (id: milage)
        1: (id: radius)
    ax : plt.Axes
        the axes to plot onto.
    """
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.set_xlabel('time')
    ax.set_xlim(data.index[0], data.index[-1])

    if curves is None:
        return

    ax.plot(data.index, data.milage)
    ax.set_ylabel('Milage / km')

    milage = data.milage.to_numpy()
    ascending = 1 if milage[-1] > milage[0] else -1
    if ascending == -1:
        milage = milage[::-1]

    ymin, ymax = ax.get_ylim()
    h = ymax - ymin
    yh = ymin + 0.50 * h
    yl = ymin + 0.25 * h

    for i, c in curves.iterrows():
        start, finish, radius = c[['start', 'finish', 'radius']]
        s = ascending * np.searchsorted(milage, start)
        f = ascending * np.searchsorted(milage, finish)
        try:
            st = data.index[s]
            ft = data.index[f]
        except IndexError:
            continue

        if st > ft:
            st, ft = ft, st
        ax.vlines(st, ymin, ymax, ls='-.', color='g')
        ax.vlines(ft, ymin, ymax, ls=':', color='r')

        if detail == 0:
            ax.text(st, yh, f'({i}: {start:.2f})', ha='left')
            ax.text(ft, yl, f'({i}: {finish:.2f})', ha='right')
        elif detail == 1:
            ax.text(st, yh, f'({i}: {radius:.0f})', ha='left')
            ax.text(ft, yl, f'({i}: {radius:.0f})', ha='right')
        else:
            pass


def show_curves(data, curves):
    """
    Show vertical and laterl force with curves bound. This is useful to check if the curves calculated
    based on milage are correct.

    Parameters
    ----------
    data : pd.DataFrame
        data with its columns contans `LP`, `RP`, `LQ`, `RQ` and `milage`.
    curves : pd.DataFrame
        curves information with its columns caontan `start` and `finish`.

    Returns
    -------
    ax: np.ndarray
        the axes of plots.
    """
    fig, ax = plt.subplots(3, 1, sharex=True)

    ax[0].plot(data.index, data.LP, data.index, data.RP)
    ax[0].set_ylabel('Vertical Load / kN')

    ax[1].plot(data.index, data.LQ, data.index, data.RQ)
    ax[1].set_ylabel('Lateral Load / kN')

    _plot_curves(data, curves, detail=0, ax=ax[2])

    return ax


def _emphasize_abs(data, threshold, ax, *, abs_plot=True):
    """
    plot data, and marker the points that its absolute value is greater that threshold.
    Parameters
    ----------
    data : pd.DataFrame
        the data
    threshold : float
        the threshold
    ax : plt.Axes
        the axes to plot onto.
    abs_plot : bool, default True
        if False, the plot the original values instead of its absolute values.
    """
    if abs_plot:
        abs = data.abs().to_numpy()
        big = abs > threshold
    else:
        abs = data  # the variable name `abs` is a little bit confusing here.
        big = abs.abs() > threshold

    ax.plot(data.index, abs)
    if np.any(big):
        ax.plot(data.index[big], abs[big], 'rx')


def show_drh(samples, curves=None, title=None, limit_d=(0.8, 1.0)):
    """
    Show Derailment, Reduction Ratio, Laterl Force and vertical, laterl force.

    Parameters
    ----------
    samples : pd.DataFrame
        data with its columns contains `Q/P`, `DP/P`, `H`, `P` and `Q`.
    curves : pd.DataFrame, optional
        if presents, plot milage and curves information.
    title : str, optional
        the subtitle of each figure.
    limit_d : tuple[float], default (0.8, 1.0).
        the limit of derailtment in the form of (limit R > 400, limit R <= 400).

    Returns
    -------
    figs: list[plt.Figure]
        the list of figures.
    """
    figs = []
    if curves is None:
        fig, ax = plt.subplots(3, 1, sharex=True)
    else:
        fig, ax = plt.subplots(4, 1, sharex=True)
    figs.append(fig)

    kw = dict(ha='right', va='top')

    _emphasize_abs(samples['Q/P'], limit_d[0], ax=ax[0])
    ax[0].hlines(limit_d, *ax[0].get_xlim(), ls='--', colors=['g', 'r'])
    ax[0].text(samples.index[-1], limit_d[0], r'$R > 400\mathrm{m}$', **kw)
    ax[0].text(samples.index[-1], limit_d[1], r'$R\leq 400\mathrm{m}$', **kw)
    ax[0].set_ylim(ymin=0)
    ax[0].set_ylabel('Q/P')

    # reduction ratio
    _emphasize_abs(samples['DP/P'], 0.65, ax=ax[1])
    ax[1].hlines([0.65, 0.80], *ax[1].get_xlim(),  ls='--', colors=['g', 'r'])
    ax[1].text(samples.index[-1], 0.65, r'$v\leq 160\mathrm{km/h}$', **kw)
    ax[1].text(samples.index[-1], 0.80, r'$v > 160\mathrm{km/h}$', **kw)
    ax[1].set_ylim(ymin=0)
    ax[1].set_ylabel(r'$\Delta$P/P')

    hmax = 15 + samples['P'].mean() / 3
    _emphasize_abs(samples['H'], hmax, ax=ax[2], abs_plot=False)
    ax[2].set_ylabel('H / kN')
    ax[2].hlines([hmax, -hmax], *ax[2].get_xlim(),  ls='--', colors='r')

    _plot_curves(samples, curves, detail=1, ax=ax[-1])

    if title:
        fig.suptitle(title)

    if curves is None:
        fig, ax = plt.subplots(2, 1, sharex=True)
    else:
        fig, ax = plt.subplots(3, 1, sharex=True)
    figs.append(fig)

    ax[0].plot(samples.index, samples['P'])
    ax[0].set_ylabel('P / kN')

    ax[1].plot(samples.index, samples['Q'])
    ax[1].set_ylabel('Q / kN')

    _plot_curves(samples, curves, detail=1, ax=ax[-1])

    if title:
        fig.suptitle(title)

    return figs


def show_lateral_stability(filtered, original=None, labels=None):
    if original is None:
        fig, ax = plt.subplots()
    else:
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(original.index, original)
        axs[0].set_title('original')
        axs[0].set_ylabel(r'm / s$^2$')
        ax = axs[1]

    ax.plot(filtered.index, filtered, label=labels)
    ax.set_title('0.5--10 Hz bandpass')
    ax.set_ylabel(r'm / s$^2$')
    ax.set_xlabel('Time')
    _plot_curves(filtered, None, 0, ax=ax)   # adjust x-axis

    if labels:
        ax.legend()

    if original is None:
        return ax
    else:
        return axs

