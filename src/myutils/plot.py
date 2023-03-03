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
