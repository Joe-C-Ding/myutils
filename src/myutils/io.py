from siewrapper import Sie

import time
import pathlib

import pandas as pd
import numpy as np


_sie_path = None
_sie = None


def _open_sie(sie):
    """
    cause open sie could be very slow, this function hold a sie instance for successive usage.

    Parameters
    ----------
    sie : str or pathlib.Path or Sie

    Returns
    -------
    sie : Sie
        the Sie instance.
    """
    global _sie, _sie_path

    if isinstance(sie, pathlib.Path):
        path = sie
    elif isinstance(sie, str):
        path = pathlib.Path(sie)
    elif isinstance(sie, Sie):
        path = pathlib.Path(sie.file_name)
    else:
        raise TypeError(f'unsupported type of sie: {type(sie)}: {sie}')

    if path != _sie_path:
        _sie_path = path
        if _sie is not None:
            _sie.__del__()

        # This may take a long time.
        tic = time.time()
        _sie = Sie(str(_sie_path.absolute()))
        print(f'open sie file takes {time.time() - tic:.2f}s.')

    return _sie


def extract_channels(sie, channels, columns=None, *, test_id=0):
    """
    extract channel data into DataFrame from sie file.

    Parameters
    ----------
    sie : str or Sie or pathlib.Path
        the sie file.
    channels : int or list[int]
        channel indexes to be extracted. NOTE: The index is NOT the id of this channel.
    columns : str or list[str], optional
        the name of each channel.
    test_id : int
        the test_id from which the data will be extracted, default 0.

    Returns
    -------
    df : pd.DataFrame
        the extracted data.
    """
    if isinstance(channels, int):
        channels = [channels]

    if columns is None:
        columns = [str(i) for i in channels]
    elif isinstance(columns, str):
        columns = [columns]

    if len(channels) != len(columns):
        raise ValueError('size of channels and columns are mismatched.')

    sie = _open_sie(sie)

    start = sie.tests[test_id].start
    chs = sie.tests[test_id].channels

    collection = dict()
    for i, c in enumerate(channels):
        data = list(chs[c].data)
        collection[columns[i]] = np.concatenate(data, axis=0)

    df = None
    for c, v in collection.items():
        if df is None:
            t = start + pd.TimedeltaIndex(v[:, 0], 's')
            df = pd.DataFrame(index=t, columns=columns)
        df.loc[:, c] = v[:, 1]

    del chs
    return df


def extract_acceleration(sie, acceleration_chs, coefficients, *, test_id=0, multiply_g=True):
    """
    Extract acceleration and speed data from sie file. And perform unit conversion on acceleration data.

    Parameters
    ----------
    sie : str or pathlib.Path or Sie
        the sie file.
    acceleration_chs : list[int]
        channel indexes of the acceleration data in the order of longitudinal, lateral and vertical directions. If only
        2 indexes are passed, it is considered that the direction of longitudinal is omitted.
    coefficients : list[float]
        the acceleration data will be multiplied by those coefficients for unit conversion.
    multiply_g : bool
        if True (default), the acceleration will be multiplied by g after unit conversion.
        Multiplying the unit converting coefficients may result in g, which is not suitable for GB/T 5599 calculation.
        This option make it easy to convert the result into m/s**2 further. If the `coefficients` is already for m/s**2,
        or it's intentionally to convert data into g, pass False instead.
    test_id : int
        the test_id from which the data will be extracted, default 0.

    Returns
    -------
    accel : pd.DataFrame
        the extracted acceleration data.
    """
    if len(acceleration_chs) >= 3:
        columns = list('xyz')
    elif len(acceleration_chs) == 2:
        columns = list('yz')
    else:
        raise ValueError(f'invalid value: {acceleration_chs = }')

    if len(coefficients) != len(columns):
        raise ValueError('`acceleration_chs` and `coefficients` size mismatch.')
    coefficients = np.array(coefficients)

    if multiply_g:
        coefficients *= 9.80665

    accel = coefficients * extract_channels(sie, acceleration_chs[:3], columns=columns, test_id=test_id)
    return accel


def infer_fs(data):
    """
    infer sampling frequency from data.

    Parameters
    ----------
    data : pd.Sereis or pd.DataFrame
        the data must be indexed by time, and should have const sample rate.

    Returns
    -------
    fs : float
        the sampling frequency of data.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError('data should be indexed by time')

    if isinstance(data, pd.Series):
        data = data.to_frame()

    max_length = 10000
    if data.shape[0] > max_length:
        data = data.iloc[:max_length, :]

    t = data.index.to_numpy().astype('M8[ns]')
    dt = np.diff(t).astype('d') / 1e9
    if np.allclose(dt, dt[0], atol=1e-6):
        return 1. / dt[0]
    else:
        raise ValueError('data are not sampled in const frequency.')

