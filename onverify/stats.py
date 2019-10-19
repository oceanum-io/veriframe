"""Stats functions."""
import numpy as np
from scipy.stats import ks_2samp

def mad(x, y, norm=False):
    """Mean absolute difference MAD.

    :math:`MAD = \\frac{1}{N}{\\sum_{i=1}^N {\\left|A_i-B_i \\right|}}}`

    Args:
        x (array): x values, usually observations.
        y (array): y values, usually model.
        norm (bool): Normalise MAD by xmean.

    """
    ret = np.mean(np.abs(y - x))
    if norm:
        ret /= np.mean(x)
    return ret


def mrad(x, y, **kwargs):
    """Mean Relative Absolute Deviation MRAD.

    :math:`MRAD = {\\frac 1 N}{\\sum_{i=1}^N {|\\frac {A_i-B_i} {B_i}|}}`

    Args:
        x (array): x values, usually observations.
        y (array): y values, usually model.

    """
    xmask = np.ma.masked_values(x, 0.)
    return np.mean(np.abs((y - x) / xmask))


def rmsd(x, y, norm=False):
    """Root-mean-square difference.

    :math:`RMSD = \\sqrt{\\frac{1}{N}{\\sum_{i=1}^N {\\left(A_i-B_i \\right)^2}}}`

    Args:
        x (array): x values, usually observations.
        y (array): y values, usually model.
        norm (bool): Normalise MAD by xmean.

    """
    ret = np.sqrt(np.mean((y - x) ** 2))
    if norm:
        ret /= np.mean(x)
    return ret


def bias(x, y, norm=False):
    """Bias.

    :math:`Bias = {\\frac 1 N}{\\sum_{i=1}^N {A_i-B_i}}`

    Args:
        x (array): x values, usually observations.
        y (array): y values, usually model.
        norm (bool): Normalise MAD by xmean.

    """
    ret = np.mean(y - x)
    if norm:
        ret /= np.mean(x)
    return ret


def si(x, y):
    """Scatter Index.

    :math:`SI = {\\frac { \\sqrt { {\\frac 1 N} { \\sum_{i=1}^N {\\left(\\left(A_i-{\\overline A}\\right)-\\left(B_i-{\\overline B}\\right)\\right)^2}}} }{  \overline B} }`

    Args:
        x (array): x values, usually observations.
        y (array): y values, usually model.

    """
    diff_values = y - x
    bias_values = bias(x, y)
    return np.sqrt(np.mean((diff_values - bias_values) ** 2)) / np.mean(x)


def r(x, y):
    """Pearson Correlation Coeficient.

    :math:`R = ...`
 
    Args:
        x (array): x values, usually observations.
        y (array): y values, usually model.

    """
    return np.corrcoef(y, x)[0, 1]


def ks(x, y):
    """Kolmogorov-Smirnov statistic.

    :math:`D = {\\max(|F1(x)-F2(x)|)}`

    Args:
        x (array): x values, usually observations.
        y (array): y values, usually model.

    """
    return ks_2samp(x, y)[0]
