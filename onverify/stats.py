"""Stats functions."""
import numpy as np


def mad(xarr, yarr, norm=False):
    """Mean absolute difference MAD.

    :math:`MAD = \\frac{1}{N}{\\sum_{i=1}^N {\\left|A_i-B_i \\right|}}}`

    Args:
        xarr (array): x values, usually observations.
        yarr (array): y values, usually model.
        norm (bool): Normalise MAD by xmean.

    """
    ret = np.mean(np.abs(yarr - xarr))
    if norm:
        ret /= np.mean(xarr)
    return ret


def rmsd(xarr, yarr, norm=False):
    """Root-mean-square difference.

    :math:`RMSD = \\sqrt{\\frac{1}{N}{\\sum_{i=1}^N {\\left(A_i-B_i \\right)^2}}}`

    Args:
        xarr (array): x values, usually observations.
        yarr (array): y values, usually model.
        norm (bool): Normalise MAD by xmean.

    """
    ret = np.mean(np.sqrt((yarr - xarr) ** 2))
    if norm:
        ret /= np.mean(xarr)
    return ret


def bias(xarr, yarr, norm=False):
    """Bias.

    :math:`Bias = {\\frac 1 N}{\\sum_{i=1}^N {A_i-B_i}}`

    Args:
        xarr (array): x values, usually observations.
        yarr (array): y values, usually model.
        norm (bool): Normalise MAD by xmean.

    """
    ret = np.mean(yarr - xarr)
    if norm:
        ret /= np.mean(xarr)
    return ret


def si(xarr, yarr):
    """Scatter Index.

    :math:`SI = {\\frac { \\sqrt { {\\frac 1 N} { \\sum_{i=1}^N {\\left(\\left(A_i-{\\overline A}\\right)-\\left(B_i-{\\overline B}\\right)\\right)^2}}} }{  \overline B} }`

    Args:
        xarr (array): x values, usually observations.
        yarr (array): y values, usually model.

    """
    diff_values = yarr - xarr
    bias_values = bias(xarr, yarr)
    return np.sqrt(np.mean((diff - bias) ** 2)) / np.mean(xarr)


def r(xarr, yarr):
    """Pearson Correlation Coeficient.

    :math:`R = ...`
 
    Args:
        xarr (array): x values, usually observations.
        yarr (array): y values, usually model.

    """
    return np.corrcoef(yarr, xarr)[0, 1]
