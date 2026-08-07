"""Stats functions."""
import numpy as np
from scipy.stats import ks_2samp


def _err(x, y, circular=False):
    """Differences between model and observations.

    Args:
        x (array): x values, usually observations.
        y (array): y values, usually model.
        circular (bool): for circular arrays such as directions.

    """
    if circular:
        err0 = np.abs(y % 360 - x % 360)
        errmin = np.minimum(err0, 360 - err0)
        errneg = np.logical_xor(y > x, err0 < 180)
        signchanger = 1 - 2 * errneg
        err = signchanger * errmin
    else:
        err = y - x
    return err


def mad(x, y, norm=False, circular=False):
    """Mean absolute difference MAD.

    :math:`MAD = \\frac{1}{N}{\\sum_{i=1}^N {\\left|A_i-B_i \\right|}}}`

    Args:
        x (array): x values, usually observations.
        y (array): y values, usually model.
        norm (bool): Normalise MAD by xmean.
        circular (bool): for circular arrays such as directions.

    """
    ret = np.mean(np.abs(_err(x, y, circular)))
    if norm:
        ret /= np.mean(x)
    return ret


def mrad(x, y, circular=False):
    """Mean Relative Absolute Deviation MRAD.

    :math:`MRAD = {\\frac 1 N}{\\sum_{i=1}^N {|\\frac {A_i-B_i} {B_i}|}}`

    Args:
        x (array): x values, usually observations.
        y (array): y values, usually model.
        circular (bool): for circular arrays such as directions.

    """
    xmask = np.ma.masked_values(x, 0.0)
    return np.mean(np.abs(_err(x, y, circular) / xmask))


def rmsd(x, y, norm=False, circular=False):
    """Root-mean-square difference.

    :math:`RMSD = \\sqrt{\\frac{1}{N}{\\sum_{i=1}^N {\\left(A_i-B_i \\right)^2}}}`

    Args:
        x (array): x values, usually observations.
        y (array): y values, usually model.
        norm (bool): Normalise MAD by xmean.
        circular (bool): for circular arrays such as directions.

    """
    ret = np.sqrt(np.mean(_err(x, y, circular) ** 2))
    if norm:
        ret /= np.mean(x)
    return ret


def bias(x, y, norm=False, circular=False):
    """Bias.

    :math:`Bias = {\\frac 1 N}{\\sum_{i=1}^N {A_i-B_i}}`

    Args:
        x (array): x values, usually observations.
        y (array): y values, usually model.
        norm (bool): Normalise MAD by xmean.
        circular (bool): for circular arrays such as directions.

    """
    ret = np.mean(_err(x, y, circular))
    if norm:
        ret /= np.mean(x)
    return ret


def si(x, y, circular=False):
    """Scatter Index.

    :math:`SI = {\\frac { \\sqrt { {\\frac 1 N} { \\sum_{i=1}^N {\\left(\\left(A_i-{\\overline A}\\right)-\\left(B_i-{\\overline B}\\right)\\right)^2}}} }{  {\\overline B} }`

    Args:
        x (array): x values, usually observations.
        y (array): y values, usually model.
        circular (bool): for circular arrays such as directions.

    """
    diff_values = _err(x, y, circular)
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


def usi(x, y):
    """Unexplained Scatter Index USI.

    :math:`USI = \\frac{\\sigma_A \\sqrt{1 - R^2}}{\\overline A}`

    The scatter that remains once *both* systematic components of the error --
    the mean offset and the amplitude error -- have been accounted for. It is
    the root-mean-square residual of the least-squares regression of the
    observations on the model, normalised by the observed mean.

    Args:
        x (array): x values, usually observations.
        y (array): y values, usually model.

    Why this in addition to :func:`si`:
        :func:`si` removes the mean offset but **not** an amplitude error. Its
        square is the error variance, which decomposes exactly as

        .. math::
            \\mathrm{var}(y-x) = (\\sigma_B - \\sigma_A)^2
                               + 2\\,\\sigma_A\\sigma_B\\,(1 - R)

        The first term is a systematic mismatch in variability: a model that
        reproduces every event in phase but swings 25% too hard scores a large
        SI with no random error at all. The second is the genuinely
        unexplained part. USI isolates that second component.

        USI is invariant under any affine transform of the model,
        :math:`y \\rightarrow \\alpha y + \\beta`, so neither a bias nor a
        gain error can move it. Comparing SI and USI therefore separates
        "the model is noisy" from "the model is mis-scaled", which the two
        indices conflate when reported alone.

    Note:
        Linear variables only. It is built on the Pearson correlation, which
        is not meaningful for directions -- there is deliberately no
        ``circular`` argument.

    References:
        Murphy, A.H. (1988). Skill scores based on the mean square error and
        their relationships to the correlation coefficient. *Monthly Weather
        Review*, 116(12), 2417-2424. The MSE decomposition this rests on.

        Taylor, K.E. (2001). Summarizing multiple aspects of model performance
        in a single diagram. *Journal of Geophysical Research*, 106(D7),
        7183-7192. The same geometry, as the relationship between the centred
        RMS difference, the standard-deviation ratio and R.

        Mentaschi, L., Besio, G., Cassola, F., Mazzino, A. (2013). Problems in
        RMSE-based wave model validations. *Ocean Modelling*, 72, 53-58. On
        why RMSE-normalised indicators mislead when a model is mis-scaled.

        Hanna, S.R., Heinold, D.W. (1985). Development and application of a
        simple method for evaluating air quality models. API Publication 4409,
        American Petroleum Institute.

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.std(x) * np.sqrt(1.0 - r(x, y) ** 2) / np.mean(x)


def mse_decomposition(x, y):
    """Split the mean square error into its three additive components.

    :math:`MSE = \\mathrm{bias}^2
                 + (\\sigma_B - \\sigma_A)^2
                 + 2\\,\\sigma_A\\sigma_B\\,(1 - R)`

    Args:
        x (array): x values, usually observations.
        y (array): y values, usually model.

    Returns:
        dict with keys ``bias2`` (systematic offset), ``amplitude``
        (mismatch in variability) and ``decorrelation`` (the unexplained
        remainder), plus ``mse`` and the share each term carries.

    The identity is exact, so the three terms sum to the MSE to within
    floating-point error. Which one dominates is what tells you where to look:
    ``bias2`` points at a mean offset, ``amplitude`` at a gain error such as a
    dissipation term scaling wrongly with sea state, ``decorrelation`` at
    timing or forcing.

    References:
        See :func:`usi`.

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sa, sb = np.std(x), np.std(y)
    rr = r(x, y)
    terms = {
        "bias2": bias(x, y) ** 2,
        "amplitude": (sb - sa) ** 2,
        "decorrelation": 2.0 * sa * sb * (1.0 - rr),
    }
    total = sum(terms.values())
    terms["mse"] = float(np.mean((y - x) ** 2))
    for key in ("bias2", "amplitude", "decorrelation"):
        terms[f"{key}_share"] = terms[key] / total if total else np.nan
    return terms
