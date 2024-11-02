import numpy as np
import numpy.ma as ma
from numpy import polyfit, polyval
from scipy import stats


def linear_regression(xin, yin, order=1, dtype="float64"):
    """Performs linear regressions of xin onto yin."""
    return linear_regression_poly(xin, yin, order=order, dtype=dtype)


def linear_regression_poly(xin, yin, order=1, dtype="float64"):
    """Performs linear regressions of xin onto yin."""
    x = ma.fix_invalid(xin.astype(dtype))
    y = ma.fix_invalid(yin.astype(dtype))

    # Ensure common mask
    x.mask = y.mask = x.mask | y.mask
    coeffs = polyfit(x.compressed(), y.compressed(), order)
    yr = linear_correct(coeffs, yin)
    return coeffs, yr


def linear_regression_scipy(xin, yin, order=1, dtype="float64"):
    """Performs linear regressions of xin onto yin."""
    x = ma.fix_invalid(xin.astype(dtype))
    y = ma.fix_invalid(yin.astype(dtype))

    # Ensure common mask
    x.mask = y.mask = x.mask | y.mask
    gradient, intercept, r_value, p_value, std_err = stats.linregress(
        x.compressed(), y.compressed()
    )
    coeffs = [gradient, intercept]
    yr = linear_correct(coeffs, yin)
    return coeffs, yr


def linear_regression_zero(xin, yin, order=1, dtype="float64"):
    """Performs linear regressions of xin onto yin."""
    x = ma.fix_invalid(xin.astype(dtype))
    y = ma.fix_invalid(yin.astype(dtype))
    # Ensure common mask
    x.mask = y.mask = x.mask | y.mask
    xy = ma.sum(x * y)
    xx = ma.sum(x * x)
    coef = xy / xx
    return coef, yin / coef


def linear_correct(coeffs, model):
    """Corrects model based in linear regression coefficients."""
    corr = (model - coeffs[1]) / coeffs[0]
    return corr


def linear_regression_manual(xin, yin, dtype="float64"):
    """Performs linear regressions of xin onto yin."""
    x = ma.fix_invalid(xin.astype(dtype))
    y = ma.fix_invalid(yin.astype(dtype))
    # Ensure common mask
    x.mask = y.mask = x.mask | y.mask
    xym = ma.mean(x * y)
    xxm = ma.mean(x * x)
    # yym = ma.mean(y*y)
    xm = ma.mean(x)
    ym = ma.mean(y)
    a = (xym - xm * ym) / (xxm - xm**2)
    b = ym - a * xm
    coeffs = [a, b]
    yr = polyval(coeffs, yin)
    return coeffs, yr
