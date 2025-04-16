"""Pandas DataFrame for model verification."""
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
from matplotlib.projections.polar import PolarAxes
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde
from tabulate import tabulate
import logging
import cartopy.crs as ccrs

from veriframe import stats
from veriframe.regression import linear_regression
from veriframe.taylor import df2taylor


logger = logging.getLogger(__name__)


HERE = Path(__file__).parent
ETOPO = "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO1_Bed_g_gmt4.nc"
FONTSIZE_LABEL = 14

with open(HERE / "vardef.yml") as stream:
    VARDEF = yaml.load(stream, Loader=yaml.Loader)

with open(HERE / "defaults.yml") as stream:
    DEFAULTS = yaml.load(stream, Loader=yaml.Loader)


def set_docstring(fun):
    """format and return the docstring of function fun."""
    docstring_lines = getattr(stats, fun).__doc__.split("\n")
    docstring_lines = [
        line
        for line in docstring_lines
        if "x (array)" not in line and "y (array)" not in line
    ]
    return "\n".join(docstring_lines)


class AxisVerify:
    """Class for Formatting axis in VeriFrame."""

    def _extract_min_max_coords(self, ax):
        from matplotlib.collections import PolyCollection, PathCollection
        from matplotlib.lines import Line2D
        from matplotlib.patches import Polygon
        from matplotlib.contour import QuadContourSet

        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")

        for child in ax.get_children():
            if isinstance(child, Line2D):
                x_data = child.get_xdata()
                y_data = child.get_ydata()
            elif isinstance(child, PolyCollection):  # For hexbin
                for path in child.get_paths():
                    vertices = path.vertices
                    x_data = vertices[:, 0]
                    y_data = vertices[:, 1]
                    min_x = min(min_x, np.min(x_data))
                    max_x = max(max_x, np.max(x_data))
                    min_y = min(min_y, np.min(y_data))
                    max_y = max(max_y, np.max(y_data))
            elif isinstance(child, PathCollection):  # For scatter
                offsets = child.get_offsets()
                x_data = offsets[:, 0]
                y_data = offsets[:, 1]
                min_x = min(min_x, np.min(x_data))
                max_x = max(max_x, np.max(x_data))
                min_y = min(min_y, np.min(y_data))
                max_y = max(max_y, np.max(y_data))
            elif isinstance(child, Polygon):  # Sometimes contours use Polygon
                vertices = child.get_xy()
                x_data = vertices[:, 0]
                y_data = vertices[:, 1]
                min_x = min(min_x, np.min(x_data))
                max_x = max(max_x, np.max(x_data))
                min_y = min(min_y, np.min(y_data))
                max_y = max(max_y, np.max(y_data))
            elif isinstance(child, QuadContourSet):  # For contour and contourf
                try:
                    paths = child.collections.get_paths()
                except AttributeError:
                    paths = child.get_paths()
                for path in paths:
                    vertices = path.vertices
                    x_data = vertices[:, 0]
                    y_data = vertices[:, 1]
                    try:
                        min_x = min(min_x, np.min(x_data))
                        max_x = max(max_x, np.max(x_data))
                        min_y = min(min_y, np.min(y_data))
                        max_y = max(max_y, np.max(y_data))
                    except ValueError:
                        continue
            else:
                continue

        return min_x, max_x, min_y, max_y

    def set_xylimit(self, ax, equal=True, offset=0.02, xlim=None, ylim=None):
        """Set limit for axis based on plotted data.

        Retrieves data from axis to define x,y extensions.

        Args:
            - ``ax`` (handle): axis handle with plotted data to set limits to.
            - ``equal`` (bool): if True same limits are applied to x and y.
            - ``offset`` (float): proportion for extending limits beyond data.
            - ``xlim`` (tuple): x-limits for axis, if None inferred from data.
            - ``ylim`` (tuple): y-limits for axis, if None inferred from data.

        """
        min_x, max_x, min_y, max_y = self._extract_min_max_coords(ax)
        if equal:
            min_x = min(min_x, min_y)
            max_x = max(max_x, max_y)
            min_y = min_x
            max_y = max_x
        xoff = offset * (max_x - min_x)
        yoff = offset * (max_y - min_y)
        xlim = xlim or (min_x - xoff, max_x + xoff)
        ylim = ylim or (min_y - yoff, max_y + yoff)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return ax

    def legend(self, ax=None, **kwargs):
        """Standarised legend configuration with no outside frame.

        Args:
            - ``ax``: axis handle.
            - ``kwargs``: options to pass Matplotlib's legend.

        """
        kwargs = {**DEFAULTS["text_kwargs"]["legend"], **kwargs}

        font_prop = kwargs.pop("prop")
        if ax:
            lgd = ax.legend(**kwargs)
        else:
            lgd = plt.legend(**kwargs)
        lgd.get_frame().set_linewidth(0)
        lgd.get_frame().set_alpha(0.2)
        plt.setp(lgd.get_texts(), **font_prop)
        return lgd

    def add_text(self, ax, text, loc=2, **kwargs):
        """Add text to axis.

        Args:
            - ``ax``: axis handle.
            - ``text`` (str): text to show on axis.
            - ``loc`` (int, string or pair of floats): define location, analog to legend.
                - ``0``: ``'best'``.
                - ``1``: ``'upper right'``.
                - ``2``: ``'upper left'``.
                - ``3``: ``'lower left'``.
                - ``4``: ``'lower right'``.
                - ``5``: ``'right'``.
                - ``6``: ``'center left'``.
                - ``7``: ``'center right'``.
                - ``8``: ``'lower center'``.
                - ``9``: ``'upper center'``.
                - ``10``: ``'center'``.

        """
        ob = offsetbox.AnchoredText(text, loc=loc, prop=kwargs, frameon=False)
        ax.add_artist(ob)
        return ax


class VeriFrame(pd.DataFrame, AxisVerify):
    """DataFrame for model verification.

    Main Pandas DataFrame kwargs:
        ``data``: ndarray, dict or DataFrame.
        ``index``: Index or array-like for index in resulting frame,
            default is np.arange(n).
        ``columns``: Index or array-like for column labels in resulting
            frame, default is np.arange(n).

    Required kwargs:
        ``ref_col`` (str): name of column with observation values.
        ``verify_col`` (str): name of column with model values.

    Optional kwargs:
        ``var`` (str): id of variable to verify, 'hs' by default (supported
            options need to be defined in vardef.yml).
        ``circular`` (bool): use True for circular arrays such as directions.
        ``lat`` (float): latitude of site to validate, ignored if already a column.
        ``lon`` (float): longitude of site to validate, ignored if already a column.
        ``ref_label`` (str): used for labelling obs in plots if provided,
            otherwise constructed from ref_col, var, units.
        ``verify_label`` (str): used for labelling model in plots if provided,
            otherwise constructed from verify_col, var, units.

    """

    # These properties will propagate when subclassing
    _metadata = [
        "ref_col",
        "verify_col",
        "var",
        "circular",
        "ref_label",
        "verify_label",
    ]

    @property
    def _constructor(self):
        """Ensure sliced VeriFrame will preserve attributes."""
        return VeriFrame

    def __init__(self, *args, **kwargs):
        # Required arguments
        ref_col = kwargs.pop("ref_col", "ref_col has not been assigned")
        verify_col = kwargs.pop("verify_col", "verify_col has not been assigned")
        # Optional arguments
        var = kwargs.pop("var", "hs")
        circular = kwargs.pop("circular", False)
        lat = kwargs.pop("lat", "lat")
        lon = kwargs.pop("lon", "lon")
        ref_label = kwargs.pop("ref_label", ref_col)
        verify_label = kwargs.pop("verify_label", verify_col)

        super().__init__(*args, **kwargs)
        setattr(self, "ref_col", ref_col)
        setattr(self, "verify_col", verify_col)
        setattr(self, "ref_label", ref_label)
        setattr(self, "verify_label", verify_label)
        setattr(self, "var", var)
        setattr(self, "time_fmt", "%Y-%m-%d")
        setattr(self, "circular", circular)
        if self.circular:
            setattr(self, "ref_col", self.ref_col % 360)
            setattr(self, "verify_col", self.verify_col % 360)
        # Setting lat/lon cols if a string is provided
        if isinstance(lat, str) and lat != "lat":
            self["lat"] = self[lat]
        if isinstance(lon, str) and lon != "lon":
            self["lon"] = self[lon]

    def __repr__(self):
        return f"<{self.__class__.__name__}>\n{super().__repr__()}"

    def _set_axis_label(self, ax):
        """Set xy labels for axis."""
        xlabel = "${}_{{{}}}$ ({})".format(
            self.var.title(), self.ref_label.replace("_", "-"), self.units
        )
        ylabel = "${}_{{{}}}$ ({})".format(
            self.var.title(), self.verify_label.replace("_", "-"), self.units
        )
        ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
        return ax

    def _show_equality(self, ax, color="0.5"):
        """Show equality line on axis."""
        limits = list(ax.get_xlim()) + list(ax.get_ylim())
        vmin, vmax = min(limits), max(limits)
        ax.plot((vmin, vmax), (vmin, vmax), "--", color=color, label=None)
        return ax

    def _density_matrix(self, binsize):
        """Density matrix for contour plot.

        Args:
            - ``binsize`` (float): size of bins over which density is estimated.

        """
        x = np.arange(self.vmin - binsize, self.vmax + (2 * binsize), binsize)
        density, __, __ = np.histogram2d(
            self[self.verify_col], self[self.ref_col], bins=(x, x)
        )
        density /= density.max()
        return x[0:-1], np.ma.masked_less(density, 0)

    def _gaussian_kde(self, sample_size=None):
        """Kernel Density Estimation for density scatter plot."""
        # Optionally subsample for KDE calculation
        if sample_size is not None and len(self) > sample_size:
            df = self.sample(n=sample_size, random_state=42)
        else:
            df = self

        # Calculate KDE using sample points
        xy = np.vstack([df[self.ref_col], df[self.verify_col]])
        kde = gaussian_kde(xy)

        # Apply KDE to all points
        xy_all = np.vstack([self[self.ref_col], self[self.verify_col]])
        z = kde(xy_all)

        # Sort by density
        idx = z.argsort()
        return self[self.ref_col][idx], self[self.verify_col][idx], z[idx]

    def _get_cdf(self, col):
        """Return arrays for a CDF plot."""
        x = self[col].sort_values()
        n = np.arange(1, len(x) + 1) / float(len(x))
        return x, n

    def _fill_times(self, times):
        """Fill dataframe with interpolated values at specific times.

        Ensures suplots in plot_subtimeseries are continuous.

        """
        df = self.copy(deep=True)
        for t in times:
            row = pd.DataFrame(data={c: np.nan for c in vf.columns}, index=[t])
            if t not in df.index:
                df = df.append(row)
        return df.sort_index().interpolate()

    def _stats_frame(self, label="stats"):
        """Returns dataframe with stats."""
        ret = {}
        for col in DEFAULTS["_stats_table"]:
            if col == "n":
                ret[col.upper()] = int(self.nsamp)
            else:
                try:
                    ret[col.upper()] = getattr(self, col)()
                except:
                    try:
                        ret[col.upper()] = getattr(self, col[1:])(normalised=True)
                    except Exception:
                        raise NotImplementedError(
                            "Stat {} has not been defined".format(col)
                        )
        return pd.DataFrame(data=list(ret.values()), index=ret.keys(), columns=[label])

    @property
    def nsamp(self):
        """Returns the sample size n."""
        return self[[self.ref_col, self.verify_col]].dropna().shape[0]

    @property
    def vmin(self):
        """Returns the lowest value of the verification columns."""
        return self[[self.ref_col, self.verify_col]].min().min()

    @property
    def vmax(self):
        """Returns the lowest value of the verification columns."""
        return self[[self.ref_col, self.verify_col]].max().max()

    @property
    def units(self):
        """Returns the units of the verification variable."""
        try:
            return VARDEF["vars"][self.var]["units"]
        except KeyError:
            print(
                f"Variable {self.var} not defined in {HERE / 'vardef.yml'}. "
                f"Available variables: {','.join(VARDEF['vars'].keys())}"
            )
            return ""

    @property
    def coeffs(self):
        """Returns the coefficients of the linear regression y = ax + b."""
        coef, __ = linear_regression(self[self.ref_col], self[self.verify_col], order=1)
        return coef

    def r(self):
        """Returns the correlation coefficient."""
        return np.corrcoef(self[self.ref_col], self[self.verify_col])[0, 1]

    def n(self, *kwargs):
        """The number of collocated samples."""
        return self.nsamp

    def bias(self, norm=False, **kwargs):
        return stats.bias(x=self[self.ref_col], y=self[self.verify_col], norm=norm)

    bias.__doc__ = set_docstring("bias")

    def rmsd(self, norm=False, **kwargs):
        return stats.rmsd(x=self[self.ref_col], y=self[self.verify_col], norm=norm)

    rmsd.__doc__ = set_docstring("rmsd")

    def mad(self, norm=False, **kwargs):
        return stats.mad(x=self[self.ref_col], y=self[self.verify_col], norm=norm)

    mad.__doc__ = set_docstring("mad")

    def si(self, **kwargs):
        return stats.si(x=self[self.ref_col], y=self[self.verify_col])

    si.__doc__ = set_docstring("si")

    def mrad(self, **kwargs):
        return stats.mrad(x=self[self.ref_col], y=self[self.verify_col])

    mrad.__doc__ = set_docstring("mrad")

    def ks(self, **kwargs):
        return stats.ks(x=self[self.ref_col], y=self[self.verify_col])

    ks.__doc__ = set_docstring("ks")

    def plot_scatter(self, showeq=True, colorbar=False, xlim=None, ylim=None, **kwargs):
        """Scatter plot of model vs observations.

        Args:
            - ``showeq`` (bool): show equality line if True.
            - ``colorbar`` (bool): show colorbar if True.
            - ``xlim`` (tuple): x-limits for axis, if None inferred from data.
            - ``ylim`` (tuple): y-limits for axis, if None inferred from data.
            - ``kwargs``: options to pass to matplotlib plotting method.

        Returns:
            - axis instance.

        Note:
            - If the argument ``c`` is the name of a dataframe column, the values
              of that column are used to color each point (see pandas doc).
            - The ``colorbar`` argument requires ``c`` to be prescribed.
            - If ``c`` is prescribed it is used to sort frame so larger values
              in the scatter are highlighted.

        """
        kwargs = {**DEFAULTS["plot_kwargs"]["scatter"], **kwargs}

        if "ax" not in kwargs:
            fig = plt.figure()
            kwargs["ax"] = fig.add_subplot(111)
        if "c" in kwargs:
            kwargs["colorbar"] = False
            kwargs.pop("color", None)
            df = self.sort_values(by=kwargs["c"], ascending=True)
        else:
            df = self.copy(deep=True)
        df.plot(x=self.ref_col, y=self.verify_col, kind="scatter", **kwargs)
        mappable = [
            chi
            for chi in kwargs["ax"].get_children()
            if isinstance(chi, PathCollection)
        ][-1]
        self.set_xylimit(kwargs["ax"], equal=True, xlim=xlim, ylim=ylim)
        self._set_axis_label(kwargs["ax"])
        if showeq:
            self._show_equality(kwargs["ax"])
        if colorbar and mappable:
            plt.colorbar(mappable, label=kwargs["c"].title())
        return kwargs["ax"]

    def plot_qq(
        self, increment=0.01, showeq=True, xlim=None, ylim=None, label=None, **kwargs
    ):
        """Quantile-quantile plot of model vs observations.

        Args:
            - ``increment`` (float): percentile increment for defining quantiles.
            - ``showeq`` (bool): show equality line if True.
            - ``xlim`` (tuple): x-limits for axis, if None inferred from data.
            - ``ylim`` (tuple): y-limits for axis, if None inferred from data.
            - ``kwargs``: options to pass to matplotlib plotting method.

        Returns:
            - axis instance.

        """
        kwargs = {**DEFAULTS["plot_kwargs"]["qq"], **kwargs}

        if "ax" not in kwargs:
            fig = plt.figure(figsize=(6, 6))
            kwargs["ax"] = fig.add_subplot(111)
        # It will break otherwise:
        kwargs.pop("c", None)

        perc = np.arange(0, 1 + increment, increment)
        qq = self[[self.ref_col, self.verify_col]].dropna().quantile(perc)
        qq.plot(
            x=self.ref_col, y=self.verify_col, kind="scatter", label=label, **kwargs
        )
        self.set_xylimit(kwargs["ax"], equal=True, xlim=xlim, ylim=ylim)
        self._set_axis_label(kwargs["ax"])
        if showeq:
            self._show_equality(kwargs["ax"])
        return kwargs["ax"]

    def plot_scatter_qq(
        self,
        increment=0.01,
        showeq=True,
        xlim=None,
        ylim=None,
        scatter_kw={},
        qq_kw={},
        **kwargs,
    ):
        """Plot scatter and qq of model vs observations.

        Args:
            - ``increment`` (float): percentile increment for defining quantiles.
            - ``showeq`` (bool): show equality line if True.
            - ``xlim`` (tuple): x-limits for axis, if None inferred from data.
            - ``ylim`` (tuple): y-limits for axis, if None inferred from data.
            - ``scatter_kw`` (dict): kwargs to pass to plot_scatter.
            - ``qq_kw`` (dict): kwargs to pass to plot_qq.
            - ``kwargs``: extra options to pass to both plot_scatter and plot_qq.

        Returns:
            - axis instance.

        """
        if "ax" not in kwargs:
            fig = plt.figure(figsize=(6, 6))
            kwargs["ax"] = fig.add_subplot(111)

        scatter_kw = {**DEFAULTS["plot_kwargs"]["scatter"], **scatter_kw, **kwargs}
        qq_kw = {**DEFAULTS["plot_kwargs"]["qq"], **qq_kw, **kwargs}

        self.plot_scatter(showeq=False, **scatter_kw)
        self.plot_qq(increment=increment, showeq=showeq, xlim=xlim, ylim=ylim, **qq_kw)
        return kwargs["ax"]

    def plot_scatter_polar(
        self,
        theta,
        radius,
        colorbar=False,
        cbar_kwargs=dict(pad=0.1, shrink=1.0),
        oceanographic=True,
        show_label=False,
        **kwargs,
    ):
        """Polar scatter plot.

        Args:
            - ``theta`` (str): name of column to use for polar angles.
            - ``radius`` (str): name of column to use for radius.
            - ``colorbar`` (bool): show colorbar if True.
            - ``cbar_pad`` (float): axis fraction for padding colorbar beyond plot.
            - ``oceanographic`` (bool): True for Nautical, False for Cartesian.
            - ``show_label`` (bool): if True theta and radius labels are shown.
            - ``kwargs``: options to pass to matplotlib plotting method.

        Returns:
            - axis instance.

        Note:
            - If the argument ``c`` is the name of a dataframe column, the values
              of that column are used to color each point (see pandas doc).
            - The ``colorbar`` argument requires ``c`` to be prescribed.

        """
        kwargs = {**DEFAULTS["plot_kwargs"]["polar"], **kwargs}

        if "ax" not in kwargs:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="polar")
        else:
            ax = kwargs.pop("ax")
            assert isinstance(ax, PolarAxes), "Axis must be polar."

        if oceanographic:
            ax.set_theta_direction("clockwise")
            ax.set_theta_zero_location("N")

        if "c" in kwargs:
            kwargs["colorbar"] = False
            kwargs.pop("color", None)
            df = self.sort_values(by=kwargs["c"], ascending=True)
        else:
            df = self.copy(deep=True)

        # Angles must be in radians
        df[theta] = np.radians(df[theta])

        df.plot(kind="scatter", x=theta, y=radius, ax=ax, **kwargs)

        mappable = [
            chi for chi in ax.get_children() if isinstance(chi, PathCollection)
        ][-1]

        if not show_label:
            ax.set_xlabel("")
            ax.set_ylabel("")

        if colorbar and mappable:
            if "label" not in cbar_kwargs:
                cbar_kwargs["label"] = kwargs["c"].title()
            plt.colorbar(mappable, **cbar_kwargs)

        return ax

    def plot_density_contour(
        self, binsize=None, showeq=True, colorbar=True, xlim=None, ylim=None, **kwargs
    ):
        """Contour density countour of model vs observations.

        Args:
            - ``binsize`` (float): size of bins over which density is calculated,
              by default calculated such that there are 12 bins.
            - ``showeq`` (bool): show equality line if True.
            - ``colorbar`` (bool): show colorbar if True.
            - ``xlim`` (tuple): x-limits for axis, if None inferred from data.
            - ``ylim`` (tuple): y-limits for axis, if None inferred from data.
            - ``kwargs``: options to pass to matplotlib plotting method.

        Returns:
            - axis instance.

        Note:
            - The colours represent the relative density of datapoints.

        """
        kwargs = {**DEFAULTS["plot_kwargs"]["density_contour"], **kwargs}
        isgrid = kwargs.pop("grid", False)

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)

        binsize = binsize or (self.vmax - self.vmin) / 12
        xx, density = self._density_matrix(binsize=binsize)

        levels = np.arange(0.1, 1.1, 0.1)
        ax.contour(xx, xx, density, colors="0.5", levels=levels, alpha=0.4)
        cobj = ax.contourf(xx, xx, density, levels=levels, **kwargs)

        self.set_xylimit(ax, equal=True, xlim=xlim, ylim=ylim)
        self._set_axis_label(ax)
        if showeq:
            self._show_equality(ax)
        if colorbar:
            plt.colorbar(mappable=cobj)
        ax.grid(isgrid)
        return ax

    def plot_density_scatter(
        self, sample_size=None, showeq=True, colorbar=True, xlim=None, ylim=None, **kwargs
    ):
        """Scatter density plot of model vs observations.

        Args:
            - ``sample_size`` (int): number of points to use for KDE calculation.
            - ``showeq`` (bool): show equality line if True.
            - ``colorbar`` (bool): show colorbar if True.
            - ``xlim`` (tuple): x-limits for axis, if None inferred from data.
            - ``ylim`` (tuple): y-limits for axis, if None inferred from data.
            - ``kwargs``: options to pass to matplotlib plotting method.

        Returns:
            - axis instance.

        Note:
            - The colours represent the relative density of datapoints.

        """
        kwargs = {**DEFAULTS["plot_kwargs"]["density_scatter"], **kwargs}

        ax = kwargs.pop("ax", None)
        is_grid = kwargs.pop("grid")
        if not ax:
            plt.figure(figsize=(6, 6))
            ax = plt.subplot(111)
        xs, ys, zs = self._gaussian_kde(sample_size=sample_size)
        if np.any(~np.isnan(zs)):
            pobj = ax.scatter(xs, ys, c=zs, **kwargs)
        else:
            pobj = ax.scatter(xs, ys, c="black", **kwargs)
            colorbar = False
            logger.warning(
                "No value returned from kde, density scatter cannot "
                "be calculated {}--{}".format(self.ref_col, self.verify_col)
            )
        ax.grid(is_grid)
        self.set_xylimit(ax, equal=True, xlim=xlim, ylim=ylim)
        self._set_axis_label(ax)
        if showeq:
            self._show_equality(ax)
        if colorbar:
            plt.colorbar(mappable=pobj)
        return ax

    def plot_pdf(
        self,
        show_mod=True,
        show_obs=True,
        show_hist=False,
        show_legend=True,
        loc=1,
        xlim=None,
        ylim=None,
        **kwargs,
    ):
        """Probability density function plot.

        Args:
            - ``show_mod`` (bool): if True model timeseries is plotted.
            - ``show_obs`` (bool): if True obs timeseries is plotted.
            - ``show_hist`` (bool): if True normilised histogram is plotted.
            - ``show_legend`` (bool): if True legend is shown.
            - ``loc`` (int, str): location code for legend.
            - ``xlim`` (tuple): x-limits for axis, if None inferred from data.
            - ``ylim`` (tuple): y-limits for axis, if None inferred from data.
            - ``kwargs``: options to pass to matplotlib plotting method.

        Returns:
            - axis instance.

        """
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)

        kwargs_mod = {**DEFAULTS["plot_kwargs"]["pdf_mod"], **kwargs}
        # If color kwarg is provided, use for model only
        kwargs.pop("color", None)
        kwargs_obs = {**DEFAULTS["plot_kwargs"]["pdf_obs"], **kwargs}
        kwargs_hist = {**DEFAULTS["plot_kwargs"]["pdf_hist"], **kwargs}
        kwargs_hist.pop("facecolor", None)

        if show_obs:
            ax = self[self.ref_col].plot(
                kind="density", label=self.ref_label, ax=ax, **kwargs_obs
            )
            if show_hist:
                ax.hist(
                    self[self.ref_col],
                    density=True,
                    facecolor=kwargs_obs["color"],
                    label=None,
                    **kwargs_hist,
                )
        if show_mod:
            ax = self[self.verify_col].plot(
                kind="density", label=self.verify_label, ax=ax, **kwargs_mod
            )
            if show_hist:
                ax.hist(
                    self[self.verify_col],
                    density=True,
                    facecolor=kwargs_mod["color"],
                    color=kwargs_mod["color"],
                    label=None,
                    **kwargs_hist,
                )

        offset = 0.1 * (self.vmax - self.vmin)
        xlim = xlim or (self.vmin - offset, self.vmax + offset)
        ylim = ylim or (0, 1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel(
            "${}$ ({})".format(self.var.title(), self.units), fontsize=FONTSIZE_LABEL
        )
        ax.set_ylabel("Density", fontsize=FONTSIZE_LABEL)

        if show_legend:
            lgd = self.legend(ax=ax, loc=loc)

        return ax

    def plot_cdf(
        self, show_mod=True, show_obs=True, show_legend=True, loc=2, xlim=None, ylim=None, **kwargs
    ):
        """Cumulative density function plot.

        Args:
            - ``show_mod`` (bool): if True model timeseries is plotted.
            - ``show_obs`` (bool): if True obs timeseries is plotted.
            - ``show_legend`` (bool): if True legend is shown.
            - ``loc`` (int, str): location code for legend.
            - ``xlim`` (tuple): x-limits for axis, if None inferred from data.
            - ``ylim`` (tuple): y-limits for axis, if None inferred from data.
            - ``kwargs``: options to pass to matplotlib plotting method.

        Returns:
            - axis instance.

        """
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)

        kwargs_mod = {**DEFAULTS["plot_kwargs"]["cdf_mod"], **kwargs}
        # If color kwarg is provided, use for model only
        kwargs.pop("color", None)
        kwargs_obs = {**DEFAULTS["plot_kwargs"]["cdf_obs"], **kwargs}

        if show_obs:
            x, n = self._get_cdf(self.ref_col)
            ax.step(x, n, label=self.ref_label, **kwargs_obs)
        if show_mod:
            x, n = self._get_cdf(self.verify_col)
            ax.step(x, n, label=self.verify_label, **kwargs_mod)

        xlim = xlim or (self.vmin, self.vmax)
        ylim = ylim or (0, 1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel(
            "${}$ ({})".format(self.var.title(), self.units), fontsize=FONTSIZE_LABEL
        )
        ax.set_ylabel("Cumulative density", fontsize=FONTSIZE_LABEL)

        if show_legend:
            lgd = self.legend(ax=ax, loc=loc)

        return ax

    def plot_timeseries(
        self, show_mod=True, show_obs=True, fill_under_obs=False, tcol="time", **kwargs
    ):
        """Timeseries plot of model and|or observations.

        Args:
            - ``show_mod`` (bool): if True model timeseries is plotted.
            - ``show_obs`` (bool): if True obs timeseries is plotted.
            - ``fill_under_obs`` (bool): if True obs is shown as a fill patch.
            - ``kwargs``: options to pass to matplotlib plotting method.

        Returns:
            - axis instance.

        """
        from dateutil.parser import parse

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            fig = plt.figure(figsize=(14, 6))
            ax = fig.add_subplot(111)
        is_grid = kwargs.pop("grid", True)

        kwargs_mod = {**DEFAULTS["plot_kwargs"]["timeseries_mod"], **kwargs}
        # If color kwarg is provided, use for model only
        kwargs.pop("color", None)
        kwargs_obs = {**DEFAULTS["plot_kwargs"]["timeseries_obs"], **kwargs}
        kwargs_fill = {**DEFAULTS["plot_kwargs"]["timeseries_fill"], **kwargs}

        for key in ["marker"]:  # kwargs that break fill
            kwargs_fill.pop(key, None)

        try:
            dtimes = [t.to_pydatetime() for t in self.index]
        except AttributeError:
            try:
                dtimes = [parse(t) for t in self[tcol]]
            except TypeError:
                dtimes = self[tcol].values

        if show_obs:
            if fill_under_obs:
                kwargs_lgd = {
                    x: kwargs_fill[x] for x in kwargs_fill if x not in ["edgecolor"]
                }
                ax.plot(dtimes, self[self.ref_col], label=self.ref_label, **kwargs_lgd)
                ax.fill_between(x=dtimes, y1=self[self.ref_col], y2=0, **kwargs_fill)
            else:
                ax.plot(dtimes, self[self.ref_col], label=self.ref_label, **kwargs_obs)
        if show_mod:
            ax.plot(
                dtimes, self[self.verify_col], label=self.verify_label, **kwargs_mod
            )
        ax.grid(is_grid)
        ax.set_ylabel(
            "${}$ ({})".format(self.var.title(), self.units), fontsize=FONTSIZE_LABEL
        )
        lgd = self.legend(ax=ax)
        return ax

    def plot_regression(self, x=None, **kwargs):
        """Add regression line to plot.

        Args:
            - ``x`` (tuple): x-values specifying range for regression line, by
              default the axis xlim.
            - ``kwargs``: options to pass to matplotlib plotting method.

        Returns:
            - axis instance.

        Note:
            - this method should be called after another plot so axis limit is set.

        """
        kwargs = {**DEFAULTS["plot_kwargs"]["regression"], **kwargs}

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
        is_grid = kwargs.pop("grid", False)

        x = x or np.array(ax.get_xlim())
        y = self.coeffs[0] * np.array(x) + self.coeffs[1]
        ax.plot(x, y, **kwargs)
        # self.set_xylimit(ax, equal=True)
        ax.grid(is_grid)
        return ax

    def add_regression(
        self, ax, loc=2, show_eqn=True, show_r2=True, show_n=True, **kwargs
    ):
        """Add regression text to plot.

        Args:
            - ``ax`` (object): axis instance to show regression on.
            - ``loc`` (int, str): legend locaction, see options in self.add_text.
            - ``show_eqn`` (bool): if True add equation line to text.
            - ``show_r2`` (bool): if True add r2 line to text.
            - ``show_n`` (bool): if True add n-sample line to text.
            - ``kwargs``: options to pass to matplotlib text method.

        Returns:
            - axis instance.

        """
        kwargs = {**DEFAULTS["text_kwargs"]["regression"], **kwargs}

        text = ""
        if show_eqn:
            text += "$y = {:0.3f} x + {:0.3f}$\n".format(self.coeffs[0], self.coeffs[1])
        if show_r2:
            r2 = np.corrcoef(self[self.ref_col], self[self.verify_col])[0, 1] ** 2
            text += "$R^2 = {:0.3f}$\n".format(r2)
        if show_n:
            text += "$N = {:d}$".format(self.nsamp)
        ax = self.add_text(ax, text, loc=loc, **kwargs)
        return ax

    def add_stats(self, ax, decimals=3, loc=2, stats=DEFAULTS["stats"], **kwargs):
        """Adds stats to a plot axis.

        Args:
            - ``ax`` (object): axis instance to show stats on.
            - ``decimals`` (int): number of decimal places in stats.
            - ``loc`` (int, str): legend locaction, see options in self.add_text.
            - ``stats`` (list): list of stats to display.

        Returns:
            - axis instance.

        Note:
            - stats to display are defined in `self.stats` list. To change the
              defaults, update this list before calling this method.
            - preappend `"n"` to the stat name for normalised stat
              (if supported), e.g., `nbias`, `nrmsd`.

        """
        kwargs = {**DEFAULTS["text_kwargs"]["stats"], **kwargs}

        text = ""
        non_supported = set(stats) - set(DEFAULTS["_stats_table"])
        if non_supported:
            raise ValueError(f"The following stats are supported: {non_supported}")
        for stat in stats:
            if (
                stat.startswith("n")
                and len(stat) > 1
                and getattr(self, f"{stat[1:]}", None)
            ):
                val = getattr(self, f"{stat[1:]}")(normalised=True)
            else:
                val = getattr(self, f"{stat}")()
            text += f"${stat.upper()} = {str(np.around(val, decimals=decimals))}$"
            if stat in ["bias", "mad", "rmsd"]:
                text += f" {self.units}"
            text += "\n"
        ax = self.add_text(ax, text, loc=loc, **kwargs)
        return ax

    def _month_to_season(self, month):
        if month in [12, 1, 2]:
            return "DJF"
        elif month in [3, 4, 5]:
            return "MAM"
        elif month in [6, 7, 8]:
            return "JJA"
        elif month in [9, 10, 11]:
            return "SON"

    def stats_table(self, freq=None, outfile=None, **kwargs):
        """Return DataFrame of verification statistics.

        Args:
            - ``freq`` (str): frequency for calculating stats, by default None
              (full dataframe only). Example valid frequencies:
                - ``M``: month end frequency.
                - ``MS``: month start frequency.
                - ``A``, ``Y``: year end frequency.
                - ``AS``, ``YS``: year start frequency.
            - ``outfile`` (str): name of output file for saving stats to.
            - ``kwargs``: options to pass to pd.to_csv method.

        Returns:
            - Pandas DataFrame with stats summary.

        Note:
            - full frequency options can be checked in: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects.
            - ``season`` is accepted as a frequency, in which case the stats are calculated for each season.

        """
        df = self.copy(deep=True)

        kwargs = {**DEFAULTS["to_csv"], **kwargs}

        ret = self._stats_frame("all")
        if freq is not None:
            if freq == "season":
                df["season"] = df.index.month.map(self._month_to_season)
                grouped = df.groupby("season")
            else:
                grouped = self.groupby(pd.Grouper(freq=freq))
            for group, df in grouped:
                vf = VeriFrame(
                    df, ref_col=self.ref_col, verify_col=self.verify_col, var=self.var
                )
                ret = pd.concat((ret, vf._stats_frame(group)), axis=1)
        if outfile is not None:
            ret.to_csv(outfile, **kwargs)
        return ret

    def pretty_stats_table(self, freq=None, outfile=None, **kwargs):
        """Pretty stats table using tabulate.

        Args:
            - ``freq`` (str): frequency for calculating stats, see self.stats_table.
            - ``outfile`` (str): name of output file for saving pretty table to.
            - ``kwargs``: options to pass to tabulate function.

        Returns:
            - String with stats summary.

        """
        kwargs = {**DEFAULTS["tabulate"], **kwargs}

        stats = self.stats_table(freq=freq, outfile=None)
        table = tabulate(stats, headers="keys", **kwargs)
        if outfile is not None:
            with open(outfile, "w") as stream:
                stream.write(table)
        return table

    def gridstats(self, boxsize, lats=None, lons=None):
        """Calculate the stats for each grid point of given box size (lat,lon).

        Args:
            boxsize (float): Length of grid cells.
            lats (array): Custom latitude array for grid.
            lons (array): Custom longitude array for grid.

        Returns:
            Xarray dataset with gridded stats.

        """
        if not ("lat" in self.columns and "lon" in self.columns):
            raise ValueError("gridstats requires lon, lat columns in VeriFrame.")

        logger.info("Calculating gridded statistics...")

        df = self[[self.ref_col, self.verify_col, "lat", "lon"]].dropna()
        obs = df[self.ref_col]
        model = df[self.verify_col]

        if lats is None:
            latmin = df.lat.min() + (abs(df.lat.min()) % boxsize)
            latmax = df.lat.max() - (abs(df.lat.min()) % boxsize)
            lats = np.arange(latmin, latmax + boxsize, boxsize)

        if lons is None:
            lonmin = df.lon.min() + (abs(df.lon.min()) % boxsize)
            lonmax = df.lon.max() - (abs(df.lon.min()) % boxsize)
            lons = np.arange(lonmin, lonmax + boxsize, boxsize)

        # Calculating gridded sums
        diff = model - obs
        diff2 = model**2 - obs**2
        msdall = diff**2

        n = np.histogram2d(df.lat, df.lon, bins=(lats, lons))[0]
        bias_sum = np.histogram2d(df.lat, df.lon, weights=diff, bins=(lats, lons))[0]
        obs_sum = np.histogram2d(df.lat, df.lon, weights=obs, bins=(lats, lons))[0]
        mod_sum = np.histogram2d(df.lat, df.lon, weights=model, bins=(lats, lons))[0]
        msd_sum = np.histogram2d(df.lat, df.lon, weights=msdall, bins=(lats, lons))[0]

        # Masking
        bias_sum = np.ma.masked_equal(bias_sum, 0)
        obs_sum = np.ma.masked_equal(obs_sum, 0)
        mod_sum = np.ma.masked_equal(mod_sum, 0)
        msd_sum = np.ma.masked_equal(msd_sum, 0)

        msd = msd_sum / n

        # Assert coordinates are matching
        nlat, nlon = n.shape
        if lats.size > nlat:
            lats = lats[:-1]
        if lons.size > nlon:
            lons = lons[:-1]

        # Testing this as coordinates look shifted
        lats += (lats[1] - lats[0]) / 2
        lons += (lons[1] - lons[0]) / 2

        # Assign gridded dataset
        dset = xr.Dataset(coords={"lat": lats, "lon": lons})

        dset["nobs"] = xr.DataArray(n, coords=dset.coords)
        dset["obsmean"] = xr.DataArray(obs_sum / n, coords=dset.coords)
        dset["modmean"] = xr.DataArray(mod_sum / n, coords=dset.coords)
        dset["bias"] = xr.DataArray(bias_sum / n, coords=dset.coords)
        dset["rmsd"] = xr.DataArray(np.sqrt(msd), coords=dset.coords)
        dset["si"] = xr.DataArray(
            data=np.sqrt((msd - dset["bias"] ** 2)) / (np.abs(obs_sum) / n),
            coords=dset.coords,
        )
        dset["nbias"] = xr.DataArray(
            data=dset["bias"] / (np.abs(obs_sum) / n), coords=dset.coords
        )
        dset["nrmsd"] = xr.DataArray(
            data=dset["rmsd"] / (np.abs(obs_sum) / n), coords=dset.coords
        )

        return dset

    @classmethod
    def from_file(
        cls,
        filename,
        kind,
        ref_col="obs",
        verify_col="model",
        var=None,
        circular=False,
        ref_label=None,
        verify_label=None,
        **kwargs,
    ):
        """Alternative constructor to create a ``VeriFrame`` from a file.

        Args:
            - ``filename`` (str): name of CSV file to read.
            - ``kind`` (str): type of file to read from, must correspond to a valid
              `read_{kind}` method in pandas, e.g., `csv`, `pickle`, `excel`, etc.
            - ``ref_col`` (str): name of column with observation values.
            - ``verify_col`` (str): name of column with model values.
            - ``var`` (str): id of variable to verify, 'hs' by default (needs to be
              defined in vardef.yml file).
            - ``circular`` (bool): use True for circular arrays such as directions.
            - ``ref_label`` (str): used for labelling obs in plots if provided,
              otherwise constructed from ref_col, var, units.
            - ``verify_label`` (str): used for labelling model in plots if provided,
              otherwise constructed from verify_col, var, units.
            - ``kwargs``: options to pass to `pandas.read_{kind}` method.

        Returns:
            - VeriFrame instance.

        """
        verify_kw = {"ref_col": ref_col, "verify_col": verify_col}
        if var is not None:
            verify_kw.update({"var": var})
        if circular is not None:
            verify_kw.update({"circular": circular})
        if ref_label is not None:
            verify_kw.update({"ref_label": ref_label})
        if verify_label is not None:
            verify_kw.update({"verify_label": verify_label})

        df = getattr(pd, "read_" + kind)(filename, **kwargs)
        return cls(df, **verify_kw)


class VeriFrameMulti(VeriFrame):

    # These properties will propagate when subclassing
    _metadata = ["ref_col", "ref_label", "verify_cols", "verify_labels", "plot_colors"]

    @property
    def _constructor(self):
        """Ensure sliced VeriFrameMulti will preserve attributes."""
        return VeriFrameMulti

    def __init__(self, *args, **kwargs):
        """VeriFrame with methods for verifying multiple models.

        Arguments are identical to VeriFrame, except for the following:

        required kwarg:
            - ``verify_cols`` (list): list of columns with model values.

        Optional kwarg:
            - ``verify_labels`` (list): list of labels for plotting
              corresponding to verify_cols.

        """
        # Required arguments
        kwargs.setdefault(
            "ref_col", "ref_col has not been assigned"
        )  # Required in both classes
        verify_cols = kwargs.pop(
            "verify_cols", ["verify_cols has not been assigned"]
        )  # Required in VeriFrameMulti
        kwargs.update({"verify_col": verify_cols[0]})  # Required in VeriFrame
        # Optional arguments
        verify_labels = kwargs.pop(
            "verify_labels", verify_cols
        )  # Optional in VeriFrameMulti

        super().__init__(*args, **kwargs)
        self.plot_colors = ["b", "r", "g", "c", "y", "m"]
        self.verify_cols = verify_cols
        self.verify_labels = verify_labels

        assert isinstance(self.verify_cols, list), "verify_cols must be a list"
        assert isinstance(self.verify_labels, list), "verify_labels must be a list"
        assert len(self.verify_cols) == len(
            self.verify_labels
        ), "length of verify_cols and verify_labels do not match"

    def __repr__(self):
        return "<{}>\n{}".format(self.__class__.__name__, str(self))

    def _rename_df_columns(self):
        """rename columns because plot dataframes doesn't recognize labels"""
        newcols = {self.obsname: self.obslabel}
        for i in range(len(self.verify_cols)):
            newcols[self.verify_cols[i]] = self.modlabels[i]
        self = self.rename(columns=newcols)
        self.obsname = self.obslabel
        self.verify_cols = self.modlabels

    def _multi_plot(self, plot_function, **kwargs):
        method = getattr(super(), plot_function)
        ax = kwargs.pop("ax", None)
        loc = kwargs.pop("loc", 4)
        if not ax:
            plt.figure()
            ax = plt.axes()
        ii = 0
        for self.verify_col, self.verify_label in zip(
            self.verify_cols, self.verify_labels
        ):
            ax = method(
                ax=ax, color=self.plot_colors[ii], label=self.verify_label, **kwargs
            )
            ii += 1
        self.legend(loc=loc)
        return ax

    def _multi_plot_obs(self, plot_function, **kwargs):
        method = getattr(super(), plot_function)
        ax = kwargs.pop("ax", None)
        loc = kwargs.pop("loc", 4)
        if not ax:
            plt.figure()
            ax = plt.axes()
        ii = 0
        show_obs = True
        for self.verify_col, self.verify_label in zip(
            self.verify_cols, self.verify_labels
        ):
            ax = method(ax=ax, color=self.plot_colors[ii], show_obs=show_obs, **kwargs)
            ii += 1
            show_obs = False
        try:
            self.legend(loc=loc)
        except:
            pass
        return ax

    def stats_table(self, **kwargs):
        frames = []
        for self.mod_col in self.verify_cols:
            self.verify_col = self.mod_col
            tbm = super().stats_table(**kwargs)
            tbm.columns = [self.mod_col]
            frames += [tbm]
        tb = pd.concat(frames, axis=1)
        return tb

    def plot_qq(self, ax=None, **kwargs):
        """Quantile-quantile plot of models vs observations.

        Args:
            - ``increment`` (float): percentile increment for defining quantiles.
            - ``showeq`` (bool): show equality line if True.
            - ``xlim`` (tuple): x-limits for axis, if None inferred from data.
            - ``ylim`` (tuple): y-limits for axis, if None inferred from data.
            - ``kwargs``: options to pass to matplotlib plotting method.

        Returns:
            - axis instance.

        """
        return self._multi_plot("plot_qq", ax=ax, **kwargs)

    def plot_timeseries(self, ax=None, **kwargs):
        """Timeseries plot of models and|or observations.

        Args:
            - ``show_mod`` (bool): if True model timeseries is plotted.
            - ``show_obs`` (bool): if True obs timeseries is plotted.
            - ``fill_under_obs`` (bool): if True obs is shown as a fill patch.
            - ``kwargs``: options to pass to matplotlib plotting method.

        Returns:
            - axis instance.

        """
        return self._multi_plot_obs("plot_timeseries", ax=ax, **kwargs)

    def plot_pdf(self, ax=None, **kwargs):
        """Probability density function plot.

        Args:
            - ``show_mod`` (bool): if True model timeseries is plotted.
            - ``show_obs`` (bool): if True obs timeseries is plotted.
            - ``show_hist`` (bool): if True normilised histogram is plotted.
            - ``loc`` (int, str): location code for legend.
            - ``xlim`` (tuple): x-limits for axis, if None inferred from data.
            - ``ylim`` (tuple): y-limits for axis, if None inferred from data.
            - ``kwargs``: options to pass to matplotlib plotting method.

        Returns:
            - axis instance.

        """
        return self._multi_plot_obs("plot_pdf", ax=ax, **kwargs)

    def plot_cdf(self, ax=None, **kwargs):
        """Cumulative density function plot.

        Args:
            - ``show_mod`` (bool): if True model timeseries is plotted.
            - ``show_obs`` (bool): if True obs timeseries is plotted.
            - ``loc`` (int, str): location code for legend.
            - ``xlim`` (tuple): x-limits for axis, if None inferred from data.
            - ``ylim`` (tuple): y-limits for axis, if None inferred from data.
            - ``kwargs``: options to pass to matplotlib plotting method.

        Returns:
            - axis instance.

        """
        return self._multi_plot_obs("plot_cdf", ax=ax, **kwargs)

    def plot_set(self, map_buff=20):
        logger.info("   Plotting set ... ")
        nmods = len(self.verify_cols)
        nr = max(nmods + 1, 3)
        nc = 3

        # Set up axes
        self.fig = plt.figure(figsize=(14, 6 * nr))
        mapax = plt.subplot2grid((nr, nc), (0, 0), projection=ccrs.PlateCarree())
        self.plot_map(ax=mapax, buff=map_buff)
        self.tsax = plt.subplot2grid((nr, nc), (0, 1), colspan=2)
        self.denax = plt.subplot2grid((nr, nc), (2, 1), colspan=2)
        self.qqax = plt.subplot2grid((nr, nc), (1, 1), aspect=1)
        self.plot_timeseries(ax=self.tsax, alpha=0.6, lw=2)
        self.plot_pdf(ax=self.denax, alpha=0.6)
        self.plot_qq(ax=self.qqax, alpha=0.6)

        df2taylor(
            self,
            fig=plt.gcf(),
            obslabel=self.ref_col,
            mod_cols=self.verify_cols,
            verify_labels=self.verify_labels,
            rect="%i%i%i" % (nr, nc, 6),
            colors=self.plot_colors,
        )

        ii = 1
        for self.verify_col in self.verify_cols:
            self.scatax = plt.subplot2grid((nr, nc), (ii, 0), aspect=1)
            # self.plot_contour(ax=self.scatax)
            self.plot_scatter(ax=self.scatax, alpha=0.6)
            self.add_stats(ax=self.scatax)
            ii += 1

        plt.tight_layout()
        return

    def plot_set_scatter_density(self, map_buff=20):
        logger.info("   Plotting set scatter density... ")
        nmods = len(self.verify_cols)
        nr = 2 + int(np.ceil(nmods / 3.0))
        nc = 3

        # Set up axes
        self.fig = plt.figure(figsize=(18, 6 * nr))
        mapax = plt.subplot2grid((nr, nc), (0, 0), projection=ccrs.PlateCarree())
        # projection=ccrs.Orthographic(central_longitude=self.lon,
        # central_latitude=self.lat))
        self.plot_map(ax=mapax, buff=map_buff)
        self.tsax = plt.subplot2grid((nr, nc), (0, 1), colspan=2)
        self.denax = plt.subplot2grid((nr, nc), (1, 0))
        self.qqax = plt.subplot2grid((nr, nc), (1, 1), aspect=1)
        self.plot_timeseries(ax=self.tsax, alpha=0.6, lw=2)
        self.plot_pdf(ax=self.denax, alpha=0.6)
        self.plot_qq(ax=self.qqax, alpha=0.6)

        # df2taylor(self.df, fig=plt.gcf(), obslabel=self.obslabel, rect=100*nr+30+((nr-1)*3), colors=self.plot_colors)
        df2taylor(
            self,
            fig=plt.gcf(),
            obslabel=self.ref_col,
            rect="%i%i%i" % (nr, nc, 6),
            colors=self.plot_colors,
        )

        ii = 7
        for self.verify_col in self.verify_cols:
            ax = plt.subplot("%i%i%i" % (nr, nc, ii))
            ax.set_aspect("equal")
            self.plot_density_scatter(ax=ax, alpha=0.6, colorbar=False)
            self.add_regression(ax=ax, show_eqn=False, color="k")
            self.add_stats(ax=ax, loc=1)
            ii += 1

        # plt.tight_layout()
        return
