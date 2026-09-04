#!/usr/bin/env python
# Copyright: This document has been placed in the public domain.

"""Taylor diagram (Taylor, 2001).

A single plot showing, for each model, its standard deviation (radius), its
correlation with the reference (angle) and -- implicitly, by the law of
cosines -- the centred RMS difference (distance from the reference point).
The three are not independent:

.. math::
    E'^2 = \\sigma_{ref}^2 + \\sigma_{mod}^2
           - 2\\,\\sigma_{ref}\\sigma_{mod}\\cos(\\theta)

so a diagram places a model by two of them and reads the third off the grey
contours. Note the diagram says nothing about the *bias*: two models with
identical spread and correlation but metres apart in the mean plot on the same
point. Read it alongside ``bias``.

Reference:
    Taylor, K.E. (2001). Summarizing multiple aspects of model performance in
    a single diagram. *Journal of Geophysical Research*, 106(D7), 7183-7192.

Originally Yannick Copin's public-domain implementation
(http://www-pcmdi.llnl.gov/about/staff/Taylor/CV/Taylor_diagram_primer.htm),
since fixed and extended: points that fall outside the axes are no longer
dropped silently, negative correlations are supported, the figure passed in is
the one drawn on, and the default colours are not ``jet``.
"""

__version__ = "Time-stamp: <2012-02-17 20:59:35 ycopin>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import logging
import os
import warnings


class TaylorDiagram(object):
    """Taylor diagram: plot model standard deviation and correlation
    to reference (data) sample in a single-quadrant polar plot, with
    r=stddev and theta=arccos(correlation).
    """

    def __init__(self, refstd, maxstd, fig=None, rect=111, label="_",
                 negative=False, std_grid=True, **kwargs):
        """Set up the Taylor diagram axes.

        Args:
            - ``refstd`` (float): standard deviation of the reference.
            - ``maxstd`` (float): outer limit of the radial axis. Samples
              beyond it cannot be drawn -- :meth:`add_sample` warns rather
              than dropping them silently.
            - ``fig`` (Figure): figure to draw on; created if None.
            - ``rect`` (int): subplot spec.
            - ``label`` (str): legend label for the reference point.
            - ``negative`` (bool): extend the diagram to the half plane
              (0 to 180 deg) so negatively correlated samples can be shown.
              A single quadrant cannot represent them.
            - ``std_grid`` (bool): draw arcs of constant standard deviation.
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd  # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        self.negative = bool(negative)
        rlocs = np.concatenate((np.arange(10) / 10.0, [0.95, 0.99]))
        if self.negative:
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        tlocs = np.arccos(rlocs)  # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)  # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent
        self.smin = 0
        self.smax = maxstd

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi if self.negative else np.pi / 2,
                      self.smin, self.smax),
            grid_locator1=gl1,
            tick_formatter1=tf1,
        )

        if fig is None:
            fig = plt.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")  # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")  # "Y axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")

        ax.axis["bottom"].set_visible(False)  # Useless

        ax.grid(std_grid)

        self._ax = ax  # Graphical axes
        self.ax = ax.get_aux_axes(tr)  # Polar coordinates

        # Add reference point and stddev contour
        logging.debug("Reference std: %s" % self.refstd)
        (l,) = self.ax.plot([0], self.refstd, "k*", ls="", ms=10, label=label)
        t = np.linspace(0, np.pi if self.negative else np.pi / 2)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, "k--", label="_")

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """Add a sample to the diagram.

        `args` and `kwargs` are passed to ``plot``. A sample that cannot be
        represented on the current axes -- a negative correlation on a
        single-quadrant diagram, or a standard deviation beyond ``maxstd`` --
        is warned about rather than drawn where it cannot be seen. Silently
        losing a model off the edge is the failure mode this guards.
        """
        label = kwargs.get("label", "sample")
        if np.isnan(stddev) or np.isnan(corrcoef):
            warnings.warn(f"Taylor: {label!r} has NaN stddev or correlation, "
                          "not plotted", stacklevel=2)
            return None
        if corrcoef < 0 and not self.negative:
            warnings.warn(
                f"Taylor: {label!r} has correlation {corrcoef:.3f} < 0, which "
                "a single-quadrant diagram cannot show. Rebuild with "
                "negative=True.", stacklevel=2)
        if stddev > self.smax:
            warnings.warn(
                f"Taylor: {label!r} has stddev {stddev:.3g} beyond the axis "
                f"limit {self.smax:.3g} and will fall outside the frame. "
                "Increase maxstd.", stacklevel=2)

        (l,) = self.ax.plot(
            np.arccos(corrcoef), stddev, *args, **kwargs
        )  # (theta,radius)
        self.samplePoints.append(l)

        return l

    def add_contours(self, levels=5, **kwargs):
        """Add constant centered RMS difference contours."""

        rs, ts = np.meshgrid(
            np.linspace(self.smin, self.smax),
            np.linspace(0, np.pi if self.negative else np.pi / 2),
        )
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd**2 + rs**2 - 2 * self.refstd * rs * np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours


#: Default categorical colours for samples. Validated against a light surface
#: for colour-vision deficiency; ``jet`` was the previous default and is a
#: rainbow ramp, which encodes no categorical order and is not CVD-safe.
SAMPLE_COLORS = ["#c9512f", "#6a4c93", "#2f8f5b", "#c98a1e", "#a05a3a",
                 "#2a7ea8"]


def _sample_colors(n):
    """One distinct colour per sample.

    :data:`SAMPLE_COLORS` while they last; beyond that, ``tab20`` rather than
    cycling, so two models never share a colour in a calibration sweep.
    """
    if n <= len(SAMPLE_COLORS):
        return SAMPLE_COLORS[:n]
    return [plt.cm.tab20(i % 20) for i in range(n)]


def df2taylor(
    df,
    obslabel="obs",
    mod_cols=[],
    mod_labels=None,
    fig=None,
    label="Reference",
    colors=None,
    plotdir=None,
    dia=None,
    legend=True,
    maxstd=None,
    contour_fmt="%.2f",
    units="",
    legend_kw=None,
    **kwargs
):
    """Plot every model column of `df` against `obslabel` on a Taylor diagram.

    Args:
        - ``df`` (DataFrame): one column of observations, one per model.
        - ``obslabel`` (str): name of the reference column.
        - ``mod_cols`` (list): model columns to plot; all others if empty.
        - ``mod_labels`` (list): display names, defaulting to ``mod_cols``.
        - ``fig`` (Figure): figure to draw on. Everything is drawn on *this*
          figure rather than on pyplot's current one.
        - ``label`` (str): legend label for the reference point.
        - ``colors`` (list): one colour per model, defaulting to
          :data:`SAMPLE_COLORS` in order (``tab20`` beyond six models).
        - ``maxstd`` (float): radial limit. Defaults to comfortably enclosing
          the reference and every model, so no sample falls off the edge.
        - ``contour_fmt`` (str): format for the centred-RMS contour labels.
        - ``units`` (str): appended to the axis label, e.g. ``"m"``.
        - ``legend_kw`` (dict): passed to the diagram's ``legend``. By
          default it sits in the upper left: the high-standard-deviation,
          low-correlation region of a Taylor diagram is empty for any usable
          model, whereas the upper-right corner is where the "Correlation"
          axis label lives, so a legend there collides with it.

    Returns:
        The :class:`TaylorDiagram`, or None if the reference is degenerate.

    Note:
        A Taylor diagram encodes standard deviation, correlation and centred
        RMS difference. It does **not** encode bias -- two runs differing only
        in their mean plot at the same point.
    """
    refstd = df[obslabel].std(ddof=1)  # Reference standard deviation
    if not mod_cols:
        mod_cols = [
            c for c in df.columns
            if c not in (obslabel, "site", "lon", "lat")
            and is_numeric_dtype(df[c])
        ]
    if not mod_labels:
        mod_labels = mod_cols
    mapping = dict(zip(mod_cols, mod_labels))
    if np.isnan(refstd):
        logging.error("Reference stddev is NaN")
        return
    if refstd == 0.0:
        logging.error("Reference stddev is 0.0")
        return

    cols = [c for c in mod_cols if c in df.columns and not df[c].isnull().all()]
    stds = [df[c].std(ddof=1) for c in cols]
    corrs = [df[obslabel].corr(df[c]) for c in cols]

    # Scale the radial axis to the data rather than a fixed multiple of the
    # reference: a model with more than 1.5x the reference spread used to be
    # drawn outside the frame and simply not appear.
    if maxstd is None:
        maxstd = 1.15 * max([refstd] + [s for s in stds if np.isfinite(s)])
    negative = any(c < 0 for c in corrs if np.isfinite(c))

    if dia is not None:
        fig = dia._ax.figure
    else:
        if fig is None:
            fig = plt.figure()
        dia = TaylorDiagram(refstd, maxstd, fig=fig, label=label,
                            negative=negative, **kwargs)
    if colors is None:
        colors = _sample_colors(len(cols))

    for i, col in enumerate(cols):
        if np.isnan(stds[i]):
            continue
        dia.add_sample(stds[i], corrs[i], marker="o", ms=8, ls="",
                       c=colors[i], mec="white", mew=0.8,
                       label="%s" % (mapping.get(col, col)))

    # Centred RMS difference contours. Labelled, and named in the legend so
    # the grey rings are not left for the reader to guess at.
    contours = dia.add_contours(colors="0.5", linewidths=0.8)
    dia.ax.clabel(contours, inline=1, fontsize=8, fmt=contour_fmt)
    unit = f" ({units})" if units else ""
    dia._ax.axis["left"].label.set_text(f"Standard deviation{unit}")
    dia._ax.axis["top"].label.set_text("Correlation")
    # Name the grey rings in the legend rather than as figure text: an
    # axisartist FloatingSubplot does not report figure-level text to a tight
    # bounding box, so a caption placed there is silently cropped away. Kept
    # short so the legend still fits in the empty corner outside the arc.
    handles = list(dia.samplePoints) + [
        plt.Line2D([], [], color="0.5", lw=0.8, label=f"Centred RMSD{unit}")
    ]

    # Draw on the diagram's own axes and figure. plt.legend/savefig act on
    # pyplot's *current* figure and axes, which in a batch build or a
    # multi-panel figure is not necessarily this one; fig.legend would anchor
    # the legend to the whole figure rather than to this subplot.
    if legend:
        opts = {"numpoints": 1, "prop": {"size": "small"}, "loc": "upper left",
                "frameon": True, "framealpha": 0.85, "edgecolor": "none"}
        opts.update(legend_kw or {})
        dia._ax.legend(handles, [h.get_label() for h in handles], **opts)
    if plotdir:
        if not os.path.isdir(plotdir):
            os.makedirs(plotdir)
        fig.savefig(os.path.join(plotdir, label + ".png"), bbox_inches="tight")
        plt.close(fig)

    return dia
