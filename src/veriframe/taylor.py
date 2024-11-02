#!/usr/bin/env python
# Copyright: This document has been placed in the public domain.

"""
Taylor diagram (Taylor, 2001) test implementation.

http://www-pcmdi.llnl.gov/about/staff/Taylor/CV/Taylor_diagram_primer.htm
"""

__version__ = "Time-stamp: <2012-02-17 20:59:35 ycopin>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

import numpy as np
import matplotlib.pyplot as plt
import logging
import os


class TaylorDiagram(object):
    """Taylor diagram: plot model standard deviation and correlation
    to reference (data) sample in a single-quadrant polar plot, with
    r=stddev and theta=arccos(correlation).
    """

    def __init__(self, refstd, maxstd, fig=None, rect=111, label="_", **kwargs):
        """Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using mpl_toolkits.axisartist.floating_axes. refstd is
        the reference standard deviation to be compared to.
        maxstd = maximum acceptable std
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd  # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.concatenate((np.arange(10) / 10.0, [0.95, 0.99]))
        tlocs = np.arccos(rlocs)  # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)  # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent
        self.smin = 0
        self.smax = maxstd

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi / 2, self.smin, self.smax),  # 1st quadrant
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

        # Contours along standard deviations
        ax.grid(False)

        self._ax = ax  # Graphical axes
        self.ax = ax.get_aux_axes(tr)  # Polar coordinates

        # Add reference point and stddev contour
        logging.debug("Reference std: %s" % self.refstd)
        (l,) = self.ax.plot([0], self.refstd, "k*", ls="", ms=10, label=label)
        t = np.linspace(0, np.pi / 2)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, "k--", label="_")

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """Add sample (stddev,corrcoeff) to the Taylor diagram. args
        and kwargs are directly propagated to the Figure.plot
        command."""

        (l,) = self.ax.plot(
            np.arccos(corrcoef), stddev, *args, **kwargs
        )  # (theta,radius)
        self.samplePoints.append(l)

        return l

    def add_contours(self, levels=5, **kwargs):
        """Add constant centered RMS difference contours."""

        rs, ts = np.meshgrid(
            np.linspace(self.smin, self.smax), np.linspace(0, np.pi / 2)
        )
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd**2 + rs**2 - 2 * self.refstd * rs * np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours


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
    **kwargs
):
    refstd = df[obslabel].std(ddof=1)  # Reference standard deviation
    if not mod_cols:
        mod_cols = df.columns
    if not mod_labels:
        mod_labels = mod_cols
    mapping = dict(zip(mod_cols, mod_labels))
    if np.isnan(refstd):
        logging.error("Reference stddev is NaN")
        return
    if refstd == 0.0:
        logging.error("Reference stddev is 0.0")
        return
    if dia is None:
        dia = TaylorDiagram(refstd, 1.5 * refstd, fig=fig, label=label, **kwargs)
    colors = colors or plt.matplotlib.cm.jet(
        np.linspace(0, 1, (df[mod_cols].isnull().all() == False).sum())
    )

    ii = 0
    for col in df:
        if col in [obslabel, "site", "lon", "lat"]:
            continue
        if df[col].isnull().all():
            continue
        if col not in mod_cols:
            continue
        stddev = df[col].std(ddof=1)
        R = df[obslabel].corr(df[col])
        if np.isnan(stddev):
            continue

        dia.add_sample(
            stddev, R, marker="o", ls="", c=colors[ii], label="%s" % (mapping[col])
        )
        ii += 1

    # Add RMS contours, and label them
    contours = dia.add_contours(colors="0.5")
    plt.clabel(contours, inline=1, fontsize=10)

    # Add a figure legend
    if legend:
        plt.legend(
            dia.samplePoints,
            [p.get_label() for p in dia.samplePoints],
            numpoints=1,
            prop=dict(size="small"),
            loc="upper right",
        )
    if plotdir:
        if not os.path.isdir(plotdir):
            os.makedirs(plotdir)
        plt.savefig(os.path.join(plotdir, label + ".png"), bbox_inches="tight")
        plt.close()

    return dia
