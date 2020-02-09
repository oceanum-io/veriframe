import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import cmocean

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from onverify.veriframe import VeriFrame


logger = logging.getLogger(__name__)


def plot_gridded_stat(
    darr,
    vmin,
    vmax,
    cmap=cmocean.cm.balance,
    projection=ccrs.PlateCarree(),
    resolution="50m",
    figsize=(9, 7),
    label=f"$Bias$ $(m)$",
):
    """Plot gridded stats on a map."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection)
    fig.subplots_adjust(hspace=0, wspace=0, top=1.0, left=0.1)

    # Pcolor layer
    pobj = ax.pcolormesh(
        darr.lon,
        darr.lat,
        np.ma.masked_invalid(darr),
        transform=projection,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        # levels=np.arange(-vlim, vlim+vint, vint),
        # extend="both"
    )

    # Colorbar
    posn = ax.get_position()
    cax = fig.add_axes([posn.x0, posn.y0 - 0.1, posn.width, 0.04])
    cbar = plt.colorbar(
        pobj,
        cax=cax,
        orientation="horizontal",
        label=label,
        # ticks=np.arange(-0.4, 0.5, 0.1)
    )

    def resize_colobar(event):
        plt.draw()
        posn = ax.get_position()
        cax.set_position([posn.x0, posn.y0 - 0.1, posn.width, 0.04])

    fig.canvas.mpl_connect("resize_event", resize_colobar)

    # Axis settings
    ax.coastlines(resolution=resolution, color="black", facecolor="black")
    land = cfeature.LAND.with_scale(resolution)
    ax.add_feature(land, facecolor="0.8", zorder=10)

    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.ylabels_left = True
    gl.n_steps = 4
    gl.xlocator = mticker.MaxNLocator(nbins=5)
    gl.ylocator = mticker.MaxNLocator(nbins=5)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER


def creat_plots_from_gbq(dset, project_id, **kwargs):
    """Create standard onverify plots from GBQ colocs table.

    Args:
        dset (str): GBQ table name.
        project_id (str): Name of project where GBQ table is created.
        kwargs: kwarg options to pass on to create_plots.

    """
    vf = VeriFrame.from_gbq(dset=dset, project_id=project_id, columns="minimum")
    return create_plots(vf, **kwargs)


def create_plots(vf, plotdir="./plots", boxsize=2, binsize=0.2, proj="Robinson"):
    """Standard plots of colocations.

    Args:
        vf (obj): VeriFrame instance to plot.
        plotdir (str): Directory for saving figures.
        boxsize (float): Grid size in gridded stats dataset calculated from colocs.
        binsize (float): Bin size for defining scatter contourf plot.
        proj (str): Projection for maps, not implemented yet.

    Creates:
        * scatter + qq.
        * scatter-contourf + regression.
        * gridded stats (bias, nbias, rmsd, nrmsd, si, nobs).
        * stats txt file.

    """
    logger.info(f"Saving output to {plotdir}")
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    vf.stats = ["bias", "rmsd", "si", "n"]
    dset = vf.gridstats(boxsize=0.5)

    # Stats table
    vf.stats_table(outfile=os.path.join(plotdir, "stats_table.txt"))

    # Scatter / qq
    logger.info("Plotting scatter-qq")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    vf.plot_scatter_qq(ax=ax)
    vf.add_stats(ax=ax)
    plt.savefig(os.path.join(plotdir, "scatter_qq.png"), bbox_inches="tight")
    plt.close()

    # Density / regression
    logger.info("Plotting density scatter with regression")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    vf.plot_density_contour(
        ax=ax, colorbar=False, binsize=binsize, cmap=cmocean.cm.thermal
    )
    vf.plot_regression(ax=ax, color="red", linewidth=1)
    vf.add_stats(ax=ax)
    vf.add_regression(ax=ax, loc=4, show_n=False)
    plt.savefig(os.path.join(plotdir, "density_regression.png"), bbox_inches="tight")
    plt.close()

    # Bias map
    logger.info("Plotting gridded bias")
    plot_gridded_stat(
        dset["bias"],
        vmin=-0.45,
        vmax=0.45,
        cmap=cmocean.cm.balance,
        projection=ccrs.PlateCarree(),
        resolution="50m",
        figsize=(9, 6),
        label=f"$Bias$ $(m)$",
    )
    plt.savefig(os.path.join(plotdir, "grid_bias.png"), bbox_inches="tight")
    plt.close()

    # RMSD map
    logger.info("Plotting gridded RMSD")
    plot_gridded_stat(
        dset["rmsd"],
        vmin=0.0,
        vmax=0.5,
        cmap=cmocean.cm.thermal,
        projection=ccrs.PlateCarree(),
        resolution="50m",
        figsize=(9, 6),
        label=f"$RMSD$ $(m)$",
    )
    plt.savefig(os.path.join(plotdir, "grid_rmsd.png"), bbox_inches="tight")
    plt.close()

    # SI map
    logger.info("Plotting gridded SI")
    plot_gridded_stat(
        dset["si"],
        vmin=0.0,
        vmax=0.2,
        cmap=cmocean.cm.thermal,
        projection=ccrs.PlateCarree(),
        resolution="50m",
        figsize=(9, 6),
        label=f"$Scatter$ $Index$",
    )
    plt.savefig(os.path.join(plotdir, "grid_si.png"), bbox_inches="tight")
    plt.close()

    # NBIAS map
    logger.info("Plotting gridded normalised bias")
    plot_gridded_stat(
        dset["nbias"],
        vmin=-0.2,
        vmax=0.2,
        cmap=cmocean.cm.balance,
        projection=ccrs.PlateCarree(),
        resolution="50m",
        figsize=(9, 6),
        label=f"$Normalised$ $bias$",
    )
    plt.savefig(os.path.join(plotdir, "grid_nbias.png"), bbox_inches="tight")
    plt.close()

    # NRMSD map
    logger.info("Plotting gridded normalised RMSD")
    plot_gridded_stat(
        dset["nrmsd"],
        vmin=0,
        vmax=0.2,
        cmap=cmocean.cm.thermal,
        projection=ccrs.PlateCarree(),
        resolution="50m",
        figsize=(9, 6),
        label=f"$Normalised$ $RMSD$",
    )
    plt.savefig(os.path.join(plotdir, "grid_nrmsd.png"), bbox_inches="tight")
    plt.close()

    # N map
    logger.info("Plotting number of observations")
    plot_gridded_stat(
        dset["nobs"],
        vmin=0,
        vmax=None,
        cmap=cmocean.cm.dense,
        projection=ccrs.PlateCarree(),
        resolution="50m",
        figsize=(9, 6),
        label=f"$Number$ $of$ $colocations$",
    )
    plt.savefig(os.path.join(plotdir, "grid_nobs.png"), bbox_inches="tight")
    plt.close()

    return vf


if __name__ == "__main__":

    logger.setLevel("info")

    dset = "wave.weuro_st6_03_debia097"
    project_id = "oceanum-dev"
    creat_plots_from_gbq(dset, project_id)

    # pkl = "/source/paper-swan-st6/data/altverify/nweuro/new/weuro_st6_03_debias097/colocs/201601.pkl"
    # vf = VeriFrame.from_file(filename=pkl, kind="pickle")
    # create_plots(vf)
    # plt.show()
