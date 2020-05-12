import argparse
import json
import logging
import os
import resource
from datetime import datetime, timedelta
from functools import partial
from glob import glob
from multiprocessing import Pool

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from scipy.ndimage import map_coordinates
from scipy.stats import mstats

from retrying import retry
from pandas_gbq.gbq import to_gbq, GenericGBQException

# from onverify.site_base import Verify as VerifyBase
from onverify.veriframe import VeriFrame
from onverify.stats import bias, rmsd, si
from onverify.io.gbq import GBQAlt

# from onverify.io.file import open_netcdf
from oncore.dataio import get
from ontake.ontake import Ontake

# from verify.core.calc_nrt_pairs import load_nrt
logging.basicConfig(level=logging.INFO)

plt.rcParams["image.cmap"] = "viridis"

COREFIELDS = ["time", "lat", "lon", "satellite"]
COLOCFIELDS = COREFIELDS + ["obs", "model"]

RETRY_KWARGS = dict(
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    stop_max_attempt_number=5,
)

WIND_COMPONENT_NAMES = [
    ["ugrd10m", "vgrd10m"],
    ["uwnd", "vwnd"],
    ["xwnd", "ywnd"],
    ["u10", "v10"],
]


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """To show default values in help and preserve line break in epilog"""

    pass


class Parser(object):
    def __init__(self):
        super(Parser, self).__init__()
        self.run()

    def define_parser(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "ncglob", help="Glob pattern for netcdf files to process"
        )
        self.parser.add_argument(
            "-o", "--outdir", default="./output", help="path for output files"
        )
        self.parser.add_argument(
            "-nrt",
            "--nrt",
            type=str,
            default="",
            help="Use raw or ifremer near real time data. By defalt cersat is used",
        )
        self.parser.add_argument(
            "-obs", "--obsregex", default=None, help="Observations locations"
        )
        self.parser.add_argument(
            "-b",
            "--boxsize",
            type=float,
            default=2,
            help="Lat/lon grid size for stats calculations",
        )
        self.parser.add_argument(
            "-vmin", "--vmin", type=float, default=0, help="Plot min"
        )
        self.parser.add_argument(
            "-vmax", "--vmax", type=float, default=6, help="Plot max"
        )
        self.parser.add_argument(
            "-x1",
            "--lonmin",
            type=float,
            default=None,
            help="lonmin for delimitting verification area",
        )
        self.parser.add_argument(
            "-x2",
            "--lonmax",
            type=float,
            default=None,
            help="lonmax for delimitting verification area",
        )
        self.parser.add_argument(
            "-y1",
            "--latmin",
            type=float,
            default=None,
            help="latmin for delimitting verification area",
        )
        self.parser.add_argument(
            "-y2",
            "--latmax",
            type=float,
            default=None,
            help="latmin for delimitting verification area",
        )
        self.parser.add_argument(
            "-n",
            "--ncores",
            type=int,
            default=1,
            help="Number of cores to use in colocation calculations",
        )
        self.parser.add_argument(
            "-m",
            "--model_vars",
            default=[],
            nargs="*",
            help="model variables to interpolate collocs dataframe. First should be hs or wndsp",
        )
        self.parser.add_argument(
            "-k",
            "--keepna",
            action="store_true",
            help="choose it to keep nans after collocating obs and model",
        )

    def parsecla(self):
        self.args = self.parser.parse_args()

    def action_args(self):
        assert glob(self.args.ncglob), "No netcdf file located to process in %s" % (
            self.args.ncglob
        )
        colocsdir = os.path.join(self.args.outdir, "colocs")
        plotdir = os.path.join(self.args.outdir, "plots")
        if self.args.nrt == "raw":
            # use raw NRT data from several sources
            cla = VerifyNRTraw
        elif self.args.nrt == "ifremer":
            # use ifremer preprocessed NRT data
            cla = VerifyNRT
        else:
            # use cersat altimetry data (higher quality than NRT but not available for recent times)
            cla = Verify
        calcColocs(
            self.args.ncglob,
            obsregex=self.args.obsregex,
            outdir=colocsdir,
            cla=cla,
            model_vars=self.args.model_vars,
            lonmin=self.args.lonmin,
            lonmax=self.args.lonmax,
            latmin=self.args.latmin,
            latmax=self.args.latmax,
            dropna=not self.args.keepna,
            pool=self.args.ncores,
        )
        createPlots(
            hdfglob=colocsdir + "/*pkl",
            plotdir=plotdir,
            boxsize=self.args.boxsize,
            vmin=self.args.vmin,
            vmax=self.args.vmax,
        )

    def run(self):
        self.define_parser()
        self.parsecla()
        self.action_args()


class Verify(object):
    def __init__(
        self,
        ncglob,
        obsregex=None,
        test=False,
        modvar=None,
        model_vars=[],
        latmin=None,
        latmax=None,
        lonmin=None,
        lonmax=None,
        logger=logging,
        boxsize=2,
        proj="PlateCarree",
        clat=0,
        scalefactor=1,
        plotdir="./plots",
        savestats=False,
        savemonthlystats=False,
        savemonthlyplots=False,
        ** kwargs,
    ):
        """
        - model_vars :: model variables to interpolate onto colocs
        """
        self.logger = logger
        self.test = test
        self.plotdir = plotdir
        self.savemonthlystats = savemonthlystats
        self.savemonthlyplots = savemonthlyplots
        self.savestats = savestats

        self.ncglob = ncglob
        self.obsregex = obsregex or "/net/datastor1/data/obs/cersat/%Y/wm_%Y%m%d.nc"

        if not model_vars:
            model_vars = [modvar if modvar else "hs"]
        self.model_vars = set(model_vars)

        if modvar:
            self.modvar = modvar
        else:
            self.modvar = model_vars[0]

        self.units = "m"
        self.boxsize = boxsize
        self.proj = proj
        self.clat = clat
        self.scalefactor = scalefactor

        self.latmin = latmin
        self.latmax = latmax
        self.lonmin = lonmin
        self.lonmax = lonmax

        # filled in loadModel()
        self.model = []
        self.latname = []
        self.lonname = []
        self.t0 = []
        self.t1 = []
        self.tstep = []
        self.latres = []
        self.lonres = []
        self.modlatmin = []
        self.modlatmax = []
        self.modlonmin = []
        self.modlonmax = []

        # # filled in loadObs()
        # self.altnames = []
        # self.obs = []

        # # filled in calcColocs() or loadColocs()
        # self.df = []
        df = self.calcColocs()
        self.df = VeriFrame(df, ref_col="obs", verify_col="model", **kwargs)

    def calcColocs(self, dropvars=None):
        self.loadModel()
        self.loadObs(dropvars=dropvars)
        self.interpModel()
        return self.createColocs()

    def _swap_longitude_convention(self, longitudes):
        """Swap longitudes between 0 -- 360 and -180 -- 180 conventions."""
        if self._is_180(longitudes):
            return longitudes % 360
        elif self._is_360(longitudes):
            longitudes[longitudes > 180] = longitudes[longitudes > 180] - 360
        return longitudes

    def _is_180(self, array):
        """True if longitudes are in -180 -- 180 convention."""
        if array.min() < 0 and array.max() <= 180:
            return True
        return False

    def _is_360(self, array):
        """True if longitudes are in -180 -- 180 convention."""
        if array.min() >= 0 and array.max() <= 360:
            return True
        return False

    def _sel_model(self):
        """Slice model area taking care of longitude wrapping and inverted latitudes."""
        lon_range = np.array([self.lonmin, self.lonmax])
        lat_range = np.array([self.latmin, self.latmax])

        # ERA5 has latitude in descending order
        if self.model[self.latname][0] > self.model[self.latname][1]:
            lat_range = lat_range[::-1]

        slicing_dict = {
            self.latname: slice(*lat_range),
            self.lonname: slice(*lon_range),
        }

        # Take care of longitude wrapping inconsistencies
        if self._is_360(lon_range) == self._is_360(self.model[self.lonname].values):
            self.model = self.model.sel(**slicing_dict)
        else:
            if self._is_180(lon_range):
                left, right = 0, 360
            else:
                left, right = -180, 180
            # Left interval
            lon_range = self._swap_longitude_convention(lon_range)
            slicing_dict[self.lonname] = slice(lon_range[0], right)
            dsout = self.model.sel(**slicing_dict)
            # Right interval
            slicing_dict[self.lonname] = slice(left, lon_range[1])
            self.model = xr.concat(
                (dsout, self.model.sel(**slicing_dict)), dim=self.lonname
            )
            self.model[self.lonname].values = self._swap_longitude_convention(
                self.model[self.lonname].values
            )

    def loadModel(self):
        ncfiles = glob(self.ncglob)
        infonc = xr.open_dataset(ncfiles[0])
        extra_vars = {
            "uwnd",
            "vwnd",
            "ugrd10m",
            "vgrd10m",
            "ucur",
            "vcur",
            "u10",
            "v10",
        }  # vector variables can be built later
        excluded = set(infonc.data_vars) - self.model_vars - extra_vars

        self.logger.info("Loading model data %s \n" % "\n\t".join(glob(self.ncglob)))
        model = xr.open_mfdataset(ncfiles, drop_variables=excluded, engine="netcdf4")
        if self.test:
            self.logger.info(" Using first 10 timesteps only")
            model = model.isel(time=slice(None, 10))
        dsettime = model.time.to_pandas()
        inodup = np.where(dsettime.duplicated() == False)[0]
        self.model = model.isel(time=inodup)
        self._check_model_data()

    def _check_model_data(self):

        if (self.lonmin is None) != (self.lonmax is None):
            raise NotImplementedError(
                "Either provide both (lonmin, lonmax) or neither of them."
            )

        # Different in WW3 and SWAN
        if "latitude" in self.model.dims.keys():
            self.latname = "latitude"
            self.lonname = "longitude"
        else:
            self.latname = "lat"
            self.lonname = "lon"

        # Constrain model to prescribed rectangle
        x0, x1 = self.model[self.lonname][[0, -1]].values
        y0, y1 = self.model[self.latname][[0, -1]].values
        self.lonmin = self.lonmin or x0
        self.lonmax = self.lonmax or x1
        self.latmin = self.latmin or y0
        self.latmax = self.latmax or y1
        self._sel_model()

        # ensure bbox is within model grid
        if self.model[self.lonname].size == 0 or self.model[self.latname].size == 0:
            raise ValueError(
                f"The model bounds are lon=({x0}, {x1}), lat=({y0}, {y1}) but you are "
                f"trying to slice between lon=({self.lonmin}, {self.lonmax}), "
                f"lat=({self.latmin}, {self.latmax}). Not going to work!"
            )

        # Determine relevant data attributes
        dsettime = self.model.time.to_pandas()
        self.t0 = dsettime[0]
        self.t1 = dsettime[-1]
        self.tstep = dsettime[1] - self.t0

        self.latres = np.abs(
            self.model[self.latname].values[0] - self.model[self.latname].values[1]
        )
        self.lonres = np.abs(
            self.model[self.lonname].values[0] - self.model[self.lonname].values[1]
        )
        self.modlatmin = self.model[self.latname].values.min()
        self.modlatmax = self.model[self.latname].values.max()
        self.modlonmin = self.model[self.lonname].values.min()
        self.modlonmax = self.model[self.lonname].values.max()

    def loadObs(self, interval=timedelta(hours=24), dropvars=None):
        self.logger.info("Loading observations")
        self.altnames = {
            1: "ERS1",
            2: "ERS2",
            3: "ENVISAT",
            4: "TOPEX",
            5: "POSEIDON",
            6: "JASON1",
            7: "GFO",
            8: "JASON2",
            9: "CRYOSAT",
        }
        obslist = []
        dttmp = self.t0
        #'swhcor', #'lat', #'lon', #'time' #'satellite',
        obsnames = {"hs": "swh_adjusted", "wndsp": "wind_speed_cor"}
        obsvar = obsnames[self.modvar]
        obsvars = [obsnames[v] for v in {"hs", "wndsp"}.intersection(self.model_vars)]
        if not dropvars:
            dropvars = set(
                [
                    "swhcor",
                    "swhstd",
                    "sigma0_cal",
                    "pass_number",
                    "mes",
                    "sigma0",
                    "sigma0std",
                    "sigma0second",
                    "cycle",
                    "swh",
                    "wind_speed",
                    "wind_speed_cor",
                    "absolute_orbit",
                    "sigma0secondstd",
                    "sigma0second",
                ]
            ).difference(obsvars)
        while dttmp <= self.t1:
            obsfile = dttmp.strftime(self.obsregex)
            if os.path.exists(obsfile):
                self.logger.debug("Adding %s" % obsfile)
                obslist.append(dttmp.strftime(self.obsregex))
            else:
                self.logger.warning("File %s does not exist" % obsfile)
            dttmp += interval

        # Loading satellite data
        if not obslist:
            raise IOError(
                "No obs between %s-%s (%s)" % (self.t0, self.t1, self.obsregex)
            )
        obsnc = xr.open_mfdataset(obslist, drop_variables=dropvars)
        self.obs = obsnc.to_dataframe()
        obsnc.close()

        # Ensure obs is same range as model, but only if model has not been converted yet
        if self.model[self.lonname].min() >= 0 and self.model[self.lonname].max() > 180:
            self.logger.debug("Correcting obs lons to 0-360 range")
            self.obs.lon %= 360

        # ERS2 data are corrupted after 2003
        self.obs = self.obs.loc[(self.obs.satellite != 2) | (self.obs.time < "2003")]

        # Select on model area
        self.obs = self.obs.loc[
            (self.obs.lat >= self.modlatmin)
            & (self.obs.lat <= self.modlatmax)
            & (self.obs.lon >= self.modlonmin)
            & (self.obs.lon <= self.modlonmax)
        ]
        #                        is_lons_in(self.obs.lon, self.modlonmin, self.modlonmax)]

        self.obs.set_index("time", inplace=True)
        self.obs.loc[:, "obs"] = self.obs[obsvar]
        self.obs.rename(
            columns={"swhcor": "hs", "wind_speed_cor": "wndsp"}, inplace=True
        )

    def aux_interp(self, vals, coords, passes):
        for nn in np.arange(passes, -1, -1):
            self.logger.info("order " + str(nn) + "...")
            temp = map_coordinates(vals, coords, cval=np.nan, order=nn, prefilter=False)
            if nn == passes:
                ret = temp.copy()
            else:
                ret = np.where(np.isnan(ret), temp, ret)
        return ret

    def interpModel(self, passes=1):
        """
        Interpolate model to observation locations passes -- number of passes
        to perform, see below
        Note:  There exists a compromise between the order of spline
        interpolation used, and the proximity to the coast that can be acheived
        - i.e.  higher cubic spline interpolations require more surounding
        points to not be land.  To get the best of both worlds, a number of
        passes are performed here, in decending order of spline interpolation.
        Where the higher order values are not defined, they are replaced with
        lower level values. In the case of the default 2 for example, a 2nd
        order, then first, then zeroth are performed, the result being a
        quartic spline interpolation in the open ocean, a trilinear
        interpolation nearer the coast, and nearest neighbour sampling very
        close to the coast.
        """
        self.logger.info("    Interpolating model data to observation locations...")
        latindex, lonindex, timeindex = self._calc_interp_indeces()
        coords = np.array([timeindex, latindex, lonindex])
        self.logger.info("        order " + str(passes) + "...")
        self.mod_interp = {}

        # Interpolate extra variables from the model
        directional_vars = {"dp", "dir"}
        for modelvar in self.model_vars:
            self.logger.info("Interpolating model variable: %s" % (modelvar))
            if modelvar == "wndsp":
                for names in WIND_COMPONENT_NAMES:
                    if set(names).issubset(self.model.var()):
                        self.model[modelvar] = np.sqrt(
                            self.model[names[0]] ** 2 + self.model[names[1]] ** 2
                        )
                    continue
            elif modelvar == "cur" and {"ucur", "vcur"}.issubset(self.model.var()):
                self.model[modelvar] = np.sqrt(
                    self.model.ucur ** 2 + self.model.vcur ** 2
                )
            elif modelvar not in self.model.data_vars:
                self.logger.warning(
                    "%s not in model dataset, skipping interpolation" % (modelvar)
                )
                continue

            vals = self.model[modelvar].astype("float32").values
            if modelvar in directional_vars:
                ret1 = self.aux_interp(np.sin(vals * np.pi / 180), coords, passes)
                ret2 = self.aux_interp(np.cos(vals * np.pi / 180), coords, passes)
                ret = (np.arctan2(ret1, ret2) * 180 / np.pi) % 360
            else:
                ret = self.aux_interp(vals, coords, passes)

            self.mod_interp.update({modelvar: ret.astype("float32")})
        return

    def createColocs(self, dropna=True, drop_duplicates=True):
        df = self.obs
        for modelvar, modeldata in self.mod_interp.items():
            df["m_" + modelvar] = modeldata
            if modelvar == self.modvar:
                # duplicated column to make changing vars easy without modifying much code
                df["model"] = modeldata
        if dropna:
            df.dropna(inplace=True)
        if drop_duplicates:
            df.drop_duplicates(inplace=True)
        if not self.test:
            del self.obs
            del self.model
        return df

    def _calc_interp_indeces(self):
        """ Calculates the indices that need to be interpolated to
            for the given observations
            """
        # Calculating indexes depends on which way the model data is ordered
        if self.model[self.latname][0] < self.model[self.latname][1]:
            latindex = np.round(
                (1 / float(self.latres)) * (self.obs.lat.values - self.modlatmin)
            ).astype(int)
        else:
            latindex = np.round(
                (1 / float(self.latres)) * (self.modlatmax - self.obs.lat.values)
            ).astype(int)

        if self.model[self.lonname][0] < self.model[self.lonname][1]:
            lonindex = np.round(
                (1 / float(self.lonres)) * (self.obs.lon.values - self.modlonmin)
            ).astype(int)
        else:
            lonindex = np.round(
                (1 / float(self.lonres)) * (self.modlonmax - self.obs.lon.values)
            ).astype(int)

        timeindex = []
        for ttmp in pd.to_datetime(self.obs.index):
            timedelta = ttmp.tz_localize(None) - self.t0
            timeindex.append(self._timedelta_to_index(timedelta))

        return latindex, lonindex, np.array(timeindex, dtype=float)

    def _timedelta_to_index(self, timedelta):
        """
        Converts timedelta to an array index in time for interpolation
        for the given timestep in the model data array
        timedelta -- timedelta object
        """
        return ((timedelta.days) * 24.0 + (timedelta.seconds / 3600.0)) / (
            self.tstep.total_seconds() / 3600.0
        )

    def calcGriddedStats(self, boxsize=None, latedges=None, lonedges=None):
        """
        Calculate the stats for each grid point of given box size (lat/lon)
        """
        self.logger.info("    Calculating gridded statistics...")
        self.boxsize = boxsize or self.boxsize

        lat = self.df['lat']
        lon = self.df['lon']
        obs = self.df['obs']
        model = self.df['model']

        # if self.modlonmax + self.lonres == 360:
        #     self.logger.debug("calc_grid_stats adding cyclic points for global grid..")
        #     minlon, maxlon = self.modlonmin, self.modlonmax+self.lonres
        # else:
        #     minlon, maxlon = self.modlonmin, self.modlonmax
        #     minlon, maxlon = self.modlonmin, self.modlonmax
        # minlat, maxlat = self.modlatmin, self.modlatmax

        if latedges is None:
            minlat, maxlat = (
                lat.min() + (abs(lat.min()) % self.boxsize),
                lat.max() - (abs(lat.min()) % self.boxsize),
            )
            # minlat, maxlat = lat.min(), lat.max()
            lats = np.arange(
                minlat + self.boxsize / 2.0, maxlat + self.boxsize / 2.0, self.boxsize
            )
            nlats = lats.size
            latedges = np.arange(minlat, maxlat + self.boxsize, self.boxsize)

        if lonedges is None:
            minlon, maxlon = (
                lon.min() + (abs(lon.min()) % self.boxsize),
                lon.max() - (abs(lon.min()) % self.boxsize),
            )
            lons = np.arange(
                minlon + self.boxsize / 2.0, maxlon + self.boxsize / 2.0, self.boxsize
            )
            nlons = lons.size
            lonedges = np.arange(minlon, maxlon + self.boxsize, self.boxsize)

        dt = np.dtype(
            [
                ("bias", np.float32),
                ("nbias", np.float32),
                ("bias2", np.float32),
                ("rmsd", np.float32),
                ("nrmsd", np.float32),
                ("si", np.float32),
                ("obsmean", np.float32),
                ("modmean", np.float32),
                ("N", int),
            ]
        )

        diff = model - obs
        if np.isnan(diff).any():
            raise Exception("NANs found, check data for errors! ")
        diff2 = model ** 2 - obs ** 2
        bias = diff.mean()
        msdall = diff ** 2

        grid_N, xedges, yedges = np.histogram2d(lat, lon, bins=(latedges, lonedges))
        grid_bias_sum, xedges, yedges = np.histogram2d(
            lat, lon, weights=diff, bins=(latedges, lonedges)
        )
        grid_bias2_sum, xedges, yedges = np.histogram2d(
            lat, lon, weights=diff2, bins=(latedges, lonedges)
        )
        grid_bias_sum = ma.masked_equal(grid_bias_sum, 0)
        obs_sum, xedges, yedges = np.histogram2d(
            lat, lon, weights=obs, bins=(latedges, lonedges)
        )
        obs_sum = ma.masked_equal(obs_sum, 0)
        mod_sum, xedges, yedges = np.histogram2d(
            lat, lon, weights=model, bins=(latedges, lonedges)
        )
        mod_sum = ma.masked_equal(mod_sum, 0)
        grid_msd_sum, xedges, yedges = np.histogram2d(
            lat, lon, weights=msdall, bins=(latedges, lonedges)
        )
        grid_msd_sum = ma.masked_equal(grid_msd_sum, 0)

        grid_N = grid_N
        grid_bias = grid_bias_sum / grid_N
        grid_bias2 = grid_bias2_sum / grid_N
        grid_msd = grid_msd_sum / grid_N
        grid_rmsd = np.sqrt(grid_msd)
        grid_si = np.sqrt((grid_msd - grid_bias ** 2)) / (np.abs(obs_sum) / grid_N)
        nbias = grid_bias / (np.abs(obs_sum) / grid_N)
        nrmsd = grid_rmsd / (np.abs(obs_sum) / grid_N)
        obs_mean = obs_sum / grid_N
        mod_mean = mod_sum / grid_N

        stats_grid = np.rec.fromarrays(
            (
                grid_bias,
                nbias,
                grid_bias2,
                grid_rmsd,
                nrmsd,
                grid_si,
                obs_mean,
                mod_mean,
                grid_N,
            ),
            dtype=dt,
        )

        # if self.modlonmax + self.lonres == 360:
        #     # Note: This is a bit ad-hoc.  Basically adds cyclic points at
        #     # each end, as basemap can't automatically recognise cyclic grids
        #     stats_grid, lons = addcyclic_both(stats_grid, lons)
        self.stats_grid_lats = latedges
        self.stats_grid_lons = lonedges
        self.stats_grid = stats_grid

    def saveGriddedStats(self, outnc):
        self.logger.info("        Saving stats to %s" % outnc)
        nlat, nlon = self.stats_grid["bias"].shape
        # Work around because grids created from hist2d did not usually match defined coordinate array
        lat = (
            self.stats_grid_lats
            if len(self.stats_grid_lats) == nlat
            else self.stats_grid_lats[:-1]
        )
        lon = (
            self.stats_grid_lons
            if len(self.stats_grid_lons) == nlon
            else self.stats_grid_lons[:-1]
        )
        ds = xr.Dataset(coords={"lat": lat, "lon": lon})
        for stat in (
            "bias",
            "nbias",
            "bias2",
            "rmsd",
            "nrmsd",
            "si",
            "obsmean",
            "modmean",
            "N",
        ):
            ds[stat] = (("lat", "lon"), self.stats_grid[stat])
        ds.to_netcdf(outnc, engine="scipy")

    def plotGriddedStats(
        self,
        statistic,
        minobs=100,
        ptype="pcolormesh",
        title=False,
        smoothfac=False,
        scalefactor=1,
        **kwargs,
    ):
        """
        Plots requested statistic
        statistic -- bias, stdev, rmsd,
        minobs -- minimum number of observations for calculation of mask
        type -- contour or pcolormesh
        """
        self.logger.info("        Plotting gridded %s" % statistic)
        figname = statistic + "_" + str(self.boxsize)
        if statistic == "si":
            cblabel = "SI (%)"
            kwargs["vmax"] = kwargs.get("vmax", 0.2)
            kwargs["vmin"] = kwargs.get("vmin", 0)
        elif statistic == "rmsd":
            cblabel = "RMSE [" + self.units + "]"
            kwargs["vmax"] = kwargs.get("vmax", 0.8 * scalefactor)
            kwargs["vmin"] = kwargs.get("vmin", 0)
        elif statistic == "nrmsd":
            cblabel = "Normalized RMSE [" + self.units + "]"
            kwargs["vmax"] = kwargs.get("vmax", 0.20)
            kwargs["vmin"] = kwargs.get("vmin", 0)
        elif statistic == "nbias":
            cblabel = "Normalized Bias"
            kwargs["cmap"] = plt.get_cmap("RdBu_r")
            # if no vmax or vmin set, make sure automatic ones are symmetric
            kwargs["vmax"] = kwargs.get("vmax", 0.25)
            kwargs["vmin"] = kwargs.get("vmin", -kwargs["vmax"])
        elif statistic == "bias":
            cblabel = "Bias [" + self.units + "]"
            kwargs["cmap"] = plt.get_cmap("RdBu_r")
            # if no vmax or vmin set, make sure automatic ones are symmetric
            kwargs["vmax"] = kwargs.get("vmax", 0.5 * scalefactor)
            kwargs["vmin"] = kwargs.get("vmin", -kwargs["vmax"])
        elif statistic == "N":
            cblabel = "Number of Obs"
        elif statistic == "obsmean":
            cblabel = "Observation Mean " + self.units + ")"
        elif statistic == "obsstd" or statistic == "std":
            cblabel = "Observation Std (" + self.units + ")"
        else:
            cblabel = statistic
        if statistic != "N":
            data = ma.masked_where(
                self.stats_grid["N"] <= minobs, self.stats_grid[statistic]
            )
        else:
            data = ma.masked_where(
                self.stats_grid["N"] <= 0, self.stats_grid[statistic]
            )
        if smoothfac:
            import smooth

            self.logger.info("Smoothing field by a factor of %s" % smoothfac)
            data = smooth.smooth_grid(data.filled(0), smoothfac, pad=0)
            data = ma.masked_where(self.stats_grid["N"] <= 0, data)
            figname = figname + "-" + "smoothed-" + str(smoothfac)

        plotMap(data, self.stats_grid_lats, self.stats_grid_lons, ptype=ptype, **kwargs)
        plt.title(cblabel)
        return

    def calcStats(self):
        self.logger.info("Calculating stats")
        self.stats = {}
        groups = self.df.groupby("satellite")
        self.stats = self.df.groupby("satellite").apply(statsDict)

    def calcStatsFreq(self, freq="M"):
        return self.df.groupby([pd.TimeGrouper(freq=freq), "satellite"]).apply(
            statsDict
        )

    def saveStats(self, outfile):
        self.saveStatsFile(outfile)

    def saveStatsFile(self, outfile):
        if not hasattr(self, "stats"):
            self.calcStats()
        if hasattr(self, "stats"):
            self.logger.info("    Writing stats to %s" % outfile)
            self.stats.to_json(outfile)
        else:
            self.logger.warning("    No stats to save ...")

    def saveColocs(self, outfile):
        """Write matchups to pkl"""
        self.logger.info("    Writing colocs to %s" % outfile)
        self.df.to_pickle(outfile)

    def loadColocs(self, fglob, subset=None):
        if isinstance(fglob, list):
            filelist = fglob
        else:
            filelist = sorted(glob(fglob))
        self.logger.info("Loading colocs %s \n" % "\n\t".join(filelist))

        self.df = None
        for fname in filelist:
            print(" Reading %s" % fname)
            if self.df is None:
                if subset is None:
                    self.df = pd.read_pickle(fname)
                else:
                    self.df = pd.read_pickle(fname)[subset]
            else:
                if subset is None:
                    self.df = self.df.append(pd.read_pickle(fname))
                else:
                    self.df = self.df.append(pd.read_pickle(fname)[subset])

        self.obsname = "obs"
        self.modname = "model"

    def standard_plots(self, color="r"):
        if self.plotdir:
            logging.info("Saving output to %s" % self.plotdir)
            os.makedirs(self.plotdir, exist_ok=True)
        ax = self.df.plot_density_contour(cmap="viridis")
        self.df.add_stats(ax)
        self.df.plot_regression(ax=ax)
        self.df.add_regression(ax, loc=4)
        if self.plotdir:
            plt.savefig(
                os.path.join(self.plotdir, "density_contour.png"), bbox_inches="tight"
            )
        ax = self.df.plot_density_scatter(cmap="viridis")
        self.df.add_stats(ax)
        self.df.plot_regression(ax=ax)
        self.df.add_regression(ax, loc=4)
        if self.plotdir:
            plt.savefig(
                os.path.join(self.plotdir, "density_scatter.png"), bbox_inches="tight"
            )
        ax = self.df.plot_qq(
            increment=0.001, color=color, alpha=0.1, label="0.1 % increments"
        )
        self.df.plot_qq(
            increment=0.01, color=color, ax=ax, alpha=0.2, label="1 % increments"
        )
        self.df.plot_qq(
            increment=0.1, color=color, ax=ax, alpha=1.0, label="10 % increments"
        )
        plt.legend()

        if self.plotdir:
            plt.savefig(os.path.join(self.plotdir, "qq.png"), bbox_inches="tight")

        ax = self.df.plot_scatter_qq()
        if self.plotdir:
            plt.savefig(
                os.path.join(self.plotdir, "scatter_qq.png"), bbox_inches="tight"
            )

        self.calcGriddedStats(self.boxsize)
        self.plotGriddedStats(
            "bias", proj=self.proj, clat=self.clat, scalefactor=self.scalefactor
        )
        if self.plotdir:
            plt.savefig(os.path.join(self.plotdir, "gridbias.png"), bbox_inches="tight")
        self.plotGriddedStats(
            "rmsd", proj=self.proj, clat=self.clat, scalefactor=self.scalefactor
        )
        if self.plotdir:
            plt.savefig(os.path.join(self.plotdir, "gridrmsd.png"), bbox_inches="tight")
        self.plotGriddedStats("si", proj=self.proj, clat=self.clat)
        if self.plotdir:
            plt.savefig(os.path.join(self.plotdir, "gridsi.png"), bbox_inches="tight")
        self.plotGriddedStats("nrmsd", proj=self.proj, clat=self.clat)
        if self.plotdir:
            plt.savefig(
                os.path.join(self.plotdir, "gridnrmsd.png"), bbox_inches="tight"
            )
        self.plotGriddedStats("nbias", proj=self.proj, clat=self.clat)
        if self.plotdir:
            plt.savefig(
                os.path.join(self.plotdir, "gridnbias.png"), bbox_inches="tight"
            )
        self.plotGriddedStats("N", proj=self.proj, clat=self.clat)
        if self.plotdir:
            plt.savefig(os.path.join(self.plotdir, "gridN.png"), bbox_inches="tight")
        if self.plotdir:
            self.saveGriddedStats(os.path.join(self.plotdir, "gridded_stats.nc"))
        if self.plotdir:
            self.saveStatsFile(os.path.join(self.plotdir, "stats.json"))


class VerifyNRT(Verify):
    def __init__(
        self,
        logger=logging,
        obsregex=None,
        test=False,
        modvar=None,
        latmin=None,
        latmax=None,
        lonmin=None,
        lonmax=None,
        **kwargs,
    ):
        obsregex = obsregex or "/net/diskserver1/volume1/data/obs/ifremer/*%Y%m%d*.nc"
        super(VerifyNRT, self).__init__(
            logger=logging,
            obsregex=obsregex,
            test=test,
            modvar=modvar,
            latmin=latmin,
            latmax=latmax,
            lonmin=lonmin,
            lonmax=lonmax,
            **kwargs,
        )

    def loadObs(self, interval=timedelta(hours=24), dropvars=None):
        self.logger.info("Loading observations")
        self.altnames = {
            1: "ERS1",
            2: "ERS2",
            3: "ENVISAT",
            4: "TOPEX",
            5: "POSEIDON",
            6: "JASON1",
            7: "GFO",
            8: "JASON2",
            9: "CRYOSAT2",
        }
        self.altfnames = {
            1: "ERS1",
            2: "ERS2",
            3: "ENV",
            4: "TOPEX",
            5: "POSEIDON",
            6: "JAS1",
            7: "GFO",
            8: "JAS2",
            9: "CRYO",
        }
        #'swhcor', #'lat', #'lon', #'time' #'satellite',

        obsnames = {"hs": "swh_calibrated", "wndsp": "wind_speed_alt_calibrated"}
        obsvar = obsnames[self.modvar]
        if not dropvars:
            dropvars = set(
                [
                    "short swh",
                    "swh_calibrated",
                    "swh_standard_error",
                    "swh_2nd",
                    "swh_2nd_calibrated",
                    "sigma0",
                    "sigma0_calibrated",
                    "sigma0_2nd",
                    "sigma0_2nd_calibrated",
                    "wind_speed_alt",
                    "wind_speed_alt_calibrated",
                    "wind_speed_rad",
                    "wind_speed_model_u",
                    "wind_speed_model_v",
                    "swh_rms",
                    "swh_rms_2nd",
                    "sigma0_rms",
                    "sigma0_rms_2nd",
                    "off_nadir_angle_wf",
                    "off_nadir_angle_pf",
                    "range_rms",
                    "range_rms_2nd",
                    "bathymetry",
                    "distance_to_coast",
                    "sea_surface_temperature",
                    "surface_air_temperature",
                    "surface_air_pressure",
                    "swh_num_valid_2nd",
                    "sigma0_num_valid",
                    "swh_2nd_quality",
                    "swh_num_valid",
                    "sigma0_num_valid_2nd",
                    "swh",
                    "rejection_flags",
                    "sigma0_2nd_quality",
                    "sigma0_quality",
                    "cycle",
                ]
            ).difference([obsvar])
        obslistall = []
        dttmp = self.t0
        first = True
        while dttmp <= self.t1:
            obsfiles = glob(dttmp.strftime(self.obsregex))
            for obsfile in obsfiles:
                if os.path.exists(obsfile):
                    self.logger.debug("Adding %s" % obsfile)
                    obslistall.append(obsfile)
                else:
                    self.logger.warning("File %s does not exist" % obsfile)
            dttmp += interval

        for key, alt in self.altfnames.items():
            obslist = []
            for fname in obslistall:
                if alt in fname:
                    obslist.append(fname)
            nfiles = len(obslist)
            if nfiles == 0:
                self.logger.debug("No files for %s" % alt)
                continue
            elif nfiles < 1000:
                self.logger.debug("Reading files for %s" % alt)
                obsfiles = xr.open_mfdataset(obslist, drop_variables=dropvars)
                if first:
                    obs = obsfiles.to_dataframe()
                    obs["satellite"] = key
                    first = False
                else:
                    obstmp = obsfiles.to_dataframe()
                    obstmp["satellite"] = key
                    obs = pd.concat([obs, obstmp])
                obsfiles.close()
            else:
                fac = nfiles / 1000 + 1
                self.logger.info("Splitting files for %s" % alt)
                split = int(len(obslist) / fac)
                for ii in np.arange(0, fac):
                    self.logger.info(
                        "  Opening files %s to %s of %s"
                        % (ii * split, (ii + 1) * split, len(obslist))
                    )
                    obsfiles1 = xr.open_mfdataset(
                        obslist[ii * split : (ii + 1) * split], drop_variables=dropvars
                    )
                    if first:
                        obs = obsfiles1.to_dataframe()
                        obs["satellite"] = key
                        first = False
                    else:
                        obstmp = obsfiles1.to_dataframe()
                        obstmp["satellite"] = key
                        obs = pd.concat([obs, obstmp])
                    obsfiles1.close()
        obs = obs[obs.swh_quality == 0]
        if self.model[self.lonname].max() > 180:
            self.logger.debug("Correcting obs lons to 0-360 range")
            obs.lon %= 360

        # Select on model area
        self.obs = obs.loc[
            (obs.lat > self.modlatmin)
            & (obs.lat < self.modlatmax)
            & is_lons_in(obs.lon, self.modlonmin, self.modlonmax)
        ]
        # self.obs.set_index('time',inplace=True)
        self.obs.rename(columns={obsvar: "obs"}, inplace=True)


class VerifyNRTraw(Verify):
    def __init__(
        self,
        logger=logging,
        obsregex=None,
        test=False,
        modvar=None,
        latmin=None,
        latmax=None,
        lonmin=None,
        lonmax=None,
        **kwargs,
    ):
        # not exactly obsregex, here the path to the folder with the missions is used
        obsregex = obsregex or "/net/datastor1/data/obs/altimetry/"
        if not os.path.isdir(obsregex):
            raise Exception("{} is not a directory".format(obsregex))
        super(VerifyNRTraw, self).__init__(
            logger=logging,
            obsregex=obsregex,
            test=test,
            modvar=modvar,
            latmin=latmin,
            latmax=latmax,
            lonmin=lonmin,
            lonmax=lonmax,
            **kwargs,
        )

    def loadObs(self,):
        self.logger.info("Loading NRT raw observations")
        self.altnames = {}
        obsvar = self.modvar  # 'hs' or 'wndspd'
        # Loading satellite data
        self.obs = load_nrt(
            tlims=[self.t0, self.t1],
            xlims=[self.modlonmin, self.modlonmax],
            ylims=[self.modlatmin, self.modlatmax],
            reqvars=[obsvar],
            satpath0=self.obsregex,
        )

        if self.model[self.lonname].max() > 180:  # redundant with load_nrt?
            self.logger.debug("Correcting obs lons to 0-360 range")
            self.obs.lon %= 360
        else:
            self.obs.lon = ((self.obs.lon + 180) % 360) - 180
        self.obs.rename(columns={obsvar: "obs"}, inplace=True)


def statsDict(group):
    stats = {}
    stats["obsmean"] = group.obs.mean()
    stats["modmean"] = group.model.mean()
    stats["modstd"] = group.model.std()
    stats["obsstd"] = group.obs.std()
    stats["N"] = group.shape[0]
    stats["bias"] = bias(group.obs, group.model)
    stats["nbias"] = stats["bias"] / stats["modmean"]
    stats["std"] = (group.model - group.obs).std()
    stats["rmsd"] = rmsd(group.obs, group.model)
    stats["nrmsd"] = stats["rmsd"] / stats["modmean"]
    stats["si"] = si(group.obs, group.model)
    stats["r"] = np.corrcoef(group.obs, group.model)[0, 1]
    P = np.arange(0, 1, 0.01)
    Qx = mstats.mquantiles(group.obs, P)
    Qy = mstats.mquantiles(group.model, P)
    for ii in range(0, len(P)):
        stats["Qx_" + str(ii)] = Qx[ii]
        stats["Qy_" + str(ii)] = Qy[ii]
    # return pd.DataFrame.from_dict(stats,index=0)
    return stats


def plotMap(
    data,
    lats,
    lons,
    ptype="pcolormesh",
    proj="Robinson",
    clon=None,
    clat=None,
    **kwargs,
):
    if clon == None:
        clon = (lons.min() + lons.max()) / 2.0
    if clat == None:
        clat = (lats.min() + lats.max()) / 2.0
    if proj == "PlateCarree":
        projection = ccrs.PlateCarree(central_longitude=clon)
    elif proj == "Orthographic":
        projection = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)
    elif proj == "Robinson":
        projection = ccrs.Robinson(central_longitude=clon)
    else:
        raise ("Projection not recognized")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={"projection": projection})
    # cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
    # fig.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)
    if ptype == "pcolormesh":
        lats, lons = adjust_latlon_pcolormesh(lats, lons)
        mpl = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), **kwargs)
    elif ptype == "contour":
        mpl = ax.contour(
            lons, lats, data, transform=ccrs.PlateCarree(), zorder=2, **kwargs
        )
        mpl = ax.contourf(
            lons, lats, data, transform=ccrs.PlateCarree(), zorder=1, **kwargs
        )
    else:
        raise Exception("Plot type %s not recognised" % ptype)

    # if proj == 'Orthographic':
    #    ax.set_global()

    # def resize_colobar(event):
    #     plt.draw()
    #     posn = ax.get_position()
    #     cbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0+0.1,
    #                          0.02, posn.height-0.2])
    # fig.canvas.mpl_connect('resize_event', resize_colobar)
    ax.coastlines(zorder=3, color="gray")
    ax.add_feature(cfeature.LAND, facecolor="gray", zorder=2)
    plt.colorbar(mpl, fraction=0.046, pad=0.04, orientation="horizontal")
    try:
        gl = ax.gridlines(draw_labels=True)
        gl.xlabels_top = gl.ylabels_right = False
        gl.yformatter = LATITUDE_FORMATTER
        gl.xformatter = LONGITUDE_FORMATTER
    except Exception as e:
        gl = ax.gridlines(draw_labels=False)
    # resize_colobar(None)


def configure_logging(loglevel=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(loglevel)
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def adjust_latlon_pcolormesh(lat, lon):
    """
    Appropriately shifts given lat and lon grids by half a grid point to
    accomodate the peculiarities of pcolormesh plots
    """
    latres = lat[1] - lat[0]
    lonres = lon[1] - lon[0]
    # plat = np.hstack((lat-latres/2., lat[-1]+latres/2.))
    # plon = np.hstack((lon-lonres/2., lon[-1]+lonres/2.))
    plat = lat - latres / 2.0
    plon = lon - lonres / 2.0
    return plat, plon


def addcyclic_both(arrin, lonsin):
    """
    ``arrout, lonsout = addcyclic(arrin, lonsin)``
    adds cyclic (wraparound) point in longitude to ``arrin`` and ``lonsin``.
    same as matplotlib function, only it adds an extra longitude column on
    each side.
    """
    nlats = arrin.shape[0]
    nlons = arrin.shape[1]
    if hasattr(arrin, "mask"):
        arrout = ma.zeros((nlats, nlons + 2), arrin.dtype)
    else:
        arrout = np.zeros((nlats, nlons + 2), arrin.dtype)
    arrout[:, 1 : nlons + 1] = arrin[:, :]
    arrout[:, nlons + 1] = arrin[:, 0]
    arrout[:, 0] = arrin[:, nlons - 1]
    if hasattr(lonsin, "mask"):
        lonsout = ma.zeros(nlons + 2, lonsin.dtype)
    else:
        lonsout = np.zeros(nlons + 2, lonsin.dtype)
    lonsout[1 : nlons + 1] = lonsin[:]
    lonsout[nlons + 1] = lonsin[-1] + lonsin[1] - lonsin[0]
    lonsout[0] = lonsin[0] - (lonsin[1] - lonsin[0])
    return arrout, lonsout


def createPlots(
    ncglob=None,
    hdfglob=None,
    dset=None,
    project_id="oceanum-dev",
    plotdir="./plots",
    boxsize=2,
    vmin=0,
    vmax=8,
    proj="Robinson",
    clat=None,
    clon=None,
    scalefactor=1,
    start=None,
    end=None,
    **kwargs,
):
    logging.info("Saving output to %s" % plotdir)
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)
    if sum([ncglob == None, hdfglob == None, dset == None]) < 2:
        raise Exception("Only specify one of ncglob, hdfglob or dset")
    if not ncglob and not hdfglob and not dset:
        raise Exception("Need to specify one of ncglob, hdfglob or dest")
    if dset:
        verif = VerifyGBQ(project_id=project_id, **kwargs)
        verif.loadColocs(start=start, end=end, dset=dset)
    else:
        verif = Verify(**kwargs)
        if ncglob:
            verif.calcColocs(ncglob)
        if hdfglob:
            verif.loadColocs(hdfglob, subset=["lon", "lat", "obs", "model"])
    verif.vmin = vmin
    verif.vmax = vmax
    verif.obslabel = "Observed $H_s$"
    verif.modlabel = "Modelled $H_s$"
    ax = verif.plot_contour(cmap="jet")
    verif.add_regression(ax)
    # verif.add_stats(ax=ax, loc='lower right')
    pos, step = verif.add_stats(ax)
    pos[1] = step
    pos[0] = 0.6 * ax.get_xlim()[1]
    verif.add_regression(ax, print_pos=pos)
    plt.savefig(os.path.join(plotdir, "propden.png"), bbox_inches="tight")
    ax = verif.plot_qq(increment=0.01, color="k")
    verif.plot_qq(increment=0.1, color="r", ax=ax)
    plt.savefig(os.path.join(plotdir, "qq.png"), bbox_inches="tight")
    verif.calcGriddedStats(boxsize)
    verif.plotGriddedStats("bias", proj=proj, clat=clat, scalefactor=scalefactor)
    plt.savefig(os.path.join(plotdir, "gridbias.png"), bbox_inches="tight")
    verif.plotGriddedStats("rmsd", proj=proj, clat=clat, scalefactor=scalefactor)
    plt.savefig(os.path.join(plotdir, "gridrmsd.png"), bbox_inches="tight")
    verif.plotGriddedStats("si", proj=proj, clat=clat)
    plt.savefig(os.path.join(plotdir, "gridsi.png"), bbox_inches="tight")
    verif.plotGriddedStats("nrmsd", proj=proj, clat=clat)
    plt.savefig(os.path.join(plotdir, "gridnrmsd.png"), bbox_inches="tight")
    verif.plotGriddedStats("nbias", proj=proj, clat=clat)
    plt.savefig(os.path.join(plotdir, "gridnbias.png"), bbox_inches="tight")
    verif.plotGriddedStats("N", proj=proj, clat=clat)
    plt.savefig(os.path.join(plotdir, "gridN.png"), bbox_inches="tight")
    verif.saveGriddedStats(os.path.join(plotdir, "gridded_stats.nc"))
    return verif


def calcColocs(
    ncglob,
    cla=Verify,
    outdir="./out",
    overwrite=False,
    dropna=True,
    pool=1,
    plot_kwargs=None,
    **kwargs,
):
    logger = configure_logging()
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if isinstance(ncglob, list):
        flist = ncglob
    else:
        flist = sorted(glob(ncglob))

    if pool == 1:
        for fname in flist:
            calcColocsFile(
                fname,
                cla=cla,
                outdir=outdir,
                overwrite=overwrite,
                dropna=dropna,
                plot_kwargs=plot_kwargs,
                **kwargs,
            )
    else:
        p = Pool(pool)
        func = partial(
            calcColocsFile,
            cla=cla,
            outdir=outdir,
            overwrite=overwrite,
            dropna=dropna,
            plot_kwargs=plot_kwargs,
            **kwargs,
        )
        p.map(func, flist)


class VerifyGBQ(Verify):
    def __init__(
        self,
        ncglob,
        logger=logging,
        obsdset="oceanum-prod.cersat.data",
        project_id="oceanum-prod",
        test=False,
        modvar=None,
        latmin=None,
        latmax=None,
        lonmin=None,
        lonmax=None,
        **kwargs,
    ):
        self.project_id = project_id
        super().__init__(
            ncglob,
            logger=logging,
            obsregex=obsdset,
            test=test,
            modvar=modvar,
            latmin=latmin,
            latmax=latmax,
            lonmin=lonmin,
            lonmax=lonmax,
            **kwargs,
        )

    @property
    def gbq_fields(self):
        """The GBQ fields to write."""
        try:
            fields = ["time"] + list(self.df.columns)
        except:
            fields = COLOCFIELDS
        return fields

    def loadObs(
        self, interval=timedelta(hours=24), dropvars=None, use_bqstorage_api=True
    ):
        self.logger.info("Loading observations")

        obsnames = {"hs": "swh_adjusted", "wndsp": "wind_speed_cor"}
        self.obsvar = obsnames[self.modvar]
        obsq = GBQAlt(
            variables=COREFIELDS + [self.obsvar],
            dset=self.obsregex,
            project_id=self.project_id,
            use_bqstorage_api=use_bqstorage_api,
        )
        obsq.get(
            start=self.t0,
            end=self.t1,
            x0=self.lonmin or self.modlonmin,
            x1=self.lonmax or self.modlonmax,
            y0=self.latmin or self.modlatmin,
            y1=self.latmax or self.modlatmax,
        )
        self.obs = obsq.df
        self.obs.set_index("time", inplace=True)

        # Select on model area - TODO implement in query
        # self.obs = obs.loc[(obs.lat > self.modlatmin) &
        # (obs.lat < self.modlatmax) &
        # is_lons_in(obs.lon, self.modlonmin, self.modlonmax)]
        # self.obs.set_index('time',inplace=True)
        self.obs.rename(columns={self.obsvar: "obs"}, inplace=True)

        # Ensure obs is same range as model, but only if model has not been converted yet
        if self.model[self.lonname].min() >= 0 and self.model[self.lonname].max() > 180:
            self.logger.debug("Correcting obs lons to 0-360 range")
            self.obs.lon %= 360

    @retry(**RETRY_KWARGS)
    def saveColocs(self, table, if_exists="append", project_id="oceanum-dev", **kwargs):
        to_gbq(
            self.df.reset_index()[self.gbq_fields],
            table,
            project_id=project_id,
            if_exists=if_exists,
            **kwargs,
        )

    @retry(**RETRY_KWARGS)
    def saveStats(
        self, table, time=None, if_exists="append", project_id="oceanum-dev", **kwargs
    ):
        if not hasattr(self, "stats"):
            self.calcStats()
        self.logger.info("    Writing stats to %s" % table)
        dfs = []
        nn = 0
        for key, items in self.stats.items():
            tmp = pd.DataFrame(items, index=[nn])
            if time:
                tmp["time"] = time
            tmp["satellite"] = key
            dfs.append(tmp)
            nn += 1
        stats = pd.concat(dfs)
        to_gbq(
            stats.reset_index(),
            table,
            project_id=project_id,
            if_exists=if_exists,
            **kwargs,
        )

    @retry(**RETRY_KWARGS)
    def saveStatsMonthly(
        self, table, time=None, if_exists="append", project_id="oceanum-dev", **kwargs
    ):
        dftmp = self.df.copy()
        for date, self.df in self.df.groupby(pd.Grouper(freq="M")):
            if hasattr(self, "stats"):
                del self.stats
            self.saveStats(table, time=datetime(date.year, date.month, 1))
        self.df = dftmp

    def standard_plots_monthly(self):
        dftmp = self.df.copy()
        plotdir = self.plotdir
        for date, self.df in self.df.groupby(pd.Grouper(freq="M")):
            if hasattr(self, "stats"):
                del self.stats
            self.plotdir = f"{plotdir}/{date.strftime('%Y%m')}"
            self.standard_plots()
        self.df = dftmp
        self.plotdir = plotdir

    def loadColocs(
        self, start=None, end=None, dset="wave.test", use_bqstorage_api=True
    ):
        obsq = GBQAlt(
            dset=dset,
            variables=self.gbq_fields,
            project_id=self.project_id,
            use_bqstorage_api=use_bqstorage_api,
        )
        obsq.get(start, end)
        self.df = obsq.df
        self.df.set_index("time", inplace=True)
        self.obsname = "obs"
        self.modname = "model"

    def loadModel(self):
        self.logger.info("Loading model data {}".format(self.ncglob))
        model = open_netcdf(self.ncglob)
        if self.test:
            self.logger.info(" Using first 10 timesteps only")
            model = model.isel(time=slice(None, 10))
        dsettime = model.time.to_pandas()
        inodup = np.where(dsettime.duplicated() == False)[0]
        self.model = model.isel(time=inodup)
        self._check_model_data()
        # local = get(fname, "./")
        # super().loadModel(local)

    def __call__(self):
        self.calcGriddedStats()
        self.standard_plots()
        if self.savestats:
            self.saveStats(self.savestats)
        if self.savemonthlystats:
            self.saveStatsMonthly(self.savemonthlystats)
        if self.savemonthlyplots:
            self.standard_plots_monthly()


class VerifyZarr(VerifyGBQ):
    def __init__(
        self,
        moddset,
        start,
        end,
        *args,
        namespace="hindcast",
        master_url="gs://oceanum-catalog/oceanum.yml",
        **kwargs,
    ):
        self.moddset = moddset
        self.start = start
        self.end = end
        self.namespace = namespace
        self.master_url = master_url
        super().__init__(self, *args, **kwargs)

    def loadModel(self,):
        self.logger.info(f"Loading zarr model data from {self.moddset}")
        ot = Ontake(master_url=self.master_url, namespace=self.namespace)
        metadata = ot.dataset(self.moddset)
        latres = metadata.latitude[1] - metadata.latitude[0]
        lonres = metadata.longitude[1] - metadata.longitude[0]
        self.model = (
            ot.dataset(self.moddset).sel(
                # latitude=slice(self.latmin-latres, self.latmax+latres),
                # longitude=slice(self.lonmin-latres, self.lonmax+lonres),
                time=slice(self.start, self.end),
            )
            # .load()
        )
        dsettime = self.model.time.to_pandas()
        self._check_model_data()
        self.model.load()


class VerifyDAP(VerifyGBQ):
    def __init__(
        self, start, end, *args, **kwargs,
    ):
        self.start = start
        self.end = end
        super().__init__(*args, **kwargs)

    def loadModel(self):
        if not isinstance(self.ncglob, list):
            ncfiles = [self.ncglob]
        else:
            ncfiles = self.ncglob
        infonc = xr.open_dataset(ncfiles[0])
        extra_vars = {
            "uwnd",
            "vwnd",
            "ugrd10m",
            "vgrd10m",
            "ucur",
            "vcur",
            "u10",
            "v10",
        }  # vector variables can be built later
        excluded = set(infonc.data_vars) - self.model_vars - extra_vars
        latres = infonc.lat[1].values - infonc.lat[0].values
        lonres = infonc.lon[1].values - infonc.lon[0].values

        self.logger.info("Loading model data %s \n" % "\n\t".join(ncfiles))
        model = (
            xr.open_mfdataset(ncfiles, drop_variables=excluded, engine="netcdf4")
            .sel(
                lat=slice(self.latmin - latres, self.latmax + latres),
                lon=slice(self.lonmin - latres, self.lonmax + lonres),
                time=slice(self.start, self.end),
            )
            .load()
        )
        dsettime = model.time.to_pandas()
        inodup = np.where(dsettime.duplicated() == False)[0]
        self.model = model.isel(time=inodup)
        self._check_model_data()


def calcColocsFile(
    fname,
    cla=Verify,
    outdir="./out",
    overwrite=False,
    dropna=True,
    plot_kwargs=None,
    **kwargs,
):
    # try:
    print(fname)
    verif = cla(**kwargs)
    verif.loadModel(fname)
    colocstmpl = os.path.join(outdir, "%Y%m.pkl")
    colocsfile = verif.t0.strftime(colocstmpl)
    if os.path.isfile(colocsfile):
        if overwrite:
            logging.info("Colocs file %s exists, overwriting" % colocsfile)
        else:
            logging.info("Colocs file %s exists, skipping" % colocsfile)
            return

    verif.loadObs()
    verif.interpModel()
    verif.createColocs(dropna=dropna)
    verif.calcStats()
    verif.saveStats(colocsfile.replace("pkl", "json"))
    verif.saveColocs(colocsfile)
    if plot_kwargs:
        plotdir = colocsfile[:-4]
        if not os.path.isdir(plotdir):
            os.makedirs(plotdir)
        createPlots(hdfglob=colocsfile, plotdir=plotdir, **plot_kwargs)
        plt.close("all")
    # except Exception as e:
    # print e
    # pass


def is_lons_in(lons, lon_start, lon_end):

    if lon_end - lon_start >= 360:
        return np.full(np.size(lons), True)

    lons360 = lons % 360
    lon_start360 = lon_start % 360
    lon_end360 = lon_end % 360

    minx = min(lon_start360, lon_end360)
    maxx = max(lon_start360, lon_end360)
    inminmax = (lons360 > minx) & (lons360 < maxx)
    onlimits = (lons360 == minx) | (lons360 == maxx)

    if lon_start360 < lon_end360:
        lonsIn = inminmax
    else:
        lonsIn = not inminmax

    return lonsIn | onlimits


def test():
    obsregex = "/net/diskserver1/volume1/data/obs/cersat/%Y/wm_%Y%m%d.nc"
    hdfglob = "/local_home/refcst/ec/st4_mf/201509.pkl"
    v = Verify()
    v.loadColocs(hdfglob)
    # v.calcGriddedStats(2)
    # v.plotGriddedStats('bias',vmin=-0.5,vmax=0.5)
    # plt.savefig('test.png')
    # plt.savefig('test.png')
    # hdfglob='./out/st4/2012*hdf5'
    # v=Verify(loglevel=20)
    # v.loadColocs(hdfglob)
    v.calcGriddedStats(2)
    # v.saveGriddedStats('out.nc')
    v.plotGriddedStats(
        "bias", vmin=-0.5, vmax=0.5, clon=0, clat=-90, proj="Orthographic"
    )
    plt.show()
    # plt.savefig('test.png',bbox_inches='tight')


if __name__ == "__main__" and __package__ is None:
    # # test()
    # logging.basicConfig(level=logging.INFO)
    # fname = "/scratch/glob-20120101T00.nc"
    # dset = "oceanum-prod.cersat.data"
    # project_id = "oceanum-prod"
    # v = VerifyGBQ(
    # obsdset=dset,
    # project_id="oceanum-prod",
    # model_vars=["wndsp", "hs", "tps", "dpm", "botl"],
    # modvar="wndsp",
    # lonmin=170.0,
    # lonmax=190.0,
    # latmin=48.5,
    # latmax=55.5,
    # )
    # v.loadModel(fname)
    # v.loadObs()
    # v.interpModel()
    # v.createColocs()
    # v.saveColocs(
    #     "wave.test",
    #     project_id='oceanum-dev',
    #     if_exists='append'
    # )

    # from os import sys, path

    # sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    Parser()
