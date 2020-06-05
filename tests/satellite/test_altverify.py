import datetime

import matplotlib.pyplot as plt
from datetime import datetime
import logging

from onverify.satellite.verify import (
    VerifyGBQ,
    createPlots,
    VerifyZarr,
    Verify,
    VerifyDAP,
)

logging.basicConfig(level=logging.INFO)


def test_gbq():
    dset = "oceanum-prod.cersat.data"
    project_id = "oceanum-prod"
    modfile = (
        "gs://oceanum-data-dev/ww3/glob05_era5_prod/output/grid/glob-20120101T00.nc"
    )
    v = VerifyGBQ(
        modfile,
        obsdset=dset,
        project_id=project_id,
        test=True,
        upload="gs://oceanum-data-dev/test/%Y%m",
    )
    # v.calcGriddedStats(2)
    # # # v.saveGriddedStats('out.nc')
    # v.plotGriddedStats(
    # "bias", vmin=-0.5, vmax=0.5, clon=0, clat=-90, proj="Orthographic"
    # )
    # v.upload_output()
    # v.saveColocs("wave.test", project_id="oceanum-dev")
    # plt.show()
    v()


def test_gds():
    obsdset = "oceanum-prod.cersat.data_v0"
    project_id = "oceanum-prod"
    modfile = [
        "http://gds-hindcast.metoceanapi.com:80/dods/wrf/2012/nzra1_sfc_d02.2012"
    ]
    v = VerifyDAP(
        ncglob=modfile,
        start=datetime(2012, 1, 1),
        end=datetime(2012, 1, 3),
        obsdset=obsdset,
        modvar="wndsp",
        latmin=-44,
        latmax=-30,
        lonmin=160,
        lonmax=176,
        savecolocsfile=True,
    )
    # v.calcGriddedStats(2)
    # # v.saveGriddedStats('out.nc')
    # v.plotGriddedStats(
    # "bias", vmin=-0.5, vmax=0.5, clon=0, clat=-90, proj="Orthographic"
    # )
    # v.saveColocs("wave.test", project_id="oceanum-dev")
    # v.standard_plots()
    v()
    # plt.show()


def test_cawcr():
    obsdset = "oceanum-prod.cersat.data_%Y%m"
    project_id = "oceanum-prod"
    modfile = [
        "http://opendap.bom.gov.au:8080/thredds/dodsC/paccsapwaves_gridded/ww3.glob_24m.%Y%m.nc"
    ]
    v = VerifyDAP(
        ncglob=modfile,
        start=datetime(2012, 1, 1),
        end=datetime(2012, 1, 3),
        obsdset=obsdset,
        savecolocsfile=True,
        model_subsample=3,
    )
    # v.calcGriddedStats(2)
    # # v.saveGriddedStats('out.nc')
    # v.plotGriddedStats(
    # "bias", vmin=-0.5, vmax=0.5, clon=0, clat=-90, proj="Orthographic"
    # )
    # v.saveColocs("wave.test", project_id="oceanum-dev")
    # v.standard_plots()
    v()
    # plt.show()


def test_monthly():
    obsdset = "oceanum-prod.cersat.data_v0"
    project_id = "oceanum-prod"
    modfile = [
        "http://gds-hindcast.metoceanapi.com:80/dods/wrf/2012/nzra1_sfc_d02.2012"
    ]
    v = VerifyDAP(
        ncglob=modfile,
        start=datetime(2012, 1, 1),
        end=datetime(2012, 12, 31),
        obsdset=obsdset,
        modvar="wndsp",
        latmin=-41,
        latmax=-38,
        lonmin=172,
        lonmax=175,
        savemonthlystats="wave.teststast",
        savemonthlyplots=True,
    )
    v()


def test_zarr():
    obsdset = "oceanum-prod.cersat.data_v0"
    project_id = "oceanum-prod"
    moddset = "era5_wind10m"
    master_url = "/source/ontake/tests/catalog_example/oceanum.yml"
    v = VerifyZarr(
        obsdset=obsdset,
        project_id=project_id,
        modvar="wndsp",
        latmin=-44,
        latmax=-30,
        lonmin=160,
        lonmax=176,
        moddset=moddset,
        start=datetime(2012, 1, 1),
        end=datetime(2012, 1, 3),
        # master_url=master_url,
    )
    v.calcGriddedStats(2)
    v.plotGriddedStats("bias", vmin=-2.0, vmax=2.0, proj="PlateCarree")
    # plt.show()


def test_zarr_hs():
    project_id = "oceanum-prod"
    moddset = ""
    v = VerifyZarr(
        project_id=project_id,
        latmin=-44,
        latmax=-30,
        lonmin=160,
        lonmax=176,
        moddset="oceanum_wave_glob05_era5_v1.0_grid",
        start=datetime(2012, 1, 1),
        end=datetime(2012, 1, 3),
        # master_url=master_url,
        upload="gs://oceanum-data-dev/ww3/test/verification/%Y%m",
    )
    # v.calcGriddedStats(2)
    # v.plotGriddedStats("bias", vmin=-2.0, vmax=2.0, proj="PlateCarree")
    plt.show()
    v()


def test_plot():
    dset = "wave.glob05_era5_prod"
    project_id = "oceanum-dev"
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2000, 1, 10)
    v = VerifyGBQ(project_id=project_id)
    v.loadColocs(start=start, end=end, dset=dset)
    v.calcGriddedStats(2)
    v.plotGriddedStats(
        "bias", vmin=-0.5, vmax=0.5, clon=0, clat=-90, proj="Orthographic"
    )
    plt.show()


def test_write_stats():
    dset = "wave.glob05_era5_prod"
    project_id = "oceanum-dev"
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2000, 1, 2)
    v = VerifyGBQ(project_id=project_id)
    v.loadColocs(start=start, end=end, dset=dset)
    v.saveStats("test-stats")


def test_full():
    dset = "oceanum-prod.cersat.data"
    project_id = "oceanum-prod"
    v = VerifyGBQ(obsdset=dset, project_id=project_id)
    v.loadModel("gs://oceanum-data-dev/ww3/glob3_era5/grids/glob3-20111201T00.nc")
    # v.loadModel('gs://oceanum-data-dev/ww3/glob05/grids/glob-20120101T00.nc')
    v.loadObs()
    v.interpModel()
    v.createColocs()
    v.vmin = 0
    v.vmax = 8
    v.obslabel = "Observed $H_s$"
    v.modlabel = "Modelled $H_s$"
    v.obsname = "obs"
    v.modname = "model"
    v.plot_scatter()
    # v.saveColocs('wave.test2.colocs', project_id='oceanum-dev')
    v.saveStats(
        "wave.test2-stats", time=datetime(2011, 12, 1), project_id="oceanum-dev"
    )
    # plt.show()


def test_createplots():
    dset = "wave.test2"
    createPlots(dset=dset)
