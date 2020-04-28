import datetime

import matplotlib.pyplot as plt
from datetime import datetime
import logging

from onverify.satellite.verify import VerifyGBQ, createPlots, VerifyZarr

logging.basicConfig(level=logging.INFO)


def test_gbq():
    dset = "oceanum-prod.cersat.data"
    project_id = "oceanum-prod"
    v = VerifyGBQ(obsdset=dset, project_id=project_id)
    v.loadModel("/home/tdurrant/Downloads/ww3_glob05_grids_glob-20120101T00.nc")
    v.loadObs()
    v.interpModel()
    v.createColocs()
    # v.calcGriddedStats(2)
    # # v.saveGriddedStats('out.nc')
    # v.plotGriddedStats('bias', vmin=-0.5, vmax=0.5, clon=0,
    # clat=-90, proj='Orthographic')
    v.saveColocs("wave.test", project_id="oceanum-dev")
    # plt.show()


def test_zarr():
    obsdset = "oceanum-prod.cersat.data_v0"
    project_id = "oceanum-prod"
    moddset = "era5_wind10m_360"
    master_url = "/source/ontake/tests/catalog_example/oceanum.yml"
    v = VerifyZarr(
        obsdset=obsdset,
        project_id=project_id,
        modvar="wndsp",
        latmin=-50,
        latmax=-10,
        lonmin=120,
        lonmax=175,
        moddset=moddset,
        start=datetime(2012, 1, 1),
        end=datetime(2012, 1, 12),
        master_url=master_url,
    )
    v.loadModel()
    v.loadObs()
    v.interpModel()
    v.createColocs()
    v.calcGriddedStats(2)
    v.plotGriddedStats("bias", vmin=-2.0, vmax=2.0, proj="PlateCarree")
    plt.show()


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
