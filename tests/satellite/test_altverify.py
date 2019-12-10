from onverify.satellite.verify import VerifyGBQ
from onverify.satellite.verify import createPlots
import matplotlib.pyplot as plt

def test_gbq():
    dset="oceanum-prod.cersat.data"
    project_id="oceanum-prod",
    v = VerifyGBQ(obsdset=dset, project_id=project_id)
    v.loadModel('/home/tdurrant/Downloads/ww3_glob05_grids_glob-20120101T00.nc')
    v.loadObs()
    v.interpModel()
    v.createColocs()
    # v.calcGriddedStats(2)
    # # v.saveGriddedStats('out.nc')
    # v.plotGriddedStats('bias', vmin=-0.5, vmax=0.5, clon=0,
                       # clat=-90, proj='Orthographic')
    v.saveColocs('wave.test', project_id='oceanum-dev')
    # plt.show()

def test_plot():
    dset="wave.test2"
    project_id="oceanum-dev",
    v = VerifyGBQ(project_id=project_id)
    v.loadColocs(dset=dset)
    v.calcGriddedStats(2)
    v.plotGriddedStats('bias', vmin=-0.5, vmax=0.5, clon=0,
                        clat=-90, proj='Orthographic')
    plt.show()

def test_full():
    dset="oceanum-prod.cersat.data"
    project_id="oceanum-prod",
    v = VerifyGBQ(obsdset=dset, project_id=project_id)
    v.loadModel('gs://oceanum-data-dev/ww3/glob3_era5/grids/glob3-20111201T00.nc')
    # v.loadModel('gs://oceanum-data-dev/ww3/glob05/grids/glob-20120101T00.nc')
    v.loadObs()
    v.interpModel()
    v.createColocs()
    v.vmin = 0
    v.vmax = 8
    v.obslabel = "Observed $H_s$"
    v.modlabel = "Modelled $H_s$"
    v.obsname = 'obs'
    v.modname = 'model'
    v.plot_scatter()
    v.saveColocs('wave.test2', project_id='oceanum-dev')
    #plt.show()

def test_createplots():
    dset="wave.test2"
    createPlots(dset=dset)
