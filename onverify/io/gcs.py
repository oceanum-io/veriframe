import os

import gcsfs
import h5netcdf
import h5py
import matplotlib.pyplot as plt
import xarray as xr


def open_netcdf(fname):
    """
    Opens local or remote netcdf file and returns an xarray.Dataset

    Args:
        fname (str): file path
    Returns:
        xarray.Dataset
    """
    fs = gcsfs.GCSFileSystem()
    fileObj = fs.open(fname)
    ds = xr.open_dataset(fileObj, engine='h5netcdf')
    return ds

# ds = open_netcdf('gs://oceanum-data-dev/ww3/glob3_era5/grids/glob3-20111201T00.nc')
# ds
# ds.hs.isel(time=0,).plot()
# plt.show()

