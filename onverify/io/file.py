import os

import fsspec
import h5netcdf
import h5py
import xarray as xr


def open_netcdf(fname, **kwargs):
    """
    Opens local or remote netcdf file and returns an xarray.Dataset.

    Args:
        fname (str): file path.

    Returns:
        xarray.Dataset.

    """
    fs = fsspec.open_files(fname)[0].fs
    fileObj = fs.open(fname)
    ds = xr.open_dataset(fileObj, engine='h5netcdf', **kwargs)
    return ds

