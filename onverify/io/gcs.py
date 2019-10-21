import xarray as xr
import h5netcdf
import h5py
import gcsfs
import os


fs = gcsfs.GCSFileSystem()
fileObj = fs.open('gs://oceanum-era5/wind_10m_netcdf/wind_10m-197902.nc')
ds = xr.open_dataset(fileObj, engine='h5netcdf')
ds
ds.u10.isel(time=0,).plot()


