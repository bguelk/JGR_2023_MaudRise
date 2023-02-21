#!/usr/bin/env python
# coding: utf-8
import numpy as np
import xarray as xr
from pathlib import Path


xr.set_options(keep_attrs=True)

path_out=('./post_pro2/')

path_in=Path('./post_pro')
if __name__ == '__main__':

    from dask.distributed import Client
    client = Client(n_workers=25)
    ds = xr.open_mfdataset(path_in.glob(f'sig0_l2_20*.nc'),chunks='auto')
    xloc_tc36=268
    yloc_tc36=553
    ds['sig0_ro']=ds.sig0_l2.rolling(t=91,center=True).mean()
    cor4=xr.corr(ds.sig0_ro.isel(t=slice(500,len(ds.t)-500)),ds.sig0_ro.isel(y_c=yloc_tc36+3,x_c=181,t=slice(500,len(ds.t)-500)),dim='t')
    cor4=cor4.rename("cor4")
    cor4=cor4.assign_attrs(long_name='correlation of rolled sig0 with ref loc x_c 181, y_c 556')
    cor4.to_netcdf(path_out+'cor4_sig0_10d_rolled90_m36.nc')





