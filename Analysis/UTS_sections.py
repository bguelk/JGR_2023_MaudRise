#!/usr/bin/env python
# coding: utf-8
import xnemogcm
import xgcm
import numpy as np
import gsw
import xarray as xr
from pathlib import Path

xr.set_options(keep_attrs=True)
def prime(x):
        return x-x.mean(dim='t')

path_out=('./processed/')
path_in=Path('./monthly_fields/')
years=2009
yeare=2014
if __name__ == '__main__':

    from dask.distributed import Client
    client = Client(n_workers=15, threads_per_worker=1)
    print(client)
    for year in [years,yeare]:
        ds = xnemogcm.open_nemo_and_domain_cfg(nemo_files=list(path_in.glob(f'MAUD36_{year}*.nc')), domcfg_files=['./1_domain_cfg_50levels.nc'],nemo_kwargs=dict(datadir='./',chunks={'time_counter':1}))
        ds['bathy_meter']=ds.bathy_meter.swap_dims({'x':'x_c','y':'y_c'})
        ds['x_c'].attrs['axis'] = 'X'
        ds['y_c'].attrs['axis'] = 'Y'
        ds = ds.drop_vars('ncatice', errors='ignore')

        metrics = {
            ('X',): ['e1t', 'e1u', 'e1v', 'e1f'], # X distances
            ('Y',): ['e2t', 'e2u', 'e2v', 'e2f'], # Y distances
            ('Z',): ['e3t', 'e3u', 'e3v', 'e3w'], # Z distances
        }
        grid = xgcm.Grid(ds, metrics=metrics, periodic=False)
        # extract a section of U,T and S across Maud Rise as yearly mean
        xloc=268 # this represents in the model domain 2.4 E
        T_2E=ds.thetao.isel(x_c=xloc).mean(dim='t')
        T_2E.to_netcdf(Path(path_out)/f'T_2E_{year}_m36.nc')
        S_2E=ds.so.isel(x_c=xloc).mean(dim='t')
        S_2E.to_netcdf(Path(path_out)/f'S_2E_{year}_m36.nc')
        # U Section needs to be interpolated 
        U_2E=grid.interp(ds.uo,'X',boundary='extend').isel(x_c=xloc).mean(dim='t')
        U_2E=U_2E.rename("uo")

        U_2E.to_netcdf(Path(path_out)/f'U_2E_{year}_m36.nc')
