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
path_in=Path('../monthly_fields/')
years=2009
yeare=2014
# relative vorticity of sea ice: 
# vort_ice= dv_ice/dx -du_ice/dy

if __name__ == '__main__':

    from dask.distributed import Client
    client = Client(n_workers=15, threads_per_worker=1)
    print(client)
    for year in [years,yeare]:
        ds = xnemogcm.open_nemo_and_domain_cfg(nemo_files=list(path_in.glob(f'MAUD36_{year}*_icemod_grid_T.nc')), domcfg_files=['./1_domain_cfg_50levels.nc'],nemo_kwargs=dict(datadir='./',chunks={'time_counter':1}))
        ds['bathy_meter']=ds.bathy_meter.swap_dims({'x':'x_c','y':'y_c'})
        ds['x_c'].attrs['axis'] = 'X'
        ds['y_c'].attrs['axis'] = 'Y'
        ds = ds.drop_vars('ncatice', errors='ignore')

        metrics = {
            ('X',): ['e1t', 'e1u', 'e1v', 'e1f'], # X distances
            ('Y',): ['e2t', 'e2u', 'e2v', 'e2f'] # Y distances
        }
        print(len(ds.t))
        grid = xgcm.Grid(ds, metrics=metrics, periodic=False)
        # compute the vorticty
        dvi_dx = grid.derivative(ds.sivelv,'X',boundary='fill')
        dui_dy = grid.derivative(ds.sivelu,'Y',boundary='fill')
         # multiply vorticity with 86400 for unit [1/day]
        vort_ice=86400*(grid.interp(dvi_dx,'X',boundary='fill')-grid.interp(dui_dy,'Y',boundary='fill'))
        vort_ice_ym=vort_ice.mean(dim='t')
        vort_ice_ym=vort_ice_ym.rename("vort_ice_ym")
        vort_ice_ym=vort_ice_ym.assign_attrs(long_name=' mean Sea Ice Vorticity')
        vort_ice_ym.to_netcdf(Path(path_out)/f'Vort_ice_{year}_fullyear_m36.nc')
