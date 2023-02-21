#!/usr/bin/env python
# coding: utf-8
import xgcm
import numpy as np
import xarray as xr
from pathlib import Path

xr.set_options(keep_attrs=True)
years=2014
yeare=2019

path_out=('./post_pro2')

path_in=Path('./post_pro/')

bat=xr.open_dataset('1_domain_cfg_50levels_new.nc')
bat.coords['glamt']=bat.nav_lon.mean(dim='y')
bat.coords['gphit']=bat.nav_lat.mean(dim='x')
bat=bat.swap_dims({'x':'x_c','y':'y_c'})
mask=xr.open_dataset('./post_pro_data/mask_tc_m36.nc')

ds=xr.merge((bat,mask.mask_tc),compat='override')

grid = xgcm.Grid(ds, coords={"X": {"center": "x_c", "right": "x_f"},"Y": {"center": "y_c", "right": "y_f"},"T":{"center":"t"},"Z":{"center":"z_c", "left":"z_f"}})

ds['dx_tc'] = grid.diff(ds.mask_tc, 'X', boundary='fill')
ds['dy_tc'] = grid.diff(ds.mask_tc, 'Y', boundary='fill')
area=ds.e1t*ds.e2t
ds['area']=area
mask=ds
cp=4000
rho=1026
dt=86400
if __name__ == '__main__':

    from dask.distributed import Client
    client = Client(n_workers=20, threads_per_worker=1)
    print(client)
    for year in range(years,yeare):
        print(year)
        ds1 = xr.open_mfdataset(path_in.glob(f'Tint*_{year}*_m36.nc'))
        ds2 = xr.open_mfdataset(path_in.glob(f'dT_eddy_adv_*_{year}*_m36.nc'))
        del ds2['glamu']
        del ds2['gphiu']
        del ds2['glamv']
        del ds2['gphiv']
        ds3 = xr.open_mfdataset(path_in.glob(f'eddy_hfw*_{year}*_m36.nc'))
        ds4 = xr.open_mfdataset(path_in.glob(f'mld*_{year}*_m36.nc'))      
        ds=xr.merge((ds1,ds2,ds3,ds4,mask),compat='override')

        print('Level1')
        # the heat in the upper 200m is controlled by the surface fluxes, lateral changes and fluxes through the 200m depth
        dT_eddy_l1_tc=(cp*rho*((ds.dT_eddy_adv_l1).where(ds.mask_tc==1).sum(dim=['x_c','y_c'])))
        eddy_hfw_200_tc=(ds.eddy_hfw_l1).where(ds.mask_tc==1).sum(dim={'x_c','y_c'}) # defined upwards
        eddy_hf_up_200_tc=(cp*rho*eddy_hfw_200_tc)

        dT_eddy_l1_tc=dT_eddy_l1_tc.rename("dT_eddy_l1_tc")
        dT_eddy_l1_tc=dT_eddy_l1_tc.assign_attrs(long_name='eddy advective heat fluxes in taylor Column- level 1')
        dT_eddy_l1_tc.to_netcdf(Path(path_out)/f'dT_eddy_l1_tc_{year}_m36.nc')


        eddy_hf_up_200_tc=eddy_hf_up_200_tc.rename("eddy_hf_up_200_tc")
        eddy_hf_up_200_tc=eddy_hf_up_200_tc.assign_attrs(long_name='vertical eddy heat fluxes in taylor Column at 200m')
        eddy_hf_up_200_tc.to_netcdf(Path(path_out)/f'eddy_hf_up_200_tc_{year}_m36.nc')

        print('Level 2')
        dT_eddy_l2_tc=(cp*rho*((ds.dT_eddy_adv_l2).where(ds.mask_tc==1).sum(dim=['x_c','y_c'])))
        eddy_hfw_1000_tc=(ds.eddy_hfw_l2).where(ds.mask_tc==1).sum(dim={'x_c','y_c'}) # defined upwards
        eddy_hf_up_1000_tc=(cp*rho*eddy_hfw_1000_tc)

        dT_eddy_l2_tc=dT_eddy_l2_tc.rename("dT_eddy_l2_tc")
        dT_eddy_l2_tc=dT_eddy_l2_tc.assign_attrs(long_name='advective eddy heat fluxes in taylor Column- level 2')
        dT_eddy_l2_tc.to_netcdf(Path(path_out)/f'dT_eddy_l2_tc_{year}_m36.nc')


        eddy_hf_up_1000_tc=eddy_hf_up_1000_tc.rename("eddy_hf_up_1000_tc")
        eddy_hf_up_1000_tc=eddy_hf_up_1000_tc.assign_attrs(long_name='vertical eddy heat fluxes in taylor Column at 1000m')
        eddy_hf_up_1000_tc.to_netcdf(Path(path_out)/f'eddy_hf_up_1000_tc_{year}_m36.nc')

        print('level 3')
        dT_eddy_l3_tc=(cp*rho*((ds.dT_eddy_adv_l3).where(ds.mask_tc==1).sum(dim=['x_c','y_c'])))

        dT_eddy_l3_tc=dT_eddy_l3_tc.rename("dT_eddy_l3_tc")
        dT_eddy_l3_tc=dT_eddy_l3_tc.assign_attrs(long_name='advective eddy heat fluxes in taylor Column- level 3')
        dT_eddy_l3_tc.to_netcdf(Path(path_out)/f'dT_eddy_l3_tc_{year}_m36.nc')

 

