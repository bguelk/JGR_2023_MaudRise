#!/usr/bin/env python
# coding: utf-8
import xgcm
import numpy as np
import xarray as xr
from pathlib import Path

xr.set_options(keep_attrs=True)
years=2007
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
        ds2 = xr.open_mfdataset(path_in.glob(f'dT*_{year}*_m36.nc'))
        del ds2['glamu']
        del ds2['gphiu']
        del ds2['glamv']
        del ds2['gphiv']
        ds3 = xr.open_mfdataset(path_in.glob(f'hfw*_{year}*_m36.nc'))
        ds4 = xr.open_mfdataset(path_in.glob(f'mld*_{year}*_m36.nc'))      
        ds=xr.merge((ds1,ds2,ds3,ds4,mask),compat='override')

        print('compute the heat content and fluxes for the full depth of the Taylor Column')
        dT_tc = (cp*rho*((ds.dT_adv_l1+ds.dT_adv_l2+ds.dT_adv_l3).where(ds.mask_tc==1).sum(dim=['x_c','y_c'])))
        dh_tc = ((((ds.Tint_l1+ds.Tint_l2+ds.Tint_l3)*ds.area).where(ds.mask_tc==1).sum(dim=['x_c','y_c']))*rho*cp)        

        dT_tc=dT_tc.rename("dT_tc")
        dT_tc=dT_tc.assign_attrs(long_name='advective heat fluxes in taylor Column- full depth')
        dT_tc.to_netcdf(Path(path_out)/f'dT_tc_{year}_m36.nc')

        dh_tc =dh_tc.rename("dh_tc")
        dh_tc =dh_tc.assign_attrs(long_name=' change in Heat content in the Taylor Column- full depth')
        dh_tc.to_netcdf(Path(path_out)/f'dh_tc_{year}_m36.nc')

        print('Level1')
        # the heat in the upper 200m is controlled by the surface fluxes, lateral changes and fluxes through the 200m depth
        dT_l1_tc=(cp*rho*((ds.dT_adv_l1).where(ds.mask_tc==1).sum(dim=['x_c','y_c'])))
        hfw_200_tc=(ds.hfw_l1).where(ds.mask_tc==1).sum(dim={'x_c','y_c'}) # defined upwards
        hf_up_200_tc=(cp*rho*hfw_200_tc)
        dh_l1_tc=((((ds.Tint_l1)*ds.area).where(ds.mask_tc==1).sum(dim=['x_c','y_c']))*rho*cp)

        dT_l1_tc=dT_l1_tc.rename("dT_l1_tc")
        dT_l1_tc=dT_l1_tc.assign_attrs(long_name='advective heat fluxes in taylor Column- level 1')
        dT_l1_tc.to_netcdf(Path(path_out)/f'dT_l1_tc_{year}_m36.nc')

        dh_l1_tc =dh_l1_tc.rename("dh_l1_tc")
        dh_l1_tc =dh_l1_tc.assign_attrs(long_name=' change in Heat content in the Taylor Column- level 1')
        dh_l1_tc.to_netcdf(Path(path_out)/f'dh_dt_l1_tc_{year}_m36.nc')

        hf_up_200_tc=hf_up_200_tc.rename("hf_up_200_tc")
        hf_up_200_tc=hf_up_200_tc.assign_attrs(long_name='vertical heat fluxes in taylor Column at 200m')
        hf_up_200_tc.to_netcdf(Path(path_out)/f'hf_up_200_tc_{year}_m36.nc')

        print('Level 2')
        dT_l2_tc=(cp*rho*((ds.dT_adv_l2).where(ds.mask_tc==1).sum(dim=['x_c','y_c'])))
        hfw_1000_tc=(ds.hfw_l2).where(ds.mask_tc==1).sum(dim={'x_c','y_c'}) # defined upwards
        hf_up_1000_tc=(cp*rho*hfw_1000_tc)
        dh_l2_tc=((((ds.Tint_l2)*ds.area).where(ds.mask_tc==1).sum(dim=['x_c','y_c']))*rho*cp)

        dT_l2_tc=dT_l2_tc.rename("dT_l2_tc")
        dT_l2_tc=dT_l2_tc.assign_attrs(long_name='advective heat fluxes in taylor Column- level 2')
        dT_l2_tc.to_netcdf(Path(path_out)/f'dT_l2_tc_{year}_m36.nc')

        dh_l2_tc =dh_l2_tc.rename("dh_l2_tc")
        dh_l2_tc =dh_l2_tc.assign_attrs(long_name=' change in Heat content in the Taylor Column- level 2')
        dh_l2_tc.to_netcdf(Path(path_out)/f'dh_dt_l2_tc_{year}_m36.nc')

        hf_up_1000_tc=hf_up_1000_tc.rename("hf_up_1000_tc")
        hf_up_1000_tc=hf_up_1000_tc.assign_attrs(long_name='vertical heat fluxes in taylor Column at 1000m')
        hf_up_1000_tc.to_netcdf(Path(path_out)/f'hf_up_1000_tc_{year}_m36.nc')

        print('level 3')
        dT_l3_tc=(cp*rho*((ds.dT_adv_l3).where(ds.mask_tc==1).sum(dim=['x_c','y_c'])))
        dh_l3_tc=((((ds.Tint_l3)*ds.area).where(ds.mask_tc==1).sum(dim=['x_c','y_c']))*rho*cp)

        dT_l3_tc=dT_l3_tc.rename("dT_l3_tc")
        dT_l3_tc=dT_l3_tc.assign_attrs(long_name='advective heat fluxes in taylor Column- level 3')
        dT_l3_tc.to_netcdf(Path(path_out)/f'dT_l3_tc_{year}_m36.nc')

        dh_l3_tc =dh_l3_tc.rename("dh_l3_tc")
        dh_l3_tc =dh_l3_tc.assign_attrs(long_name=' change in Heat content in the Taylor Column- level 3')
        dh_l3_tc.to_netcdf(Path(path_out)/f'dh_dt_l3_tc_{year}_m36.nc')
 

