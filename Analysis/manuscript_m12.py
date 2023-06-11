#!/usr/bin/env python
# coding: utf-8
import xnemogcm
import xgcm
import numpy as np
import gsw
import xarray as xr
from pathlib import Path
from skimage.morphology import dilation
from skimage.morphology import disk

xr.set_options(keep_attrs=True)
def prime(x):
        return x-x.mean(dim='t')

path_out=('./post_pro_data')

path_in=Path('./data_years/')
years=2007
yeare=2018

if __name__ == '__main__':
    	
    from dask.distributed import Client
    client = Client(n_workers=20, threads_per_worker=1)
    print(client)
    for year in range(years,yeare):   
        print(year)
        ds = xnemogcm.open_nemo_and_domain_cfg(nemo_files=list(path_in.glob(f'MAUD12_{year}*.nc')), domcfg_files=['./1_domain_cfg_50levels.nc'],nemo_kwargs=dict(datadir='./',chunks={'time_counter':1}))
        #print('data saved to disk')
        #ds = xr.open_dataset('xnemogcm.nc').chunk({'t':30})
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

        # split the region in 4 parts.
        # The Taylor Column with a shallower than 2500m Bathymetry,
        # the Transition Zone between 2500 m and 3500 m Depth
        # the Halo, 14 grid points outside of the Transition Zone
        # the Remnant, everything else expect, the 10 outest grid points and land
        ds['mask_tc']=np.zeros((len(ds.y_c),len(ds.x_c)))*(ds.bathy_meter.where(ds.bathy_meter<2500))+1
        ds['mask_tc']=ds.mask_tc.where(((ds.mask_tc==1)&(ds.glamt<4.3)&(ds.glamt>0.5)&(ds.gphit>-65.5)),other=0)
        mask_tc=ds.mask_tc
        mask_tc[173,98]=1
        mask_tc[178,109]=0
        ds['mask_tc']=mask_tc
        mask_tc=mask_tc.rename("mask_tc")
        mask_tc.to_netcdf(Path(path_out)/f'mask_tc_m12.nc')

        mean_bathy = ds.bathy_meter.rolling(x_c=13,center=True).mean().rolling(y_c=13,center=True).mean()
        ds['mask_tr_all']=np.zeros((len(ds.y_c),len(ds.x_c)))*(mean_bathy.where(mean_bathy<3500))+1
        ds['mask_tr_all']=ds.mask_tr_all.where(((ds.mask_tr_all==1)&(ds.glamt<8)&(ds.gphit>-66.5)),other=0)
        ds['mask_tr']=ds.mask_tr_all-ds.mask_tc
        
        mask_tr=ds.mask_tr
        mask_tr=mask_tr.rename("mask_tr")
        #mask_tr.to_netcdf(Path(path_out)/f'mask_tr_{year}_m12.nc')

        ds['mask_ha'] = xr.DataArray(dilation(ds.mask_tr_all, disk(14)) - ds.mask_tr_all,dims=('y_c', 'x_c'))
        ds['mask_ha_all'] = ds.mask_ha+ds.mask_tr_all
        mask_ha=ds.mask_ha
        mask_ha=mask_ha.rename("mask_ha")
        mask_ha.to_netcdf(Path(path_out)/f'mask_ha_m12.nc')
        mask_ha_mod=xr.where((ds.gphit>-65)&(ds.glamt<5)&(ds.bathy_meter<5000),ds.mask_ha,0)
        ds['ha_mod']=mask_ha_mod
        
        print('compute Halo')
        deptht=ds.e3t_1d.cumsum(dim='z_c')- 0.5*ds.e3t_1d.isel(z_c=0)
        deptht=deptht.rename("deptht")
        #depth.to_netcdf(Path(path_out)/f'deptht_m12.nc')
        halo=ds.thetao.where((deptht>ds.mldr10_1)&(deptht<ds.bathy_meter)).max(dim='z_c')
        halo=halo.rename("halo")
        halo=halo.assign_attrs(long_name='maximum_temperature_below_MLD')
        halo.to_netcdf(Path(path_out)/f'halo_{year}_m12.nc')
        
        #compute  mean values of the Tmax in the Halo and Taylor Cap
        T_max_mean_ha=halo.where(ds.ha_mod==1).mean(dim={'x_c','y_c'})#.mean(dim='t').values
        T_max_mean_tc=halo.where(ds.mask_tc==1).mean(dim={'x_c','y_c'})#.mean(dim='t').values

        T_max_mean_ha=T_max_mean_ha.rename("T_max_mean_ha")
        T_max_mean_ha=T_max_mean_ha.assign_attrs(long_name='mean maximum_temperature_below_MLD over the modified Halo region')
        T_max_mean_ha.to_netcdf(Path(path_out)/f'Tmax_mean_ha_{year}_m12.nc')
        T_max_mean_tc=T_max_mean_tc.rename("T_max_mean_tc")
        T_max_mean_tc=T_max_mean_tc.assign_attrs(long_name='mean maximum_temperature_below_MLD over the Taylor Cap region')
        T_max_mean_tc.to_netcdf(Path(path_out)/f'Tmax_mean_tc_{year}_m12.nc')

