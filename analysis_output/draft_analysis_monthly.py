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

path_out=('./post_pro')

path_in=Path('./data_mons/')
years=2016
yeare=2017


if __name__ == '__main__':
    	
    from dask.distributed import Client
    client = Client(n_workers=15, threads_per_worker=1)
    print(client)
    for year in range(years,yeare):   
        print(year)
        for mon in [str(i).zfill(2) for i in range(1,3 )]:#13)]:
            ds1 = xnemogcm.open_nemo_and_domain_cfg(nemo_files=list(path_in.glob('*mean_fullsim_grid*.nc')), domcfg_files=['./1_domain_cfg_50levels.nc'],nemo_kwargs=dict(datadir='./'))
            ds1=ds1.mean(dim={'t','bnds'})
            ds = xnemogcm.open_nemo_and_domain_cfg(nemo_files=list(path_in.glob(f'MAUD36_{year}{mon}*.nc')), domcfg_files=['./1_domain_cfg_50levels.nc'],nemo_kwargs=dict(datadir='./',chunks={'time_counter':1}))
            ds=xr.merge((ds1.vo_fm,ds1.uo_fm,ds1.wo_fm,ds1.so_fm,ds1.thetao_fm,ds1.ssh_fm,ds))
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
  
            # just extracting fields from the dataset
            ds.mldr10_1.to_netcdf(Path(path_out)/f'mld_{year}{mon }_m36.nc')
            ds.zos.to_netcdf(Path(path_out)/f'ssh_{year}{mon}_m36.nc')
            ds.siconc.to_netcdf(Path(path_out)/f'sic_{year}{mon}_m36.nc')
            ds.sithic.to_netcdf(Path(path_out)/f'sit_{year}{mon}_m36.nc')

            # Define the Masks for Taylor Cap and Halo
            # The Taylor Column with a shallower than 2500m Bathymetry,
            # the Transition Zone between 2500 m and 3500 m Depth
            # the Halo, 39 grid points outside of the Transition Zone, this alings with the approx. the 5000m isobath
            ds['mask_tc']=np.zeros((len(ds.y_c),len(ds.x_c)))*(ds.bathy_meter.where(ds.bathy_meter<2500))+1
            ds['mask_tc']=ds.mask_tc.where(((ds.mask_tc==1)&(ds.glamt<4.3)&(ds.glamt>0.6)&(ds.gphit>-65.5)),other=0)
            mask_tc=ds.mask_tc
            # manipulate mask, so there are no single gird cells marked as Taylor Cap
            mask_tc[595,262]=1
            mask_tc[528:530,299]=1
            mask_tc[504,288]=1
            mask_tc[493,296]=0
            mask_tc[509,297]=0
            mask_tc[515:519,299]=0
            mask_tc[535:537,304]=0
            mask_tc[536,305]=0
            mask_tc[530:535,325:329]=0
            mask_tc=mask_tc.rename("mask_tc")
            ds['mask_tc']=mask_tc
            mask_tc.to_netcdf(Path(path_out)/f'mask_tc_m36.nc')
            mean_bathy = ds.bathy_meter.rolling(x_c=39,center=True).mean().rolling(y_c=39,center=True).mean()
            ds['mask_tr_all']=np.zeros((len(ds.y_c),len(ds.x_c)))*(mean_bathy.where(mean_bathy<3500))+1
            ds['mask_tr_all']=ds.mask_tr_all.where(((ds.mask_tr_all==1)&(ds.glamt<8)&(ds.gphit>-66.5)),other=0)
            ds['mask_tr']=ds.mask_tr_all-ds.mask_tc
            mask_tr=ds.mask_tr
            mask_tr=mask_tr.rename("mask_tr")
            mask_tr.to_netcdf(Path(path_out)/f'mask_tr_m36.nc')
            # general Halo mask, fully encircling Maud Rise  
            ds['mask_ha'] = xr.DataArray(dilation(ds.mask_tr_all, disk(42)) - ds.mask_tr_all,dims=('y_c', 'x_c'))
            ds['mask_ha_all'] = ds.mask_ha+ds.mask_tr_all
            mask_ha=ds.mask_ha
            mask_ha=mask_ha.rename("mask_ha")
            mask_ha.to_netcdf(Path(path_out)/f'mask_ha_m36.nc')
            # modified Halo mask, north of 65°S and west of 5°E - used for Halo properties 
            mask_ha_mod=xr.where((ds.gphit>-65)&(ds.glamt<5)&(ds.bathy_meter<5000),ds.mask_ha,0)
            ds['ha_mod']=mask_ha_mod
            mask_ha_mod=mask_ha_mod.rename("mask_ha_mod")
            mask_ha_mod.to_netcdf(Path(path_out)/f'mask_ha_mod_m36.nc')

            print('compute Tmax')
            # this section extracts the maximum temperature below the Mixed layer depth, this field is called "halo" in the code
            depth=ds.e3t_1d.cumsum(dim='z_c')- 0.5*ds.e3t_1d.isel(z_c=0)
            depth=depth.rename("deptht")
            #depth.to_netcdf(Path(path_out)/f'deptht_m36.nc')
            
            halo=ds.thetao.where((depth>ds.mldr10_1)&(depth<ds.bathy_meter)).max(dim='z_c')
            halo=halo.rename("halo")
            halo=halo.assign_attrs(long_name='maximum_temperature_below_MLD')
            halo.to_netcdf(Path(path_out)/f'halo_{year}{mon}_m36.nc')
        
            #compute spatial means values of the Tmax in the Halo and Taylor Cap
            T_max_mean_ha=halo.where(ds.ha_mod==1).mean(dim={'x_c','y_c'})
            T_max_mean_tc=halo.where(ds.mask_tc==1).mean(dim={'x_c','y_c'})
            T_max_mean_ha=T_max_mean_ha.rename("T_max_mean_ha")
            T_max_mean_ha=T_max_mean_ha.assign_attrs(long_name='mean maximum_temperature_below_MLD over the modified Halo region')
            T_max_mean_ha.to_netcdf(Path(path_out)/f'Tmax_mean_ha_{year}{mon}_m36.nc')
            T_max_mean_tc=T_max_mean_tc.rename("T_max_mean_tc")
            T_max_mean_tc=T_max_mean_tc.assign_attrs(long_name='mean maximum_temperature_below_MLD over the Taylor Cap region')
            T_max_mean_tc.to_netcdf(Path(path_out)/f'Tmax_mean_tc_{year}{mon}_m36.nc')


            print('compute heatflux')
            print('Temperature change of one grid cell')
            # this includes, when summed over volume lateral fluxes and induction 
            dT_adv_u = -1 * (ds.uo * ds.e2u * ds.e3u * grid.interp(ds.thetao,'X',boundary='extend')).rename({'x_f':'x_c'}).assign_coords({'x_c':ds.x_c})+ 1 * (ds.uo * ds.e2u * ds.e3u * grid.interp(ds.thetao,'X',boundary='extend')).shift(x_f=1).rename({'x_f':'x_c'}).assign_coords({'x_c':ds.x_c})
            dT_adv_v =(- 1 * (ds.vo * ds.e1v * ds.e3v * grid.interp(ds.thetao,'Y',boundary='extend')).rename({'y_f':'y_c'}) + 1 * (ds.vo * ds.e1v * ds.e3v * grid.interp(ds.thetao,'Y',boundary='extend')).shift(y_f=1).rename({'y_f':'y_c'})).assign_coords({'y_c':ds.y_c})
            dT_adv_3D=dT_adv_u+dT_adv_v

   
            print('split water column in 3 different levels')
            # level 1 --> surface  to 200m
            # level 2 --> 200m to 1000m
            # level 3 --> below 1000m
            lev1_mask=xr.where(ds.e3t_0.cumsum(dim='z_c')<=200, 1, ds.e3t_0*0)
            lev2_mask=xr.where((ds.e3t_0.cumsum(dim='z_c')>200)&(ds.e3t_0.cumsum(dim='z_c')<=1000), 1, ds.e3t_0*0)
            lev3_mask=xr.where((ds.e3t_0.cumsum(dim='z_c')>1000), 1, ds.e3t_0*0)

            # Sum of Temperature change of each level in the vertical 
            dT_adv_l1=dT_adv_3D.where(lev1_mask==1).sum('z_c')
            dT_adv_l2=dT_adv_3D.where(lev2_mask==1).sum('z_c')
            dT_adv_l3=dT_adv_3D.where(lev3_mask==1).sum('z_c')
            dT_adv_l1=dT_adv_l1.rename("dT_adv_l1")
            dT_adv_l2=dT_adv_l2.rename("dT_adv_l2")
            dT_adv_l3=dT_adv_l3.rename("dT_adv_l3")
            dT_adv_l1=dT_adv_l1.assign_attrs(long_name='Temperature change in each grid cell due to horizontal advection, vertical sum over the top 200m')
            dT_adv_l2=dT_adv_l2.assign_attrs(long_name='Temperature change in each grid cell due to horizontal advection, vertical sum over 200m to 1000m ')
            dT_adv_l3=dT_adv_l3.assign_attrs(long_name='Temperature change in each grid cell due to horizontal advection, vertical sum over below 1000m ')
            dT_adv_l1.to_netcdf(Path(path_out)/f'dT_adv_l1_{year}{mon}_m36.nc')
            dT_adv_l2.to_netcdf(Path(path_out)/f'dT_adv_l2_{year}{mon}_m36.nc')
            dT_adv_l3.to_netcdf(Path(path_out)/f'dT_adv_l3_{year}{mon}_m36.nc')
            # Compute the vertical transport between the levels
            hfw_l1=(ds.wo*ds.e2t*ds.e1t*grid.interp(ds.thetao,'Z',boundary='extend')).where(grid.interp(lev1_mask,'Z',boundary='extend')==0.5,other=0).sum('z_f')
            hfw_l2=(ds.wo*ds.e2t*ds.e1t*grid.interp(ds.thetao,'Z',boundary='extend')).where(grid.interp(lev3_mask,'Z',boundary='extend')==0.5,other=0).sum('z_f')
            hfw_l1=hfw_l1.rename("hfw_l1")
            hfw_l2=hfw_l2.rename("hfw_l2")
            hfw_l1=hfw_l1.assign_attrs(long_name='Upward temperature transport at a deep of 200m ')
            hfw_l2=hfw_l2.assign_attrs(long_name='Upward temperature transport at a deep of 1000m ')
            hfw_l1.to_netcdf(Path(path_out)/f'hfw_200_{year}{mon}_m36.nc')
            hfw_l2.to_netcdf(Path(path_out)/f'hfw_1000_{year}{mon}_m36.nc')
            # Compute the heat content of each level
            Tint_l1=grid.integrate(ds.thetao.where(lev1_mask==1),['Z'])
            Tint_l2=grid.integrate(ds.thetao.where(lev2_mask==1),['Z'])
            Tint_l3=grid.integrate(ds.thetao.where(lev3_mask==1),['Z'])
            Tint_l1=Tint_l1.rename('Tint_l1')
            Tint_l2=Tint_l2.rename('Tint_l2')
            Tint_l3=Tint_l3.rename('Tint_l3')
            Tint_l1=Tint_l1.assign_attrs(long_name='vertical integrated Temperature for the upper 200m')
            Tint_l2=Tint_l2.assign_attrs(long_name='vertical integrated Temperature for 200m to 1000m')
            Tint_l3=Tint_l3.assign_attrs(long_name='vertical integrated Temperature forbelow 1000m')
            Tint_l1.to_netcdf(Path(path_out)/f'Tint_l1_{year}{mon}_m36.nc')
            Tint_l2.to_netcdf(Path(path_out)/f'Tint_l2_{year}{mon}_m36.nc')
            Tint_l3.to_netcdf(Path(path_out)/f'Tint_l3_{year}{mon}_m36.nc')
            print('surface flux in the Taylor Cap')
            qtoai_tc=grid.integrate(ds.qt_oce_ai.where(ds.mask_tc==1),['X','Y'])
            qtoai_tc=qtoai_tc.rename("qtoai_tc")
            qtoai_tc=qtoai_tc.assign_attrs(long_name='total heat flux at ocean surface for Taylor Column interface oce-(ice+atm)')
            qtoai_tc.to_netcdf(Path(path_out)/f'qtoai_tc_{year}{mon}_m36.nc')

            print(' compute eddy fluxes of Temperature and Salinity')
            ds['u_prime']=ds.uo-ds.uo_fm
            ds['v_prime']=ds.vo-ds.vo_fm
            ds['w_prime']=ds.wo-ds.wo_fm

            # u'*S'
            ds['S_prime']=ds.so-ds.so_fm
            ds['T_prime']=ds.thetao-ds.thetao_fm
            print('Temperature change of one parcel')
            # this includes, when summed over volume lateral fluxes and induction 
            dT_eddy_adv_u = -1 * (ds.u_prime * ds.e2u * ds.e3u * grid.interp(ds.T_prime,'X',boundary='extend')).rename({'x_f':'x_c'}).assign_coords({'x_c':ds.x_c})+ 1 * (ds.u_prime * ds.e2u * ds.e3u * grid.interp(ds.T_prime,'X',boundary='extend')).shift(x_f=1).rename({'x_f':'x_c'}).assign_coords({'x_c':ds.x_c})
            dT_eddy_adv_v =(- 1 * (ds.v_prime * ds.e1v * ds.e3v * grid.interp(ds.T_prime,'Y',boundary='extend')).rename({'y_f':'y_c'}) + 1 * (ds.v_prime * ds.e1v * ds.e3v * grid.interp(ds.T_prime,'Y',boundary='extend')).shift(y_f=1).rename({'y_f':'y_c'})).assign_coords({'y_c':ds.y_c})
            dT_eddy_adv_3D=dT_eddy_adv_u+dT_eddy_adv_v
            # Sum of Temperature change of each level in the vertical 
            dT_eddy_adv_l1=dT_eddy_adv_3D.where(lev1_mask==1).sum('z_c')
            dT_eddy_adv_l2=dT_eddy_adv_3D.where(lev2_mask==1).sum('z_c')
            dT_eddy_adv_l3=dT_eddy_adv_3D.where(lev3_mask==1).sum('z_c')
            dT_eddy_adv_l1=dT_eddy_adv_l1.rename("dT_eddy_adv_l1")
            dT_eddy_adv_l2=dT_eddy_adv_l2.rename("dT_eddy_adv_l2")
            dT_eddy_adv_l3=dT_eddy_adv_l3.rename("dT_eddy_adv_l3")
            dT_eddy_adv_l1=dT_eddy_adv_l1.assign_attrs(long_name='Temperature change in each grid cell due to horizontal advection of the eddy component, vertical sum over the top 200m')
            dT_eddy_adv_l2=dT_eddy_adv_l2.assign_attrs(long_name='Temperature change in each grid cell due to horizontal advection of the eddy component, vertical sum over 200m to 1000m ')
            dT_eddy_adv_l3=dT_eddy_adv_l3.assign_attrs(long_name='Temperature change in each grid cell due to horizontal advection of the eddy component, vertical sum over below 1000m ')
            dT_eddy_adv_l1.to_netcdf(Path(path_out)/f'dT_eddy_adv_l1_{year}{mon}_m36.nc')
            dT_eddy_adv_l2.to_netcdf(Path(path_out)/f'dT_eddy_adv_l2_{year}{mon}_m36.nc')
            dT_eddy_adv_l3.to_netcdf(Path(path_out)/f'dT_eddy_adv_l3_{year}{mon}_m36.nc')

            # Vertical eddy transport between the levels
            eddy_hfw_l1=(ds.w_prime*ds.e2t*ds.e1t*grid.interp(ds.T_prime,'Z',boundary='extend')).where(grid.interp(lev1_mask,'Z',boundary='extend')==0.5,other=0).sum('z_f')
            eddy_hfw_l2=(ds.w_prime*ds.e2t*ds.e1t*grid.interp(ds.T_prime,'Z',boundary='extend')).where(grid.interp(lev3_mask,'Z',boundary='extend')==0.5,other=0).sum('z_f')
            eddy_hfw_l1=eddy_hfw_l1.rename("eddy_hfw_l1")
            eddy_hfw_l2=eddy_hfw_l2.rename("eddy_hfw_l2")
            eddy_hfw_l1=eddy_hfw_l1.assign_attrs(long_name='Upward temperature transport at a deep of 200m by the eddy component')
            eddy_hfw_l2=eddy_hfw_l2.assign_attrs(long_name='Upward temperature transport at a deep of 1000m by the eddy component')
            eddy_hfw_l1.to_netcdf(Path(path_out)/f'eddy_hfw_200_{year}{mon}_m36.nc')
            eddy_hfw_l2.to_netcdf(Path(path_out)/f'eddy_hfw_1000_{year}{mon}_m36.nc')
            
            # Compute T and S mean values for the subsurface level (200-1000m) used to estimate sigma0
            Tmean_l2=((ds.thetao*ds.e3t).where(lev2_mask==1)).sum('z_c')/(ds.e3t.where(lev2_mask==1)).sum('z_c')
            Tmean_l2=Tmean_l2.rename('Tmean_l2')
            Tmean_l2=Tmean_l2.assign_attrs(long_name='vertical averaged Temperature for 200m to 1000m')
            Tmean_l2.to_netcdf(Path(path_out)/f'Tmean_l2_{year}{mon}_m36.nc')

            Smean_l2=((ds.so*ds.e3t).where(lev2_mask==1)).sum('z_c')/(ds.e3t.where(lev2_mask==1)).sum('z_c')
            Smean_l2=Smean_l2.rename('Smean_l2')
            Smean_l2=Smean_l2.assign_attrs(long_name='vertical averaged Salinity for 200m to 1000m')
            Smean_l2.to_netcdf(Path(path_out)/f'Smean_l2_{year}{mon}_m36.nc')
         
            sig0_l2=gsw.sigma0(Smean_l2,Tmean_l2)
            sig0_l2=sig0_l2.rename("sig0_l2")
            sig0_l2=sig0_l2.assign_attrs(long_name='Sigma0 in the subsurface layer estimated from T and S means')
            sig0_l2.to_netcdf(Path(path_out)/f'sig0_l2_{year}{mon}_m36.nc')

            # Extract vertical water properties for the Taylor Cap (at 2.4°E, 64.7°S) and Halo ( at 2.4°E, 63.6°S)
            xloc_tc=268
            yloc_tc=553
            t_tc=ds.thetao.isel(x_c=xloc_tc,y_c=yloc_tc)
            s_tc=ds.so.isel(x_c=xloc_tc,y_c=yloc_tc)
            m_tc=ds.mldr10_1.isel(x_c=xloc_tc,y_c=yloc_tc)
            sic_tc=ds.siconc.isel(x_c=xloc_tc,y_c=yloc_tc)
            xloc_ha=268
            yloc_ha=643
            t_ha=ds.thetao.isel(x_c=xloc_ha,y_c=yloc_ha)
            s_ha=ds.so.isel(x_c=xloc_ha,y_c=yloc_ha)
            m_ha=ds.mldr10_1.isel(x_c=xloc_ha,y_c=yloc_ha)
            sic_ha=ds.siconc.isel(x_c=xloc_ha,y_c=yloc_ha)
            t_tc.to_netcdf(Path(path_out)/f't_tc_{year}{mon}_m36.nc')
            s_tc.to_netcdf(Path(path_out)/f's_tc_{year}{mon}_m36.nc')
            t_ha.to_netcdf(Path(path_out)/f't_ha_{year}{mon}_m36.nc')
            s_ha.to_netcdf(Path(path_out)/f's_ha_{year}{mon}_m36.nc')
            m_tc.to_netcdf(Path(path_out)/f'm_tc_{year}{mon}_m36.nc')
            m_ha.to_netcdf(Path(path_out)/f'm_ha_{year}{mon}_m36.nc')
            sic_tc.to_netcdf(Path(path_out)/f'sic_tc_{year}{mon}_m36.nc')
            sic_ha.to_netcdf(Path(path_out)/f'sic_ha_{year}{mon}_m36.nc')
            
