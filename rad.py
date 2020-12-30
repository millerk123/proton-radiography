import h5py
# %matplotlib inline
# %matplotlib notebook
import numpy as np
import sys
import time
import glob
from joblib import Parallel, delayed
import gc
import os
import subprocess

# sys.path.append('/global/homes/m/millerk1/han-pyVisOS/')
# sys.path.append('/home/ripper/kyle/pyVisOS/')
sys.path.append('/u/home/m/millerk1/pyVisOS/')

import osh5io
import osh5def
import osh5vis
import osh5utils
import osh5visipy
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.stats as st
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.signal import hilbert

clrs=plt.rcParams['axes.prop_cycle'].by_key()['color'];
mpl.rcParams['font.serif'] = 'Latin Modern Roman'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

def main():
    # Input parameters, only sigz allowed, specify experimental parameters
    prop = 'x' # Propagation direction is x or y
    fldr = '.'
    sigma_t0 = 100e-15 # Standard deviation in time of the beam
    T0 = 55 # Initial temperature in MeV of the beam
    z_scr = 0.50 # Distance to screen [m]
    npart = int(1e7) # Total number of particles in the simulation
    rqm = -1.0 # Mass to charge ratio of particles
    dt_cour = 0.9 # fraction of a courant limit to make dt
    rcap = 27.0 # distance to cap propagation
    if_move = True
    n_threads = 12
    x_final = 50 # final distance to propagate (cm)
    n_frames = 101 # number of frames in the movie

    if_prop = True # Propagate the electrons
    if_plot = True # Make the plots
    if_par  = True # Parallelize
    if_movie = True # Make movie
    if_ion = True # Include ion density

    abs_unit = True # False # To plot in mm or not
    rpmax = 3 # None # ylim of radiograph plot
    rpmax_den = 0.17 # ylim of density plot

    if if_prop:

        # p_temp = np.sqrt(T0/0.511)
        p_temp = np.sqrt( np.square(T0/0.511+1) - 1 )
        gamma = np.sqrt(1+np.square(p_temp))
        if if_move:
            vd = 1
        else:
            vd = 0

        # Either specify the distance of the focus and the spot size at that location, along with beta_star
        # d_focus = np.array([0.8637, 0.7595]) # Distance of focus behind plasma, [m], [transverse,horizontal]
        # sigma_x0 = np.array([0.0104e-3, 0.0416e-3]) # Standard deviation in z,x [m]
        # beta_star = np.array([0.0039, 0.0711]) # Beta star value in z,x [m]
        # sigma_v0 = sigma_x0 / beta_star
        # M = np.sqrt((np.square(d_focus+z_scr)+np.square(beta_star))/(np.square(d_focus)+np.square(beta_star)))

        # Or specify the emittance, spot size at the screen, and desired magnification (z,x)
        em = np.array([2.89e-6,2.62e-6])/gamma # Numbers are the normalized emittance [m]
        sigma_s = np.array([4e-3,4e-4]) # Spot size at the screen [m]
        M = np.array([3.0,3.0]) # Desired magnification
        m2=np.square(M); m4=np.power(M,4); s2=np.square(sigma_s); s4=np.power(sigma_s,4); e2=np.square(em)
        d_focus = (-m4*e2-2*s4+2*m2*s4-M*s2*np.sqrt(4*s4-m2*e2)+np.power(M,3)*s2*np.sqrt(4*s4-m2*e2))/(4*m4*e2+4*np.square(m2-1)*s4)
        sigma_x0 = M*em/(2*np.sqrt((m2+1)*s2-M*np.sqrt(4*s4-m2*e2)))
        beta_star = np.square(sigma_x0)/em
        sigma_v0 = em/sigma_x0

        # Info that doesn't change
        flds = ['e1','e2','e3','b1','b2','b3']
        mode_sub = ['0-re','1-re','1-im']
        c = 3e8
        c_wp = np.load(fldr+'/c_wp.npy')[0]
        npart_str = '{:1.0E}'.format(npart).replace('+','')
        # Prepare the fields
        dz, zmin, zmax, dr, rmax, fields, mag_e, mag_b = prep_fields( fldr, flds, mode_sub )
        ximax = zmax - zmin
        rmax = np.min((rmax,rcap))
        if prop == 'x':
            i_prop = 1
            i_trans = 2
        elif prop == 'y':
            i_prop = 2
            i_trans = 1
        else:
            print("prop needs to be either 'x' or 'y'")
            raise
        suff = '_exp'

        # Convert inputs to simulation parameters
        sigma_t0_ = sigma_t0*c/c_wp
        sigma_x0_ = sigma_x0/c_wp
        # z_scr_ = z_scr/c_wp/gamma
        d_focus_ = d_focus/c_wp
        sigma_p0 = sigma_v0*gamma
        print(sigma_p0/p_temp*100)
        dt = dt_cour / np.sqrt( 1.0/np.square(dz) + 1.0/np.square(dr) + 1.0/np.square(2.0*dr*np.pi) )

        # Calculate momentum spread in the transverse direction based on beam evolution
        # sigma_p0 = sigma_x0 * np.sqrt(np.square(Mx)-1) / z_scr_

        # Initialize all transverse and longitudinal aspects of beam as if at center of plasma
        x = np.zeros((npart,5)) # z,x,y,r,xi=z-t

        # We choose a certain number of standard deviations in time that particles will actually go through the fields
        # Outside of that, the particles will just end up free streaming
        # zpmin = -rmax - std_t_; zpmax = zmax - zmin + rmax + std_t_

        # Alternatively, we can select a spread in z, just like we do in time
        # sigma_z0 = (zpmax - zpmin) / std_z
        x[:,0] = np.random.normal( loc=(zmax - zmin)/2, scale=sigma_x0_[0], size=npart )
        # sigma_pz0 = sigma_z0 * np.sqrt(np.square(Mz)-1) / z_scr_

        x[:,i_trans] = np.random.normal( scale=sigma_x0_[1], size=npart ) # Initialize transverse position

        # Initialize momentum; Gaussian in z and transversely, hot in propagation direction
        # print('Transverse momentum spread: {:.2f}%'.format(sigma_p0/p_temp*100))
        # print('Longitudinal momentum spread: {:.2f}%'.format(sigma_pz0/p_temp*100))
        p = np.zeros((npart,3))
        p[:,0] = np.random.normal( scale=sigma_p0[0], size=npart )
        p[:,i_trans] = np.random.normal( scale=sigma_p0[1], size=npart )
        p[:,i_prop] = -p_temp

        # We want to start all particles at x[:,i_prop] = r_max.  So we calculate how far to push particles back
        rgamma = 1.0 / np.sqrt( 1.0 + np.sum( np.square(p), axis=1 ) )
        # We need to push the particles to the r=0 plane, but we have to do it separately for each dimension
        # Calculate push time in longitudinal direction
        t_push = (0 - d_focus_[0]) / (p[:,i_prop] * rgamma)
        x[:,0] = x[:,0] + p[:,0] * rgamma * t_push
        # Calculate xi at this point
        x[:,4] = x[:,0] - vd * np.random.normal( scale=sigma_t0_, size=npart ) # Initialize the spread in time when particles are at center
        # Next calculate push time in transverse direction
        t_push = (0 - d_focus_[1]) / (p[:,i_prop] * rgamma)
        x[:,i_trans] = x[:,i_trans] + p[:,i_trans] * rgamma * t_push
        # Next we push all particles back to rmax
        t_push = rmax / (p[:,i_prop] * rgamma)
        x[:,:3] = x[:,:3] + p * np.tile(np.expand_dims(rgamma*t_push,1),3)
        x[:,4] = x[:,4] - vd * t_push
        x[:,3] = np.sqrt( np.square(x[:,1]) + np.square(x[:,2]) ) # Calculate r based on initial position

        # Simulate particles crossing the plasma
        nmin = 2*rmax / dt # Minimum number of time steps to traverse directly across the box

        if if_par:
            idx = ((np.arange(n_threads+1)*npart)/n_threads).astype(int)
            # max_nbytes=None,
            out=Parallel(n_jobs=n_threads,backend='multiprocessing') \
            (delayed(sim)(x[idx[i]:idx[i+1],:],p[idx[i]:idx[i+1],:],fields,T0,nmin,ximax,rmax,dt,rqm,dz,dr,vd,i_prop,
                          disp=i-np.floor(n_threads/2).astype(int),perc_mod=5,m0=1,m1=1) for i in np.arange(n_threads))
            for i, (xx,pp) in enumerate(out):
                x[idx[i]:idx[i+1],:]=xx
                p[idx[i]:idx[i+1],:]=pp
        else:
            x,p = sim(x,p,fields,T0,nmin,ximax,rmax,dt,rqm,dz,dr,vd,i_prop,perc_mod=5,m0=1,m1=1)

        print('All done')

        # Save output data to plot later if desired
        np.savez('{}/data_{}_T_{}_prop_{}{}'.format(fldr,npart_str,T0,prop,suff),x=x,p=p,sigma_x0=sigma_x0,sigma_t0=sigma_t0,
                 T0=T0,z_scr=z_scr,npart=npart,rqm=rqm,dt_cour=dt_cour,i_prop=i_prop,i_trans=i_trans,
                 d_focus=d_focus,beta_star=beta_star,sigma_p0=sigma_p0,M=M,rcap=rcap)

    if if_plot:
        # Make a movie of the final results propagating out
        npart_str = '{:1.0E}'.format(npart).replace('+','')
        c_wp = np.load(fldr+'/c_wp.npy')[0]
        suff = '_exp'

        title = 'radiography_density_{}_T_{}_prop_{}{}'.format(npart_str,T0,prop,suff)
        if not os.path.exists('{}/{}'.format(fldr,title)):
            os.makedirs('{}/{}'.format(fldr,title))
        dists = np.linspace(0,x_final,n_frames)

        dat_ele = osh5io.read_h5(glob.glob(fldr+'/charge*electrons*0-re*')[0])
        size_x = 2*(dat_ele.shape[0]-1)
        charge_e = np.zeros((size_x,dat_ele.shape[1]))
        charge_e[int(size_x/2):,:] = dat_ele.values[1:,:]
        charge_e[:int(size_x/2),:] = np.flip( dat_ele.values[1:,:], axis=0 )
        dat_fld = osh5io.read_h5(glob.glob(fldr+'/e2*1-re*')[0])
        fields = np.zeros_like(charge_e)
        fields[int(size_x/2):,:] = dat_fld.values[1:,:]
        fields[:int(size_x/2),:] = np.flip( dat_fld.values[1:,:], axis=0 )

        if if_ion:
            dat_ion = osh5io.read_h5(glob.glob(fldr+'/charge*ions*0-re*')[0])
            size_x = 2*(dat_ion.shape[0]-1)
            charge_ion = np.zeros((size_x,dat_ion.shape[1]))
            charge_ion[int(size_x/2):,:] = dat_ion.values[1:,:]
            charge_ion[:int(size_x/2),:] = np.flip( dat_ion.values[1:,:], axis=0 )

        zmin = dat_ele.axes[1].min
        zmax = dat_ele.axes[1].max
        # Load in saved data
        data=np.load('{}/data_{}_T_{}_prop_{}{}.npz'.format(fldr,npart_str,T0,prop,suff))
        rmax = data['rcap']
        sigma_x0 = data['sigma_x0']
        beta_star = data['beta_star']
        d_focus = data['d_focus']

        gc.collect()
        for i in np.arange(n_frames):
            if if_ion:
                plt.figure(figsize=(7,11))
                plt.subplot(311)
            else:
                plt.figure(figsize=(7,7))
                plt.subplot(211)
            sigma_x = proton_pulse( c_wp, data, rmax, zmin, T0, dat_ele.run_attrs['TIME'][0], dist=dists[i], perc=0.95, n_z=1200, n_r=1200, abs_unit=abs_unit, rpmax=rpmax )
            xlm = plt.xlim()
            ylm = plt.ylim()
            if if_ion:
                plt.subplot(312)
            else:
                plt.subplot(212)
            mag = np.sqrt( (1+np.square((d_focus+dists[i]*0.01)/beta_star)) / (1+np.square(d_focus/beta_star)) )
            print(mag)
            ext=np.array([0.5*((1-mag[0])*zmax+(1+mag[0])*zmin),0.5*((1+mag[0])*zmax+(1-mag[0])*zmin),-rmax*mag[1],rmax*mag[1]])
            if abs_unit:
                ext = ext * c_wp * 1e3
            plt.imshow(np.abs(charge_e),extent=ext,
                       vmax=1.2,cmap='CMRmap_r',aspect='auto')
            if abs_unit:
                plt.xlabel('$z$ [mm]')
                plt.ylabel('$x$ [mm]')
            else:
                plt.xlabel('$z$ [$c/\omega_p$]')
                plt.ylabel('$x$ [$c/\omega_p$]')
            plt.xlim(xlm)
            if rpmax_den == None:
                plt.ylim(ylm)
            else:
                plt.ylim([-rpmax_den,rpmax_den])
            cb=plt.colorbar(pad=0.09)
            cb.set_label('electron charge $[n_0]$')

            if if_ion:
                plt.subplot(313)
                plt.imshow(np.abs(charge_ion),extent=ext,
                           vmax=1.2,cmap='CMRmap_r',aspect='auto')
                if abs_unit:
                    plt.xlabel('$z$ [mm]')
                    plt.ylabel('$x$ [mm]')
                else:
                    plt.xlabel('$z$ [$c/\omega_p$]')
                    plt.ylabel('$x$ [$c/\omega_p$]')
                plt.xlim(xlm)
                if rpmax_den == None:
                    plt.ylim(ylm)
                else:
                    plt.ylim([-rpmax_den,rpmax_den])
                cb=plt.colorbar(pad=0.09)
                cb.set_label('ion charge $[n_0]$')
            plt.tight_layout()
            plt.savefig('{}/{}/{}_{:03d}.png'.format(fldr,title,title,i+1),dpi=300)
            plt.close()

        gc.collect()
        if if_movie:
            x, y = 300 * 7, 300 * 7
            if (x * y > 4000 * 2000):
                x, y = x / 2, y / 2

            stdout = subprocess.check_output(['ffmpeg', '-encoders', '-v', 'quiet'])
            for encoder in [b'libx264', b'mpeg4', b'mpeg']:
                if encoder in stdout:
                    break
            else:
                print('unsupported')
            subprocess.call(
                ["ffmpeg", "-framerate", "10", "-pattern_type", "glob", "-i", '{}/{}/*.png'.format(fldr,title), '-vcodec', encoder, '-vf',
                 'scale=' + str(x) + ':' + str(y) + ',format=yuv420p', '-y', '{}/{}/{}.mp4'.format(fldr,title,title)])


def save_c_wp(fldr,c_wp):
    # Save the c/w_p value to a certain directory
    c_wp = np.array([c_wp])
    np.save(fldr+'/c_wp.npy',c_wp)

def my_plot(data, os_data, window=20, title='', ax=None, clbl='', rlim=None, hilb=False, **kwargs):
    extent = [ os_data.axes[1].ax.min(), os_data.axes[1].ax.max(),
               -os_data.axes[0].ax.max(), os_data.axes[0].ax.max() ]
    dz = (os_data.axes[1].max - os_data.axes[1].min) / os_data.axes[1].size
    xlbl = osh5vis.axis_format(os_data.axes[1].attrs['LONG_NAME'], os_data.axes[1].attrs['UNITS'])
    ylbl = osh5vis.axis_format(os_data.axes[0].attrs['LONG_NAME'], os_data.axes[0].attrs['UNITS'])
    if ax==None:
        plt.imshow(data, extent=extent, aspect='auto', **kwargs)
        plt.xlabel(xlbl)
        plt.ylabel(ylbl)
        if rlim!=None:
            plt.ylim(-rlim,rlim)
        cb = plt.colorbar(pad=0.09)
        cb.set_label(clbl)
        plt.title(title)
        plt.autoscale(False)
        plt.twinx()
        if hilb:
            vals = np.abs(hilbert(data[data.shape[0]/2,:]))
            plt.plot(np.linspace(extent[0],extent[1],vals.size),vals,alpha=0.5,color='r')
            plt.ylabel('Hilbert envelope')
        else:
            weights = np.repeat(1.0, window)/window
            vals = np.convolve(data[data.shape[0]/2,:], weights, 'valid')
            plt.plot(np.linspace(extent[0]+dz*window/2,extent[1]-dz*window/2,vals.size),vals,alpha=0.5,color='r')
            plt.ylabel('Moving average')
        plt.ylim(top=vals.max()+1.4*(vals.max()-vals.min()))
    else:
        ax.imshow(data, extent=extent, aspect='auto', **kwargs)
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        if rlim!=None:
            ax.ylim(-rlim,rlim)
        cb = ax.colorbar(pad=0.09)
        cb.set_label(clbl)
        ax.set_title(title)
        ax.autoscale(False)
        ax1=ax.twinx()
        if hilb:
            vals = np.abs(hilbert(data[data.shape[0]/2,:]))
            ax1.plot(np.linspace(extent[0],extent[1],vals.size),vals,alpha=0.5,color='r')
            ax1.set_ylabel('Hilbert envelope')
        else:
            weights = np.repeat(1.0, window)/window
            vals = np.convolve(data[data.shape[0]/2,:], weights, 'valid')
            ax1.plot(np.linspace(extent[0]+dz*window/2,extent[1]-dz*window/2,vals.size),vals,alpha=0.5,color='r')
            ax1.set_ylabel('Moving average')
        plt.ylim(top=vals.max()+1.4*(vals.max()-vals.min()))
        
# Use fancy scipy interpolation function – very slow
def interp_fields( x, interp, fld='e', calc=True ):
    if fld=='e':
        idx=0
    else: # fld=='b'
        idx=3
    
    flds = np.zeros((x.shape[0],3))
    if calc:
        for f in np.arange(3): # Go over field components
            # Add contributions from the 3 modes
            inds = x[:,3] <= rmax
            flds[inds,f] = interp[f+idx][0]( x[inds,3], x[inds,0], grid=False ) + \
                      ( interp[f+idx][1]( x[inds,3], x[inds,0], grid=False ) * x[inds,1] + \
                        interp[f+idx][2]( x[inds,3], x[inds,0], grid=False ) * x[inds,2] ) / x[inds,3]
    
    return flds

# Just use nearest grid to get the field, much faster
def interp_nearest( x, fields, e, b, n, ximax, rmax, dz, dr, m0=1, m1=1 ):

    e[:] = 0.0
    b[:] = 0.0

    inds = np.logical_and( x[:,3] < rmax-dr/2, np.logical_and( x[:,4]<ximax-dz/2, x[:,4]>0.0 ) )

    i_r_ezp_br = (x[inds,3]/dr + 0.5).round().astype(int)
    i_r_er_bzp = (x[inds,3]/dr).round().astype(int)
    i_z_erp_bz = (x[inds,4]/dz).round().astype(int)
    i_z_ez_brp = (x[inds,4]/dz - 0.5).round().astype(int)

    x_r = x[inds,1]/x[inds,3]
    y_r = x[inds,2]/x[inds,3]
    x2_r2 = np.square(x_r)
    xy_r2 = x_r * y_r
    y2_r2 = np.square(y_r)
    
    # E first
    idx = 0
    # z component
    e[inds,0] = fields[0+idx,0,i_r_ezp_br,i_z_ez_brp] * m0 + \
                fields[0+idx,1,i_r_ezp_br,i_z_ez_brp] * m1 * x_r + \
                fields[0+idx,2,i_r_ezp_br,i_z_ez_brp] * m1 * y_r
    # x component
    e[inds,1] = fields[1+idx,0,i_r_er_bzp,i_z_erp_bz] * m0 * x_r - \
                fields[2+idx,0,i_r_ezp_br,i_z_erp_bz] * m0 * y_r + \
                fields[1+idx,1,i_r_er_bzp,i_z_erp_bz] * m1 * x2_r2 - \
                fields[2+idx,1,i_r_ezp_br,i_z_erp_bz] * m1 * xy_r2 + \
                fields[1+idx,2,i_r_er_bzp,i_z_erp_bz] * m1 * xy_r2 - \
                fields[2+idx,2,i_r_ezp_br,i_z_erp_bz] * m1 * y2_r2
    # y component
    e[inds,2] = fields[2+idx,0,i_r_ezp_br,i_z_erp_bz] * m0 * x_r + \
                fields[1+idx,0,i_r_er_bzp,i_z_erp_bz] * m0 * y_r + \
                fields[2+idx,1,i_r_ezp_br,i_z_erp_bz] * m1 * x2_r2 + \
                fields[1+idx,1,i_r_er_bzp,i_z_erp_bz] * m1 * xy_r2 + \
                fields[2+idx,2,i_r_ezp_br,i_z_erp_bz] * m1 * xy_r2 + \
                fields[1+idx,2,i_r_er_bzp,i_z_erp_bz] * m1 * y2_r2
                
    # B next
    idx = 3
    # z component
    b[inds,0] = fields[0+idx,0,i_r_er_bzp,i_z_erp_bz] * m0 + \
                fields[0+idx,1,i_r_er_bzp,i_z_erp_bz] * m1 * x_r + \
                fields[0+idx,2,i_r_er_bzp,i_z_erp_bz] * m1 * y_r
    # x component
    b[inds,1] = fields[1+idx,0,i_r_ezp_br,i_z_ez_brp] * m0 * x_r - \
                fields[2+idx,0,i_r_er_bzp,i_z_ez_brp] * m0 * y_r + \
                fields[1+idx,1,i_r_ezp_br,i_z_ez_brp] * m1 * x2_r2 - \
                fields[2+idx,1,i_r_er_bzp,i_z_ez_brp] * m1 * xy_r2 + \
                fields[1+idx,2,i_r_ezp_br,i_z_ez_brp] * m1 * xy_r2 - \
                fields[2+idx,2,i_r_er_bzp,i_z_ez_brp] * m1 * y2_r2
    # y component
    b[inds,2] = fields[2+idx,0,i_r_er_bzp,i_z_ez_brp] * m0 * x_r + \
                fields[1+idx,0,i_r_ezp_br,i_z_ez_brp] * m0 * y_r + \
                fields[2+idx,1,i_r_er_bzp,i_z_ez_brp] * m1 * x2_r2 + \
                fields[1+idx,1,i_r_ezp_br,i_z_ez_brp] * m1 * xy_r2 + \
                fields[2+idx,2,i_r_er_bzp,i_z_ez_brp] * m1 * xy_r2 + \
                fields[1+idx,2,i_r_ezp_br,i_z_ez_brp] * m1 * y2_r2

def dudt_boris( p_in, ep, bp, dt, rqm ):

    tem = 0.5 * dt / rqm

    ep = ep * tem
    utemp = p_in + ep

    gam_tem = tem / np.sqrt( 1.0 + np.sum( np.square(utemp), axis=1 ) )

    bp = bp * np.tile(np.expand_dims(gam_tem,1),3)

    p_in[:,:] = utemp + np.cross(utemp,bp,axis=1)

    bp = bp * np.tile(np.expand_dims( 2.0 / ( 1.0 + np.sum( np.square(bp), axis=1 ) ), 1), 3)

    utemp = utemp + np.cross(p_in,bp,axis=1)

    p_in[:,:] = utemp + ep

def dudt_boris_e( p_in, ep ):

    tem = dt / rqm

    p_in = p_in + ep * tem
    
    return p_in

def dudt_boris_b( p_in, bp ):

    tem = 0.5 * dt / rqm

    utemp = p_in

    gam_tem = tem / np.sqrt( 1.0 + np.sum( np.square(utemp), axis=1 ) )

    bp = bp * np.tile(np.expand_dims(gam_tem,1),3)

    p_in = utemp + np.cross(utemp,bp,axis=1)

    bp = bp * np.tile(np.expand_dims( 2.0 / ( 1.0 + np.sum( np.square(bp), axis=1 ) ), 1), 3)

    p_in = utemp + np.cross(p_in,bp,axis=1)
    
    return p_in

def sim(x,p,fields,T0,nmin,ximax,rmax,dt,rqm,dz,dr,vd,i_prop,disp=0,perc_mod=5,m0=1,m1=1):
    # Run simulation
    if nmin==0:
        return x,p
    n = 0
    print_percent = -1
    e = np.zeros((x.shape[0],3))
    b = np.zeros((x.shape[0],3))
    x = x.copy()
    p = p.copy()

    if disp==0: print("Running with T={}".format(T0))
    while n <= nmin or np.any(x[:,i_prop]>=-rmax):
        interp_nearest( x, fields, e, b, n, ximax, rmax, dz, dr, m0=m0, m1=m1 )
        dudt_boris( p, e, b, dt, rqm )
        rgamma = 1.0 / np.sqrt( 1.0 + np.sum( np.square(p), axis=1 ) )
        x[:,:3] = x[:,:3] + p * np.tile(np.expand_dims(rgamma*dt,1),3)
        x[:,3] = np.sqrt( np.square(x[:,1]) + np.square(x[:,2]) )
        x[:,4] = x[:,4] + p[:,0] * rgamma * dt - vd * dt

        percent = np.round(n/nmin*100).astype(int)
        if percent%perc_mod==0 and percent > print_percent:
            if disp==0: print('{}%'.format(percent))
            print_percent = percent
        n += 1
        
    return x,p

def prep_fields( fldr, flds, mode_sub ):
    # Get and store r/z arrays to use with all fields
    data=osh5io.read_h5(glob.glob(fldr+'/e1*0-re*')[0])
    nr = data.axes[0].size
    nz = data.axes[1].size
    dr = (data.axes[0].max - data.axes[0].min) / nr
    dz = (data.axes[1].max - data.axes[1].min) / nz
    rmax = data.axes[0].max
    zmin = data.axes[1].min; zmax = data.axes[1].max

    # Get interpolation function for each component
    # interp = [] # will be interp[fld_type,mode]
    fields = np.zeros((len(flds),len(mode_sub),nr,nz))
    for i,fld in enumerate(flds):
        # interp.append([]) # Append empty list to get a sort of two-dimensional list
        for j,mode in enumerate(mode_sub):
            data=osh5io.read_h5(glob.glob('{}/{}_cyl_m-{}*'.format(fldr,fld,mode))[0])
            # Use data array with zeros on the edges to fix interpolation outside of accepted values
            # interp_data = np.zeros((r_interp.size,z_interp.size))
            # interp_data[:-1,1:-1] = data.values
            # interp[i].append(RBS(r_interp,z_interp,interp_data,kx=3,ky=3))
            fields[i,j,:,:] = data.values # Or just store it in this array since interpolation is slow

    mag_e_half = np.sqrt( np.sum( np.square( np.sum( fields[:3,:2,:,:], axis=1 ) ), axis=0 ) )
    size_x = 2*(mag_e_half.shape[0]-1)
    mag_e = np.zeros((size_x,mag_e_half.shape[1]))
    mag_e[int(size_x/2):,:] = mag_e_half[1:,:]
    mag_e[:int(size_x/2),:] = np.flip( mag_e_half[1:,:], axis=0 )
    mag_b_half = np.sqrt( np.sum( np.square( np.sum( fields[3:,:2,:,:], axis=1 ) ), axis=0 ) )
    mag_b = np.zeros((size_x,mag_b_half.shape[1]))
    mag_b[int(size_x/2):,:] = mag_b_half[1:,:]
    mag_b[:int(size_x/2),:] = np.flip( mag_b_half[1:,:], axis=0 )
    return dz, zmin, zmax, dr, rmax, fields, mag_e, mag_b

def plot_proton( c_wp, T_init, t_sim, dist, x_all, i_trans, perc, n_z, n_r, abs_unit, rpmax ):
    if abs_unit:
        x_all[:,i_trans] = x_all[:,i_trans] * c_wp * 1e3
        x_all[:,0] = x_all[:,0] * c_wp * 1e3

    if rpmax is None:

        hist, edges = np.histogram(np.abs(x_all[:,i_trans]),bins=1500,normed=True)
        w_ind_r = np.argmax(np.cumsum(hist)*(edges[1]-edges[0])>perc)
        wind_r = edges[w_ind_r+1]

        hist, edges = np.histogram(x_all[:,0],bins=1500,normed=True)
        w_ind_u = np.argmax(np.cumsum(hist)*(edges[1]-edges[0])>perc)
        wind_u = edges[w_ind_u+1]
        w_ind_l = np.argmax(np.cumsum(np.flip(hist,axis=0))*(edges[1]-edges[0])>perc)
        wind_l = np.flip(edges,axis=0)[w_ind_l+1]

    else:

        wind_r = rpmax

        hist, edges = np.histogram(x_all[:,0],bins=1500,normed=True)
        w_ind_u = np.argmax(np.cumsum(hist)*(edges[1]-edges[0])>perc)
        wind_u = edges[w_ind_u+1]
        w_ind_l = np.argmax(np.cumsum(np.flip(hist,axis=0))*(edges[1]-edges[0])>perc)
        wind_l = np.flip(edges,axis=0)[w_ind_l+1]
        mean = (wind_l+wind_u)/2.0

        wind_l = -rpmax + mean
        wind_u = rpmax + mean

    vals = plt.hist2d(x_all[:,0],x_all[:,i_trans],bins=[np.linspace(wind_l,wind_u,n_z),
                                                 np.linspace(-wind_r,wind_r,n_r)],cmap='gray')
    cb=plt.colorbar(pad=0.09)
    cb.set_label('charge [a.u.]')
    if abs_unit:
        plt.xlabel('$z$ [mm]')
        plt.ylabel('$x$ [mm]')
    else:
        plt.xlabel('$z$ [$c/\omega_p$]')
        plt.ylabel('$x$ [$c/\omega_p$]')
    plt.title('Radiograph at {:4.1f} cm, $t=${:.2f} ps, $\sigma_x=${:.3f} mm, $\sigma_z=${:.3f} mm'.format(dist,t_sim*c_wp/3e8*1e12,
                                                                                         np.std(x_all[:,i_trans])*c_wp*1e3,np.std(x_all[:,0])*c_wp*1e3))
    return np.std(x_all[:,0])

def proton_pulse( c_wp, data, rmax, zmin, T_init, t_sim, dist=2, perc=0.95, n_z=400, n_r=220, abs_unit=False, rpmax=None ):

    # Get correct data
    x=data['x']; p=data['p']
    i_prop=data['i_prop']; i_trans=data['i_trans']

    # Calculate positions at the screen
    d = -dist * 1e-2 / c_wp - rmax
    if np.any(x[:,i_prop]-d<0):
        d = np.min(x[:,i_prop])
    rgamma = 1.0 / np.sqrt( 1.0 + np.sum( np.square(p), axis=1 ) )
    t = ( d - x[:,i_prop] ) / ( p[:,i_prop] * rgamma )
    xfinal = x[:,:3] + p * np.tile(np.expand_dims(rgamma*t,1),3)
    # Add the zmin offset to align with the simulation data
    xfinal[:,0] = xfinal[:,0] + zmin

    return plot_proton( c_wp, T_init, t_sim, dist, xfinal, i_trans, perc, n_z, n_r, abs_unit, rpmax )

if __name__ == "__main__":
    main()
