#!/usr/bin/env python
#  QG_pert_channel.m
#
# Solve the 1-Layer Quasi-Geostrophic (QG) Model
#
# Geometry: periodic in x and a channel in y
#
# Fields:
#   q : Potential Vorticity
#   u : zonal velocity
#   v : meridional velocity
# psi : streamfunction
#
# Parameters:
#  U  : background velocity
#  F  : Froude number
#  Q_y: F*U + beta
#
# Evolution Eqns:
#   q_t = - (u + U) q_x - (q_y + Q_y) v
#
# Potential Vorticity:
#   q = psi_xx + psi_yy - F psi
#   q_hat = - (K2 + F) psi_hat
#
# Geostrophy:
#   u = -psi_y
#   v =  psi_x
#
#   u_hat = -il*psi_hat =  il/(K2 + F)*q_hat
#   v_hat =  ik*psi_hat = -ik/(K2 + F)*b_hat
#
# Numerical Method:
# 1) FFT to compute the derivatives in spectral space
# 2) Adams-Bashforth for Advection
#
# Requires scripts:
#        flux_qg.py  - compute flux for the qg model

# Import libraries
from __future__ import division
#import numpy as np
from numpy import linalg as LA
import scipy as np
import matplotlib.pyplot as plt
#from scipy.fftpack import fft, ifft, fftn, ifftn, fftshift, fftfreq
from scipy.fftpack import fftshift, fftfreq
import sys
from time import gmtime, strftime
import os


try:
    import pyfftw
    from numpy import zeros as nzeros

    # Keep fft objects in cache for efficiency
    nthreads = 1
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(1e8)
    def empty(N, dtype="float", bytes=16):
        return pyfftw.n_byte_align_empty(N, bytes, dtype=dtype)

    def zeros(N, dtype="float", bytes=16):
        return pyfftw.n_byte_align(nzeros(N, dtype=dtype), bytes)

    # Monkey patches for fft
    ifft = pyfftw.interfaces.numpy_fft.ifft
    fft = pyfftw.interfaces.numpy_fft.fft
    fft2 = pyfftw.interfaces.numpy_fft.fft2
    ifft2 = pyfftw.interfaces.numpy_fft.ifft2
    irfft = pyfftw.interfaces.numpy_fft.irfft
    rfft = pyfftw.interfaces.numpy_fft.rfft
    rfft2 = pyfftw.interfaces.numpy_fft.rfft2
    irfft2 = pyfftw.interfaces.numpy_fft.irfft2
    ifftn = pyfftw.interfaces.numpy_fft.ifftn
    fftn = pyfftw.interfaces.numpy_fft.fftn
    irfftn = pyfftw.interfaces.numpy_fft.irfftn
    rfftn = pyfftw.interfaces.numpy_fft.rfftn

except:
    print Warning("Install pyfftw, it is much faster than numpy fft")

# Make directory to store pictures in
fname = strftime("%Y-%m-%d %H;%M;%S", gmtime())
os.mkdir(fname)
cnt = 0

def filt(k,kcut,beta,alph):

    #SPINS default parameters: 0.6, 2.0, 20.0

    knyq = max(k)
    kxcut = kcut*knyq
    filt = np.ones_like(k)
    filt = np.exp(-alph*((np.absolute(k) - kxcut)/(knyq - kxcut))**beta)*(np.absolute(k)>kxcut) + (np.absolute(k)<=kxcut)

    return filt

def plot_q_qhat(q, t):

    # Plot Potential Vorticity
    plt.clf()
    plt.subplot(2,1,1)
    plt.pcolormesh(xx/1e3,yy/1e3,q)
    plt.colorbar()
    plt.axes([-Lx/2e3, Lx/2e3, -Ly/2e3, Ly/2e3])
    name = "PV at t = %5.2f" % (t/(3600.0*24.0))
    plt.title(name)

    # compute power spectrum and shift ffts
    qe = np.vstack((q0,-np.flipud(q)))
    qhat = (np.absolute(fftn(qe)))
    kx = fftshift((parms.ikx/parms.ikx[0,1]).real)
    ky = fftshift((parms.iky/parms.iky[1,0]).real)
    qhat = fftshift(qhat)

    Sx, Sy = int(parms.Nx/2), parms.Ny

    # Plot power spectrum
    plt.subplot(2,1,2)
    #plt.pcolor(kx[Sy:Sy+20,Sx:Sx+20],ky[Sy:Sy+20,Sx:Sx+20],qhat[Sy:Sy+20,Sx:Sx+20])
    plt.pcolor(kx[Sy:int(1.5*Sy),Sx:int(1.5*Sx)],ky[Sy:int(1.5*Sy),Sx:int(1.5*Sx)],
               qhat[Sy:int(1.5*Sy),Sx:int(1.5*Sx)])
    #plt.axis([0, 10, 0, 10])
    plt.colorbar()
    name = "PS at t = %5.2f" % (t/(3600.0*24.0))
    plt.title(name)

    plt.draw()
    plt.savefig(fname + os.sep + '%03d.png' % (cnt))

def flux_qg(q, parms):

    # - (u + U) q_x - (q_y + Q_y) v
    qe = np.vstack((q,-np.flipud(q)))
    qe_hat = fftn(qe)

    # Compute gradient of PV
    q_x = (ifftn( parms.ikx*qe_hat)).real
    q_y = (ifftn( parms.iky*qe_hat)).real

    # Compute streamfunction
    psie_hat = parms.K2Fi*qe_hat
    psi = (ifftn(psie_hat)).real

    # Compute physical velocities
    u = (ifftn(-parms.iky*psie_hat)).real
    v = (ifftn( parms.ikx*psie_hat)).real

    # Restrict to physical domain
    q_x = q_x[0:parms.Ny,:]
    q_y = q_y[0:parms.Ny,:]
    u   = u[0:parms.Ny,:]
    v   = v[0:parms.Ny,:]
    psi = psi[0:parms.Ny,:]

    # Compute flux
    flux = - (u + parms.U)*q_x - (q_y + parms.Q_y)*v

    # FJP: energy should include potential energy
    energy = 0.5*np.mean(u**2 + v**2) + np.mean(parms.F*psi**2)
    enstr  = np.mean(q**2)
    mass   = np.mean(psi)

    return flux, energy, enstr, mass


#######################################################
#        Parameters Class                             #
#######################################################

class Parms(object):
    """A class to solve the one-layer QG model in a channel."""

    def __init__(
        self,
        # Grid size parameters
        Nx=128,                     # x-grid resolution
        Ny=128,                     # y-grid resolution
        Lx=1000e3,                  # zonal domain size
        Ly=1000e3,                  # meridional domain size

        # Physical parameters
        g0  = 9.81,                 # (reduced) gravity
        H0  = 1000,                 # mean depth
        f0  = 1e-4,                 # Coriolis parameter
        beta= 1e-11,                # gradient of coriolis parameter

        # Timestepping parameters
        t0  = 0.0,                   # Initial time
        dt  = 300.,                  # Timestep
        tf  = 20.*3600.*24.,         # Final time
        npt = 12,                    # Frequency of plotting

        # Filter Parameters (Following SPINS)
        kcut = 0.6,
        bet  = 2.0,
        alph = 20.0,
    ):

        # Save parameters
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.g0 = g0
        self.H0 = H0
        self.beta = beta
        self.t0 = t0
        self.dt = dt
        self.tf = tf
        self.npt = npt
        self.tplot = dt*npt

        # Physical parameters
        dx  = Lx/Nx
        dy  = Ly/Ny
        F   = 0*(f0/(g0*H0))**2
        U   = 0

        self.dx = dx
        self.dy = dy
        self.F   = F
        self.U   = U
        self.Q_y = F*U + beta

        # Define Grid
        x = np.linspace(-Lx/2+dx/2,Lx/2-dx/2,Nx)
        y = np.linspace(-Ly/2+dy/2,Ly/2-dy/2,Ny)
        xx,yy = np.meshgrid(x,y)
        self.xx = xx
        self.yy = yy

        #  Define wavenumber (frequency)
        kx = 2*np.pi/Lx*np.hstack([range(0,int(Nx/2)), range(-int(Nx/2),0)])
        ky = np.pi/Ly*np.hstack([range(0,Ny), range(-Ny,0)])
        kxx, kyy = np.meshgrid(kx,ky)
        K2Fi = -1./(kxx**2 + kyy**2 + F)
        if F == 0:
            K2Fi[0,0] = 0.
        else:
            K2Fi[0,0] = -1./F

        # Save parameters
        self.ikx = 1j*kxx
        self.iky = 1j*kyy
        self.K2Fi = K2Fi
        self.xx = xx
        self.yy = yy

        # Exponential Filter
        sfiltx = filt(kx,kcut,bet,alph)
        sfilty = filt(ky,kcut,bet,alph)
        [sfiltxs, sfiltys] = np.meshgrid(sfiltx,sfilty)
        self.sfilt = sfiltxs*sfiltys


#######################################################
#        Solve 1-Layer QG model in a channel          #
#######################################################

def solve_qg(parms, q0):

    global cnt
    # Set parameters
    dt = parms.dt
    Nx = parms.Nx
    Ny = parms.Ny

    # initialize fields
    Nt = int(parms.tf/parms.dt)
    energy = np.zeros(Nt)
    enstr  = np.zeros(Nt)
    mass   = np.zeros(Nt)

    # Euler step
    t,ii = 0., 0
    NLnm, energy[0], enstr[0], mass[0] = flux_qg(q0, parms)
    q  = q0 + dt*NLnm;

    # AB2 step
    t,ii = parms.dt, 1
    NLn, energy[1], enstr[1], mass[1] = flux_qg(q, parms)
    q   = q + 0.5*dt*(3*NLn - NLnm)

    kx = fftshift((parms.ikx/parms.ikx[0,1]).real)
    ky = fftshift((parms.iky/parms.iky[1,0]).real)

    cnt = 1
    for ii in range(3,Nt+1):

        # AB3 step
        t = (ii-1)*parms.dt
        NL, energy[ii-1], enstr[ii-1], mass[ii-1] = flux_qg(q, parms)
        q  = q + dt/12*(23*NL - 16*NLn + 5*NLnm).real

        # Exponential Filter
        qe = np.vstack((q,-np.flipud(q)))
        qe = (ifftn(parms.sfilt*fftn(qe))).real
        q  = qe[0:Ny,:]

        # Reset fluxes
        NLnm = NLn
        NLn  = NL

        if (ii-0)%parms.npt==0:

            t = ii*dt
            plot_q_qhat(q, t)
            cnt += 1

    return q, energy, enstr, mass

#######################################################
#         Main Program                                #
#######################################################

# Numerical parameters

#FJP: fix up diagnostics
#FJP: filter does not stabilize.  Why not?
#FJP: copy pyfftw from 2dstrat_vort.py
# Set parameters
parms = Parms(Nx=64, Ny=64, dt=300, npt = 6*20, tf = 20*3600*24.)

# Initial Conditions
Lx = parms.Lx
Ly = parms.Ly
xx = parms.xx
yy = parms.yy
q0 = 1.0e-4*np.sin(-1.0*np.pi*(yy+Ly/2)/Ly)*np.cos(0*np.pi*(xx+Lx/2)/Lx)

# Prepare animation
plt.ion()
plt.clf()

plot_q_qhat(q0,0)
plt.draw()

# Find Solution
q, energy, enstr, mass = solve_qg(parms,q0)

plt.ioff()
plt.show()
