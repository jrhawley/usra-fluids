#!/usr/bin/env python
#  QG.m
#
# Solve the Barotropic Quasi-Geostrophic (QG) Model
#
# Geometry: periodic in x and y
#
# Fields:
#   q : Potential Vorticity
#   u : zonal velocity
#   v : meridional velocity
# psi : streamfunction (not used)
#
# Evolution Eqns:
#    q_t = - u q_x - v q_y
#
# Potential Vorticity:
#   q = psi_xx + psi_yy = v_x - u_y
#   q_hat = - K2 psi_hat
#
# Geostrophy:
#   u = -psi_y
#   v =  psi_x
#
#   u_hat = -il*psi_hat = il/K2*q_hat
#   v_hat =  ik*psi_hat =-il/K2*b_hat
#
# Numerical Method:
# 1) FFT to compute the derivatives in spectral space
# 2) Adams-Bashforth for Advection
#
# Requires scripts:
#        flux_qg.py  - compute flux for the sqg model

# Import libraries
import numpy as np
import scipy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftn, ifftn
import pylab
import sys

class wavenum:
    pass

# FJP: to do
# 1) Make Arakawa version with FD2
# 2) Try spectral Arakawa model???

def flux_qg(q, parms):

    q_hat = fftn(q)

    # Compute physical velocities
    u = (ifftn(-parms.ikyoK2*q_hat)).real
    v = (ifftn( parms.ikxoK2*q_hat)).real

    # Compute gradient of buoyancy
    q_x = (ifftn( parms.ikx*q_hat)).real
    q_y = (ifftn( parms.iky*q_hat)).real

    # Compute flux
    flux = - u*q_x - v*q_y;

    energy = 0.5*np.mean(u**2 + v**2)
    enstr  = np.mean(q**2)

    return flux, energy, enstr

def main():

    # Grid Parameters
    sc  = 2
    Lx  = 200e3
    Ly  = Lx
    Nx  = 128*sc
    Ny  = Nx
    dx  = Lx/Nx
    dy  = Ly/Ny

    # Physical parameters
    #N0 = 1e-2

    # Temporal Parameters
    t0  = 0.0
    tf  = 10.*3600.*24.
    dt  = 3600/2/sc
    Nt  = int(tf/dt)
    npt = 12*sc
    tt  = np.arange(Nt)*dt

    print "Time parameters", t0, tf, dt, Nt

    # Define Grid
    x = np.linspace(-Lx/2+dx/2,Lx/2-dx/2,Nx)
    y = np.linspace(-Ly/2+dy/2,Ly/2-dy/2,Ny)
    xx,yy = np.meshgrid(x,y)

    #  Define wavenumber (frequency)
    kx = 2*np.pi/Lx*np.hstack([range(0,Nx/2+1), range(-Nx/2+1,0)])
    ky = 2*np.pi/Ly*np.hstack([range(0,Ny/2+1), range(-Ny/2+1,0)])
    kxx, kyy = np.meshgrid(kx,ky)
    K2i = 1/(kxx**2 + kyy**2)
    K2i[0,0] = 0

    # Modify class
    parms = wavenum()
    parms.ikx = 1j*kxx
    parms.iky = 1j*kyy
    parms.ikxoK2 = -1j*kxx*K2i
    parms.ikyoK2 = -1j*kyy*K2i

    # Filter Parameters
    kmax = max(kx);
    ks = 0.4*kmax;
    km = 0.5*kmax;
    alpha = 0.69*ks**(-1.88/np.log(km/ks));
    beta  = 1.88/np.log(km/ks);
    sfilt = np.exp(-alpha*(kxx**2 + kyy**2)**(beta/2.0));

    # Initial Conditions with plot
    q0  = 1.e-4*np.exp(-(xx**2 + (4.0*yy)**2)/(Lx/6.0)**2);
    energy = np.zeros(Nt)
    enstr = np.zeros(Nt)

    # Prepare animation
    plt.ion()
    plt.clf()
    plt.pcolormesh(xx/1e3,yy/1e3,q0)
    plt.colorbar()
    plt.title("buoyancy at t = 0.00")
    plt.draw()
    plt.show()

    # Euler step
    NLnm, energy[0], enstr[0] = flux_qg(q0, parms)
    q    = q0 + dt*NLnm;
    q    = (ifftn(sfilt*fftn(q))).real

    # AB2 step
    NLn, energy[1], enstr[1] = flux_qg(q, parms)
    q   = q + 0.5*dt*(3*NLn - NLnm)
    q   = (ifftn(sfilt*fftn(q))).real

    cnt = 2
    for ii in range(3,Nt+1):

        # AB3 step
        NL, energy[ii-1], enstr[ii-1] = flux_qg(q, parms);
        q  = q + dt/12*(23*NL - 16*NLn + 5*NLnm).real
        q  = (ifftn(sfilt*fftn(q))).real

        # Reset fluxes
        NLnm = NLn
        NLn  = NL

        if (ii-0)%npt==0:

            # make title
            t = ii*dt/(3600.0*24.0)
            name = "PV at t = %5.2f" % (t)

            # Plot PV (or streamfunction)
            plt.clf()
            plt.pcolormesh(xx/1e3,yy/1e3,q)
            plt.colorbar()
            plt.title(name)
            plt.draw()

            cnt = cnt+1
            pylab.savefig('images/foo'+str(ii).zfill(3)+'.png', bbox_inches='tight')
    plt.show()

    print "Error in energy is ", np.amax(energy-energy[0])/energy[0]
    print "Error in enstrophy is ", np.amax(enstr-enstr[0])/enstr[0]

    plt.ioff()
    plt.figure()
    fig, axarr = plt.subplots(2, sharex=True)
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    ax1.plot((energy-energy[0])/energy[0],'-ob',linewidth=2, label='Energy')
    ax1.set_title('Energy')
    ax2.plot((enstr-enstr[0])/enstr[0],'-or', linewidth=2, label='Enstrophy')
    ax2.set_title('Enstrophy')
    plt.show()

main()
