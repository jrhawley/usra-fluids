#!/usr/bin/env python
#  QG_pert_channel.m
#
# Solve the 2-Layer Quasi-Geostrophic (QG) Model
#
# Geometry: periodic in x and a channel in y
#
# Fields: (for each layer)
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

# Import libraries
from __future__ import division
#import numpy as np
from numpy import linalg as LA
import scipy as np
import matplotlib.pyplot as plt
#from scipy.fftpack import fft, ifft, fftn, ifftn, fftshift, fftfreq
from scipy.fftpack import fftshift, fftfreq
import sys

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
    sys.exit()

def filt(k,kcut,beta,alph):

    #SPINS default parameters: 0.6, 2.0, 20.0

    knyq = max(k)
    kxcut = kcut*knyq
    filt = np.ones_like(k)
    filt = np.exp(-alph*((np.absolute(k) - kxcut)/(knyq - kxcut))**beta)*(np.absolute(k)>kxcut) + (np.absolute(k)<=kxcut)

    return filt

def flux_qg(q, parms):

    Nx = parms.Nx
    Ny = parms.Ny

    psi = np.zeros((2*Ny,Nx,2),dtype=float)
    u = np.zeros((2*Ny,Nx,2),dtype=float)
    v = np.zeros((2*Ny,Nx,2),dtype=float)
    qe  = np.zeros((2*Ny,Nx,2),dtype=float)
    flux = np.zeros((Ny,Nx,2),dtype=float)
    q_x = np.zeros((2*Ny,Nx,2),dtype=float)
    q_y = np.zeros((2*Ny,Nx,2),dtype=float)
    qe_hat = np.zeros((2*Ny,Nx,2),dtype=complex)
    psie_hat = np.zeros((2*Ny,Nx,2),dtype=complex)

    # - (u + U) q_x - (q_y + Q_y) v
    for ii in range(2):
        # Extend and take FFT
        qe[:,:,ii] = np.vstack((q[:,:,ii],-np.flipud(q[:,:,ii])))
        qe_hat[:,:,ii] = fftn(qe[:,:,ii])

        # Compute gradient of PV
        q_x[:,:,ii] = (ifftn( parms.ikx*qe_hat[:,:,ii])).real
        q_y[:,:,ii] = (ifftn( parms.iky*qe_hat[:,:,ii])).real

    # Compute streamfunction
    psie_hat[:,:,0] = parms.W1[:,:,0]*qe_hat[:,:,0] + parms.W1[:,:,1]*qe_hat[:,:,1]
    psie_hat[:,:,1] = parms.W2[:,:,0]*qe_hat[:,:,0] + parms.W2[:,:,1]*qe_hat[:,:,1]

    for ii in range(2):
        psi[:,:,ii] = (ifftn(psie_hat[:,:,ii])).real

        # Compute physical velocities
        u[:,:,ii] = (ifftn(-parms.iky*psie_hat[:,:,ii])).real
        v[:,:,ii] = (ifftn( parms.ikx*psie_hat[:,:,ii])).real

        # Restrict to physical domain
        #q_x[:,:,ii] = q_x[0:parms.Ny,:,ii]
        #q_y[:,:,ii] = q_y[0:parms.Ny,:,ii]
        #u[:,:,ii]   = u[0:parms.Ny,:,ii]
        #v[:,:,ii]   = v[0:parms.Ny,:,ii]
        #psi[:,:,ii] = psi[0:parms.Ny,:,ii]

        # Compute flux
        flux[:,:,ii] = - (u[0:Ny,:,ii] + parms.U[ii])*q_x[0:Ny,:,ii]- (q_y[0:Ny,:,ii] + parms.Q_y[ii])*v[0:Ny,:,ii]

    # FJP: energy should include potential energy
    #energy = 0.5*np.mean(u**2 + v**2) + np.mean(parms.F*psi**2)
    #enstr  = np.mean(q**2)
    #mass   = np.mean(psi)

    return flux


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
        H1  = 500,                  # mean upper depth
        H2  = 500,                  # mean lower depth
        f0  = 1e-4,                 # Coriolis parameter        H0  = 1000,                 # mean depth
        beta= 1e-11,                # gradient of coriolis parameter
        rho1= 1010,                 # upper layer density
        rho2= 1020,                 # lower layer density

        # Timestepping parameters
        t0  = 0.0,                   # Initial time
        dt  = 300.,                  # Timestep
        tf  = 50.*3600.*24.,         # Final time
        npt = 60,                    # Frequency of plotting

        # Filter Parameters (Following SPINS)
        kcut = 0.4,
        bet  = 2.0,
        alph = 20.0,
    ):

        # Save parameters
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.gp = g0*(rho2 - rho1)/rho2
        self.H  = np.array([H1,H2])
        self.F  = f0**2/(self.gp*self.H)
        self.beta = beta
        self.t0 = t0
        self.dt = dt
        self.tf = tf
        self.npt = npt
        self.tplot = dt*npt

        # Physical parameters
        dx  = Lx/Nx
        dy  = Ly/Ny
        U   = np.array([1,-1])

        self.dx = dx
        self.dy = dy
        self.U   = U
        self.Q_y = self.F*U + beta

        # Define Grid
        x = np.linspace(-Lx/2+dx/2,Lx/2-dx/2,Nx)
        y = np.linspace(-Ly/2+dy/2,Ly/2-dy/2,Ny)
        xx,yy = np.meshgrid(x,y)
        self.xx = xx
        self.yy = yy

        #  Define wavenumber (frequency)
        kx = 2*np.pi/Lx*np.hstack([range(0,int(Nx/2)+1), range(-int(Nx/2)+1,0)])
        ky = np.pi/Ly*np.hstack([range(0,Ny+1), range(-Ny+1,0)])
        kxx, kyy = np.meshgrid(kx,ky)
        K2 = kxx**2 + kyy**2
        W1 = np.zeros((2*Ny,Nx,2))
        W2 = np.zeros((2*Ny,Nx,2))
        W1[:,:,0] = -(self.F[1] + K2)/(K2*(K2 + self.F[0] + self.F[1]))
        W1[:,:,1] = -self.F[0]/(K2*(K2 + self.F[0] + self.F[1]))
        W2[:,:,0] = -self.F[1]/(K2*(K2 + self.F[0] + self.F[1]))
        W2[:,:,1] = -(self.F[0] + K2)/(K2*(K2 + self.F[0] + self.F[1]))
        W1[0,0,0] = 0.0
        W1[0,0,1] = 0.0
        W2[0,0,0] = 0.0
        W2[0,0,1] = 0.0


        # Save parameters
        self.ikx = 1j*kxx
        self.iky = 1j*kyy
        self.W1 = W1
        self.W2 = W2
        self.xx = xx
        self.yy = yy

        # Exponential Filter (SPINS)
        sfiltx = filt(kx,kcut,bet,alph)
        sfilty = filt(ky,kcut,bet,alph)
        [sfiltxs, sfiltys] = np.meshgrid(sfiltx,sfilty)
        self.sfilt = sfiltxs*sfiltys

        # Filter Parameters (Eric)
        #kmax = max(kx);
        #ks = 0.4*kmax;
        #km = 0.5*kmax;
        #alpha = 0.69*ks**(-1.88/np.log(km/ks));
        #beta  = 1.88/np.log(km/ks);
        #self.sfilt = np.exp(-alpha*(kxx**2 + kyy**2)**(beta/2.0))

        #plt.figure()
        #plt.plot(kx/kx[1], sfiltx,'or')
        #plt.plot(kx/kx[1], np.exp(-alpha*(kx**2)**(beta/2.0)),'ob')
        #plt.show()
        #plt.figure()
        #plt.plot(ky/ky[1], sfilty,'or')
        #plt.plot(ky/ky[1], np.exp(-alpha*(ky**2)**(beta/2.0)),'ob')
        #plt.show()
        #sys.exit()

#######################################################
#        Solve 2-Layer QG model in a channel          #
#######################################################

def solve_qg(parms, q0):

    # Set parameters
    dt = parms.dt
    Nx = parms.Nx
    Ny = parms.Ny

    # initialize fields
    Nt = int(parms.tf/parms.dt)
    #energy = np.zeros(Nt)
    #enstr  = np.zeros(Nt)
    #mass   = np.zeros(Nt)

    # Euler step
    t,ii = 0., 0
    NLnm = flux_qg(q0, parms)
    q  = q0 + dt*NLnm;

    # AB2 step
    t,ii = parms.dt, 1
    NLn = flux_qg(q, parms)
    q   = q + 0.5*dt*(3*NLn - NLnm)

    qe = np.zeros((2*Ny,Nx,2), dtype=float)
    cnt = 2
    for ii in range(3,Nt+1):

        # AB3 step
        t = (ii-1)*parms.dt
        NL = flux_qg(q, parms)
        q  = q + dt/12*(23*NL - 16*NLn + 5*NLnm).real

        # Exponential Filter
        for jj in range(2):
            qe[:,:,jj] = np.vstack((q[:,:,jj],-np.flipud(q[:,:,jj])))
            qe[:,:,jj] = (ifftn(parms.sfilt*fftn(qe[:,:,jj]))).real
            q[:,:,jj]  = qe[0:Ny,:,jj]

        # Reset fluxes
        NLnm = NLn
        NLn  = NL

        if (ii-0)%parms.npt==0:

            # make title
            name = "PV at t = %5.2f" % (t/(3600.0*24.0))

            # Plot PV (or streamfunction)
            plt.clf()
            for jj in range(2):
                plt.subplot(1,2,jj+1)
                plt.pcolormesh(xx/1e3,yy/1e3,q[:,:,jj])
                plt.colorbar()
                plt.title(name)
                plt.axes([-Lx/2, Lx/2, -Ly/2, Ly/2])
            plt.draw()

            cnt += 1

    return q

#######################################################
#         Main Program                                #
#######################################################

# Numerical parameters

# Set parameters
parms = Parms()
Nx = parms.Nx
Ny = parms.Ny

# Initial Conditions
Lx = parms.Lx
Ly = parms.Ly
xx = parms.xx
yy = parms.yy

q0 = np.zeros((Ny,Nx,2))
q0[:,:,0]  = 1e-8*np.sin(1.0*np.pi*(yy+Ly/2)/Ly)*np.cos(2*np.pi*(xx+Lx/2)/Lx)
q0[:,:,1]  = 1e-8*np.sin(1.0*np.pi*(yy+Ly/2)/Ly)*np.cos(2*np.pi*(xx+Lx/2)/Lx)

q0 = 1e-10*np.random.rand(Ny,Nx,2)

# Prepare animation
plt.ion()
plt.clf()

# Plot Potential Vorticity
for jj in range(2):
    plt.subplot(1,2,jj+1)
    plt.pcolormesh(xx/1e3,yy/1e3,q0[:,:,jj])
    plt.colorbar()
    #plt.title(name)
    plt.axes([-Lx/2, Lx/2, -Ly/2, Ly/2])
    plt.draw()

plt.show()

# Find Solution
q = solve_qg(parms,q0)

plt.ioff()
plt.show()
