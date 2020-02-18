"""odeintegrate.py: Implements a few time integration routines for ODEs."""

__author__      = "Tony Saad"
__copyright__   = "Copyright 2018, Tony Saad under the MIT license"

import numpy as np
from scipy.optimize import fsolve

def forward_euler(rhs, f0, tend, dt):
    ''' Computes the forward_euler method '''
    nsteps = int(tend/dt)
    f = np.zeros(nsteps)
    f[0] = f0
    time = np.linspace(0,tend,nsteps)
    for n in np.arange(nsteps-1):
        f[n+1] = f[n] + dt * rhs(f[n], time[n])
    return time, f

def forward_euler_system(rhsvec, f0vec, tend, dt):
    '''
    Solves a system of ODEs using the Forward Euler method
    '''
    nsteps = int(tend/dt)
    neqs = len(f0vec)
    f = np.zeros( (neqs, nsteps) )
    f[:,0] = f0vec
    time = np.linspace(0,tend,nsteps)
    for n in np.arange(nsteps-1):
        t = time[n]
        f[:,n+1] = f[:,n] + dt * rhsvec(f[:,n], t)
    return time, f

def be_residual(fnp1, rhs, fn, dt, tnp1):
    '''
    Nonlinear residual function for the backward Euler implicit time integrator
    '''    
    return fnp1 - fn - dt * rhs(fnp1, tnp1)

def backward_euler(rhs, f0, tend, dt):
    ''' 
    Computes the backward euler method 
    :param rhs: an rhs function
    '''
    nsteps = int(tend/dt)
    f = np.zeros(nsteps)
    f[0] = f0
    time = np.linspace(0,tend,nsteps)
    for n in np.arange(nsteps-1):
        fn = f[n]
        tnp1 = time[n+1]
        fnew = fsolve(be_residual, fn, (rhs, fn, dt, tnp1))
        f[n+1] = fnew
    return time, f

def cn_residual(fnp1, rhs, fn, dt, tnp1, tn):
    '''
    Nonlinear residual function for the Crank-Nicolson implicit time integrator
    '''
    return fnp1 - fn - 0.5 * dt * ( rhs(fnp1, tnp1) + rhs(fn, tn) )

def crank_nicolson(rhs,f0,tend,dt):
    nsteps = int(tend/dt)
    f = np.zeros(nsteps)
    f[0] = f0
    time = np.linspace(0,tend,nsteps)
    for n in np.arange(nsteps-1):
        fn = f[n]
        tnp1 = time[n+1]
        tn = time[n]
        fnew = fsolve(cn_residual, fn, (rhs, fn, dt, tnp1, tn))
        f[n+1] = fnew
    return time, f