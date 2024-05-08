import numpy as np
import matplotlib.pyplot as plt
import yt

def readData(fileroot):
    arr = np.loadtxt(fileroot)
    lev = arr[:,0]
    x = arr[:,1]
    d = arr[:,2]
    u = arr[:,3]
    P = arr[:,4]

    return lev, x, d, u, P



def display_tube1d(lev, x, d, u, P, title):
    '''for single data'''

    plt.subplot(221)
    plt.plot(x, u, 'o', markerfacecolor='none')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')

    plt.subplot(222)
    plt.title(title)
    plt.step(x, lev)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$level$')

    plt.subplot(223)
    plt.plot(x, P, 'o', markerfacecolor='none')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$P$')

    plt.subplot(224)
    plt.plot(x, d, 'o', markerfacecolor='none')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\rho$')

    plt.tight_layout()
    plt.show()

def display_tube1d_multi(lev, x, d, u, P):
    '''for multiple data'''
    pass

def display_tube1d_compare(lev, x, d, u, P, title):
    from shocktube1dcalc import solver_analytic

    # by default it will create a the shock tube based on Sod's classic condition.
    shocktube = solver_analytic.ShockTube(rho_left=1.0, u_left=0.0, p_left=1.0, rho_right=0.125, u_right=0.0, p_right=0.1)

    mesh = np.linspace(0.0, 1.0, 50)
    analytic_solution = shocktube.get_analytic_solution(
        mesh, t=0.4
    )

    analytic_solution = np.array(analytic_solution)

    ref_x = analytic_solution[:,0]
    ref_d = analytic_solution[:,1]
    ref_u = analytic_solution[:,2]
    ref_P = analytic_solution[:,3]

    plt.subplot(221)
    plt.plot(x, u, 'o', markerfacecolor='none')
    plt.plot(ref_x, ref_u, '-r')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')

    plt.subplot(222)
    plt.title(title)
    plt.step(x, lev)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$level$')

    plt.subplot(223)
    plt.plot(x, P, 'o', markerfacecolor='none')
    plt.plot(ref_x, ref_P, '-r')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$P$')

    plt.subplot(224)
    plt.plot(x, d, 'o', markerfacecolor='none')
    plt.plot(ref_x, ref_d, '-r')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\rho$')

    plt.tight_layout()
    plt.show()