
#--------------------------------------------------------------------
# Test the Strong shock problem with the MUSCL-Hancock scheme
#--------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def demo(scheme:str, nghost:int, cfl:float):
   '''scheme:  (1) Lax-Friedrichs 
               (2) Lax_Wendroff
               (3) MUSCL-Hancock

      nghost = 1: Lax-Friedrichs, Lax_Wendroff
      nghost = 2: MUSCL-Hancock
      
      cfl = 1.0: Lax-Friedrichs, MUSCL-Hancock
      cfl = 0.8: Lax_Wendroff'''
   return scheme, nghost, cfl

scheme, nghost, cfl= demo(scheme='(3)', nghost=2, cfl=1.0)   # which scheme to domonstrate
mod = 2 # parm: 0, 1, 2
#--------------------------------------------------------------------
# parameters
#--------------------------------------------------------------------
# constants
L        = 1.0       # 1-D computational domain size
N_In     = 128       # number of computing cells
#cfl      = 1.0       # Courant factor
#nghost   = 2        # number of ghost zones
gamma    = 5.0/3.0   # ratio of specific heats
end_time = 0.4       # simulation time

# derived constants
N  = N_In + 2*nghost    # total number of cells including ghost zones
dx = L/N_In             # spatial resolution

# plotting parameters
nstep_per_image = 10     # plotting frequency


# -------------------------------------------------------------------
# reference solution
# -------------------------------------------------------------------
ref_solution = np.loadtxt('HW02/package/strong_shock.txt')
ref_x = ref_solution[:,0]
ref_d = ref_solution[:,1]
ref_u = ref_solution[:,2]
ref_v = ref_solution[:,3]
ref_w = ref_solution[:,4]
ref_P = ref_solution[:,5]

from scipy.interpolate import CubicSpline
def ref_func(x_arr, y_arr, x_new):
    f = CubicSpline(x_arr, y_arr) # build data frame
    y_new = f(x_new)

    return y_new


# -------------------------------------------------------------------
# define initial condition
# -------------------------------------------------------------------
def InitialCondition( x ):
#  Strong shock
   if ( x < 0.5*L ):
      d = 1.25e3  # density
      u = 0.0  # velocity x
      v = 0.0  # velocity y
      w = 0.0  # velocity z
      P = 5.0e2  # pressure
      E = P/(gamma-1.0) + 0.5*d*( u**2.0 + v**2.0 + w**2.0 )   # energy density
   else:
      d = 1.25e2
      u = 0.0
      v = 0.0
      w = 0.0
      P = 5.0
      E = P/(gamma-1.0) + 0.5*d*( u**2.0 + v**2.0 + w**2.0 )

#  conserved variables [0/1/2/3/4] <--> [density/momentum x/momentum y/momentum z/energy]
   return np.array( [d, d*u, d*v, d*w, E] )


# -------------------------------------------------------------------
# define boundary condition by setting ghost zones
# -------------------------------------------------------------------
def BoundaryCondition( U ):
#  outflow
   U[0:nghost]   = U[nghost]
   U[N-nghost:N] = U[N-nghost-1]


# -------------------------------------------------------------------
# compute pressure
# -------------------------------------------------------------------
def ComputePressure( d, px, py, pz, e ):
   P = (gamma-1.0)*( e - 0.5*(px**2.0 + py**2.0 + pz**2.0)/d )
   assert np.all( P > 0 ), "negative pressure !!"
   return P


# -------------------------------------------------------------------
# compute time-step by the CFL condition
# -------------------------------------------------------------------
def ComputeTimestep( U ):
   P = ComputePressure( U[:,0], U[:,1], U[:,2], U[:,3], U[:,4] )
   a = ( gamma*P/U[:,0] )**0.5
   u = np.abs( U[:,1]/U[:,0] )
   v = np.abs( U[:,2]/U[:,0] )
   w = np.abs( U[:,3]/U[:,0] )

#  maximum information speed in 3D
   max_info_speed = np.amax( u + a )
   dt_cfl         = cfl*dx/max_info_speed
   dt_end         = end_time - t

   return min( dt_cfl, dt_end )


# -------------------------------------------------------------------
# compute limited slope
# -------------------------------------------------------------------
def ComputeLimitedSlope( L, C, R, mod=0 ):
#  compute the left and right slopes
   slope_L = C - L
   slope_R = R - C

#  apply the van-Leer limiter
   slope_LR      = slope_L*slope_R
   if (mod==0):
      slope_limited = np.where( slope_LR>0.0, 2.0*slope_LR/(slope_L+slope_R), 0.0 )
   elif (mod==1):
      slope_limited = np.where( slope_LR>0.0, (slope_L+slope_R)/2, 0.0 )
   elif (mod==2):
      slope_limited = np.where( slope_LR>0.0, slope_LR**0.5, 0.0 )

   return slope_limited


# -------------------------------------------------------------------
# convert conserved variables to primitive variables
# -------------------------------------------------------------------
def Conserved2Primitive( U ):
   W = np.empty( 5 )

   W[0] = U[0]
   W[1] = U[1]/U[0]
   W[2] = U[2]/U[0]
   W[3] = U[3]/U[0]
   W[4] = ComputePressure( U[0], U[1], U[2], U[3], U[4] )

   return W


# -------------------------------------------------------------------
# convert primitive variables to conserved variables
# -------------------------------------------------------------------
def Primitive2Conserved( W ):
   U = np.empty( 5 )

   U[0] = W[0]
   U[1] = W[0]*W[1]
   U[2] = W[0]*W[2]
   U[3] = W[0]*W[3]
   U[4] = W[4]/(gamma-1.0) + 0.5*W[0]*( W[1]**2.0 + W[2]**2.0 + W[3]**2.0 )

   return U


# -------------------------------------------------------------------
# piecewise-linear data reconstruction
# -------------------------------------------------------------------
def DataReconstruction_PLM( U, mod ):

#  allocate memory
   W = np.empty( (N,5) )
   L = np.empty( (N,5) )
   R = np.empty( (N,5) )

#  conserved variables --> primitive variables
   for j in range( N ):
      W[j] = Conserved2Primitive( U[j] )

   for j in range( 1, N-1 ):
#     compute the left and right states of each cell
      slope_limited = ComputeLimitedSlope( W[j-1], W[j], W[j+1], mod )

#     get the face-centered variables
      L[j] = W[j] - 0.5*slope_limited
      R[j] = W[j] + 0.5*slope_limited

#     ensure face-centered variables lie between nearby volume-averaged (~cell-centered) values
      L[j] = np.maximum( L[j], np.minimum( W[j-1], W[j] ) )
      L[j] = np.minimum( L[j], np.maximum( W[j-1], W[j] ) )
      R[j] = 2.0*W[j] - L[j]

      R[j] = np.maximum( R[j], np.minimum( W[j+1], W[j] ) )
      R[j] = np.minimum( R[j], np.maximum( W[j+1], W[j] ) )
      L[j] = 2.0*W[j] - R[j]

#     primitive variables --> conserved variables
      L[j] = Primitive2Conserved( L[j] )
      R[j] = Primitive2Conserved( R[j] )

   return L, R


# -------------------------------------------------------------------
# convert conserved variables to fluxes
# -------------------------------------------------------------------
def Conserved2Flux( U ):
   flux = np.empty( 5 )

   P = ComputePressure( U[0], U[1], U[2], U[3], U[4] )
   u = U[1] / U[0]

   flux[0] = U[1]
   flux[1] = u*U[1] + P
   flux[2] = u*U[2]
   flux[3] = u*U[3]
   flux[4] = u*( U[4] + P )

   return flux


# -------------------------------------------------------------------
# Roe's Riemann solver
# -------------------------------------------------------------------
def Roe( L, R ):
#  compute the enthalpy of the left and right states: H = (E+P)/rho
   P_L = ComputePressure( L[0], L[1], L[2], L[3], L[4] )
   P_R = ComputePressure( R[0], R[1], R[2], R[3], R[4] )
   H_L = ( L[4] + P_L )/L[0]
   H_R = ( R[4] + P_R )/R[0]

#  compute Roe average values
   rhoL_sqrt = L[0]**0.5
   rhoR_sqrt = R[0]**0.5

   u  = ( L[1]/rhoL_sqrt + R[1]/rhoR_sqrt ) / ( rhoL_sqrt + rhoR_sqrt )
   v  = ( L[2]/rhoL_sqrt + R[2]/rhoR_sqrt ) / ( rhoL_sqrt + rhoR_sqrt )
   w  = ( L[3]/rhoL_sqrt + R[3]/rhoR_sqrt ) / ( rhoL_sqrt + rhoR_sqrt )
   H  = ( rhoL_sqrt*H_L  + rhoR_sqrt*H_R  ) / ( rhoL_sqrt + rhoR_sqrt )
   V2 = u*u + v*v + w*w
#  check negative pressure
   assert H-0.5*V2 > 0.0, "negative pressure!"
   a  = ( (gamma-1.0)*(H - 0.5*V2) )**0.5

#  compute the amplitudes of different characteristic waves
   dU     = R - L
   amp    = np.empty( 5 )
   amp[2] = dU[2] - v*dU[0]
   amp[3] = dU[3] - w*dU[0]
   amp[1] = (gamma-1.0)/a**2.0 \
            *( dU[0]*(H-u**2.0) + u*dU[1] - dU[4] + v*amp[2] + w*amp[3] )
   amp[0] = 0.5/a*( dU[0]*(u+a) - dU[1] - a*amp[1] )
   amp[4] = dU[0] - amp[0] - amp[1]

#  compute the eigenvalues and right eigenvector matrix
   EigenValue    = np.array( [u-a, u, u, u, u+a] )
   EigenVector_R = np.array(  [ [1.0, u-a,   v,   w,  H-u*a],
                              [1.0,   u,   v,   w, 0.5*V2],
                              [0.0, 0.0, 1.0, 0.0,      v],
                              [0.0, 0.0, 0.0, 1.0,      w],
                              [1.0, u+a,   v,   w,  H+u*a] ]  )

#  compute the fluxes of the left and right states
   flux_L = Conserved2Flux( L )
   flux_R = Conserved2Flux( R )

#  compute the Roe flux
   amp *= np.abs( EigenValue )
   flux = 0.5*( flux_L + flux_R ) - 0.5*amp.dot( EigenVector_R )

   return flux


# -------------------------------------------------------------------
# initialize animation
# -------------------------------------------------------------------
def init():
   line_d.set_xdata( x )
   line_u.set_xdata( x )
   line_p.set_xdata( x )
   return line_d, line_u, line_p


# -------------------------------------------------------------------
# update animation
# -------------------------------------------------------------------
def update( frame, mod=mod ):
   global t, U

#  for frame==0, just plot the initial condition
   if frame > 0:
      for step in range( nstep_per_image ):

#        set the boundary conditions
         BoundaryCondition( U )

#        estimate time-step from the CFL condition
         dt = ComputeTimestep( U )
         print( "t = %13.7e --> %13.7e, dt = %13.7e" % (t,t+dt,dt) )


         flux = np.empty( (N,5) )
         if (scheme=='(1)'):
            for j in range( nghost, N-nghost+1 ):
   #           the Lax-Friedrichs scheme: flux[j] is defined at j-1/2
               flux[j] = 0.5*(  Conserved2Flux( U[j] ) + Conserved2Flux( U[j-1] ) \
                              -dx/dt*( U[j] - U[j-1] )  )
         if (scheme=='(2)'):
            for j in range( nghost, N-nghost+1 ):
   #           Lax_Wendroff scheme
               U_half = 0.5*(  U[j] + U[j-1] ) \
                              - 0.5*dt/dx*( Conserved2Flux( U[j] ) - Conserved2Flux( U[j-1] ) )
               flux[j] = Conserved2Flux( U_half )
         if (scheme=='(3)'):
   #        data reconstruction
            L, R = DataReconstruction_PLM( U, mod )

   #        update the face-centered variables by 0.5*dt
            for j in range( 1, N-1 ):
               flux_L = Conserved2Flux( L[j] )
               flux_R = Conserved2Flux( R[j] )
               dflux  = 0.5*dt/dx*( flux_R - flux_L )
               L[j]  -= dflux
               R[j]  -= dflux

   #        compute fluxes
            for j in range( nghost, N-nghost+1 ):
   #           R[j-1] is the LEFT state at the j+1/2 inteface
               flux[j] = Roe( R[j-1], L[j]  )

#        update the volume-averaged input variables by dt
         U[nghost:N-nghost] -= dt/dx*( flux[nghost+1:N-nghost+1] - flux[nghost:N-nghost] )

#        update time
         t = t + dt
         if ( t >= end_time ):
            anim.event_source.stop()
            break

#  plot
   d = U[nghost:N-nghost,0]
   u = U[nghost:N-nghost,1] / U[nghost:N-nghost,0]
   P = ComputePressure( U[nghost:N-nghost,0], U[nghost:N-nghost,1], U[nghost:N-nghost,2],
                        U[nghost:N-nghost,3], U[nghost:N-nghost,4] )
   line_d.set_ydata( d )
   line_u.set_ydata( u )
   line_p.set_ydata( P )
#  ax[0].legend( loc='upper right', fontsize=12 )
#  ax[1].legend( loc='upper right', fontsize=12 )
#  ax[2].legend( loc='upper right', fontsize=12 )
   ax[0].set_title( 't = %6.3f' % (t) )

#  calculate the reference analytical solution and estimate errors
   u_ref = ref_func( ref_x, ref_u, x )
   u_err   = np.abs( u_ref - u ).sum()/N
   d_ref = ref_func( ref_x, ref_d, x )
   d_err   = np.abs( d_ref - d ).sum()/N
   P_ref = ref_func( ref_x, ref_P, x )
   P_err   = np.abs( P_ref - P ).sum()/N
   text.set( text='error = %6.3f' % (u_err) )
   text1.set( text='error = %6.3f' % (d_err) )
   text2.set( text='error = %6.3f' % (P_err) )

   return line_d, line_u, line_p


#--------------------------------------------------------------------
# main
#--------------------------------------------------------------------

# set initial condition
t = 0.0
x = np.empty( N_In )
U = np.empty( (N,5) )
for j in range( N_In ):
   x[j] = (j+0.5)*dx    # cell-centered coordinates
   U[j+nghost] = InitialCondition( x[j] )

# create figure
fig, ax = plt.subplots( 3, 1, sharex=True, sharey=False, dpi=140 )
fig.subplots_adjust( hspace=0.1, wspace=0.0 )
#fig.set_size_inches( 6.4, 12.8 )
ax[0].plot(ref_x, ref_d, '-')
line_d, = ax[0].plot( [], [], 'r-o', ls='-', markeredgecolor='k', markersize=3 )
ax[1].plot(ref_x, ref_u, '-')
line_u, = ax[1].plot( [], [], 'b-o', ls='-', markeredgecolor='k', markersize=3 )
ax[2].plot(ref_x, ref_P, '-')
line_p, = ax[2].plot( [], [], 'g-o', ls='-', markeredgecolor='k', markersize=3 )
ax[2].set_xlabel( 'x' )
ax[0].set_ylabel( 'Density' )
ax[1].set_ylabel( 'Velocity' )
ax[2].set_ylabel( 'Pressure' )
ax[0].set_xlim( 0.30, L )
ax[0].set_ylim( +0.0*np.max(ref_d), 1.0*np.max(ref_d) )
ax[1].set_ylim( -0.2*np.max(ref_u), 1.2*np.max(ref_u) )
ax[2].set_ylim( -0.2*np.max(ref_P), 1.0*np.max(ref_P) )
text = ax[0].text( 0.32, 200, '' )
text1 = ax[1].text( 0.32, 0.10, '' )
text2 = ax[2].text( 0.32, 50, '' )


# create movie
nframe = 99999999 # arbitrarily large
anim   = animation.FuncAnimation( fig, func=update, init_func=init,
                                 frames=nframe, interval=200, repeat=False )
plt.show()


