import numpy as np
import matplotlib.pyplot as plt
import time


# constants
L   = 1.0   # 1-D computational domain size
N   = 16   # number of computing cells
u0  = 1.0   # background density
amp = 0.5   # sinusoidal amplitude
cfl = 1.0   # Courant condition factor


# derived constants
dx     = L/(N-1)               # spatial resolution
dy     = L/(N-1)               # spatial resolution
dt      = cfl*dx**2.0/4      # time interval for data update



# define a reference analytical solution
def ref_func( x, y ):
    k = 2.0*np.pi/L   # wavenumber
    X = amp*np.sin( k*x )
    Y = amp*np.sin( k*y )
    return X*Y # u0 + amp*np.sin( k*x )*np.exp( -k**2.0*D*t )


# define a initial density distribution
def init_rho ( x, y ):
    k = 2*np.pi/L
    return -2*(amp**2)*(k**2)*np.sin( k*x )*np.sin( k*y )


# initial condition
t = 0.0
x = np.linspace( 0.0, L, N )   # cell-centered coordinates
y = np.linspace( 0.0, L, N )   # cell-centered 
X, Y = np.meshgrid( x, y )
rho = init_rho(X, Y)
u_ref = ref_func( X, Y ) +np.ones(np.shape(X)) # initial density distribution
u = np.ones(np.shape(X))
method = '(3)' # (1)Jacobi (2)Gauss Seidel (3)SOR


# create figure
fig, ax = plt.subplots(2, 1, figsize = (3, 6))
im_num = ax[0].imshow( u    , norm= "linear", vmax=1.5, vmin=0.0, label='Numerical' )
im_ref = ax[1].imshow( u_ref, norm= "linear", vmax=1.5, vmin=0.0, label='Reference' )
fig.colorbar(im_num, ax=ax)


#def init():
#   im_num.set_data( u )
#   im_ref.set_data( u_ref )
#   return im_num, im_ref


def main():
    global t, u

    count = 0
    error = 0.
    

    t1 = time.time()
    while True:
#     back up the input data
        u_in = u.copy()

        err = 0.
#     update all cells
        if method == '(1)':
            for i in range( N-1 ):
                for j in range( N-1 ):
                    res=( u_in[i+1,j] + u_in[i-1,j] + u_in[i,j+1] + u_in[i,j-1] \
                             - 4*u_in[i,j] - dx**2*rho[i,j] )/(dx**2)
                    u[i,j] = u_in[i,j] + dt*res
                    err = err +dx**2*np.abs((res/u[i,j]))/N**2      # calculate error in this step
                    
        elif method == '(2)':
            for i in range( N-1 ):
                for j in range( N-1 ):
                    res=( u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] \
                            - 4*u_in[i,j] - dx**2*rho[i,j] )/(dx**2)
                    u[i,j] = u_in[i,j] + dx**2*res/4
                    err = err +dx**2*np.abs((res/u[i,j]))/N**2
                    
        elif method == '(3)':
            for i in range( N-1 ):
                for j in range( N-1 ):
                    res=( u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] \
                            - 4*u_in[i,j] - dx**2*rho[i,j] )/(dx**2)
                    w=1.7
                    u[i,j] = u_in[i,j] + w*dx**2*res/4
                    err = err +dx**2*np.abs((res/u[i,j]))/N**2      # calculate error in this step
                                    

#     update time
        t = t + dt
        count = count +1
        if ( err < 1.e-7 ):
            print(err)
            t2 = time.time()   
            print("wall-clock time = %2.5fs" % (t2-t1),"iteration=%s" %(count))       
            break

#  calculate the reference analytical solution and estimate errors
    error   = np.abs( u_ref - u ).sum()/N

#  plot
    im_num.set_data( u )
    im_ref.set_data( u_ref )
    ax[0].set_title("error = %2.3e \n Iterations = %d" % (error, count))
    plt.show()

    return error

### perform
main()