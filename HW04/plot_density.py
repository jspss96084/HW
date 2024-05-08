import numpy as np
import scipy
import matplotlib.pyplot as plt

filename = 'lecture05_HW/density.dat'
N        = 1024
density  = np.fromfile( filename, 'float32' ).reshape(N,N)

#plt.imshow( density )
#plt.colorbar()
#plt.savefig( 'lecture05_HW/fig__density.png', bbox_inches='tight' )
#plt.show()

# define a convolution filter
def gaussian_filter(N=N, sigma=1):
    n = 3*sigma
    x = np.linspace(-n, n, 2*n+1)
    y = np.linspace(-n, n, 2*n+1)
    X, Y = np.meshgrid( x, y )
    mu_x, mu_y = np.mean(x), np.mean(y)
    factor = -((X-mu_x)**2 + (Y-mu_y)**2) / (2*(sigma**2))
    norm = 1/(2 *np.pi *(sigma**2))
    return norm*np.exp(factor)

def convolution(f):   
    f /= f.sum()                  # normalization
    f_pad0 = np.zeros( density.shape )   # zero-padded filter
    f_pad0[ 0:len(f[0]), 0:len(f[1]) ] = f
    f_pad0 = np.roll( f_pad0, -(len(f)//2) , axis=0)  # f_pad0 = [0.4, 0.2, 0.1, 0.0, ..., 0.0, 0.1, 0.2]
    f_pad0 = np.roll( f_pad0, -(len(f)//2) , axis=1)

    # convolution
    uk = np.fft.rfft2( density )
    fk = np.fft.rfft2( f_pad0 )
    u_con = np.fft.irfft2( uk*fk )

    return u_con

def display(density, u_con, variable):
    plt.subplot(2, 1, 1)
    plt.title('Before convolution')
    plt.imshow(density)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(2, 1, 2)
    plt.title('After convolution\n'+ r'Gaussian filter with $\sigma$ = {0} cells'.format(variable))
    plt.imshow(u_con)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.savefig( 'lecture05_HW/con_sigma{0}.png'.format(variable)) 

def power_spetrum(uk):
    uk_center = np.fft.fftshift(uk)
    uk_amp = np.abs(uk_center)
    uk_power = np.log(np.power(uk_amp, 2))
    
    return uk_power

u_con1 = convolution(f=gaussian_filter(sigma=10))
display(density=density, u_con=u_con1, variable=10)
u_con2 = convolution(f=gaussian_filter(sigma=100))
display(density=density, u_con=u_con2, variable=100)

uk = np.fft.fft2( density )
fk1 = np.fft.fft2(u_con1)
fk2 = np.fft.fft2(u_con2)

plt.subplot(2, 2, 1)
plt.title('Power spectrum of the original data')
uk_power = power_spetrum(uk)
plt.imshow(uk_power, \
           extent=[-len(uk_power[0])/2.0, len(uk_power[0])/2.0, -len(uk_power[0])/2.0,len(uk_power[0])/2.0])
plt.xlabel(r'$k_x$')
plt.ylabel(r'$k_y$')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.title('Power spectrum of the data from (b)')
fk_power1 = power_spetrum(fk1)
plt.imshow(fk_power1, \
           extent=[-len(fk_power1[0])/2.0, len(fk_power1[0])/2.0, -len(fk_power1[0])/2.0,len(fk_power1[0])/2.0])
plt.xlabel(r'$k_x$')
plt.ylabel(r'$k_y$')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.title('Power spectrum of the data from (c)')
fk_power2 = power_spetrum(fk2)
plt.imshow(fk_power2, \
           extent=[-len(fk_power2[0])/2.0, len(fk_power2[0])/2.0, -len(fk_power2[0])/2.0,len(fk_power2[0])/2.0])
plt.xlabel(r'$k_x$')
plt.ylabel(r'$k_y$')
plt.colorbar()
plt.tight_layout()
plt.savefig('lecture05_HW/power_spectra.png')
plt.show()