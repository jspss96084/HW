import matplotlib.pyplot as plt
import numpy as np

# performance scale with the number of threads
f = np.loadtxt("HW05/data_txtfile/CHEN_HW_128.txt")
#f = np.loadtxt("HW05/data_txtfile/WU_HW_128.txt")
t = f[:,0:2]
n = f[:,3]
grid = f[:,4]


t = np.mean(t, axis=1)
l = np.linspace(1,8,7)


plt.close()
a, b = np.polyfit(np.log10(n), np.log10(t), 1)
plt.plot(np.log10(n), np.log10(t), ls='-', c='k', marker='.', label='data')
plt.plot(np.log10(l), a*np.log10(l)+b, ls='--', c='g', label=str(format(a,'.2f'))+'x')
plt.xlabel('number of threads [log]', fontsize=14)
plt.ylabel('wall-clock time [log]', fontsize=14)
plt.legend(frameon=False)
plt.savefig("HW05/performance.pdf")

# parallel efficiency,
f = np.loadtxt("HW05/data_txtfile/CHEN_HW_eff.txt")
#f = np.loadtxt("HW05/data_txtfile/WU_HW_eff.txt")
t = f[:,0:2]
n = f[:,3]
grid = f[:,4]

f1 = np.loadtxt("HW05/data_txtfile/CHEN_HW_eff1.txt")
#f1 = np.loadtxt("HW05/data_txtfile/WU_HW_eff1.txt")
t1 = f1[:,0:2]
n1 = f1[:,3]
grid1 = f1[:,4]

t = np.mean(t, axis=1)
t1 = np.mean(t1, axis=1)
l = np.linspace(1,520,8)

eff = t1/t/n *100
plt.close()
#a, b = np.polyfit(np.log10(n), np.log10(t), 1)
plt.plot(grid, eff, ls='-', c='k', marker='.', label='data')
#plt.plot(l, a*np.log10(l)+b-0.1, ls='--', c='g', label=str(format(a,'.2f'))+'x + constant')
plt.xlabel('grids for each side (N)', fontsize=14)
plt.ylabel('efficiency [speed up / number of threads]', fontsize=14)
plt.legend(frameon=False)
plt.savefig("HW05/efficiency.pdf")