import numpy as np
import matplotlib.pyplot as plt

#VALORS CONSTANTS
Natoms = 500                #Fixat a 500 atoms
L = 1                       # container is a cube L on a side
V=L**3
mass = 4E-3/6E23            # helium mass
k = 1.4E-23                 # Boltzmann constant
T = 300                     # around room temperature
# Bruce Sherwood
win = 500
Natoms = 500                #Fixat a 500 atoms
Ratom = 0.03                # wildly exaggerated size of helium atom


#DEFINIM LA FÒRMULA PER CALCULAR TAU
def tau(d):
    return np.sqrt((mass)/(4*np.pi*k*T))*(V/(d*Natoms))**2


#CACLULEM tau(d) PER UN RANG DE d DES DE Ratom (utilitzat a la simulacio) FINS A dHe (mida real)
dsim=2*Ratom
dHe=1e-10
valors = np.linspace(dsim, dHe, 1000)

tau_segons_d=[]
for i in valors:
    tau_segons_d.append(tau(i))

print(tau(dsim))
print(tau(dHe))
plt.plot(valors,tau_segons_d,color='c', label='$\\tau(d)$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Diàmetre efectiu $d$ [m]')
plt.ylabel('Temps mitjà entre col·lisions $\\tau$ [s]')
plt.title('Dependència de $\\tau$ amb el diàmetre efectiu $d$')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.scatter(dsim,115e-9, color='r', label='$\\tau(0.03)$', s=10)
plt.legend()
plt.show()