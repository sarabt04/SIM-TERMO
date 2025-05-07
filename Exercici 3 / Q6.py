#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 16:09:46 2025

@author: rosatrullcastanyer
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configuració per usar LaTeX
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

num_particules = 5000

d = 3 # pot ser 1,2 o 3
T = 300
m = 6.65*10**(-27)  # massa He
k_B = 1.38*10**(-23)
beta = 1 / (k_B*T)


def v_mitjana(d, m, beta):
    if d == 1:
        v1 = np.sqrt(2 / (np.pi * m * beta))
        return v1
    elif d == 2:
        v2 = np.sqrt(np.pi / (2 * m * beta))
        return v2
    elif d == 3:
        v3 = np.sqrt(8/ (np.pi * m * beta))
        return v3
    else:
        raise ValueError("Dimensió d ha de ser 1, 2 o 3")

v_avg = v_mitjana(d,m,beta)
print(v_avg)
# Atribuïm velocitats aleatories a cada partícula 
# Com que ens trobem en 3D --> 3 components de velocitats
v = np.random.uniform(- 1.5 * v_avg, 1.5 * v_avg, size=(num_particules, d)) # Genera una matriu de 100 files i d columnes || (x,y,z) per a cada partícula


def E_c(v):
    E = 1/2 * m * np.linalg.norm(v)**2  # si triga molt puc fer que directament sumi les compoentns de v pero al quadrat ja que l'arrel al estar ^2 s'elimina
    # E = 1/2 * m * np.sum(v**2)
    return E

# Mètode de MonteCarlo
n_passes = 200000
acceptat = 0

for passes in range(n_passes):
   particula = np.random.randint(num_particules) # Triem una partícula aleatòria
   v_particula = v[particula].copy() # Important fer copy per evitar que canvii el valor de la llista inicial que hem generat i aixi que no afecti si el pas no ha estat acceptat.
   E_particula = E_c(v_particula)
   
   # Nova velocitat candidata (fem una petita pertorbació aleatòria)
   delta_v = np.random.uniform(-(v_avg + 500), v_avg + 500, size = d)
   v_nova = v_particula + delta_v
   E_nova = E_c(v_nova)
   
   # Condició per acceptar el canvi o no que delta_E sigui negativa (faborable energeticament) o compleixi la condició de Metropolis per col·lectivitats canòniques
   
   delta_E = E_nova - E_particula
   if np.random.uniform(0,1) < min(1, np.exp(-beta * delta_E)): # Acceptem el canvi amb una certa probabilitat.
                       # el np.random.rand genera un numero aleatori entre 0 i 1
                       # np.random.rand > 0.2 fa que s'acceptin aprox el 20% de les vegades --> Fem el mateix pero amb la condicio de Metropolis per la col·lectivitat canònica.
       v[particula] = v_nova
       acceptat += 1
       # Si no es compleix la condició es manté la velocitat inicial

# La Taxa d'acceptació hauria de ser 30-60% per a una exploració eficient de l'espai de fases
print(f"Taxa d'acceptació: {acceptat / n_passes:.2f}")

# Comprovacions de que funciona, esperem trobar una distribució de velocitats de Maxwell-Boltzmann 

def distribucio_velocitats(v, d, m, beta):   # Distribució de velocitats: A v^(d-1) e^(1/2 (-m v^2 b))
    if d == 1:
        A1 = np.sqrt((2 * m * beta) / (np.pi))           # En 1D --> A = sqrt(2mb/pi): Distribució Gaussiana (com que moduls sempre postius --> Mitja gaussiana)      
        return A1 * np.exp(-0.5 * m * v**2 * beta)
    elif d == 2:
        A2 = m * beta
        return A2 * v * np.exp(-0.5 * m * v**2 * beta)             # En 2D --> A = mb
    elif d == 3:
        A3 = np.sqrt(2/np.pi) * (m * beta)**(3/2)  # En 3D --> A = sqrt(2/pi) (mb)^(3/2)--> Distribució de Maxwell - Boltzmann
        return A3 * v**2 * np.exp(-0.5 * m * v**2 * beta)
    else:
        raise ValueError("Dimensió d ha de ser 1, 2 o 3")

# Mòdul de cada velocitat
velocitats_modul = np.linalg.norm(v, axis=1)
# Escalat segons la distribució de Maxwell (teoria)
v_vals = np.linspace(0, np.max(velocitats_modul), 100)
dist_v = distribucio_velocitats(v_vals, d, m, beta)

# Crear la gráfica
plt.figure(figsize=(5, 4), dpi= 500)

plt.hist(velocitats_modul, bins=50, density=True, alpha=0.7, label='Simulació')
plt.plot(v_vals, dist_v, 'r--', label="Distribució teòrica")

plt.title(fr'Distribució de velocitats per $d=${d}')
plt.xlabel(r'Velocitat $|v|$')
plt.ylabel(r'Densitat de probabilitat')

plt.grid(False)
plt.legend()
plt.tick_params(axis='both', direction='out', which='both', top=True, right=True)
plt.tight_layout()

plt.show()
