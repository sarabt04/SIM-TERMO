#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 11:54:47 2025

@author: rosatrullcastanyer
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Configuració per usar LaTeX
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# DADES
x = [-0.83876, -0.93039, -1.03129, -1.14351, -1.26994, -1.4147, -1.58403, -1.78798, -2.04448, -2.39035, -2.92346, -4.14162]
y = [-0.23319, -0.25231, -0.27575, -0.30788, -0.35098, -0.39899, -0.45571, -0.543, -0.63111, -0.75502, -0.93395, -1.28013]

# Càlcul de la regressió lineal
coefficients, cov_matrix = np.polyfit(x, y, 1, cov = True)  # Ajust lineal de grau 1
polynomial = np.poly1d(coefficients)
m, n = coefficients

m_err, n_err = np . sqrt ( np . diag ( cov_matrix ) )
print(f'm = {m} ± {m_err}')
print(f'n= {n} ± {n_err}')

correlation_coefficient = np.corrcoef(x, y)[0, 1]
r_squared = correlation_coefficient ** 2
print(f'Coeficient de correlacio (R^2): {r_squared:.4f}')

# n = ln(A) --> A = e^n
A = np.e**n
inc_A = np.e**n_err
print(f'A = {A} ± {inc_A}')

# Calcular els valors ajustats (recta de regresió)
y_fit = polynomial(x)

# Crear la gráfica
plt.figure(figsize=(5, 4), dpi= 500)
plt.errorbar(x, y, fmt='o', color='b', label='Punts trobats amb la simulació', markersize=4)
plt.plot(x, y_fit, 'r--', label=f'Regressió lineal: $y = {coefficients[0]:.3f}x + {coefficients[1]:.3f}$')

#plt.title(r'$\ln{(\rho_l - \rho_g)}$ en función de $\ln{\left(1 - \frac{T}{T_c}\right)}$')
plt.xlabel(r'$\ln{\left(1 - \frac{T}{T_c}\right)}$')
plt.ylabel(r'$\ln{(\rho_l - \rho_g)}$')
plt.grid(False)
plt.legend()

plt.tick_params(axis='both', direction='out', which='both', top=True, right=True)

plt.tight_layout()
plt.show()
