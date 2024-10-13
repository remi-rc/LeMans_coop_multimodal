# -*- coding: utf-8 -*-
"""
Generate mode profiles using 1D LEE
"""

# %% Modules
import numpy as np
import matplotlib.pyplot as plt

import opal as op

# %% Environment
T = 20
P = 1.013e5
env = op.Environment(T, P)

# --------------- Set parameters related to the geometry  --------------- #
env.duct_type = '1Dplan'  # type of duct config
env.H_duct = 0.28  # Duct height (m)  (or radius if duct_type = 1Daxi)
env.L_liner = 1  # Liner length (m)

# ------------------ Update flow profile information -------------------- #
Mb = 0.3  # flow bulk Mach number
delta_shear = 1  # flow shear parameter
flow_model = 'Rienstra'  # Can be RIENSTRA, UNIFORM, not case sensitive
[U, dU] = op.get_1D_flow_profile([Mb, delta_shear], env, flow_model)

env.U = U    # Velocity profile
env.dU = dU  # Derivative of Velocity profile

# Impedance
Zs = 1e10 

# Frequency
freq = 1000

# Axial wavenumber
k0x = freq * 2*np.pi / env.c_f

# Top Mach
Mach_max = U(env.H_duct/2) / env.c_f
Mach_mean = Mb

# %% Numerical calculation of the wavenumbers
num = {'N_cheb': 250,     # Number of points in the mesh
       'parallel': False, # Parallel calculation
       'N_max': 1,       # Maximum number of modes to observe
       'm_a': [0]}        # Number of azimuthal modes (only for axisym cases)

# Set up LEE eigenvalue problem
study_case = op.StudyCase([], env, num)
study_case.set_1DLEE_matrices() # Set global matrices of LEE eigen problem

omega = 2*np.pi*freq
omega_adim = omega / env.c_f * env.H_duct
k_w = omega_adim
beta = np.sqrt(1-Mb**2)

# %% Calculate wavenumbers and eigenfunctions
k_mn, V_mn = study_case.solve_eigenproblem(freq, Zs)  # get the normalized wavenumbers
k_mn = k_mn.flatten()

# Denormalize the k_mn
k_mn = k_mn / env.H_duct

fig = plt.figure(1)
plt.plot(np.real(k_mn), np.imag(k_mn), marker = 'o', 
         fillstyle='none', linestyle='none')
plt.xlabel('real(k_mn)')
plt.ylabel('imag(k_mn)')
plt.xlim([-4*k_w/beta, 4*k_w/beta])
plt.ylim([-4*k_w/beta, 4*k_w/beta])

# %% Mode shapes
# Use Jinyue Yang heuristics (see fig 4.5 of her PhD thesis)
idx_Yang_heuristics = np.imag(k_mn) < np.real(k_mn) -k0x * Mach_mean / (Mach_mean**2 - 1)
idx_Yang_heuristics *= np.real(k_mn) < k0x/2 * (1/(Mach_mean+1) + 1/(Mach_max))
idx_Yang_heuristics *= np.imag(k_mn) <= 1e-5
idx_Yang_heuristics *= np.imag(k_mn) >= -1

k_mn = k_mn[idx_Yang_heuristics]
V_mn = V_mn[:, idx_Yang_heuristics]

plt.plot(np.real(k_mn), np.imag(k_mn), marker = '*', color="r",
         fillstyle='none', linestyle='none')

# %% Sort by descending value of real(k_mn) -> from least negative
# (least absorptive) to most negative (most absorptive).
I_sort = np.argsort(np.real(k_mn))  # ascending order
I_sort = I_sort[::-1]  # descending order
k_mn_sort = k_mn[I_sort]

# Extract mode shapes for each field
p_mn = V_mn[0::3, :]
u_mn = V_mn[1::3, :]
v_mn = V_mn[2::3, :]

fig = plt.figure()
for ij in range(len(k_mn_sort)):
    print(k_mn_sort[ij])
    plt.subplot(2, 2, 1)
    plt.plot(study_case.y_i, np.real(p_mn[:, ij]) / np.max(np.abs(p_mn[:, ij])))
    plt.subplot(2, 2, 2)
    plt.plot(study_case.y_i, np.imag(p_mn[:, ij]) / np.max(np.abs(p_mn[:, ij])))
    plt.subplot(2, 2, 3)
    plt.plot(study_case.y_i, np.real(u_mn[:, ij])  / np.max(np.abs(u_mn[:, ij])))
    plt.subplot(2, 2, 4)
    plt.plot(study_case.y_i, np.imag(u_mn[:, ij])  / np.max(np.abs(u_mn[:, ij])))
plt.xlabel('y (m)')
plt.ylabel('eigen modes')

plt.figure()
plt.plot(study_case.y_i)