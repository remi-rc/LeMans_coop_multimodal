# -*- coding: utf-8 -*-
"""
Generate mode profiles using 1D LEE : functions
"""

# %% Modules
import numpy as np
import opal as op

def get_multimodal_injection(T, P, H_duct, U, dU, freq, N_cheb):
    # %% Environment
    env = op.Environment(T, P)
    
    # --------------- Set parameters related to the geometry  --------------- #
    env.duct_type = '1Dplan'  # type of duct config
    env.H_duct = H_duct  # Duct height (m)  (or radius if duct_type = 1Daxi)
    env.L_liner = 1  # Liner length (m)
    
    # ------------------ Update flow profile information -------------------- #
    env.U = U    # Velocity profile
    env.dU = dU  # Derivative of Velocity profile
    
    # Impedance is "rigid wall"
    Zs = 1e10 
    
    # Axial wavenumber
    k0x = freq * 2*np.pi / env.c_f
    
    # Top Mach
    Mach_max = U(H_duct/2) / env.c_f
    
    y_test = np.linspace(0, H_duct, 5000)
    Mach_mean = np.mean(U(y_test))
    
    # %% Numerical calculation of the wavenumbers
    num = {'N_cheb': N_cheb,     # Number of points in the mesh
           'parallel': False, # Parallel calculation
           'N_max': 1,       # Maximum number of modes to observe
           'm_a': [0]}        # Number of azimuthal modes (only for axisym cases)
    
    # Set up LEE eigenvalue problem
    study_case = op.StudyCase([], env, num)
    study_case.set_1DLEE_matrices() # Set global matrices of LEE eigen problem
    
    # %% Calculate wavenumbers and eigenfunctions
    k_mn, V_mn = study_case.solve_eigenproblem(freq, Zs)  # get the normalized wavenumbers
    k_mn = k_mn.flatten()
    
    # Denormalize the k_mn
    k_mn = k_mn / env.H_duct
    
    # %% Mode shapes
    # Use Jinyue Yang heuristics (see fig 4.5 of her PhD thesis)
    idx_Yang_heuristics = np.imag(k_mn) < np.real(k_mn) -k0x * Mach_mean / (Mach_mean**2 - 1)
    idx_Yang_heuristics *= np.real(k_mn) < k0x/2 * (1/(Mach_mean+1) + 1/(Mach_max))
    idx_Yang_heuristics *= np.imag(k_mn) <= 1e-5
    idx_Yang_heuristics *= np.imag(k_mn) >= -1e-2
    
    k_mn = k_mn[idx_Yang_heuristics]
    V_mn = V_mn[:, idx_Yang_heuristics]
    
    # %% Sort by descending value of real(k_mn) -> from least negative
    # (least absorptive) to most negative (most absorptive).
    I_sort = np.argsort(np.real(k_mn))  # ascending order
    I_sort = I_sort[::-1]  # descending order
    k_mn_sort = k_mn[I_sort]
    
    # Extract mode shapes for each field
    p_mn = V_mn[0::3, :]
    u_mn = V_mn[1::3, :]
    v_mn = V_mn[2::3, :]
    
    # Normalize by the max of pressure
    
    return k_mn_sort, p_mn, u_mn, v_mn