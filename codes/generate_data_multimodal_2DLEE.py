#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate data for a multimodal bench configuration (from Le Mans)
"""

# %% Modules
import numpy as np
import matplotlib.pyplot as plt

import opal as op

# %% Mesh information
mesh_path = '../mesh/duct_CTTM_1.msh'

markers = [
    {'ID': [1], 'type': 'source', 'source_idx': 1},
    {'ID': [3], 'type': 'outlet'},
    {'ID': [2, 5], 'type': 'wall'},
    {'ID': [4], 'type': 'impedance', 'impedance_idx': 1},
    {'ID': [60], 'type': 'interior'}
]

H_duct = 0.28
x_liner = 1
L_liner = 0.4

# %% Environment
T = 20  # Temperature in Celsius
P = 1.013e5  # Pressure in Pa
env = op.Environment(T, P)  # Create environment
c_f = env.c_f  # m/s : speed of sound (could be a function of x, y)

# %% TODO: code the correct velocity function of Jinyue
Mc = 0.3  # Bulk Mach number
delta = 0.1  # Shear parameter
ymin = 0  # Min y
ymax = H_duct

U, dudx, dudy, V, dvdx, dvdy = op.flow_profile_Rienstra(Mc, delta, c_f, ymin, ymax)

Ux = lambda x, y:  U(x, y)
du_xdx = lambda x, y:  0
du_xdy = lambda x, y:  dudy(x, y)

# Create flow_data dict
flow_data = {
  "U": Ux, "dudx": du_xdx, "dudy": du_xdy,
  "V": V, "dvdx": dvdx, "dvdy": dvdy
}


# %% Instantiate the HarmonicLEE class
N_DG = 3  # DG order (> 1)
case_type = '2D_LEE_PLAN'
harmonic_LEE_problem = op.HarmonicLEE(mesh_path, N_DG, markers, env,
                                      flow_data, case_type)

harmonic_LEE_problem.initialize_problem()

# %% Define measurement locations
# Upstream with respect of the flow
x_meas = np.linspace(0.4, 0.8, 10)
y_meas = np.linspace(0.05, ymax-0.05, 5)
X_meas_upstream, Y_meas_upstream = np.meshgrid(x_meas, y_meas)

# Liner section (the two microphone lines of Jinyue, approx)
x_meas = np.linspace(x_liner-0.05, x_liner+L_liner+0.05, 20)
y_meas = np.linspace(H_duct/2, H_duct/3, 2)
X_meas_lining, Y_meas_lining = np.meshgrid(x_meas, y_meas)

# Downstream with respect of the flow
x_meas = np.linspace(x_liner+L_liner+0.2, x_liner+L_liner+0.6, 10)
y_meas = np.linspace(0.05, ymax-0.05, 5)
X_meas_downstream, Y_meas_downstream = np.meshgrid(x_meas, y_meas)

# Assemble in single arrays
X_meas = np.hstack((X_meas_upstream.flatten(),
                    X_meas_lining.flatten(),
                    X_meas_downstream.flatten()))
Y_meas = np.hstack((Y_meas_upstream.flatten(),
                    Y_meas_lining.flatten(),
                    Y_meas_downstream.flatten()))

# %% Create a spatially dependent acoustic source function of modes

# TODO : need to adapt to the flow velocity profile with MAMOUT
def multi_modal_source_xy(y, f): 
    """For a given frequency, return the sum of all propagating modes with
    same amplitude each
    """
    support = np.array([1, 0, 1])
    freq_1st_cuton = env.c_f / (2*ymax)
    N_modes = int(f//freq_1st_cuton + 1)
    
    sum_modes = 0
    for ij in range(N_modes):
        sum_modes += np.cos(ij*np.pi/ymax * y)
    
    return sum_modes * support

# %% Solve the problem
H_cav = 3e-2  # cavity thickness
freq = 1000
k0x = 2*np.pi*freq
Z = 1 - 1.j * np.cotan(k0x*H_cav)  # impedance used by Jinyue in her thesis
R = op.Z2R(Z)  # reflection coefficient

harmonic_LEE_problem.set_impedance_BC(R=R)


# Create the source, given the frequency
def source_xy(x, y):
    return multi_modal_source_xy(y, freq)

# Set the source
source = {"source_idx": 1,
          "source_basis": source_xy,
          "source_amplitude": 1}

harmonic_LEE_problem.set_source_BC(A_source=source)
harmonic_LEE_problem.freq = freq  # frequency

harmonic_LEE_problem.solve_harmonic()

# Get pressure
p_meas = harmonic_LEE_problem.get_data(X_meas, Y_meas, var_name="p")
N_meas = len(p_meas)

# Add noise to the pressure
std_p_r = 0.05  # standard deviation of noise applied to real part
std_p_i = 0.05  # standard deviation of noise applied to imaginary part

p_exp = (np.real(p_meas) * (1 + std_p_r*np.random.randn(N_meas))
         + 1.j*np.imag(p_meas) * (1 + std_p_i*np.random.randn(N_meas)))

# %% Plot the solution
harmonic_LEE_problem.plot_var_2D(var_idx=2, transform_field=np.abs,
                                 grid_resolution_x=400,
                                 grid_resolution_y=50)

# %% Use the snapshot POD to accelerate future optimization
N_snapshots = 20  # number of snapshots to generate

# Randomly select the values of the reflection coefficient
R_list = (np.random.rand(N_snapshots)
          * np.exp(1.j * 2*np.pi * np.random.rand(N_snapshots)))
harmonic_LEE_problem.compute_snapshot_POD_basis(R=R_list, verbose=True)

# Set variable to true to enable subsequent faster calculations
harmonic_LEE_problem.use_snapshot_POD = True

# %% Create a dataset from experimental data, optionally include uncertainties
dataset_exp_p = op.DataSet(x=X_meas, y=Y_meas, field=p_exp,
                           std_real=0.03, std_imag=0.03,
                           var_name="p")

# %% Create a list of dictionaries containing information on the variables
variables = [
    {'type': 'source', 'source_idx': 1,
     "min_amplitude": 0, "max_amplitude": 5},
    {'type': 'impedance', 'impedance_idx': 1},
]

# %% Instantiate the Eduction class
eduction = op.Eduction(harmonic_LEE_problem, dataset_exp_p, variables,)

# %% Run the Eduction
eduction.run_eduction(verbose=True)

# Get the pseudo-uncertainty of each variables (assuming Gaussian uncertainty)
# Make sure that the datasets have "std_real" and "std_imag" assigned.
std_Gaussian = eduction.get_Gaussian_std_via_Hessian(eduction.res.X, h=1e-4)

# Make sure to recalculate solution at optimum
harmonic_LEE_problem.update_BCs(eduction.res.X, variables)
harmonic_LEE_problem.solve_harmonic()

# %% Plot results
p_meas = harmonic_LEE_problem.get_data(X_meas, Y_meas, var_name="p")

plt.figure()
plt.scatter(X_meas, np.abs(p_meas), label="Eduction")
plt.scatter(X_meas, np.abs(p_exp), color="k", label="Exp.")
plt.xlabel("x (m)")
plt.ylabel("abs(p) (Pa)")
plt.legend()
