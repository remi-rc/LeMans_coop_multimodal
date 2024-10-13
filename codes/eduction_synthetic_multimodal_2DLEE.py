#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate data for a multimodal bench configuration (from Le Mans)
"""

# %% Modules
import numpy as np
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination

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
N_DG = 2  # DG order (> 1)
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
def modal_source_xy(y, n): 
    return np.cos(n*np.pi/H_duct * y) * np.array([1, 0, 1])

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
freq = 3000
k0x = 2*np.pi*freq / env.c_f
Z = 1 - 1.j / np.tan(k0x*H_cav)  # impedance used by Jinyue in her thesis
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

# %% Now create a more complex eduction problem from scratch
# How many modes to consider :
freq_1st_cuton = env.c_f / (2*H_duct)
N_modes = int(freq//freq_1st_cuton + 1)

# For each individual mode, calculate a reduced basis and store it
dict_Vm = {}
dict_F_source = {}

N_snapshots = 20  # number of snapshots to generate

# Randomly select the values of the reflection coefficient
R_list = (np.random.rand(N_snapshots)
          * np.exp(1.j * 2*np.pi * np.random.rand(N_snapshots)))

harmonic_LEE_problem.use_snapshot_POD = False
for ij in range(N_modes):
    # Redefine the source for this given mode
    def source_xy(x, y):
        return modal_source_xy(y, ij)
    
    # Set the source
    source = {"source_idx": 1,
              "source_basis": source_xy,
              "source_amplitude": 1}
    
    harmonic_LEE_problem.set_source_BC(A_source=source)
    
    # Compute the reduced basis
    harmonic_LEE_problem.compute_snapshot_POD_basis(R=R_list, verbose=True)
    
    # Store for future uses
    dict_F_source[ij] = harmonic_LEE_problem.dict_LEE["F_source"][1]
    dict_Vm[ij] = harmonic_LEE_problem.Vm

# Set variable to true to enable subsequent faster calculations
harmonic_LEE_problem.use_snapshot_POD = True

# %% Create a dataset from experimental data, optionally include uncertainties
dataset_exp_p = op.DataSet(x=X_meas, y=Y_meas, field=p_exp,
                           std_real=0.05, std_imag=0.05,
                           var_name="p")

# %% Create functions (temporary, try to implement in OPAL code)

def log_likelihood(params, harmonic_LEE_problem, all_data_exp, N_modes):
    """Compute the logarithm of the likelihood.

        params : contains the optim variables (starts with reflection coeff)
        and then it's the source parameters for each mode
        
        harmonic_LEE_problem : the harmonic class
        
        all_data_exp : multi dataset, potentially just one
        
        N_modes : number of modes considered

    """
    # Initialize harmonic solution (sum of modes)
    q_tot = 0
    
    # Update the simulation with current parameters
    R = params[0] * np.exp(1j*params[1])
    harmonic_LEE_problem.set_impedance_BC(R=R)
    
    # Iterate on modes
    for ij in range(N_modes):
        # Calibrate the source
        harmonic_LEE_problem.dict_LEE["F_source"][1] = dict_F_source[ij] * (params[2+2*ij] * np.exp(1j*params[3+2*ij]))
        
        # Apply the correct snapshot POD matrix
        harmonic_LEE_problem.Vm = dict_Vm[ij]

        # Solve the harmonic problem at the given frequency
        harmonic_LEE_problem.solve_harmonic()
        
        # Add the contribution
        q_tot += harmonic_LEE_problem.q_sol

    # Assign total harmonic solution
    harmonic_LEE_problem.q_sol = q_tot
    
    # Initialize total error
    total_error = 0

    # Iterate through each experimental dataset
    for data_exp in all_data_exp:
        x_loc = data_exp.x
        y_loc = data_exp.y
        field_num = harmonic_LEE_problem.get_data(x_loc, y_loc,
                                               var_name=data_exp.var_name)

        N_meas = data_exp.N_meas  # dimension of the measurement

        # Compute the differencse between the simulated and expe data
        error_real = np.real(field_num - data_exp.field)
        error_imag = np.imag(field_num - data_exp.field)

        # Add real and imaginary contributions
        log_like_pre_exp = -N_meas/2* (2*np.log(2*np.pi)
                                       + np.log(data_exp.std_real**2)
                                       + np.log(data_exp.std_imag**2))
        log_like_exp = (
            -1 / (2 * data_exp.std_real**2) * np.sum(error_real**2)
            -1 / (2 * data_exp.std_imag**2) * np.sum(error_imag**2)
                        )

        total_error += log_like_pre_exp + log_like_exp

    return total_error

def objective_function(params, harmonic_LEE_problem, all_data_exp, N_modes):
    """Compute the objective function (the cost function to minimize).
    It is simply "minus the log-likelihood".
    """
    return -log_likelihood(params, harmonic_LEE_problem, all_data_exp, N_modes)


# %% Setup the eduction

# Define the pyMOO problem
class EductionOptim(ElementwiseProblem):
    def __init__(self, param_bounds, harmonic_LEE_problem, all_data_exp, N_modes):
        super().__init__(n_var=len(param_bounds),
                         n_obj=1,  # Single-objective optimization
                         xl=[bound[0] for bound in param_bounds],  # Lower bounds
                         xu=[bound[1] for bound in param_bounds])  # Upper bounds

        self.param_bounds = param_bounds
        self.harmonic_LEE_problem = harmonic_LEE_problem
        self.all_data_exp = all_data_exp
        self.N_modes = N_modes
        
    def evaluate(self, x, *args, **kwargs):
        # in pyMOO, x always contains the whole population
        n_pop = np.size(x, 0)
        f1 = np.zeros(n_pop)

        # Iterate over the population
        for ij in range(n_pop):
            # Objective function to minimize
            f1[ij] = objective_function(x[ij, :], self.harmonic_LEE_problem,
                                        self.all_data_exp, self.N_modes)

        # Assemble "out" dictionary
        out = {}
        out["F"] = np.reshape(f1, (-1, 1))
        return out

    
# Define bounds
param_bounds = [(0, 1), (0, 2*np.pi)]  # reflection coefficient
for ij in range(N_modes):
    # mode amplitude and phase
    param_bounds += [(0, 5), (0, 2*np.pi)]
    
# %% Run eduction
def run_eduction(param_bounds, harmonic_LEE_problem, all_data_exp, N_modes, 
                 seed=1,
                 verbose=True):
    """Perform optimization to find the best-fit impedance and
    source parameters."""

    # Instantiate the problem
    problem = EductionOptim(param_bounds, harmonic_LEE_problem, 
                            all_data_exp, N_modes)

    # Define the NSGA-II algorithm
    algorithm = NSGA2(
        pop_size=4*len(param_bounds),
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    # Define the termination criterion
    termination = get_termination("n_gen", 50)

    # Run the optimization
    print("Starting optimization...")
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=seed,
                   save_history=True,
                   verbose=verbose)

    # Print the results
    print("Optimization successful!")
    print("Optimized parameters:", res.X)
        
    return res

res = run_eduction(param_bounds, harmonic_LEE_problem, [dataset_exp_p], N_modes)