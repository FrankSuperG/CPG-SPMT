#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Control-Oriented Parameter-Grouped Simplified Particle Model with Thermal effects (CPG-SPMT).

Copyright 2025 Feng Guo

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

If you use this code in your research or projects, please cite the related publications.
For any questions, contact: feng.guo [at] vito [dot] be
"""

import numpy as np
import math
from scipy.signal import cont2discrete

def ocp_pos(xp):
    """
    Compute the open-circuit potential (OCP) for the positive electrode.
    
    Uses a combination of exponential and hyperbolic tangent functions
    to accurately model the thermodynamic behavior of the positive electrode.

    Parameters
    ----------
    xp : float or array-like
        Normalized surface concentration (SOC) of the positive electrode [0, 1].

    Returns
    -------
    u_pos : float or array-like
        Open-circuit potential (V) for the positive electrode.
    """
    return (1.1127 * np.exp(-1.9474 - 2000.6182 * xp) 
           + 0.0388 * np.exp(-7.5995 * xp) 
           + 0.1220 * np.exp(-52.9452 * xp) 
           - 0.6990 * np.tanh(15.7948 * (xp - 1)) 
           - 0.6353 * (xp ** 87.1117) 
           + 2.7313)

def ocp_neg(xn):
    """
    Compute the open-circuit potential (OCP) for the negative electrode.
    
    Uses a sophisticated combination of exponential and hyperbolic tangent functions
    to capture the complex electrochemical behavior of the negative electrode.

    Parameters
    ----------
    xn : float or array-like
        Normalized surface concentration (SOC) of the negative electrode [0, 1].

    Returns
    -------
    u_neg : float or array-like
        Open-circuit potential (V) for the negative electrode.
    """
    # Coefficients for the multi-term OCP function
    coefficients = {
        'a1': 55.26, 'a2': 9.451, 'a3': -146.4, 'a4': -0.05217, 'a5': 9.453,
        'a6': -1.783, 'a7': 0.1051, 'a8': 4.452, 'a9': -5.533, 'a10': -0.07638,
        'a11': 19.25, 'a12': -18.04, 'a13': -0.02072, 'a14': 25.87, 'a15': -14.56,
        'a16': -54.9, 'a17': 2.106, 'a18': 26.37, 'a19': 0.01532, 'a20': 63.36,
        'a21': -56.77, 'a22': -0.111, 'a23': 52.65, 'a24': -2.862, 'a25': -0.0496,
        'a26': 77.08, 'a27': -2.383
    }
    
    # Calculate OCP using multi-term expression
    return (
        coefficients['a1']
        + coefficients['a2'] * np.exp(coefficients['a3'] * xn)
        + coefficients['a4'] * np.tanh(coefficients['a5'] * xn + coefficients['a6'])
        + coefficients['a7'] * np.tanh(coefficients['a8'] * xn + coefficients['a9'])
        + coefficients['a10'] * np.tanh(coefficients['a11'] * xn + coefficients['a12'])
        + coefficients['a13'] * np.tanh(coefficients['a14'] * xn + coefficients['a15'])
        + coefficients['a16'] * np.tanh(coefficients['a17'] * xn + coefficients['a18'])
        + coefficients['a19'] * np.tanh(coefficients['a20'] * xn + coefficients['a21'])
        + coefficients['a22'] * np.tanh(coefficients['a23'] * xn + coefficients['a24'])
        + coefficients['a25'] * np.tanh(coefficients['a26'] * xn + coefficients['a27'])
    )

def clip_scalar(value, min_val, max_val):
    """
    Clip a scalar value to stay within specified bounds.
    
    Parameters
    ----------
    value : float
        Input value to be clipped.
    min_val : float
        Minimum allowed value.
    max_val : float
        Maximum allowed value.
        
    Returns
    -------
    float
        Clipped value within [min_val, max_val].
    """
    return min(max(value, min_val), max_val)

def update_temperature_dependent_parameters(temp, param_ref, activation_energies, T_ref, R_const):
    """
    Update temperature-dependent parameters using Arrhenius equation.
    
    Parameters
    ----------
    temp : float
        Current temperature in Celsius.
    param_ref : dict
        Dictionary of reference parameter values.
    activation_energies : dict
        Dictionary of activation energies.
    T_ref : float
        Reference temperature in Kelvin.
    R_const : float
        Universal gas constant.
        
    Returns
    -------
    dict
        Updated temperature-dependent parameters.
    """
    temp_kelvin = 273.15 + temp
    
    return {
        'a_n': param_ref['a_n_1'] / np.exp((activation_energies['E_1']/R_const) * (1/T_ref - 1/temp_kelvin)),
        'a_p': param_ref['a_p_1'] / np.exp((activation_energies['E_2']/R_const) * (1/T_ref - 1/temp_kelvin)),
        'd_n': param_ref['d_n_1'] * np.exp((activation_energies['E_3']/R_const) * (1/T_ref - 1/temp_kelvin)),
        'd_p': param_ref['d_p_1'] * np.exp((activation_energies['E_4']/R_const) * (1/T_ref - 1/temp_kelvin)),
        'R_ini': param_ref['R_ini_1'] / np.exp((activation_energies['E_5']/R_const) * (1/T_ref - 1/temp_kelvin))
    }

def build_state_space_matrices(temp_params, fixed_params):
    """
    Build continuous-time state-space matrices for the battery model.
    
    Parameters
    ----------
    temp_params : dict
        Temperature-dependent parameters.
    fixed_params : dict
        Fixed model parameters.
        
    Returns
    -------
    tuple
        (A, B, C, D) matrices for state-space representation.
    """
    # Extract parameters
    a_n, a_p = temp_params['a_n'], temp_params['a_p']
    b_n, b_p = fixed_params['b_n'], fixed_params['b_p']
    
    # State matrix A
    A = np.array([
        [0.0,        0.0,        0.0,        0.0       ],
        [30.0/a_n,  -30.0/a_n,   0.0,        0.0       ],
        [0.0,        0.0,        0.0,        0.0       ],
        [0.0,        0.0,        30.0/a_p,  -30.0/a_p  ]
    ])

    # Input matrix B
    B = np.array([
        [1.0/b_n],
        [19.0/(7.0*b_n)],
        [-1.0/b_p],
        [-19.0/(7.0*b_p)]
    ])

    # Output matrix C (identity for full state observation)
    C = np.eye(4)

    # Feedthrough matrix D
    D = np.array([
        [0.0],
        [a_n/(105.0*b_n)],
        [0.0],
        [-a_p/(105.0*b_p)]
    ])

    return A, B, C, D

def compute_overpotentials(current, temp_params, fixed_params, concentrations, epsilon):
    """
    Compute overpotentials for both electrodes.
    
    Parameters
    ----------
    current : float
        Applied current (A).
    temp_params : dict
        Temperature-dependent parameters.
    fixed_params : dict
        Fixed model parameters.
    concentrations : dict
        Electrode concentrations.
    epsilon : float
        Small value to prevent division by zero.
        
    Returns
    -------
    tuple
        (y_neg, y_pos) overpotential terms.
    """
    # Extract parameters
    b_n, b_p = fixed_params['b_n'], fixed_params['b_p']
    d_n, d_p = temp_params['d_n'], temp_params['d_p']
    C_n, C_p = concentrations['C_n'], concentrations['C_p']
    
    # Calculate square root terms
    sqrt_term_n = math.sqrt(C_n * (1 - C_n))
    sqrt_term_p = math.sqrt(C_p * (1 - C_p))
    
    # Calculate denominators with numerical protection
    denom_neg = max(6 * b_n * d_n * sqrt_term_n, epsilon)
    denom_pos = max(6 * b_p * d_p * sqrt_term_p, epsilon)
    
    # Compute overpotential terms
    y_neg = -current / denom_neg
    y_pos = current / denom_pos
    
    return y_neg, y_pos

def cpg_spmt(I, T, param=None):
    """
    Simulate battery voltage using Control-Oriented Parameter-Grouped 
    Simplified Particle Model with Thermal effects (CPG-SPMT).

    This advanced battery model incorporates:
    - Temperature-dependent parameter variations via Arrhenius equations
    - Sophisticated OCP functions for both electrodes  
    - Parameter grouping for computational efficiency
    - Parabolic approximation for PDE discretization

    Parameters
    ----------
    I : array-like
        Current profile over time (A). Positive for discharge, negative for charge.
    T : array-like
        Temperature profile over time (°C).
    param : list or tuple, optional
        Model parameters [a_n_1, a_p_1, b_n, b_p, d_n_1, d_p_1, soc_n, soc_p, 
                         E_1, E_2, E_3, E_4, E_5, R_ini_1].
        If None, optimized default values are used.

    Returns
    -------
    V : np.ndarray
        Terminal voltage response (V) corresponding to input profiles.

    Notes
    -----
    The model uses Zero-Order Hold (ZOH) discretization for numerical stability
    and incorporates advanced electrochemical phenomena including:
    - Solid-phase diffusion with parabolic approximation
    - Temperature-dependent transport properties
    - Nonlinear electrode kinetics

    References
    ----------
    Feng Guo. "A Control-Oriented Parameter-Grouped Simplified Particle Model 
    with Thermal effects." VITO, 2025.

    Examples
    --------
    >>> import numpy as np
    >>> I_profile = np.full(100, 2.9)  # 1C discharge
    >>> T_profile = np.full(100, 25.0)  # 25°C
    >>> voltage = cpg_spmt(I_profile, T_profile)
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # Default Parameters (Optimized for high-energy Li-ion cell)
    # ═══════════════════════════════════════════════════════════════════════════
    default_param = [
        2696.826595,    # a_n_1: Negative electrode time constants at 25°C
        1352.428491,    # a_p_1: Positive electrode time constants at 25°C 
        10999.93219,    # b_n: Negative electrode capacity
        3982.261084,    # b_p: Positive electrode capacity
        0.005475,       # d_n_1: Negative electrode grouped parameter at 25°C
        0.000343,       # d_p_1: Positive electrode grouped parameter at 25°C
        0.382917,       # soc_n: Initial negative electrode SOC
        0.000282,       # soc_p: Initial positive electrode SOC
        44042.21814,    # E_1: Activation energy for a_n (J/mol)
        52664.43272,    # E_2: Activation energy for a_p (J/mol)
        133074.8646,    # E_3: Activation energy for d_n (J/mol)
        14397.71388,    # E_4: Activation energy for d_p (J/mol)
        4097.712369,    # E_5: Activation energy for R_ini (J/mol)
        0.173289        # R_ini_1: Reference internal resistance at 25°C (Ω)
    ]
    
    # Parameter validation and assignment
    if param is None:
        param = default_param
    else:
        if len(param) != 14:
            raise ValueError("Parameter list must contain exactly 14 elements.")
    
    # Unpack parameters with descriptive names
    (a_n_1, a_p_1, b_n, b_p, d_n_1, d_p_1, soc_n, soc_p, 
     E_1, E_2, E_3, E_4, E_5, R_ini_1) = param

    # ═══════════════════════════════════════════════════════════════════════════
    # Physical Constants and Configuration
    # ═══════════════════════════════════════════════════════════════════════════
    CONSTANTS = {
        'F': 96487,           # Faraday constant (C/mol)
        'R': 8.314472,        # Universal gas constant (J/(mol·K))
        'T_ref': 273.15 + 25, # Reference temperature (K)
        'dt': 1.0,            # Time step (s)
        'epsilon': 1e-10      # Numerical stability parameter
    }
    
    # Organize parameters
    param_ref = {
        'a_n_1': a_n_1, 'a_p_1': a_p_1, 'd_n_1': d_n_1, 
        'd_p_1': d_p_1, 'R_ini_1': R_ini_1
    }
    
    fixed_params = {'b_n': b_n, 'b_p': b_p}
    
    activation_energies = {
        'E_1': E_1, 'E_2': E_2, 'E_3': E_3, 'E_4': E_4, 'E_5': E_5
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # Simulation Initialization
    # ═══════════════════════════════════════════════════════════════════════════
    num_steps = len(I)
    voltage_response = np.zeros(num_steps)
    
    # Initialize state vector [SOC_n_avg, SOC_n_surf, SOC_p_avg, SOC_p_surf]
    state_vector = np.array([[soc_n], [soc_n], [soc_p], [soc_p]])

    # ═══════════════════════════════════════════════════════════════════════════
    # Main Simulation Loop
    # ═══════════════════════════════════════════════════════════════════════════
    for step in range(num_steps):
        current = I[step]
        temperature = T[step]
        
        # Update temperature-dependent parameters
        temp_params = update_temperature_dependent_parameters(
            temperature, param_ref, activation_energies, 
            CONSTANTS['T_ref'], CONSTANTS['R']
        )
        
        # Build state-space matrices
        A, B, C, D = build_state_space_matrices(temp_params, fixed_params)
        
        # Discretize system using scipy's cont2discrete (ZOH method)
        sys_discrete = cont2discrete((A, B, C, D), CONSTANTS['dt'], method='zoh')
        Ad, Bd, Cd, Dd, _ = sys_discrete  # cont2discrete returns 5 elements: (Ad, Bd, Cd, Dd, dt)
        
        # State update (skip first iteration)
        if step > 0:
            state_vector = Ad @ state_vector + Bd * current
            
        # Compute concentrations with numerical protection
        concentrations_raw = Cd @ state_vector + Dd * current
        concentrations = {
            'C_n': clip_scalar(concentrations_raw[1, 0], CONSTANTS['epsilon'], 1 - CONSTANTS['epsilon']),
            'C_p': clip_scalar(concentrations_raw[3, 0], CONSTANTS['epsilon'], 1 - CONSTANTS['epsilon'])
        }
        
        # Compute overpotentials
        y_neg, y_pos = compute_overpotentials(
            current, temp_params, fixed_params, concentrations, CONSTANTS['epsilon']
        )
        
        # Calculate terminal voltage
        temp_kelvin = 273.15 + temperature
        RTF = (2 * CONSTANTS['R'] * temp_kelvin) / CONSTANTS['F']
        
        terminal_voltage = (
            ocp_pos(concentrations['C_p']) - ocp_neg(concentrations['C_n'])
            - RTF * (math.asinh(y_pos) + math.asinh(y_neg))
            + current * temp_params['R_ini']
        )
        
        # Apply realistic voltage bounds
        voltage_response[step] = clip_scalar(terminal_voltage, 2.0, 3.9)

    return voltage_response

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Simulation Configuration
    # ═══════════════════════════════════════════════════════════════════════════
    SIMULATION_CONFIG = {
        'discharge_current': -1.1,    # 1C discharge current (A)
        'simulation_time': 3400,      # Duration (seconds)
        'temperature': 25.0,         # Constant temperature (°C)
    }
    
    # Create input profiles
    time_vector = np.arange(SIMULATION_CONFIG['simulation_time'])
    current_profile = np.full(SIMULATION_CONFIG['simulation_time'], 
                             SIMULATION_CONFIG['discharge_current'])
    temperature_profile = np.full(SIMULATION_CONFIG['simulation_time'], 
                                 SIMULATION_CONFIG['temperature'])
    
    print("🔋 Starting CPG-SPMT Battery Simulation...")
    print(f"   Current: {SIMULATION_CONFIG['discharge_current']} A (1C discharge)")
    print(f"   Temperature: {SIMULATION_CONFIG['temperature']} °C")
    print(f"   Duration: {SIMULATION_CONFIG['simulation_time']} seconds")
    
    # Run simulation
    voltage_response = cpg_spmt(current_profile, temperature_profile)
    power_profile = voltage_response * current_profile
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Results Visualization
    # ═══════════════════════════════════════════════════════════════════════════
    plt.style.use('default')  # Clean matplotlib style
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CPG-SPMT Battery Model Simulation Results', fontsize=16, fontweight='bold')
    
    # Voltage response
    axes[0, 0].plot(time_vector, voltage_response, 'b-', linewidth=2.5, label='Terminal Voltage')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Voltage (V)')
    axes[0, 0].set_title('Battery Voltage Response')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Current profile
    axes[0, 1].plot(time_vector, current_profile, 'r-', linewidth=2.5, label='Discharge Current')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Current (A)')
    axes[0, 1].set_title('Current Profile')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Temperature profile
    axes[1, 0].plot(time_vector, temperature_profile, 'g-', linewidth=2.5, label='Temperature')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].set_title('Temperature Profile')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Power profile
    axes[1, 1].plot(time_vector, power_profile, 'm-', linewidth=2.5, label='Power Output')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Power (W)')
    axes[1, 1].set_title('Power Profile')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Performance Summary
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("📊 CPG-SPMT SIMULATION SUMMARY")
    print("="*60)
    print(f"🔋 Initial Voltage:     {voltage_response[0]:.3f} V")
    print(f"🔋 Final Voltage:       {voltage_response[-1]:.3f} V")  
    print(f"📉 Voltage Drop:        {voltage_response[0] - voltage_response[-1]:.3f} V")
    print(f"📈 Average Voltage:     {np.mean(voltage_response):.3f} V")
    print(f"⚡ Average Power:       {np.mean(power_profile):.2f} W")
    print(f"🔋 Energy Delivered:    {np.sum(power_profile)/3600:.2f} Wh")
    print(f"🌡️  Operating Temp:     {SIMULATION_CONFIG['temperature']:.1f} °C")
    print("="*60)
    print("✅ Simulation completed successfully!")
