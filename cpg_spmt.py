#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control-oriented Parameter-Grouped Single Particle Model with Thermal effects (CPG-SPMT).

Default parameters re-estimated jointly using BOLT on 24 A123 drive cycles.
BOLT reference: https://doi.org/10.1016/j.apenergy.2026.128307

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

import math

import numpy as np


REFERENCE_TEMPERATURE_C = 25.0
MIN_MODEL_VOLTAGE = 2.0
MAX_MODEL_VOLTAGE = 3.9
CONCENTRATION_EPSILON = 1e-10
PARAMETER_NAMES = (
    "a_n_1", "a_p_1", "b_n", "b_p", "d_n_1", "d_p_1", "soc_n", "soc_p",
    "E_1", "E_2", "E_3", "E_4", "E_5", "R_ini_1",
)

# Reference parameters for A123 18650 LFP A1-007.
# The calibration jointly estimates reference parameters and activation energies
# from all 24 drive cycles at eight ambient temperatures.
DEFAULT_PARAMETERS_25C = {
    'a_n': 3772.9282897370335,
    'a_p': 373.9261859400505,
    'b_n': 12351.52221150387,
    'b_p': 3974.5497932350754,
    'd_n': 0.004151106457385239,
    'd_p': 0.31295972090644797,
    'soc_n': 0.34736051269027135,
    'soc_p': 0.0002733266457685156,
    'R_ini': 0.15531009419630287,
}
DEFAULT_PARAM = (
    3772.9282897370335,  # a_n_1 [s]
    373.9261859400505,  # a_p_1 [s]
    12351.52221150387,  # b_n [C]
    3974.5497932350754,  # b_p [C]
    0.004151106457385239,  # d_n_1 [1/s]
    0.31295972090644797,  # d_p_1 [1/s]
    0.34736051269027135,  # soc_n [-]
    0.0002733266457685156,  # soc_p [-]
    37506.041582041405,  # E_1 [J/mol]
    46010.83909563133,  # E_2 [J/mol]
    125951.04639185766,  # E_3 [J/mol]
    131036.91467234145,  # E_4 [J/mol]
    3675.9817223888035,  # E_5 [J/mol]
    0.15531009419630287,  # R_ini_1 [ohm]
)

CALIBRATION = {
    'method': 'BOLT (Batch-Optimized Local-to-Global Technique)',
    'method_doi': '10.1016/j.apenergy.2026.128307',
    'calibration_date': '2026-11-19',
    'cell': 'A123 18650 LFP A1-007',
    'nominal_capacity_Ah': 1.1,
    'reference_temperature_C': 25.0,
    'condition_count': 24,
    'sample_count': 166992,
    'evaluation': 'In-sample replay of all calibration conditions',
    'sampling_interval_s': 1.0,
    'current_convention': 'Positive for charge; negative for discharge',
    'temperature_input': 'Measured cell temperature at every sample',
    'voltage_output_limits_V': [2.0, 3.9],
    'author': 'Feng Guo',
    'email': 'feng.guo@vito.be',
}


def ocp_pos(xp):
    """Return the positive-electrode OCP in V versus Li/Li+.

    xp is the normalized surface lithium concentration theta_p = c_s / c_max
    in [0, 1]. It is an electrode coordinate, not the whole-cell SOC.
    The empirical LFP OCP function is unchanged by parameter calibration.
    """
    return (1.1127 * np.exp(-1.9474 - 2000.6182 * xp)
           + 0.0388 * np.exp(-7.5995 * xp)
           + 0.1220 * np.exp(-52.9452 * xp)
           - 0.6990 * np.tanh(15.7948 * (xp - 1))
           - 0.6353 * (xp ** 87.1117)
           + 2.7313)


def ocp_neg(xn):
    """Return the negative-electrode OCP in V versus Li/Li+.

    xn is the normalized surface lithium concentration theta_n = c_s / c_max
    in [0, 1]. It is an electrode coordinate, not the whole-cell SOC.
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
    """Evaluate the temperature-dependent grouped parameters.

    temp is measured cell temperature in degrees C; T_ref is in K and
    R_const is in J/(mol K). The E_1/E_2 energies belong to the grouped
    diffusion time parameters a_n/a_p, E_3/E_4 to the grouped kinetic
    coefficients d_n/d_p, and E_5 to R_ini. For
    g = (1 / T_ref - 1 / (273.15 + temp)) / R_const, a_n, a_p and R_ini
    scale as exp(-E*g), whereas d_n and d_p scale as exp(+E*g).
    This function does not solve a heat-balance equation.
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
    """Return continuous-time matrices (A, B, C, D).

    Internal state order is negative bulk, negative surface-related state,
    positive bulk, positive surface-related state. The concentration output
    is C @ state + D * current; surface stoichiometries include feedthrough.
    a_n and a_p are grouped diffusion time parameters [s]; b_n and b_p are
    grouped capacities [C per unit electrode coordinate].
    """
    # Extract parameters
    a_n, a_p = temp_params['a_n'], temp_params['a_p']
    b_n, b_p = fixed_params['b_n'], fixed_params['b_p']

    if a_n <= 0 or a_p <= 0:
        raise ValueError(
            'Continuous-time matrices require positive time parameters. '
            'Use build_discrete_state_space_matrices for a_n = 0 or a_p = 0.'
        )

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

    # Concentration-coordinate output map; this is not an observability claim.
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
    """Return dimensionless Butler-Volmer arguments (y_neg, y_pos).

    The function name is retained for API compatibility. The returned values
    are not voltages: eta_j = (2*R*T_kelvin/F) * asinh(y_j).
    Positive current denotes charge; y_neg = -I/K_n and y_pos = I/K_p,
    with K_j = 6*b_j*d_j*sqrt(theta_j*(1-theta_j)) [A].
    concentrations contains dimensionless surface stoichiometries C_n/C_p
    in (0, 1); b_j is in C and the grouped kinetic coefficient d_j in 1/s.
    epsilon provides a positive denominator floor.
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

    # Dimensionless signed arguments; the voltage conversion is applied later.
    y_neg = -current / denom_neg
    y_pos = current / denom_pos

    return y_neg, y_pos


def build_discrete_state_space_matrices(temp_params, fixed_params, dt=1.0):
    """Return exact zero-order-hold matrices, including a_n = 0 or a_p = 0.

    State order is negative bulk, negative surface-related state, positive
    bulk, positive surface-related state. Surface concentrations include the
    D * current feedthrough. dt is in seconds (the simulator uses dt = 1).
    """
    a_n, a_p = temp_params["a_n"], temp_params["a_p"]
    b_n, b_p = fixed_params["b_n"], fixed_params["b_p"]
    if a_n < 0 or a_p < 0 or dt <= 0:
        raise ValueError("Require a_n >= 0, a_p >= 0 and dt > 0")
    weight_n = 1.0 if a_n == 0.0 else -math.expm1(-30.0 * dt / a_n)
    weight_p = 1.0 if a_p == 0.0 else -math.expm1(-30.0 * dt / a_p)
    discrete_A = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [weight_n, 1.0-weight_n, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, weight_p, 1.0-weight_p],
    ])
    discrete_B = np.array([
        [dt/b_n],
        [(dt+(2.0*a_n/35.0)*weight_n)/b_n],
        [-dt/b_p],
        [-(dt+(2.0*a_p/35.0)*weight_p)/b_p],
    ])
    output_C = np.eye(4)
    output_D = np.array([
        [0.0], [a_n/(105.0*b_n)], [0.0], [-a_p/(105.0*b_p)],
    ])
    return discrete_A, discrete_B, output_C, output_D


def cpg_spmt(I, T, param=None):
    """Simulate A123 LFP A1-007 with the calibrated CPG-SPMT model.

    I is a finite one-dimensional current array in A. Positive current
    denotes charge. T is the temperature in degrees C, supplied as a scalar
    or an array matching I. The sampling interval is one second.

    The optional parameter vector keeps the original 14-element order.
    a_n_1, a_p_1, b_n, b_p, d_n_1, d_p_1, soc_n, soc_p,
    E_1, E_2, E_3, E_4, E_5, R_ini_1

    DEFAULT_PARAM contains the fitted reference values at 25 degrees C.
    All five activation energies are estimated from the full temperature range.
    Supply the measured cell-temperature profile to replay a fitted experiment.

    The initial state is uniform. The D feedthrough applies at every sample.
    At each subsequent sample k, I[k] applies over the preceding interval.
    At a_n = 0 or a_p = 0 the corresponding surface follows its bulk.
    T[k] also applies over the interval ending at k. Both initial electrode
    stoichiometries are shared by all calibration conditions; they are not
    whole-cell SOC. b_n/b_p are in C, and d_n/d_p are grouped kinetics [1/s].
    Returned voltage is clipped to the configured limits of 2.0-3.9 V.
    The temperature is an input; no thermal-balance ODE is integrated.
    """
    current = np.asarray(I, dtype=np.float64)
    temperature = np.asarray(T, dtype=np.float64)
    if current.ndim != 1 or not np.isfinite(current).all():
        raise ValueError("I must be a finite one-dimensional current array")
    if temperature.ndim == 0:
        temperature = np.full(current.shape, float(temperature))
    if temperature.shape != current.shape or not np.isfinite(temperature).all():
        raise ValueError("T must be a finite scalar or an array matching I")
    if np.any(temperature <= -273.15):
        raise ValueError("T must exceed absolute zero")

    values = np.asarray(DEFAULT_PARAM if param is None else param, dtype=np.float64)
    if values.shape != (14,) or not np.isfinite(values).all():
        raise ValueError("param must contain 14 finite values in PARAMETER_NAMES order")
    a_n, a_p, b_n, b_p, d_n, d_p, soc_n, soc_p = values[:8]
    if a_n < 0 or a_p < 0 or np.any(values[2:6] <= 0):
        raise ValueError("Require positive capacities and kinetics, and nonnegative time parameters")
    if b_n <= b_p or values[13] < 0:
        raise ValueError("Require b_n > b_p and R_ini_1 >= 0")
    if not (0 < soc_n < 1 and 0 < soc_p < 1):
        raise ValueError("Initial electrode stoichiometries must lie in (0, 1)")

    reference = {
        "a_n_1": a_n, "a_p_1": a_p,
        "d_n_1": d_n, "d_p_1": d_p, "R_ini_1": values[13],
    }
    fixed = {"b_n": b_n, "b_p": b_p}
    energies = dict(zip(("E_1", "E_2", "E_3", "E_4", "E_5"), values[8:13]))
    state = np.array([[soc_n], [soc_n], [soc_p], [soc_p]], dtype=np.float64)
    voltage = np.empty(len(current), dtype=np.float64)
    gas_constant, faraday = 8.314472, 96487.0
    reference_kelvin = 273.15 + REFERENCE_TEMPERATURE_C
    cached_temperature = None

    for index, applied_current in enumerate(current):
        cell_temperature = float(temperature[index])
        if cell_temperature != cached_temperature:
            temp_params = update_temperature_dependent_parameters(
                cell_temperature, reference, energies, reference_kelvin, gas_constant,
            )
            if not np.isfinite(list(temp_params.values())).all():
                raise ValueError("Arrhenius parameters must remain finite")
            if temp_params["d_n"] <= 0 or temp_params["d_p"] <= 0:
                raise ValueError("Arrhenius kinetic coefficients must remain positive")
            discrete_A, discrete_B, output_C, output_D = build_discrete_state_space_matrices(
                temp_params, fixed,
            )
            rtf = 2.0 * gas_constant * (273.15+cell_temperature) / faraday
            cached_temperature = cell_temperature

        if index > 0:
            state = discrete_A @ state + discrete_B * applied_current
        surface = output_C @ state + output_D * applied_current
        concentrations = {
            "C_n": clip_scalar(surface[1, 0], CONCENTRATION_EPSILON, 1.0-CONCENTRATION_EPSILON),
            "C_p": clip_scalar(surface[3, 0], CONCENTRATION_EPSILON, 1.0-CONCENTRATION_EPSILON),
        }
        argument_n, argument_p = compute_overpotentials(
            applied_current, temp_params, fixed, concentrations, CONCENTRATION_EPSILON,
        )
        raw_voltage = (
            ocp_pos(concentrations["C_p"])
            - ocp_neg(concentrations["C_n"])
            + rtf * (math.asinh(argument_p)-math.asinh(argument_n))
            + applied_current * temp_params["R_ini"]
        )
        voltage[index] = clip_scalar(raw_voltage, MIN_MODEL_VOLTAGE, MAX_MODEL_VOLTAGE)
    return voltage


if __name__ == "__main__":
    current_A = np.full(3601, -1.1)
    voltage_V = cpg_spmt(current_A, 25.0)
    print("A123 18650 LFP A1-007")
    print(f"Samples {len(voltage_V)}")
    print(f"Initial voltage {voltage_V[0]:.6f} V")
    print(f"Final voltage {voltage_V[-1]:.6f} V")
