#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPG-SPMT Model Validation Script
================================

This script validates the CPG-SPMT battery model against experimental data
from various temperature conditions (-10°C to 50°C) and different drive cycles
(DST, US06, FUDS).

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

Author: Feng Guo
Contact: feng.guo [at] vito [dot] be
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the CPG-SPMT model
import importlib.util
import sys

# Load the CPG-SPMT module
spec = importlib.util.spec_from_file_location("cpg_spmt_module", "cpg_spmt.py")
cpg_spmt_module = importlib.util.module_from_spec(spec)
sys.modules["cpg_spmt_module"] = cpg_spmt_module
spec.loader.exec_module(cpg_spmt_module)

# Import the function
cpg_spmt = cpg_spmt_module.cpg_spmt


def load_experimental_data(file_path, sheet_name, temperature):
    """
    Load and preprocess experimental battery data from Excel files.
    
    Parameters
    ----------
    file_path : str
        Path to the Excel file containing experimental data.
    sheet_name : str
        Name of the sheet to load (e.g., 'DST', 'US06', 'FUDS').
    temperature : float
        Operating temperature in Celsius.
        
    Returns
    -------
    dict
        Dictionary containing processed experimental data.
    """
    try:
        # Read experimental data
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Extract relevant columns
        time = np.array(df["Test_Time(s)"].tolist())
        voltage = np.array(df["Voltage(V)"].tolist())
        current = np.array(df["Current(A)"].tolist())
        
        # Check if temperature column exists, otherwise use provided temperature
        if "Temperature (C)_1" in df.columns:
            temp_data = np.array(df["Temperature (C)_1"].tolist())
        else:
            temp_data = np.full(len(time), temperature)
        
        # Remove invalid data points
        valid_mask = (voltage > 1.5) & (voltage < 4.5) & np.isfinite(voltage) & np.isfinite(current)
        
        # Apply mask to filter data
        time_clean = time[valid_mask]
        voltage_clean = voltage[valid_mask]
        current_clean = current[valid_mask]
        temp_clean = temp_data[valid_mask]
        
        # Normalize time to start from zero
        time_clean = time_clean - time_clean[0]
        
        # Interpolate to 1-second intervals for consistency
        time_interp, voltage_interp, current_interp, temp_interp = interpolate_to_uniform_grid(
            time_clean, voltage_clean, current_clean, temp_clean, dt=1.0
        )
        
        return {
            'time': time_interp,
            'voltage': voltage_interp,
            'current': current_interp,
            'temperature': temp_interp,
            'duration': len(time_interp),
            'file_name': Path(file_path).name,
            'sheet_name': sheet_name,
            'operating_temp': temperature
        }
        
    except Exception as e:
        print(f"❌ Error loading {file_path} - {sheet_name}: {str(e)}")
        return None


def interpolate_to_uniform_grid(time, voltage, current, temperature, dt=1.0):
    """
    Interpolate data to a uniform time grid.
    
    Parameters
    ----------
    time, voltage, current, temperature : array-like
        Original data arrays.
    dt : float
        Desired time step in seconds.
        
    Returns
    -------
    tuple
        Interpolated data arrays.
    """
    # Create uniform time grid
    time_uniform = np.arange(time[0], time[-1], dt)
    
    # Create interpolation functions
    voltage_interp_func = interp1d(time, voltage, kind='linear', fill_value='extrapolate')
    current_interp_func = interp1d(time, current, kind='linear', fill_value='extrapolate')
    temp_interp_func = interp1d(time, temperature, kind='linear', fill_value='extrapolate')
    
    # Interpolate data
    voltage_uniform = voltage_interp_func(time_uniform)
    current_uniform = current_interp_func(time_uniform)
    temp_uniform = temp_interp_func(time_uniform)
    
    return time_uniform, voltage_uniform, current_uniform, temp_uniform


def calculate_error_metrics(y_true, y_pred):
    """
    Calculate error metrics between true and predicted values.
    
    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
        
    Returns
    -------
    dict
        Dictionary containing error metrics.
    """
    # Ensure arrays have the same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Calculate error metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    max_error = np.max(np.abs(y_true - y_pred))
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Max_Error': max_error,
        'R_squared': calculate_r_squared(y_true, y_pred)
    }


def calculate_r_squared(y_true, y_pred):
    """Calculate R-squared coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def validate_single_condition(data_file, sheet_name, temperature):
    """
    Validate model for a single temperature and drive cycle condition.
    
    Parameters
    ----------
    data_file : str
        Path to experimental data file.
    sheet_name : str
        Drive cycle sheet name.
    temperature : float
        Operating temperature.
        
    Returns
    -------
    dict
        Validation results.
    """
    print(f"🔄 Validating {temperature}°C - {sheet_name}...")
    
    # Load experimental data
    exp_data = load_experimental_data(data_file, sheet_name, temperature)
    if exp_data is None:
        return None
    
    # Run CPG-SPMT model simulation
    try:
        model_voltage = cpg_spmt(exp_data['current'], exp_data['temperature'])
    except Exception as e:
        print(f"❌ Model simulation failed: {str(e)}")
        return None
    
    # Calculate error metrics
    error_metrics = calculate_error_metrics(exp_data['voltage'], model_voltage)
    
    # Prepare results
    results = {
        'temperature': temperature,
        'drive_cycle': sheet_name,
        'duration': exp_data['duration'],
        'experimental_data': exp_data,
        'model_voltage': model_voltage,
        'error_metrics': error_metrics
    }
    
    print(f"✅ {temperature}°C - {sheet_name}: RMSE = {error_metrics['RMSE']:.4f}V, MAE = {error_metrics['MAE']:.4f}V")
    
    return results


def create_individual_plot(result, output_dir):
    """
    Create individual plot for each validation condition.
    
    Parameters
    ----------
    result : dict
        Single validation result.
    output_dir : Path
        Directory to save plots.
    """
    exp_data = result['experimental_data']
    model_voltage = result['model_voltage']
    temp = result['temperature']
    cycle = result['drive_cycle']
    rmse = result['error_metrics']['RMSE']
    mae = result['error_metrics']['MAE']
    r2 = result['error_metrics']['R_squared']
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'{temp}°C - {cycle} Drive Cycle Validation\n'
                f'RMSE: {rmse:.4f}V | MAE: {mae:.4f}V | R²: {r2:.4f}', 
                fontsize=14, fontweight='bold')
    
    # Voltage comparison
    ax1.plot(exp_data['time'], exp_data['voltage'], 
             label='Experimental', color='black', linewidth=2, alpha=0.8)
    ax1.plot(exp_data['time'], model_voltage, 
             label='CPG-SPMT Model', color='red', linewidth=2, linestyle='--', alpha=0.9)
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Voltage Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Current profile
    ax2.plot(exp_data['time'], exp_data['current'], 
             color='blue', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('Current (A)')
    ax2.set_title('Current Profile')
    ax2.grid(True, alpha=0.3)
    
    # Temperature profile
    ax3.plot(exp_data['time'], exp_data['temperature'], 
             color='green', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_title('Temperature Profile')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{temp}C_{cycle}_validation.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()  # Close to save memory
    
    print(f"💾 Saved individual plot: {filename}")


def create_error_distribution_plots(results_list, output_dir):
    """
    Create comprehensive error distribution plots.
    
    Parameters
    ----------
    results_list : list
        List of all validation results.
    output_dir : Path
        Directory to save plots.
    """
    # Organize data by temperature and drive cycle
    temperatures = sorted(list(set([r['temperature'] for r in results_list])))
    drive_cycles = ["DST", "US06", "FUDS"]
    
    # Create RMSE and MAE distribution plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CPG-SPMT Model Error Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    rmse_data = {cycle: [] for cycle in drive_cycles}
    mae_data = {cycle: [] for cycle in drive_cycles}
    temp_labels = []
    
    for temp in temperatures:
        temp_labels.append(f"{temp}°C")
        for cycle in drive_cycles:
            # Find result for this temperature and cycle
            result = next((r for r in results_list if r['temperature'] == temp and r['drive_cycle'] == cycle), None)
            if result:
                rmse_data[cycle].append(result['error_metrics']['RMSE'])
                mae_data[cycle].append(result['error_metrics']['MAE'])
            else:
                rmse_data[cycle].append(0)
                mae_data[cycle].append(0)
    
    # Plot RMSE by temperature
    x_pos = np.arange(len(temperatures))
    width = 0.25
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    for i, cycle in enumerate(drive_cycles):
        ax1.bar(x_pos + i*width, rmse_data[cycle], width, 
               label=cycle, color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('RMSE (V)')
    ax1.set_title('RMSE Distribution by Temperature and Drive Cycle')
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels(temp_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot MAE by temperature
    for i, cycle in enumerate(drive_cycles):
        ax2.bar(x_pos + i*width, mae_data[cycle], width, 
               label=cycle, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('MAE (V)')
    ax2.set_title('MAE Distribution by Temperature and Drive Cycle')
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(temp_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Overall RMSE distribution
    all_rmse = [r['error_metrics']['RMSE'] for r in results_list]
    all_mae = [r['error_metrics']['MAE'] for r in results_list]
    
    ax3.hist(all_rmse, bins=15, color='skyblue', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(all_rmse), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(all_rmse):.4f}V')
    ax3.set_xlabel('RMSE (V)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('RMSE Distribution Histogram')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Overall MAE distribution
    ax4.hist(all_mae, bins=15, color='lightcoral', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(all_mae), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(all_mae):.4f}V')
    ax4.set_xlabel('MAE (V)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('MAE Distribution Histogram')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save error distribution plot
    error_plot_path = output_dir / "error_distribution_analysis.png"
    plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved error distribution plot: error_distribution_analysis.png")


def plot_validation_results(results_list, output_dir, save_plots=True):
    """
    Create comprehensive validation plots.
    
    Parameters
    ----------
    results_list : list
        List of validation results.
    save_plots : bool
        Whether to save plots to files.
    """
    if not results_list:
        print("❌ No results to plot.")
        return
    
    # Set up plotting style
    plt.style.use('default')
    
    # Create figure for voltage comparison plots
    n_results = len(results_list)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CPG-SPMT Model Validation Results', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, result in enumerate(results_list[:4]):  # Show first 4 results
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        exp_data = result['experimental_data']
        model_voltage = result['model_voltage']
        
        # Plot voltage comparison
        ax.plot(exp_data['time'], exp_data['voltage'], 
               label='Experimental', color='black', linewidth=2, alpha=0.8)
        ax.plot(exp_data['time'], model_voltage, 
               label='CPG-SPMT Model', color=colors[i], linewidth=2, linestyle='--')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f"{result['temperature']}°C - {result['drive_cycle']}\n"
                    f"RMSE: {result['error_metrics']['RMSE']:.4f}V")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        summary_plot_path = output_dir / 'validation_summary_comparison.png'
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved summary comparison plot: validation_summary_comparison.png")
    plt.close()


def generate_validation_report(results_list, output_dir):
    """
    Generate a comprehensive validation report.
    
    Parameters
    ----------
    results_list : list
        List of validation results.
    """
    if not results_list:
        print("❌ No results to report.")
        return
    
    print("\n" + "="*80)
    print("📊 CPG-SPMT MODEL VALIDATION REPORT")
    print("="*80)
    
    # Summary table
    print(f"{'Temperature':<12} {'Drive Cycle':<12} {'RMSE (V)':<10} {'MAE (V)':<10} {'R²':<8} {'Duration (s)':<12}")
    print("-" * 80)
    
    total_rmse = 0
    total_mae = 0
    total_r2 = 0
    
    for result in results_list:
        temp = result['temperature']
        cycle = result['drive_cycle']
        rmse = result['error_metrics']['RMSE']
        mae = result['error_metrics']['MAE']
        r2 = result['error_metrics']['R_squared']
        duration = result['duration']
        
        print(f"{temp:<12.1f} {cycle:<12} {rmse:<10.4f} {mae:<10.4f} {r2:<8.4f} {duration:<12}")
        
        total_rmse += rmse
        total_mae += mae
        total_r2 += r2
    
    # Calculate averages
    n_results = len(results_list)
    avg_rmse = total_rmse / n_results
    avg_mae = total_mae / n_results
    avg_r2 = total_r2 / n_results
    
    print("-" * 80)
    print(f"{'Average':<12} {'':<12} {avg_rmse:<10.4f} {avg_mae:<10.4f} {avg_r2:<8.4f}")
    print("="*80)
    
    # Performance assessment
    print("\n🎯 MODEL PERFORMANCE ASSESSMENT:")
    if avg_rmse < 0.05:
        print("✅ Excellent accuracy (RMSE < 0.05V)")
    elif avg_rmse < 0.1:
        print("✅ Good accuracy (RMSE < 0.1V)")
    elif avg_rmse < 0.2:
        print("⚠️  Acceptable accuracy (RMSE < 0.2V)")
    else:
        print("❌ Poor accuracy (RMSE > 0.2V)")
    
    print(f"\n📈 Key Statistics:")
    print(f"   • Average RMSE: {avg_rmse:.4f} V")
    print(f"   • Average MAE:  {avg_mae:.4f} V")
    print(f"   • Average R²:   {avg_r2:.4f}")
    print(f"   • Test Cases:   {n_results}")
    
    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'Temperature_C': r['temperature'],
            'Drive_Cycle': r['drive_cycle'],
            'RMSE_V': r['error_metrics']['RMSE'],
            'MAE_V': r['error_metrics']['MAE'],
            'R_squared': r['error_metrics']['R_squared'],
            'MAPE_percent': r['error_metrics']['MAPE'],
            'Max_Error_V': r['error_metrics']['Max_Error'],
            'Duration_s': r['duration']
        }
        for r in results_list
    ])
    
    # Save results to the output directory
    csv_path = output_dir / 'cpg_spmt_validation_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")
    
    # Also save detailed error data
    detailed_results = []
    for r in results_list:
        detailed_results.append({
            'Temperature_C': r['temperature'],
            'Drive_Cycle': r['drive_cycle'],
            'RMSE_V': r['error_metrics']['RMSE'],
            'MAE_V': r['error_metrics']['MAE'],
            'R_squared': r['error_metrics']['R_squared'],
            'MAPE_percent': r['error_metrics']['MAPE'],
            'Max_Error_V': r['error_metrics']['Max_Error'],
            'Duration_s': r['duration'],
            'File_Name': r['experimental_data']['file_name']
        })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_csv_path = output_dir / 'detailed_validation_errors.csv'
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(f"💾 Detailed error data saved to: {detailed_csv_path}")


def main():
    """Main validation function."""
    print("🚀 Starting CPG-SPMT Model Validation")
    print("="*50)
    
    # Configuration
    DATA_DIR = Path("data")
    TEMPERATURES = [-10, 0, 10, 20, 25, 30, 40, 50]
    DRIVE_CYCLES = ["DST", "US06", "FUDS"]
    
    # File mapping
    FILE_MAPPING = {
        -10: "A1-007-DST-US06-FUDS-N10-20120829_modified.xlsx",
        0: "A1-007-DST-US06-FUDS-0-20120813_modified.xlsx",
        10: "A1-007-DST-US06-FUDS-10-20120815_modified.xlsx",
        20: "A1-007-DST-US06-FUDS-20-20120817_modified.xlsx",
        25: "A1-007-DST-US06-FUDS-25-20120827_modified.xlsx",
        30: "A1-007-DST-US06-FUDS-30-20120820_modified.xlsx",
        40: "A1-007-DST-US06-FUDS-40-20120822_modified.xlsx",
        50: "A1-007-DST-US06-FUDS-50-20120824_modified.xlsx"
    }
    
    # Create output directory for results
    OUTPUT_DIR = Path("validation_results")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Run validation for all conditions in temperature order
    all_results = []
    
    print("\n🔄 Running validation for all temperature conditions (-10°C to 50°C)...")
    for temp in TEMPERATURES:
        data_file = DATA_DIR / FILE_MAPPING[temp]
        if not data_file.exists():
            print(f"⚠️  File not found: {data_file}")
            continue
        
        print(f"\n📊 Processing {temp}°C conditions...")
        for cycle in DRIVE_CYCLES:
            result = validate_single_condition(str(data_file), cycle, temp)
            if result:
                all_results.append(result)
                # Create individual plot for each condition
                create_individual_plot(result, OUTPUT_DIR)
    
    # Generate comprehensive results
    if all_results:
        print(f"\n✅ Validation completed for {len(all_results)} conditions")
        
        # Generate validation report and save to output directory
        generate_validation_report(all_results, OUTPUT_DIR)
        
        # Create error distribution plots
        create_error_distribution_plots(all_results, OUTPUT_DIR)
        
        # Create summary comparison plot (optional - shows first 4 results)
        plot_validation_results(all_results[:4], OUTPUT_DIR)
        
        print(f"\n📁 All results saved to directory: {OUTPUT_DIR}")
        print(f"📊 Individual plots: {len(all_results)} condition-specific plots")
        print(f"📈 Summary plots: error_distribution_analysis.png, validation_summary_comparison.png")
        print(f"📋 Data files: cpg_spmt_validation_results.csv, detailed_validation_errors.csv")
        
    else:
        print("❌ No successful validations completed")


if __name__ == "__main__":
    main() 