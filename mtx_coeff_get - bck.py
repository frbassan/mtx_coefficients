import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# Flag to choose between original files or simulated file:
# 1 = load and validate using 'sim_t_XX.csv' (XX is the chamber temperature)
# 0 = run original multi-file processing ('t_40.csv' and 't_20.csv')
calc_check = 1

# File paths
file_paths = {
    'T40': 't_40.csv',
    'T20': 't_20.csv'
}
simulated_file_path = 'sim_t_60.csv'

spatial_resolution = 0.1  # meters per sample

# Define the specified analysis segments (in meters)
seg_free = (1000.0, 1100.0)
seg_met = (1500.0, 1600.0)

# Temperature and thermal expansion coefficient variables
T0 = 20
T1 = 60
Al_alpha = 22.8 * 1e-6  # strain per °C

results = {}

# Process T20 (Base file for T0 = 20°C)
def process_file(file_path):
    data_rows = 0
    data_columns = 0
    data_matrix = []
    freq_step = -1.953125

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('Param;'):
                parts = line.split(';')
                if len(parts) >= 3 and parts[1] == 'freqStep':
                    freq_step = float(parts[2])
                continue
            row_values = [float(x) for x in line.split(';') if x.strip() != '']
            data_matrix.append(row_values)
            data_rows += 1
            data_columns = len(row_values)

    data_matrix = np.array(data_matrix)
    distances = np.arange(data_rows) * spatial_resolution

    rayleigh_values = []
    brillouin_values = []
    rayleigh_bins = []
    brillouin_bins = []

    rayleigh_target_bin = 100
    search_window = 10

    for row in data_matrix:
        start_idx = max(0, rayleigh_target_bin - search_window)
        end_idx = min(len(row), rayleigh_target_bin + search_window + 1)
        local_max_idx = start_idx + np.argmax(row[start_idx:end_idx])
        rayleigh_values.append(row[local_max_idx])
        rayleigh_bins.append(local_max_idx)
        
        row_masked = row.copy()
        row_masked[start_idx:end_idx] = -np.inf
        brillouin_idx = np.argmax(row_masked)
        brillouin_values.append(row[brillouin_idx])
        brillouin_bins.append(brillouin_idx)

    def get_segment_stats(start_m, end_m):
        s_idx = max(0, int(start_m / spatial_resolution))
        e_idx = min(data_rows, int(end_m / spatial_resolution))
        if s_idx < e_idx:
            r_amp = np.mean(np.array(rayleigh_values)[s_idx:e_idx])
            b_amp = np.mean(np.array(brillouin_values)[s_idx:e_idx])
            b_bin = np.mean(np.array(brillouin_bins)[s_idx:e_idx])
            return {
                'mean_rayleigh_amp': r_amp,
                'mean_brillouin_amp': b_amp,
                'mean_brillouin_bin': b_bin,
                'ratio': (r_amp / b_amp) if b_amp != 0 else np.nan
            }
        return {}

    return {
        'data_rows': data_rows,
        'data_columns': data_columns,
        'freq_step': freq_step,
        'distances': distances,
        'rayleigh_values': rayleigh_values,
        'brillouin_values': brillouin_values,
        'rayleigh_bins': rayleigh_bins,
        'brillouin_bins': brillouin_bins,
        'seg_free_stats': get_segment_stats(*seg_free),
        'seg_met_stats': get_segment_stats(*seg_met)
    }

# 1. RLP0 and vB0 ALWAYS from T20 file
results['T20'] = process_file(file_paths['T20'])

# 2. RLP1 and vB1 from simulated T20 or T40 file depending on calc_check flag
if calc_check == 1 and os.path.exists(simulated_file_path):
    print(f"Validation mode (calc_check = 1): Reading {simulated_file_path} for T1 (T40 simulated)...")
    
    sim_distances = []
    sim_rayleigh_values = []
    sim_brillouin_values = []
    sim_rayleigh_bins = []
    sim_brillouin_bins = []

    with open(simulated_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)  # Skip header
        for row in reader:
            if not row:
                continue
            sim_distances.append(float(row[0]))
            sim_rayleigh_values.append(float(row[1]))
            sim_brillouin_values.append(float(row[2]))
            sim_rayleigh_bins.append(float(row[3]))
            sim_brillouin_bins.append(float(row[4]))

    sim_distances = np.array(sim_distances)
    sim_rayleigh_values = np.array(sim_rayleigh_values)
    sim_brillouin_values = np.array(sim_brillouin_values)
    sim_rayleigh_bins = np.array(sim_rayleigh_bins)
    sim_brillouin_bins = np.array(sim_brillouin_bins)

    def get_sim_segment_stats(start_m, end_m):
        s_idx = max(0, int(start_m / spatial_resolution))
        e_idx = min(len(sim_distances), int(end_m / spatial_resolution))
        if s_idx < e_idx:
            r_amp = np.mean(sim_rayleigh_values[s_idx:e_idx])
            b_amp = np.mean(sim_brillouin_values[s_idx:e_idx])
            b_bin = np.mean(sim_brillouin_bins[s_idx:e_idx])
            return {
                'mean_rayleigh_amp': r_amp,
                'mean_brillouin_amp': b_amp,
                'mean_brillouin_bin': b_bin,
                'ratio': (r_amp / b_amp) if b_amp != 0 else np.nan
            }
        return {}

    results['T40'] = {
        'data_rows': len(sim_distances),
        'data_columns': results['T20']['data_columns'],
        'freq_step': results['T20']['freq_step'],
        'distances': sim_distances,
        'rayleigh_values': sim_rayleigh_values,
        'brillouin_values': sim_brillouin_values,
        'rayleigh_bins': sim_rayleigh_bins,
        'brillouin_bins': sim_brillouin_bins,
        'seg_free_stats': get_sim_segment_stats(*seg_free),
        'seg_met_stats': get_sim_segment_stats(*seg_met)
    }
else:
    if calc_check == 1:
        print(f"Warning: '{simulated_file_path}' not found. Falling back to original t_40.csv file.")
    results['T40'] = process_file(file_paths['T40'])

# Helper function to convert bin to MHz
def bin_to_mhz(bin_val, freq_step):
    return bin_val * abs(freq_step)

# Calculate RLP and vB (in MHz) metrics
RLP0_free_t20 = results['T20']['seg_free_stats'].get('ratio', np.nan)
RLP0_met_t20 = results['T20']['seg_met_stats'].get('ratio', np.nan)

RLP1_free_t40 = results['T40']['seg_free_stats'].get('ratio', np.nan)
RLP1_met_t40 = results['T40']['seg_met_stats'].get('ratio', np.nan)

vB0_free_t20 = bin_to_mhz(results['T20']['seg_free_stats'].get('mean_brillouin_bin', np.nan), results['T20']['freq_step'])
vB0_met_t20 = bin_to_mhz(results['T20']['seg_met_stats'].get('mean_brillouin_bin', np.nan), results['T20']['freq_step'])

vB1_free_t40 = bin_to_mhz(results['T40']['seg_free_stats'].get('mean_brillouin_bin', np.nan), results['T40']['freq_step'])
vB1_met_t40 = bin_to_mhz(results['T40']['seg_met_stats'].get('mean_brillouin_bin', np.nan), results['T40']['freq_step'])

# Calculate deltas
dvB_free = vB1_free_t40 - vB0_free_t20          # in MHz
dRLP_free = (RLP1_free_t40 - RLP0_free_t20) * 100.0  # in %
dvB_met = vB1_met_t40 - vB0_met_t20            # in MHz
dRLP_met = (RLP1_met_t40 - RLP0_met_t20) * 100.0    # in %

dT = T1 - T0                                    # in °C
de_ue = (Al_alpha * dT) * 1e6                  # in ue (με)

# Calculate coefficients with correct requested units
D1 = dvB_free / dT                              # MHz / °C
D2 = (dvB_met - dvB_free) / de_ue              # MHz / ue
D3 = dRLP_free / dT                             # % / °C
D4 = (dRLP_met - dRLP_free) / de_ue            # % / ue

# Print results comparison
for label in ['T20', 'T40']:
    res = results[label]
    print(f"\n--- RESULTS FOR {label} ---")
    print(f"File Dimensions -> Rows (Distance points): {res['data_rows']}, Columns (Frequency bins): {res['data_columns']}")
    if res['seg_free_stats']:
        print(f"seg_free (1000-1100m) Rayleigh Amp Mean: {res['seg_free_stats']['mean_rayleigh_amp']:.6f}")
        print(f"seg_free (1000-1100m) Brillouin Amp Mean: {res['seg_free_stats']['mean_brillouin_amp']:.6f}")
        print(f"seg_free (1000-1100m) Brillouin Bin Mean: {res['seg_free_stats']['mean_brillouin_bin']:.6f}")
    if res['seg_met_stats']:
        print(f"seg_met (1500-1600m) Rayleigh Amp Mean: {res['seg_met_stats']['mean_rayleigh_amp']:.6f}")
        print(f"seg_met (1500-1600m) Brillouin Amp Mean: {res['seg_met_stats']['mean_brillouin_amp']:.6f}")
        print(f"seg_met (1500-1600m) Brillouin Bin Mean: {res['seg_met_stats']['mean_brillouin_bin']:.6f}")

print("\n--- NEW CALCULATED METRICS ---")
print(f"RLP0_free_t20: {RLP0_free_t20:.6f}")
print(f"RLP0_met_t20: {RLP0_met_t20:.6f}")
print(f"RLP1_free_t40: {RLP1_free_t40:.6f}")
print(f"RLP1_met_t40: {RLP1_met_t40:.6f}")
print(f"vB0_free_t20: {vB0_free_t20:.2f} MHz")
print(f"vB0_met_t20: {vB0_met_t20:.2f} MHz")
print(f"vB1_free_t40: {vB1_free_t40:.2f} MHz")
print(f"vB1_met_t40: {vB1_met_t40:.2f} MHz")

print("\n--- DELTA METRICS ---")
print(f"dvB_free: {dvB_free:.2f} MHz")
print(f"dRLP_free: {dRLP_free:.6f} %")
print(f"dvB_met: {dvB_met:.2f} MHz")
print(f"dRLP_met: {dRLP_met:.6f} %")
print(f"dT: {dT} °C")
print(f"de: {de_ue:.2f} ue")

print("\n--- COEFFICIENTS ---")
print(f"D1: {D1:.6f} MHz/°C")
print(f"D2: {D2:.6e} MHz/ue")
print(f"D3: {D3:.6f} %/°C")
print(f"D4: {D4:.6e} %/ue")

# Helper function to add red segment bars to figures
def highlight_segments():
    plt.axvspan(seg_free[0], seg_free[1], color='red', alpha=0.2, label='seg_free (1000-1100m)')
    plt.axvspan(seg_met[0], seg_met[1], color='darkred', alpha=0.2, label='seg_met (1500-1600m)')

t40_label_suffix = " (Simulated)" if (calc_check == 1 and os.path.exists(simulated_file_path)) else ""

# Plotting Figure 2: Peak Values Comparison (T20 vs T40 / Simulated T40)
plt.figure(2, figsize=(12, 6))
plt.plot(results['T20']['distances'], results['T20']['rayleigh_values'], color='blue', linestyle='--', label='T20 - Rayleigh (Raw)', linewidth=1, alpha=0.6)
plt.plot(results['T20']['distances'], results['T20']['brillouin_values'], color='cyan', linestyle='--', label='T20 - Brillouin (Raw)', linewidth=1.5)
plt.plot(results['T40']['distances'], results['T40']['rayleigh_values'], color='red', label=f'T40 - Rayleigh{t40_label_suffix}', linewidth=1, alpha=0.6)
plt.plot(results['T40']['distances'], results['T40']['brillouin_values'], color='darkorange', label=f'T40 - Brillouin{t40_label_suffix}', linewidth=1.5)
highlight_segments()
plt.title(f'Figure 2: Peak Values Comparison (calc_check = {calc_check})')
plt.xlabel('Distance (m)')
plt.ylabel('Peak Amplitude Value')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

# Plotting Figure 3: Peak Bin Positions Comparison (T20 vs T40 / Simulated T40)
plt.figure(3, figsize=(12, 6))
plt.plot(results['T20']['distances'], results['T20']['rayleigh_bins'], color='blue', linestyle='--', label='T20 - Rayleigh Bin (Raw)', linewidth=1, alpha=0.6)
plt.plot(results['T20']['distances'], results['T20']['brillouin_bins'], color='cyan', linestyle='--', label='T20 - Brillouin Bin (Raw)', linewidth=1.5)
plt.plot(results['T40']['distances'], results['T40']['rayleigh_bins'], color='red', label=f'T40 - Rayleigh Bin{t40_label_suffix}', linewidth=1, alpha=0.6)
plt.plot(results['T40']['distances'], results['T40']['brillouin_bins'], color='darkorange', label=f'T40 - Brillouin Bin{t40_label_suffix}', linewidth=1.5)
highlight_segments()
plt.title(f'Figure 3: Peak Bin Positions Comparison (calc_check = {calc_check})')
plt.xlabel('Distance (m)')
plt.ylabel('Frequency Bin Index')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

plt.show()