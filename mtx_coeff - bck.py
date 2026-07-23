import matplotlib.pyplot as plt
import numpy as np

# File paths
file_paths = {
    'T40': 't_40.csv',
    'T20': 't_20.csv'
}

spatial_resolution = 0.1  # meters per sample

# Define the specified analysis segments (in meters)
seg_free = (1000.0, 1100.0)
seg_met = (1500.0, 1600.0)

# Temperature and thermal expansion coefficient variables
T0 = 20
T1 = 40
Al_alpha = 22.8 * 1e-6  # strain per °C

def process_file(file_path):
    data_rows = 0
    data_columns = 0
    data_matrix = []
    freq_step = -1.953125  # Default fallback if not found

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
    search_window = 10  # searches bins 90 to 110

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

    # Helper to calculate mean metrics inside custom segments
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

    seg_free_stats = get_segment_stats(*seg_free)
    seg_met_stats = get_segment_stats(*seg_met)

    return {
        'data_rows': data_rows,
        'data_columns': data_columns,
        'freq_step': freq_step,
        'distances': distances,
        'rayleigh_values': rayleigh_values,
        'brillouin_values': brillouin_values,
        'rayleigh_bins': rayleigh_bins,
        'brillouin_bins': brillouin_bins,
        'seg_free_stats': seg_free_stats,
        'seg_met_stats': seg_met_stats
    }

# Process both files
results = {}
for label, path in file_paths.items():
    print(f"Processing {label} ({path})...")
    results[label] = process_file(path)

# Helper function to convert bin to MHz
def bin_to_mhz(bin_val, freq_step):
    return bin_val * abs(freq_step)

# Calculate RLP and vB (in MHz) metrics for both files
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
de_ue = (Al_alpha * dT) * 1e6          # in ue (με)

# Calculate coefficients with correct requested units
D1 = dvB_free / dT                              # MHz / °C
D2 = (dvB_met - dvB_free) / de_ue     # MHz / ue
D3 = dRLP_free / dT                             # % / °C
D4 = (dRLP_met - dRLP_free) / de_ue   # % / ue

# Print file sizes and results comparison for the segments using raw data
for label in ['T40', 'T20']:
    res = results[label]
    print(f"\n--- RESULTS FOR {label} (RAW DATA) ---")
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

# Plotting comparisons for Figure 2 with segments highlighted
plt.figure(2, figsize=(12, 6))
plt.plot(results['T40']['distances'], results['T40']['rayleigh_values'], color='red', label='T40 - Rayleigh (Raw)', linewidth=1, alpha=0.6)
plt.plot(results['T40']['distances'], results['T40']['brillouin_values'], color='cyan', label='T40 - Brillouin (Raw)', linewidth=1.5)
plt.plot(results['T20']['distances'], results['T20']['rayleigh_values'], color='darkorange', linestyle='--', label='T20 - Rayleigh (Raw)', linewidth=1, alpha=0.6)
plt.plot(results['T20']['distances'], results['T20']['brillouin_values'], color='blue', linestyle='--', label='T20 - Brillouin (Raw)', linewidth=1.5)
highlight_segments()
plt.title('Figure 2: Peak Values Comparison with Segments (T40 vs T20)')
plt.xlabel('Distance (m)')
plt.ylabel('Peak Amplitude Value')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

# Plotting comparisons for Figure 3 with segments highlighted
plt.figure(3, figsize=(12, 6))
plt.plot(results['T40']['distances'], results['T40']['rayleigh_bins'], color='red', label='T40 - Rayleigh Bin (Raw)', linewidth=1, alpha=0.6)
plt.plot(results['T40']['distances'], results['T40']['brillouin_bins'], color='cyan', label='T40 - Brillouin Bin (Raw)', linewidth=1.5)
plt.plot(results['T20']['distances'], results['T20']['rayleigh_bins'], color='darkorange', linestyle='--', label='T20 - Rayleigh Bin (Raw)', linewidth=1, alpha=0.6)
plt.plot(results['T20']['distances'], results['T20']['brillouin_bins'], color='blue', linestyle='--', label='T20 - Brillouin Bin (Raw)', linewidth=1.5)
highlight_segments()
plt.title('Figure 3: Peak Bin Positions Comparison with Segments (T40 vs T20)')
plt.xlabel('Distance (m)')
plt.ylabel('Frequency Bin Index')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

plt.show()