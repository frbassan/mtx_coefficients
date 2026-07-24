import csv
import matplotlib.pyplot as plt
import numpy as np

# File path for T20 only
file_path_T20 = 't_20.csv'
spatial_resolution = 0.1  # meters per sample

# Define the specified analysis segments (in meters)
seg_free = (1000.0, 1100.0)
seg_met = (1500.0, 1600.0)

# Given coefficients and temperature change variable
D1 = 1.07       # MHz / °C
D2 = 0.0470     # MHz / microstrain
D3 = 0.00300    # 1 / °C (% / °C)
D4 = -0.0000086 # 1 / °C (% / °C)
dT = 40.0       # °C (change this variable as needed)

data_rows = 0
data_columns = 0
data_matrix = []

with open(file_path_T20, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith('Param;'):
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

rayleigh_values = np.array(rayleigh_values)
brillouin_values = np.array(brillouin_values)
rayleigh_bins = np.array(rayleigh_bins, dtype=float)
brillouin_bins = np.array(brillouin_bins, dtype=float)

# Extract frequency step to convert coefficients into bin/amplitude shifts
freq_step = -1.953125  # default or parse if available

with open(file_path_T20, 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('Param;'):
            parts = line.split(';')
            if len(parts) >= 3 and parts[1] == 'freqStep':
                freq_step = float(parts[2])
                break

# Calculate expected shifts based on the provided coefficients and dynamic dT
bin_shift_free = (D1 * dT) / abs(freq_step)
amp_ratio_shift_free = D3 * dT 

de_microstrain = 22.8 * 1e-6 * dT * 1e6  # thermal expansion strain based on dT
bin_shift_met = (D1 * dT + D2 * de_microstrain) / abs(freq_step)
amp_ratio_shift_met = D3 * dT + D4 * de_microstrain

# Create simulated arrays based on T20 + shifts applied ONLY at the segments
rayleigh_values_sim = rayleigh_values.copy()
brillouin_bins_sim = brillouin_bins.copy()
brillouin_values_sim = brillouin_values.copy()

# Index mapping for segments
idx_free_start = int(seg_free[0] / spatial_resolution)
idx_free_end = int(seg_free[1] / spatial_resolution)
idx_met_start = int(seg_met[0] / spatial_resolution)
idx_met_end = int(seg_met[1] / spatial_resolution)

# Apply positive modifications to seg_free (Brillouin amplitude increases)
brillouin_bins_sim[idx_free_start:idx_free_end] += bin_shift_free
for i in range(idx_free_start, idx_free_end):
    brillouin_values_sim[i] = brillouin_values[i] * (1.0 + amp_ratio_shift_free)

# Apply positive modifications to seg_met (Brillouin amplitude increases)
brillouin_bins_sim[idx_met_start:idx_met_end] += bin_shift_met
for i in range(idx_met_start, idx_met_end):
    brillouin_values_sim[i] = brillouin_values[i] * (1.0 + amp_ratio_shift_met)

# Save the simulated data to CSV file using dynamic dT in the filename
output_file_name = f"sim_t_{int(dT+20)}.csv"

with open(output_file_name, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(['Distance_m', 'Rayleigh_Peak_Amplitude', 'Brillouin_Peak_Amplitude', 'Rayleigh_Peak_Bin', 'Brillouin_Peak_Bin'])
    
    for d, r_amp, b_amp, r_bin, b_bin in zip(distances, rayleigh_values_sim, brillouin_values_sim, rayleigh_bins, brillouin_bins_sim):
        writer.writerow([f"{d:.2f}", f"{r_amp:.6f}", f"{b_amp:.6f}", f"{r_bin}", f"{b_bin:.6f}"])

print(f"Successfully generated and saved: {output_file_name}")

# Helper function to add red segment bars to figures
def highlight_segments():
    plt.axvspan(seg_free[0], seg_free[1], color='red', alpha=0.2, label='seg_free (1000-1100m)')
    plt.axvspan(seg_met[0], seg_met[1], color='darkred', alpha=0.2, label='seg_met (1500-1600m)')

# Figure 1: Simulated Brillouin Peak Values (Positive Shift)
plt.figure(1, figsize=(12, 6))
plt.plot(distances, brillouin_values, color='blue', label='T20 - Brillouin (Original)', linewidth=1, alpha=0.5)
plt.plot(distances, brillouin_values_sim, color='red', label=f'Simulated - Brillouin (dT={dT}°C)', linewidth=1.5)
highlight_segments()
plt.title(f'Figure 1: Simulated Brillouin Peak Values (Positive Shift, dT = {dT}°C)')
plt.xlabel('Distance (m)')
plt.ylabel('Peak Amplitude Value')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

# Figure 2: Simulated Peak Bin Positions
plt.figure(2, figsize=(12, 6))
plt.plot(distances, brillouin_bins, color='blue', label='T20 - Brillouin Bin (Original)', linewidth=1, alpha=0.5)
plt.plot(distances, brillouin_bins_sim, color='red', label=f'Simulated - Brillouin Bin (dT={dT}°C)', linewidth=1.5)
highlight_segments()
plt.title(f'Figure 2: Simulated Peak Bin Positions (dT = {dT}°C)')
plt.xlabel('Distance (m)')
plt.ylabel('Frequency Bin Index')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

plt.show()