import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Define the Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# Load the data
data = np.loadtxt('/content/gerotti.txt')
volts = data[:, 0]
current = data[:, 1]

# Find peaks with minimum distance
min_peak_distance = 20 
prominence_value = 0.001 
peaks, _ = find_peaks(current, distance=min_peak_distance, prominence=prominence_value)

# Fit Gaussian to each peak
peak_params = []
for peak in peaks:
    fit_range = 15
    bounds = (max(0, peak-fit_range), min(len(volts)-1, peak+fit_range))
    fit_volts = volts[bounds[0]:bounds[1]]
    fit_current = current[bounds[0]:bounds[1]]
    initial_guess = [fit_current.max(), volts[peak], 1]

    # fit a Gaussian curve to the peak
    try:
        popt, _ = curve_fit(gaussian, fit_volts, fit_current, p0=initial_guess, maxfev=1000)
        peak_params.append(popt)
    except RuntimeError as e:
        print(f"Fit did not converge for peak at {volts[peak]:.2f} V: {e}")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(volts, current, 'b-', label='Measured Data')
for params in peak_params:
    fit_x = np.linspace(params[1] - 2, params[1] + 2, 100)
    fit_y = gaussian(fit_x, *params)
    plt.plot(fit_x, fit_y, 'r-', label=f'Gaussian Fit at {params[1]:.2f} V')
plt.xlabel('Acceleration Potential (Volts)')
plt.ylabel('Current')
plt.title('Gaussian Fits to Detected Peaks in Current Data')
plt.legend()
plt.show()

#peak voltages
print("Peak voltages:")
for params in peak_params:
    print(f'{params[1]:.2f} V')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#linear model
def linear_model(x, a, b):
    return a * x + b


peak_values = [params[1] for params in peak_params]
starting_index = 2
peak_indices = np.arange(starting_index, starting_index + len(peak_values))
popt, pcov = curve_fit(linear_model, peak_indices, peak_values)

#coefficients and their errors
a, b = popt
a_error, b_error = np.sqrt(np.diag(pcov))

# Plotting
plt.figure(figsize=(10, 5))
plt.scatter(peak_indices, peak_values, color='red', label='Peak Values')
plt.plot(peak_indices, linear_model(peak_indices, *popt), 'b-', label=f'Linear Fit: y = {a:.2f}x + {b:.2f}')
plt.xlabel('Peak Number')
plt.ylabel('Peak Voltage')
plt.title('Linear Fit to Peak Values')
plt.legend()
plt.show()

#coefficients and their errors
print(f"Coefficient A (slope): {a:.4f} ± {a_error:.4f}")
print(f"Coefficient B (intercept): {b:.4f} ± {b_error:.4f}")
