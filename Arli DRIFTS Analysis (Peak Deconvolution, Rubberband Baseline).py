#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 5 13:49:20 2026

@author: Arli
"""

import re
import os
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

"""
User Settings & Parameters
"""

# DRIFTS Graphing/Peak Analysis
min_wavenumber = 1888
max_wavenumber = 2144

"""
Data Unpacking
"""

pattern_time = r'time_(-?\d+\.?\d*)'
pattern_temp = r'temp_(-?\d+\.?\d*)'

files = sorted([
    file for file in os.listdir('.')
    if file.endswith('.csv') and file != 'Summary.csv'
])

if not files:
    raise FileNotFoundError('No .csv files found in the current folder.')
    
# Get the number of points we can expect to find in each spectrum
tmp = pd.read_csv(files[0], header=0, usecols=[0, 1])
points_expected = len(tmp)
spectra_expected = len(files)

# Generate empty data arrays with NaN so they are expected shape always
x = np.full((points_expected, spectra_expected), np.nan)
y = np.full((points_expected, spectra_expected), np.nan)

# Loop over every file and load it into the arrays
for i, fname in enumerate(files):
    df = pd.read_csv(fname, header=0, usecols=[0, 1])

    xi = df.iloc[:, 0].values   # First column  (e.g. wavenumber)
    yi = df.iloc[:, 1].values   # Second column (e.g. absorbance)

    # Safety check: all files must have the same number of points
    if len(xi) != points_expected:
        raise ValueError(
            f'File {fname} has {len(xi)} points; expected {points_expected}. '
            'Resample or fix input.'
        )

    x[:, i] = xi
    y[:, i] = yi
    print(fname)

"""
Crop to Window
"""

# Create a true/false filter to crop to between min and max wavenumber
crop_filter = (x[:, 0] > min_wavenumber) & (x[:, 0] < max_wavenumber)

if not np.any(crop_filter):
    raise ValueError(
        f'No points found between minimum wavenumber={min_wavenumber:.1f} and maximum wavenumber={max_wavenumber:.1f}. '
        'Check bounds and x axis units.'
    )

x_cropped = x[crop_filter, 0] # common x-axis in window
points_window = np.sum(crop_filter) # number of points in the cropped window

# Create data frames prepped for csv files wavenumbers:spectra
absorbance = np.zeros((points_window, spectra_expected))
integral_value = np.zeros(spectra_expected)
peaks = np.zeros(spectra_expected)
peak_positions = np.zeros(spectra_expected)

"""
Data Processing
"""

# Rubberband baseline function courtesy of claude
def rubberband_baseline(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # Compute full 2D convex hull of (x, y) points
    pts = np.column_stack([x, y])
    hull = ConvexHull(pts)

    # hull.vertices are unordered — get the full ordered boundary
    # ConvexHull.vertices gives indices; we need to trace the hull
    verts = hull.vertices

    # Find leftmost and rightmost vertices (by x)
    i_min = verts[np.argmin(x[verts])]
    i_max = verts[np.argmax(x[verts])]

    # Trace hull vertices in order to extract lower boundary
    # ConvexHull gives vertices counter-clockwise
    v_ordered = list(hull.vertices)
    n = len(v_ordered)

    idx_min = v_ordered.index(i_min)
    idx_max = v_ordered.index(i_max)

    # Extract lower hull segment (from i_min to i_max going clockwise,
    # i.e. the direction with lower y values)
    if idx_min < idx_max:
        lower_seg_a = v_ordered[idx_min:idx_max + 1]
        lower_seg_b = v_ordered[idx_max:] + v_ordered[:idx_min + 1]
    else:
        lower_seg_a = v_ordered[idx_min:] + v_ordered[:idx_max + 1]
        lower_seg_b = v_ordered[idx_max:idx_min + 1]

    # Pick whichever segment has the lower mean y — that's the lower hull
    mean_a = np.mean(y[lower_seg_a])
    mean_b = np.mean(y[lower_seg_b])
    lower = lower_seg_a if mean_a < mean_b else lower_seg_b

    # Sort by x for interpolation
    lower_x = x[lower]
    lower_y = y[lower]
    order = np.argsort(lower_x)
    lower_x = lower_x[order]
    lower_y = lower_y[order]

    # Piecewise-linear baseline interpolated back onto original x
    ybase = np.interp(x, lower_x, lower_y)
    return ybase

# Do baseline subtraction, get peak area
for k in range(spectra_expected):
    xw = x[crop_filter, k]
    yw = y[crop_filter, k]

    # Rubberband baseline (lower convex hull)
    ybase = rubberband_baseline(xw, yw)

    # Corrected spectrum
    ycorr = yw - ybase

    # Shift so edges touch zero if there are small negative dips
    n_edge = min(15, len(ycorr) // 10)
    edge_min = min(np.min(ycorr[:n_edge]), np.min(ycorr[-n_edge:]))
    if edge_min < 0:
        ycorr = ycorr - edge_min

    absorbance[:, k] = ycorr

    # Peak height and position
    pidx = np.argmax(ycorr)
    peaks[k] = ycorr[pidx]
    peak_positions[k] = xw[pidx]

    if xw[1] > xw[0]: # increasing x, works with np.trapz by default
        integral_value[k] = np.trapz(ycorr, xw)
    else: # decreasing x, need to reverse data for np.trapz to work
        integral_value[k] = np.trapz(ycorr[::-1], xw[::-1])

# Stack x_cropped with all baseline-subtracted spectra for export
For_origin = np.column_stack([x_cropped, absorbance])

# Get the time and temp for the file names
time_value  = np.full(spectra_expected, np.nan)
Temperature = np.full(spectra_expected, np.nan)

for i, fname in enumerate(files):
    m1 = re.search(pattern_time, fname)
    if m1:
        time_value[i] = float(m1.group(1))

    m2 = re.search(pattern_temp, fname)
    if m2:
        Temperature[i] = float(m2.group(1))

# Combine into a single array: [time, temperature, integral, peak_position]
For_time_integral_Temp = np.column_stack([
    time_value,
    Temperature,
    integral_value,
    peak_positions
])

For_time_integral_Temp = np.column_stack([time_value, Temperature, integral_value, peak_positions])
#np.savetxt('Summary.csv', For_time_integral_Temp, delimiter=',',
#           header='time,temperature,integral,peak_position', comments='')

"""
Deconvolution
"""
# Peak deconvolution
peaks_expected = 5
# Each expected peak: amplitude, center, fwhm, eta (0=Gaussian, 1=Lorentzian)
initial_guess = [
    np.max(absorbance[:, 0])/4,   1940, 50, 0.5,
    np.max(absorbance[:, 0])/4,   1980, 50, 0.5,
    np.max(absorbance[:, 0])/2.5, 2023, 50, 0.5,
    np.max(absorbance[:, 0]),     2044, 30, 0.5,
    np.max(absorbance[:, 0])/4,   2070, 20, 0.5,
]

# initial_guess = [
#     np.max(absorbance[:, 0])/4,   1940, 50, 0.5,
#     np.max(absorbance[:, 0])/4,   1980, 50, 0.5,
#     np.max(absorbance[:, 0])/2.5, 2023, 50, 0.5,
#     np.max(absorbance[:, 0])/4,   2070, 20, 0.5,
# ]

def gaussian(x, amplitude, center, fwhm):
    sigma = fwhm / 2.355
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

def lorentzian(x, amplitude, center, fwhm):
    gamma = fwhm / 2
    return amplitude * (gamma ** 2) / ((x - center) ** 2 + gamma ** 2)

def pseudo_voigt(x, amplitude, center, fwhm, eta):
    """
    Linear combination of Gaussian and Lorentzian.
    eta=0 -> pure Gaussian, eta=1 -> pure Lorentzian.
    """
    return (1 - eta) * gaussian(x, amplitude, center, fwhm) \
             +   eta  * lorentzian(x, amplitude, center, fwhm)

def multi_peak(x, params, num_peaks):
    """
    Sum of num_peaks pseudo-Voigt peaks evaluated at x.
    Each peak has 4 params: (A, center, fwhm, eta)
    """
    y = np.zeros_like(x, dtype=float)
    for i in range(num_peaks):
        A     = params[4 * i]
        c     = params[4 * i + 1]
        fwhm  = params[4 * i + 2]
        eta   = params[4 * i + 3]
        y += pseudo_voigt(x, A, c, fwhm, eta)
    return y

def fit_peaks(x, y, num_peaks, initial_guess, overlap_penalty):
    def residuals(params):
        # --- fit quality ---
        fit_loss = np.sum((y - multi_peak(x, params, num_peaks)) ** 2)

        # --- overlap penalty ---
        # Build each individual peak curve, then penalize pairwise inner products
        peaks = []
        for i in range(num_peaks):
            A    = params[4 * i]
            c    = params[4 * i + 1]
            fwhm = params[4 * i + 2]
            eta  = params[4 * i + 3]
            peaks.append(pseudo_voigt(x, A, c, fwhm, eta))

        overlap_loss = 0.0
        for i in range(num_peaks):
            for j in range(i + 1, num_peaks):
                overlap_loss += np.dot(peaks[i], peaks[j])

        return fit_loss + overlap_penalty * overlap_loss

    bounds = []
    for i in range(num_peaks):
        bounds.append((0, 0.37))                      # amplitude
        bounds.append((min_wavenumber, max_wavenumber)) # center
        bounds.append((1, 50))                         # FWHM
        bounds.append((0, 1))                          # eta (0=G, 1=L)

    result = minimize(
        residuals,
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100000, 'maxfev': 10000}
    )
    fit_params = result.x

    y_fit  = multi_peak(x, fit_params, num_peaks)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_sq   = 1 - ss_res / ss_tot
    return fit_params, r_sq

def calculate_integrals(amplitudes, fwhms, etas):
    """
    Analytical integrals for pseudo-Voigt components.
    Gaussian integral: A * sigma * sqrt(2*pi)
    Lorentzian integral: A * gamma * pi
    Combined: weighted sum by eta.
    """
    sigmas = fwhms / 2.355
    gammas = fwhms / 2
    g_integrals = amplitudes * np.sqrt(2 * np.pi) * sigmas
    l_integrals = amplitudes * np.pi * gammas
    return (1 - etas) * g_integrals + etas * l_integrals

def deconvolute_spectrum(x, y, num_peaks, initial_guess, overlap_penalty=0.001):
    """
    Fit spectrum with pseudo-Voigt peaks (Gaussian/Lorentzian mix per peak).
    Returns:
      amplitudes : peak heights
      centers    : wavenumber of peak center
      fwhms      : full width at half maximum
      etas       : mixing parameter per peak (0=Gaussian, 1=Lorentzian)
      integrals  : peak areas (analytical)
      r_sq       : R²
    """
    fit_params, r_sq = fit_peaks(x, y, num_peaks, initial_guess, overlap_penalty)

    # Unpack — step of 4 now
    amplitudes = fit_params[0::4]
    centers    = fit_params[1::4]
    fwhms      = fit_params[2::4]
    etas       = fit_params[3::4]
    integrals  = calculate_integrals(amplitudes, fwhms, etas)

    return amplitudes, centers, fwhms, etas, integrals, r_sq

"""
Save your results
"""

all_deconv = []
prev_integrals = None  # store previous spectrum's integrals
all_integrals = [] # store all integrals

for k in range(spectra_expected):
    amplitudes, centers, fwhms, etas, integrals, r_sq = deconvolute_spectrum(  # added etas, fixed missing comma
        x_cropped,
        absorbance[:, k],
        peaks_expected,
        initial_guess)
    
    for i in range(peaks_expected):
        all_deconv.append({
            'Spectrum': k + 1,
            'Peak': i + 1,
            'Amplitude': amplitudes[i],
            'Center': centers[i],
            'FWHM': fwhms[i],
            'Eta': etas[i],          # new — 0=Gaussian, 1=Lorentzian
            'Integral': integrals[i],
            'R_sq': r_sq
        })
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_cropped, absorbance[:, k],
            'b', linewidth=2, label='Baseline-corrected spectrum')

    # multi_gaussian -> multi_peak; params now stride by 4
    y_fit = multi_peak(x_cropped,
                       np.array([[amplitudes[i], centers[i], fwhms[i], etas[i]]
                                 for i in range(peaks_expected)]).ravel(),
                       peaks_expected)
    ax.plot(x_cropped, y_fit, 'r--', linewidth=2, label='Total fit')

    colors = ['green', 'orange', 'purple', 'red', 'blue']
    for i in range(peaks_expected):
        delta = integrals[i] - prev_integrals[i] if prev_integrals is not None else 0.0
        # gaussian -> pseudo_voigt, pass eta
        y_peak = pseudo_voigt(x_cropped, amplitudes[i], centers[i], fwhms[i], etas[i])
        ax.plot(x_cropped, y_peak, '--', color=colors[i % len(colors)],
                linewidth=1.5,
                label=f'Peak {i+1} ({centers[i]:.1f} cm⁻¹) η={etas[i]:.2f} Area {integrals[i]:.2f} Δ {delta:+.2f}')

    if k % 2 == 0:
        state = 'light'
    else:
        state = 'dark'
        theta = 'Not Applicable'
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Absorbance')
    ax.set_title(f'Peak Deconvolution (Spectrum {k+1}, {state}) —  R² = {r_sq:.4f}')
    ax.legend()
    plt.tight_layout()
    plt.show()
    prev_integrals = integrals.copy()
    all_integrals += [integrals]

# Uncomment if you want to save
#deconv_results = pd.DataFrame(all_deconv)
#deconv_results.to_csv('DeconvolutionResults.csv', index=False)

"""
Peak Areas over time vs. photon flux
"""

WC_integrals = [peaks[4] for peaks in all_integrals]
UC_integrals = [peaks[2] +peaks[3] for peaks in all_integrals]
HUC_integrals = [peaks[0] + peaks[1] for peaks in all_integrals]
total_integrals = [arr.sum() for arr in all_integrals]

# Uncomment the latter parts to get fraction of each site type
WC_Light = np.array(WC_integrals[::2])#/np.array(total_integrals[::2])
WC_Dark = np.array(WC_integrals[1::2])#/np.array(total_integrals[1::2])
UC_Light = np.array(UC_integrals[::2])#/np.array(total_integrals[::2])
UC_Dark = np.array(UC_integrals[1::2])#/np.array(total_integrals[1::2])
HUC_Light = np.array(HUC_integrals[::2])#/np.array(total_integrals[::2])
HUC_Dark = np.array(HUC_integrals[1::2])#/np.array(total_integrals[1::2])

# Plotting

J_dark = np.array([0, 0.15, 0.306, 0.459, 0.612])/4.52E-19/(0.81)/10**15
plt.scatter(J_dark, UC_Dark, label="UC Dark")
plt.scatter(J_dark, HUC_Dark, label="HUC Dark")
plt.scatter(J_dark, WC_Dark, label="WC Dark")
plt.legend()
plt.title("Dark WC, UC, HUC")
plt.xlabel("Photon flux")
plt.show()

# plt.scatter(range(636), UC_Dark, label="UC Dark")
# plt.scatter(range(636), HUC_Dark, label="HUC Dark")
# plt.scatter(range(636), WC_Dark, label="WC Dark")
# plt.legend()
# plt.title("Dark WC, UC, HUC")
# plt.show()

J_light = np.array([0.15, 0.306, 0.459, 0.612, 0.765])/4.52E-19/(0.81)/10**15
plt.scatter(J_light, UC_Light, label="UC Light")
plt.scatter(J_light, HUC_Light, label="HUC Light")
plt.scatter(J_light, WC_Light, label="WC Light")
plt.xlabel("Photon Flux")
plt.title("Light WC, UC, HUC")
plt.legend()
plt.show()

# plt.scatter(range(636), UC_Light, label="UC Light")
# plt.scatter(range(636), HUC_Light, label="HUC Light")
# plt.scatter(range(636), WC_Light, label="WC Light")
# plt.title("Light WC, UC, HUC")
# plt.legend()
# plt.show()

"""
integral percent change
"""

for prev, curr in zip(total_integrals, total_integrals[1:]):
    total_percent_change = (curr - prev) / prev

WC_percent_change = []
for prev, curr in zip(WC_Light, WC_Light[1:]):
    WC_percent_change += [(curr - prev) / prev *100]

UC_percent_change = []
for prev, curr in zip(UC_Light, UC_Light[1:]):
    UC_percent_change += [(curr - prev) / prev *100]
    
HUC_percent_change = []
for prev, curr in zip(HUC_Light, HUC_Light[1:]):
    HUC_percent_change += [(curr - prev) / prev *100]
    
plt.plot(J_light[1:], WC_percent_change, label= 'WC')
plt.plot(J_light[1:], UC_percent_change, label = 'UC')
plt.plot(J_light[1:], HUC_percent_change, label = 'HUC')
plt.legend()
plt.show()