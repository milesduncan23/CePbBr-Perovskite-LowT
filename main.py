import numpy as np

import config as cfg
from fitting import load_sample, noise_level, fit_free, fit_locked
from plotting import plot_spectrum, plot_amplitudes, plot_centres, plot_combined

# ==========================================
# TO-DO
# ==========================================

# Add subplot like fig 6

# Create a single fit parameter per sample, where we can vary the amplitude of the
# distributions only between powers to obtain a more consistent fit

# Investigate misfits due to limits on peak variation from guess positions

# Fit uncertainties

# ==========================================
# Fitting
# ==========================================

results = {}

for i in cfg.SAMPLES:

    # Read and process data
    x_data, spectra = load_sample(cfg.DATA_DIR / (cfg.FILES[i] + ".xlsx"))

    power_names = list(spectra) # Low -> high, as in the spreadsheet
    powers_mw = [float(power.split()[0]) for power in power_names]

    fits = [None] * len(power_names)
    offsets = [None] * len(power_names)

    # Fit from the highest power down: the brightest spectrum gives the most reliable first fit, and its peak
    # positions (the pattern) are then locked for every lower power, which may only slide them all together
    pattern = None
    prev = None

    for k in range(len(power_names) - 1, -1, -1):
        y_data = spectra[power_names[k]]

        if pattern is None:
            peaks, offset = fit_free(x_data, y_data, cfg.N_PEAKS, cfg.PEAK_OFFSETS, cfg.REF_POS)
            pattern = peaks
        else:
            peaks, offset = fit_locked(x_data, y_data, pattern, prev)
        prev = peaks

        # Store at index k, so results stay in low -> high power order
        fits[k] = peaks
        offsets[k] = offset

    fits = np.array(fits) # [power, peak, parameter], parameters: amp, center, sigma, eta

    # Noise level of each spectrum, for the detection limit below
    noise = []
    for k in range(0, len(power_names)):
        noise.append(noise_level(spectra[power_names[k]]))

    results[i] = {
        "energy": x_data,
        "spectra": list(spectra.values()), # One y array per power
        "power_names": power_names,        # e.g. "10 mW", used in titles and file names
        "powers": powers_mw,               # mW
        "peaks": fits,
        "offsets": offsets,
        "noise": noise,                    # One noise level per power, same units as the spectra
        "amps": fits[:, :, 0].copy(),      # Copy, so leaving out undetected / garbage points never touches the fits
        "fwhm": 2.355 * fits[:, :, 2],     # Math might differ, this is the equation for Gaussians
    }

# Leave out amplitudes below the detection limit (amplitude plots only): a peak smaller than DETECTION_SNR
# noise units is not really measured, and its near-zero amplitude would spike down on the log axes
for i in results:
    for k in range(0, len(results[i]["powers"])):
        limit = cfg.DETECTION_SNR * results[i]["noise"][k]
        for j in range(0, cfg.N_PEAKS):
            if results[i]["amps"][k, j] < limit:
                results[i]["amps"][k, j] = np.nan

# Leave out points flagged as garbage in config (amplitude plots only)
for (i, p, j) in cfg.GARBAGE_PEAKS:
    if i in results:
        results[i]["amps"][p, j] = np.nan

# ==========================================
# Per-Sample Plotting
# ==========================================

for i in results:
    sample = results[i]
    label = cfg.DOPE_LABELS[i]

    # One deconvolution plot per power
    for k in range(0, len(sample["powers"])):
        plot_spectrum(sample["energy"], sample["spectra"][k], sample["peaks"][k], sample["offsets"][k], label, sample["power_names"][k])

    # Amplitude & centre vs power
    plot_amplitudes(sample["powers"], sample["amps"], label)
    plot_centres(sample["powers"], sample["peaks"][:, :, 1], label)

# ==========================================
# Combined Figures
# ==========================================

# Subplot like fig 7
plot_combined(results, (0, 4), cfg.LI_LABELS, "undoped_40%_peaks.png")

# Subplot for 10, 20, 30
plot_combined(results, (1, 2, 3), cfg.LI_LABELS, "10_20_30%_peaks.png")