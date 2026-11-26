import numpy as np

import config as cfg
from fitting import load_sample, fit_peaks
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

    # No previous fit at a sample's first power (powers run low -> high)
    prev = None

    power_names, powers_mw, fits, offsets = [], [], [], []

    for power, y_data in spectra.items():

        # Fit peak parameters using initial guesses, starting from the previous power's fit
        peaks, offset = fit_peaks(x_data, y_data, cfg.N_PEAKS, cfg.PEAK_OFFSETS, cfg.REF, cfg.REF_POS, prev)
        prev = peaks

        # Collect per-power results
        power_names.append(power)
        powers_mw.append(float(power.split()[0]))
        fits.append(peaks)
        offsets.append(offset)

    fits = np.array(fits) # [power, peak, parameter], parameters: amp, center, sigma, eta

    results[i] = {
        "energy": x_data,
        "spectra": list(spectra.values()), # One y array per power
        "power_names": power_names,        # e.g. "10 mW", used in titles and file names
        "powers": powers_mw,               # mW
        "peaks": fits,
        "offsets": offsets,
        "amps": fits[:, :, 0].copy(),      # Copy, so leaving out garbage points never touches the fits
        "fwhm": 2.355 * fits[:, :, 2],     # Math might differ, this is the equation for Gaussians
    }

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