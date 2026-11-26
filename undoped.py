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
# Analysis
# ==========================================

results = {}

for i in cfg.SAMPLES:

    # ==========================================
    # Read and process data
    # ==========================================

    x_data, spectra = load_sample(cfg.DATA_DIR / (cfg.FILES[i] + ".xlsx"))

    # No previous fit at a sample's first power (powers run low -> high)
    prev = None

    powers_mw, amps, cens = [], [], []

    for power, y_data in spectra.items():

        # ==========================================
        # Fit Creation
        # ==========================================

        # Fit peak parameters using initial guesses, starting from the previous power's fit
        peaks, offset = fit_peaks(x_data, y_data, cfg.N_PEAKS, cfg.PEAK_OFFSETS, cfg.REF, cfg.REF_POS, prev)
        prev = peaks

        # Math might differ, this is the equation for Gaussians
        fwhm = 2.355 * peaks[:, 2]

        # ==========================================
        # Per-Power Spectra Plotting
        # ==========================================

        plot_spectrum(x_data, y_data, peaks, offset, cfg.DOPE_LABELS[i], power)

        # Collect per-power data for plotting
        powers_mw.append(float(power.split()[0]))
        amps.append(peaks[:, 0])
        cens.append(peaks[:, 1])

    # ==========================================
    # Plot amplitude & centre vs power
    # ==========================================

    amps = np.array(amps)
    cens = np.array(cens)   # rows: powers, columns: peaks

    # Leave out points flagged as garbage in config
    for (k, p, j) in cfg.GARBAGE_PEAKS:
        if k == i:
            amps[p, j] = np.nan

    results[i] = {"powers": powers_mw, "amps": amps}

    plot_amplitudes(powers_mw, amps, cfg.DOPE_LABELS[i])
    plot_centres(powers_mw, cens, cfg.DOPE_LABELS[i])

# ==========================================
# Subplot like fig 7
# ==========================================

plot_combined(results, (0, 4), cfg.LI_LABELS, "undoped_40%_peaks.png")

# ==========================================
# Subplot for 10, 20, 30
# ==========================================

plot_combined(results, (1, 2, 3), cfg.LI_LABELS, "10_20_30%_peaks.png")
