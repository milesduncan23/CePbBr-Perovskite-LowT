import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from fitting import pseudo_voigt

# ==========================================
# Helpers
# ==========================================

def peak_colors(n):
    # One tab10 colour per peak (colours repeat after 10 peaks)
    return plt.cm.tab10(np.arange(n) % 10)

def sample_folder(label):
    # Output folder for one sample, e.g. "10% Doped" -> Output/10%_Doped
    folder = cfg.OUTPUT_DIR / label.replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def ref_lines(ax):
    # Fits one straight line through every point on a log-log axes, draws it under the data and labels its slope

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Collect every plotted point
    px, py = [], []
    for line in ax.lines:
        x, y = line.get_data()
        px.extend(x)
        py.extend(y)
    px = np.array(px, dtype=float)
    py = np.array(py, dtype=float)

    # Drops NaN and negative points
    ok = (px > 0) & (py > 0)
    px = px[ok]
    py = py[ok]

    # Slope of the best straight line in log-log space
    m = np.polyfit(np.log10(px), np.log10(py), 1)[0]

    # Line height: half the tightest data / line ratio, so it sits under every point
    xs = np.array([px.min(), px.max()])
    y0 = 0.5 * np.min(py / (px / xs[0])**m)
    ys = y0 * (xs / xs[0])**m

    ax.plot(xs, ys, color='tab:orange', lw=4, alpha=0.8, solid_capstyle='round')

    # Slope label, below-right of the line's midpoint
    mid = (np.sqrt(xs[0] * xs[1]), np.sqrt(ys[0] * ys[1]))
    ax.annotate(f'slope = {m:.2f}', mid, xytext=(8, -6), textcoords='offset points',
                color='tab:orange', fontsize=14, ha='left', va='top')

    # Extend the y axis down so the line and its label fit
    ax.set_ylim(bottom=min(ax.get_ylim()[0], ys.min() / 3))

# ==========================================
# Per-Sample Plots
# ==========================================

def plot_spectrum(x, y, peaks, offset, label, power):
    # Raw data, total fit and every deconvoluted peak for one power

    # Deconvoluted peaks from fit parameters
    comps = np.array([pseudo_voigt(x, *peak) for peak in peaks])
    fit_p = comps + offset

    # Total Fit
    fit_total = offset + comps.sum(axis=0)

    plt.figure(figsize=cfg.FIG_SIZE)

    # Plot raw data and total fit
    plt.plot(x, y, 'k.', alpha=0.3, label='Data')
    plt.plot(x, fit_total, 'r-', linewidth=2.5, label='Total Fit Envelope')

    # Plot individual deconvoluted peaks
    colors = peak_colors(peaks.shape[0])

    for j in range(0, peaks.shape[0]):
        plt.plot(x, fit_p[j], '--', linewidth=2, color=colors[j], label=f'Peak {j+1}: {peaks[j, 1]:.4f} eV')
        plt.fill_between(x, fit_p[j], alpha=0.2, color=colors[j])

    # Formatting
    title = f"Low-T (55K) {label} Spectral Deconvolution ({power})"

    plt.xlabel('Energy (eV)', fontsize=14, fontweight='bold')
    plt.ylabel('PL Intensity (a.u.)', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    plt.xlim(*cfg.ENERGY_RANGE) # Zoom in on the active region
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Plot Saving
    folder = sample_folder(label)
    plt.savefig(folder / (f"{power}".replace(" ", "_") + ".png"), dpi=cfg.DPI, bbox_inches="tight")
    plt.close()

def plot_amplitudes(powers_mw, amps, label):
    # Peak amplitude vs power, as a semi-log figure (peaks_linear) and a log-log figure with fitted slope (peaks_log)

    n = amps.shape[1]
    colors = peak_colors(n)

    pow_lin = plt.figure(figsize=cfg.FIG_SIZE)
    pow_log = plt.figure(figsize=cfg.FIG_SIZE)
    ax_lin = pow_lin.add_subplot()
    ax_log = pow_log.add_subplot()

    for ax in (ax_lin, ax_log):
        for j in range(0, n):
            ax.plot(powers_mw, amps[:, j], 'o-', color=colors[j], label=f'Peak {j+1}')
    ref_lines(ax_log)

    # Formatting
    for ax in (ax_lin, ax_log):
        ax.set_ylabel('Peak amplitude (log a.u.)')
        ax.set_title(f"{label} Peak Amplitude vs Power")
        ax.legend(ncol=3, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
    ax_log.set_xscale('log')
    ax_log.set_yscale('log')
    ax_log.set_xlabel('Power (log mW)')

    ax_lin.set_yscale('log')
    ax_lin.set_xlabel('Power (mW)')

    # Plot Saving (into the sample's own folder; _linear / _log keep the two file names apart)
    folder = sample_folder(label)
    for fig, tag in ((pow_lin, "linear"), (pow_log, "log")):
        fig.savefig(folder / f"peaks_{tag}.png", dpi=cfg.DPI, bbox_inches="tight")
        plt.close(fig)

def plot_centres(powers_mw, cens, label):
    # Peak centre vs power (cens rows: powers, columns: peaks)

    n = cens.shape[1]
    colors = peak_colors(n)

    pos = plt.figure(figsize=cfg.FIG_SIZE)
    ax_pos = pos.add_subplot()

    for j in range(0, n):
        ax_pos.plot(powers_mw, cens[:, j], 'o-', color=colors[j], label=f'Peak {j+1}')

    # Formatting
    ax_pos.set_xlabel('Power (mW)')
    ax_pos.set_ylabel('Peak centre (eV)')
    ax_pos.set_title(f"{label} Peak Centre vs Power")
    ax_pos.legend(ncol=3, fontsize=10)
    ax_pos.grid(True, linestyle='--', alpha=0.4)

    # Plot Saving
    folder = sample_folder(label)
    pos.savefig(folder / "peaks_position.png", dpi=cfg.DPI, bbox_inches="tight")
    plt.close(pos)

# ==========================================
# Combined Figures
# ==========================================

def plot_combined(results, samples, labels, file_name):
    # Amplitude vs power for several samples: one row per sample, semi-log on the left, log-log on the right

    # Skip the figure if one of its samples was not run (see SAMPLES in config)
    for k in samples:
        if k not in results:
            print(f"plot_combined(): skipping {file_name}, sample {k} was not run")
            return

    n = results[samples[0]]["amps"].shape[1]
    colors = peak_colors(n)
    letters = "abcdefghijklmnopqrstuvwxyz"

    # squeeze=False keeps axs 2D (rows x columns) even for a single sample
    fig, axs = plt.subplots(len(samples), 2, figsize=(14, 4.5 * len(samples)), squeeze=False)

    for r in range(0, len(samples)):
        k = samples[r]
        powers_mw = results[k]["powers"]
        amps = results[k]["amps"]

        for c in range(0, 2):
            ax = axs[r, c]

            for j in range(0, n):
                ax.plot(powers_mw, amps[:, j], 'o-', ms=4, color=colors[j], label=f'Peak {j+1}')
            ax.set_yscale('log')

            # Right column: log-log with fitted slope
            if c == 1:
                ax.set_xscale('log')
                ref_lines(ax)

            # Panel letter and sample in the bottom right corner, e.g. "(a) 0% Li @ 55 K"
            letter = letters[2 * r + c]
            ax.text(0.97, 0.05, f'({letter}) {labels[k]} @ 55 K', transform=ax.transAxes, ha='right', va='bottom', fontsize=12)
            ax.grid(True, ls='--', alpha=0.4)

    # Formatting
    for ax in axs[-1]:
        ax.set_xlabel('Power (mW)')
    for ax in axs[:, 0]:
        ax.set_ylabel('Peak amplitude (a.u.)')
    axs[0, 0].legend(ncol=3, fontsize=9, loc='center right')

    # Plot Saving
    folder = cfg.OUTPUT_DIR / "Combined"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / file_name, dpi=cfg.DPI, bbox_inches='tight')
    plt.close(fig)
