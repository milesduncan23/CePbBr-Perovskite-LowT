import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from fitting import load_sample, pseudo_voigt, fit_peaks

# ==========================================
# TO-DO
# ==========================================

# Add subplot like fig 6

# Create a single fit parameter per sample, where we can vary the amplitude of the 
# distributions only between powers to obtain a more consistent fit

# Investigate misfits due to limits on peak variation from guess positions

# Fit uncertainties

# ==========================================
# Function Declaration
# ==========================================

def ref_lines(ax):
    ax.set_xscale('log'); ax.set_yscale('log')
    px, py = np.array([(a, b) for ln in ax.lines for a, b in zip(*ln.get_data())]).T
    
    # drops NaN and negative points
    ok = (px > 0) & (py > 0)
    px, py = px[ok], py[ok]
    
    m = np.polyfit(np.log10(px), np.log10(py), 1)[0]
    xs = np.array([px.min(), px.max()])
    y0 = 0.5 * np.min(py / (px / xs[0])**m)
    ys = y0 * (xs / xs[0])**m
    ax.plot(xs, ys, color='tab:orange', lw=4, alpha=0.8, solid_capstyle='round')
    ax.annotate(f'slope = {m:.2f}', (np.sqrt(xs[0] * xs[1]), np.sqrt(ys[0] * ys[1])), xytext=(8, -6), textcoords='offset points', color='tab:orange', fontsize=14, ha='left', va='top')
    ax.set_ylim(bottom=min(ax.get_ylim()[0], ys.min() / 3))

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

    # Create per-sample peak vs power plots
    pow_lin = plt.figure(figsize=cfg.FIG_SIZE)
    pow_log = plt.figure(figsize=cfg.FIG_SIZE)
    ax_lin = pow_lin.add_subplot()
    ax_log = pow_log.add_subplot()

    pos = plt.figure(figsize=cfg.FIG_SIZE)
    ax_pos = pos.add_subplot()
    
    powers_mw, amps, cens = [], [], []

    for power, y_data in spectra.items():

        # ==========================================
        # Fit Creation
        # ==========================================

        # Fit peak parameters using initial guesses, starting from the previous power's fit
        peaks, offset = fit_peaks(x_data, y_data, cfg.N_PEAKS, cfg.PEAK_OFFSETS, cfg.REF, cfg.REF_POS, prev)
        prev = peaks

        # Deconvoluted peaks from fit parameters
        comps = np.array([pseudo_voigt(x_data, *peak) for peak in peaks])
        fit_p = comps + offset
        
        # Total Fit
        fit_total = offset + comps.sum(axis=0)
        
        # Math might differ, this is the equation for Gaussians
        fwhm = 2.355 * peaks[:, 2]
        
        # ==========================================
        # Per-Power Spectra Plotting
        # ==========================================
        
        plt.figure(figsize=cfg.FIG_SIZE)

        # Plot raw data and total fit
        plt.plot(x_data, y_data, 'k.', alpha=0.3, label='Data')
        plt.plot(x_data, fit_total, 'r-', linewidth=2.5, label='Total Fit Envelope')

        # Plot individual deconvoluted peaks
        colors = plt.cm.tab10(np.arange(peaks.shape[0]) % 10)
        
        for j in range(0, peaks.shape[0]):
            plt.plot(x_data, fit_p[j], '--', linewidth=2, color=colors[j], label=f'Peak {j+1}: {peaks[j, 1]:.4f} eV')
            plt.fill_between(x_data, fit_p[j], alpha=0.2, color=colors[j])
        
        # Formatting
        title = f"Low-T (55K) {cfg.DOPE_LABELS[i]} Spectral Deconvolution ({power})"

        plt.xlabel('Energy (eV)', fontsize=14, fontweight='bold')
        plt.ylabel('PL Intensity (a.u.)', fontsize=14, fontweight='bold')
        plt.title(title, fontsize=16)
        plt.legend(fontsize=12, loc='upper right')
        plt.xlim(*cfg.ENERGY_RANGE) # Zoom in on the active region
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()

        # Plot Saving
        folder = cfg.OUTPUT_DIR / cfg.DOPE_LABELS[i].replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(folder / (f"{power}".replace(" ", "_") + ".png"), dpi=cfg.DPI, bbox_inches="tight")
        plt.close()
        
        # Collect per-power data for plotting
        powers_mw.append(float(power.split()[0]))
        amps.append(peaks[:, 0])
        cens.append(peaks[:, 1])


    # ==========================================
    # Plot amplitude vs power
    # ==========================================
    
    amps = np.array(amps)

    # Leave out points flagged as garbage in config
    for (k, p, j) in cfg.GARBAGE_PEAKS:
        if k == i:
            amps[p, j] = np.nan

    results[i] = (powers_mw, amps)
    for ax in (ax_lin, ax_log):
        for j in range(cfg.N_PEAKS):
            ax.plot(powers_mw, amps[:, j], 'o-', color=colors[j], label=f'Peak {j+1}')
    ref_lines(ax_log)

    # Formatting
    for ax in (ax_lin, ax_log):
        ax.set_ylabel('Peak amplitude (log a.u.)')
        ax.set_title(f"{cfg.DOPE_LABELS[i]} Peak Amplitude vs Power")
        ax.legend(ncol=3, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
    ax_log.set_xscale('log')
    ax_log.set_yscale('log')
    ax_log.set_xlabel('Power (log mW)')
    
    ax_lin.set_yscale('log')
    ax_lin.set_xlabel('Power (mW)')

    # Plot Saving (into the sample's own folder; _linear / _log keep the two file names apart)
    folder = cfg.OUTPUT_DIR / cfg.DOPE_LABELS[i].replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)
    for fig, tag in ((pow_lin, "linear"), (pow_log, "log")):
        fig.savefig(folder / f"peaks_{tag}.png", dpi=cfg.DPI, bbox_inches="tight")
        plt.close(fig)

    # ==========================================
    # Plot peak centre vs power
    # ==========================================
    
    cens = np.array(cens)   # rows: powers, columns: peaks
    for j in range(cfg.N_PEAKS):
        ax_pos.plot(powers_mw, cens[:, j], 'o-', color=colors[j], label=f'Peak {j+1}')

    ax_pos.set_xlabel('Power (mW)')
    ax_pos.set_ylabel('Peak centre (eV)')
    ax_pos.set_title(f"{cfg.DOPE_LABELS[i]} Peak Centre vs Power")
    ax_pos.legend(ncol=3, fontsize=10)
    ax_pos.grid(True, linestyle='--', alpha=0.4)

    pos.savefig(folder / "peaks_position.png", dpi=cfg.DPI, bbox_inches="tight")
    plt.close(pos)

# ==========================================
# Subplot like fig 7
# ==========================================
    
cols = plt.cm.tab10(np.arange(cfg.N_PEAKS) % 10)
fig, axs = plt.subplots(2, 2, figsize=(14, 9))

for r, (k, lab) in enumerate(((0, cfg.LI_LABELS[0]), (4, cfg.LI_LABELS[4]))):
    P, A = results[k]
    for c, ax in enumerate(axs[r]):
        for j in range(cfg.N_PEAKS):
            ax.plot(P, A[:, j], 'o-', ms=4, color=cols[j], label=f'Peak {j+1}')
        ax.set_yscale('log')
        if c: ax.set_xscale('log'); ref_lines(ax)
        ax.text(0.97, 0.05, f'({"abcd"[2*r+c]}) {lab} @ 55 K', transform=ax.transAxes, ha='right', va='bottom', fontsize=12)
        ax.grid(True, ls='--', alpha=0.4)

for ax in axs[1]: ax.set_xlabel('Power (mW)')
for ax in axs[:, 0]: ax.set_ylabel('Peak amplitude (a.u.)')
axs[0, 0].legend(ncol=3, fontsize=9, loc='center right')

(cfg.OUTPUT_DIR / 'Combined').mkdir(parents=True, exist_ok=True)
fig.savefig(cfg.OUTPUT_DIR / 'Combined' / 'undoped_40%_peaks.png', dpi=cfg.DPI, bbox_inches='tight')
plt.close(fig)

# ==========================================
# Subplot for 10, 20, 30
# ==========================================

fig, axs = plt.subplots(3, 2, figsize=(14, 13.5))

for r, (k, lab) in enumerate(((1, cfg.LI_LABELS[1]), (2, cfg.LI_LABELS[2]), (3, cfg.LI_LABELS[3]))):
    P, A = results[k]
    for c, ax in enumerate(axs[r]):
        for j in range(cfg.N_PEAKS):
            ax.plot(P, A[:, j], 'o-', ms=4, color=cols[j], label=f'Peak {j+1}')
        ax.set_yscale('log')
        if c: ax.set_xscale('log'); ref_lines(ax)
        ax.text(0.97, 0.05, f'({"abcdef"[2*r+c]}) {lab} @ 55 K', transform=ax.transAxes, ha='right', va='bottom', fontsize=12)
        ax.grid(True, ls='--', alpha=0.4)

for ax in axs[2]: ax.set_xlabel('Power (mW)')
for ax in axs[:, 0]: ax.set_ylabel('Peak amplitude (a.u.)')
axs[0, 0].legend(ncol=3, fontsize=9, loc='center right')

fig.savefig(cfg.OUTPUT_DIR / 'Combined' / '10_20_30%_peaks.png', dpi=cfg.DPI, bbox_inches='tight')
plt.close(fig)