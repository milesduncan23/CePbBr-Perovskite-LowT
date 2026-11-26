import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from numpy.polynomial import Chebyshev

import config as cfg

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

def pseudo_voigt(x, amp, center, sigma, eta):
    g = np.exp(-(x - center)**2 / (2 * sigma**2)) # Gaussian
    gamma = sigma * np.sqrt(2 * np.log(2)) # Gaussian HWHM
    l = gamma**2 / ((x - center)**2 + gamma**2) # Lorentzian with Gaussian FWHM
    
    # Scale output between Gaussian and Lorentzian using factor eta
    return amp * (eta * l + (1 - eta) * g) 

def auto_guesses(x, y, n, offsets, ref, ref_pos, prev):
    # Takes in initial guesses and finds exact placement for n peaks with raw data

    # ==========================================
    # Settings
    # ==========================================
    
    deg = 60 # Degree of Chebyshev polynomial fit
    
    min_snr = 3          # A feature must stand this many noise units above the baseline to count as a peak
    min_sep = 0.004      # Minimum peak separation (eV)
    ref_win = 0.010      # Reference position search window (eV)
    ref_snap = 0.003     # Maximum distance for reference position to be placed from guess (eV)
    faded = 0.02         # previous-fit peaks below this fraction of the tallest: reset
    template_tol = 0.006 # How far a detected center may stray from guess (eV)
    
    max_shift = 0.004 # (eV) how far a centre may move from its starting position
    min_gap = 0.004   # (eV) neighbouring centres never get closer than this
    
    match_tol    = 0.004   # (eV) start on a detected feature if it lies within this
    amp_start    = 0.7     # start amplitudes below the curve height: neighbours overlap
    sigma_start  = 0.005   # (eV) starting width when no detected feature to measure
    sigma_range  = (0.003, 0.008)   # (eV) clip for widths measured from the curvature
    sigma_bounds = (0.002, 0.009)   # (eV) fit limits on sigma
    eta_start    = 0.2              # start mostly Gaussian
    eta_bounds   = (0.0, 1.0)       # fit limits on eta (fully Gaussian to fully Lorentzian)
    
    offsets = np.asarray(offsets, dtype=float) / 1000.0
    if len(offsets) != n:
        raise ValueError(f"auto_guesses(): guess has {len(offsets)} peaks but n_peaks = {n}")
    
    # ==========================================
    # Polynomial Fit & Derivative feature finder
    # ==========================================

    # Local, second order polynomial smoothing about 5 point intervals
    y_s = savgol_filter(y, 5, 2)

    # Chebyshev polynomial fit
    P = Chebyshev.fit(x, y_s, deg)
    d1 = P.deriv()
    d2 = d1.deriv()
    d3 = d2.deriv()
    d4 = d3.deriv()
    
    def real_roots(poly):
         # Takes the real part of roots where the imaginary part is near zero
        r = poly.roots()
        r = r[np.abs(r.imag) < 1e-9].real
        
        # Return roots within data range
        return r[(r > x.min()) & (r < x.max())]

    # Noise level: robust (MAD) spread of point-to-point differences, which peaks barely affect;
    # the sqrt(2) is because a difference of two noisy points has twice the variance
    d = np.diff(y)
    noise = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)

    # Threshold to define a minimum peak height: min_snr noise units above the baseline
    baseline = np.median(y[:20])
    top = y_s.max() - baseline
    tall_enough = lambda c: P(c) - baseline > min_snr * noise
    
    # Maxima which are above the min height
    maxima = [c for c in real_roots(d1) if d2(c) < 0 and tall_enough(c)]
    
    # Maxima in shoulders of found maxima, filtering existing maxima
    shoulders = [c for c in real_roots(d3)
                if d4(c) > 0 and d2(c) < 0 and tall_enough(c)
                and all(abs(c - m) > min_sep for m in maxima)]
    
    features = np.array(sorted(maxima + shoulders))
    
    # ==========================================
    # Matching Found Features & Guesses
    # ==========================================
    
    # At low power, take the largest feature of aggressively smoothed data (It will be our reference for low powers)
    x_peak = x[np.argmax(savgol_filter(y, 40, 3))]
    
    # Define template centers wrt. reference peak
    if abs(x_peak - ref_pos) <= ref_win:                                       # Peak within window
        close = [c for c in maxima if abs(c - x_peak) <= ref_snap]             # Compile candidates
        x_ref = min(close, key=lambda c: abs(c - x_peak)) if close else x_peak # Returns the closest candidate
        template = x_ref + offsets
        centres = template.copy()
    elif prev is not None:                                                           # This will be the case after the reference peak is no longer the largest
        prev_c, prev_a = (np.asarray(v, dtype=float) for v in prev)                  # Return template with previous reference
        template = prev_c[ref] + offsets
        centres = np.where(prev_a >= faded * prev_a.max(), prev_c, template)         # Return to template values for peaks that faded below %tol of the tallest previous peak
        centres = np.clip(centres, template - template_tol, template + template_tol) # Clip values which are outside allowed tol from guess
        centres = np.maximum.accumulate(centres + np.arange(n) * 1e-6)               # Keep order
    else:                                                          # Fallback condition
        pool = [c for c in maxima if abs(c - ref_pos) <= ref_win]  # Compile candidates
        x_ref = max(pool, key=lambda c: P(c)) if pool else ref_pos # Pick out max, or reference
        template = x_ref + offsets
        centres = template.copy()

    # Define window positions, 
    half_gaps = np.maximum(np.diff(centres) / 2 - min_gap / 2, 0.0005)                # Half gaps between each center, taking into account the min peak separation
    win_lo = np.maximum(centres - np.minimum(max_shift, np.r_[max_shift, half_gaps]), # Lowest allowed energy for each peak
                        template - template_tol)
    win_hi = np.minimum(centres + np.minimum(max_shift, np.r_[half_gaps, max_shift]),
                        template + template_tol)
    win_hi = np.maximum(win_hi, win_lo + 0.0005)                                      # Highest allowed energy for each peak

    # --- Starting values and bounds -------------------------------------------------
    p0, lo, hi = np.zeros((n, 4)), np.zeros((n, 4)), np.zeros((n, 4))
    for j in range(n):
        # start on a detected feature inside the window, if one is close enough
        inside = [c for c in features
                  if win_lo[j] < c < win_hi[j] and abs(c - centres[j]) <= match_tol]
        cen = min(inside, key=lambda c: abs(c - centres[j])) if inside else centres[j]

        height = max(P(cen) - baseline, 0.01 * top)
        if inside and d2(cen) < 0:
            # for a Gaussian peak, y'' at the centre = -height / sigma^2
            sigma = float(np.clip(np.sqrt(height / -d2(cen)), *sigma_range))
        else:
            sigma = sigma_start

        p0[j] = [amp_start * height, cen, sigma, eta_start]
        # amplitude may go to 0: a peak can fade out at low power and keep its index
        lo[j] = [0.0,           win_lo[j], sigma_bounds[0], eta_bounds[0]]
        hi[j] = [1.5 * y.max(), win_hi[j], sigma_bounds[1], eta_bounds[1]]
    off_lo, off_hi = -0.05 * y.max(), 0.10 * y.max()

    # keep every guess strictly inside its bounds (curve_fit rejects it otherwise)
    p0 = np.clip(p0, lo + 1e-12, hi - 1e-12)
    off = (float(np.clip(baseline, off_lo + 1e-12, off_hi - 1e-12)), off_lo, off_hi)
    return p0, lo, hi, off

def fit_peaks(x, y, n, offsets, ref, ref_pos):
    # Fits four free parameters for each distribution (amp, center, sigma, eta)
    
    tail_ratio = 0.5                            # tail peak at most this fraction of the anchor peak

    # p0, lo, hi: n x 4 (amp, centre, sigma, eta per peak); off: (start, lower, upper)
    p0, lo, hi, (off0, off_lo, off_hi) = auto_guesses(x, y, n, offsets, ref, ref_pos,
                                                      prev=getattr(fit_peaks, 'previous', None))

    # The tail peak (last row) is fitted as a ratio r = tail amplitude / ref amplitude,
    # with 0 <= r <= tail_ratio, so it can never outgrow the reference line
    p0[-1, 0] = min(p0[-1, 0] / max(p0[ref, 0], 1e-12), 0.5 * tail_ratio)
    lo[-1, 0], hi[-1, 0] = 0.0, tail_ratio

    def model(x, *q):
        # curve_fit works on a flat list: q = (4 values per peak, row by row, then the offset)
        pk = np.array(q[:-1]).reshape(n, 4)
        pk[-1, 0] *= pk[ref, 0]                  # tail: ratio -> amplitude
        y_fit = np.full_like(x, q[-1], dtype=float)          # offset
        for row in pk:
            y_fit += pseudo_voigt(x, *row)                   # amp, centre, sigma, eta
        return y_fit

    q, _ = curve_fit(model, x, y,
                     p0=np.r_[p0.ravel(), off0],
                     bounds=(np.r_[lo.ravel(), off_lo], np.r_[hi.ravel(), off_hi]),
                     maxfev=50000)

    peaks = np.array(q[:-1], dtype=float).reshape(n, 4)
    peaks[-1, 0] *= peaks[ref, 0]                # tail: ratio -> ordinary amplitude
    offset = float(q[-1])
    
    # Saves the fitted centers and amplitudes as a function attribute for the next call
    fit_peaks.previous = (peaks[:, 1], peaks[:, 0])
    return peaks, offset

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

    file_path = cfg.DATA_DIR / (cfg.FILES[i] + ".xlsx")
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.sort_values(by='Energy').reset_index(drop=True)

    power_cols = [c for c in df.columns if c != 'Energy']
    
    # Clear prev. fit objects
    fit_peaks.previous = None
    
    # Create per-sample peak vs power plots
    pow_lin = plt.figure(figsize=cfg.FIG_SIZE)
    pow_log = plt.figure(figsize=cfg.FIG_SIZE)
    ax_lin = pow_lin.add_subplot()
    ax_log = pow_log.add_subplot()

    pos = plt.figure(figsize=cfg.FIG_SIZE)
    ax_pos = pos.add_subplot()
    
    powers_mw, amps, cens = [], [], []

    for power in power_cols:
        
        x_data = df['Energy'].values
        y_data = df[power].values
        
        # ==========================================
        # Fit Creation
        # ==========================================
        
        # Fit peak parameters using initial guesses
        peaks, offset = fit_peaks(x_data, y_data, cfg.N_PEAKS, cfg.PEAK_OFFSETS, cfg.REF, cfg.REF_POS)
        
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