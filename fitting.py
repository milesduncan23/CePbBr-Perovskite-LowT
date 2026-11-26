import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from numpy.polynomial import Chebyshev

# ==========================================
# Data Loading
# ==========================================

def load_sample(file_path):
    # Reads one sample's spreadsheet, returns the energy axis and one spectrum per power column

    df = pd.read_excel(file_path, sheet_name='Sheet1')

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.sort_values(by='Energy').reset_index(drop=True)

    power_cols = [c for c in df.columns if c != 'Energy']

    x_data = df['Energy'].values
    spectra = {}
    for power in power_cols:
        spectra[power] = df[power].values # Keeps the spreadsheet's column order (low -> high power)

    return x_data, spectra

# ==========================================
# Peak Fitting
# ==========================================

def pseudo_voigt(x, amp, center, sigma, eta):
    g = np.exp(-(x - center)**2 / (2 * sigma**2)) # Gaussian
    gamma = sigma * np.sqrt(2 * np.log(2)) # Gaussian HWHM
    l = gamma**2 / ((x - center)**2 + gamma**2) # Lorentzian with Gaussian FWHM

    # Scale output between Gaussian and Lorentzian using factor eta
    return amp * (eta * l + (1 - eta) * g)

def auto_guesses(x, y, n, offsets, ref, ref_pos, prev):
    # Takes in initial guesses and finds exact placement for n peaks with raw data
    # prev: peaks from the previous power's fit (n x 4), or None at a sample's first power

    # ==========================================
    # Settings
    # ==========================================

    deg = 60 # Degree of Chebyshev polynomial fit

    min_snr = 3          # A feature must stand this many noise units above the baseline to count as a peak
    min_sep = 0.004      # Minimum peak separation (eV)
    ref_win = 0.010      # Reference position search window (eV)
    ref_snap = 0.003     # Maximum distance for reference position to be placed from guess (eV)
    faded = 0.02         # Previous-fit peaks below this fraction of the tallest go back to the template
    template_tol = 0.006 # How far a detected center may stray from guess (eV)

    max_shift = 0.004 # How far a center may move from its starting position (eV)
    min_gap = 0.004   # Neighbouring centers never get closer than this (eV)

    match_tol    = 0.004            # Start on a detected feature if it lies within this (eV)
    amp_start    = 0.7              # Start amplitudes below the curve height, since neighbours overlap
    sigma_start  = 0.005            # Starting width when there is no detected feature to measure (eV)
    sigma_range  = (0.003, 0.008)   # Clip for widths measured from the curvature (eV)
    sigma_bounds = (0.002, 0.009)   # Fit limits on sigma (eV)
    eta_start    = 0.2              # Start mostly Gaussian
    eta_bounds   = (0.0, 1.0)       # Fit limits on eta (fully Gaussian to fully Lorentzian)

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
        prev_c = prev[:, 1]                                                          # Previous fit's centers
        prev_a = prev[:, 0]                                                          # Previous fit's amplitudes
        template = prev_c[ref] + offsets                                             # Return template with previous reference
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

    # ==========================================
    # Starting Values & Bounds
    # ==========================================

    p0, lo, hi = np.zeros((n, 4)), np.zeros((n, 4)), np.zeros((n, 4))
    for j in range(n):
        # Start on a detected feature inside the window, if one is close enough
        inside = [c for c in features
                  if win_lo[j] < c < win_hi[j] and abs(c - centres[j]) <= match_tol]
        cen = min(inside, key=lambda c: abs(c - centres[j])) if inside else centres[j]

        height = max(P(cen) - baseline, 0.01 * top)
        if inside and d2(cen) < 0:
            # For a Gaussian peak, y'' at the center = -height / sigma^2
            sigma = float(np.clip(np.sqrt(height / -d2(cen)), *sigma_range))
        else:
            sigma = sigma_start

        p0[j] = [amp_start * height, cen, sigma, eta_start]
        # Amplitude may go to 0: a peak can fade out at low power and keep its index
        lo[j] = [0.0,           win_lo[j], sigma_bounds[0], eta_bounds[0]]
        hi[j] = [1.5 * y.max(), win_hi[j], sigma_bounds[1], eta_bounds[1]]
    off_lo, off_hi = -0.05 * y.max(), 0.10 * y.max()

    # Keep every guess strictly inside its bounds (curve_fit rejects it otherwise)
    p0 = np.clip(p0, lo + 1e-12, hi - 1e-12)
    off = (float(np.clip(baseline, off_lo + 1e-12, off_hi - 1e-12)), off_lo, off_hi)
    return p0, lo, hi, off

def fit_peaks(x, y, n, offsets, ref, ref_pos, prev):
    # Fits four free parameters for each distribution (amp, center, sigma, eta)
    # prev: peaks from the previous power's fit (n x 4), or None at a sample's first power

    tail_ratio = 0.5 # Tail peak at most this fraction of the reference peak

    # p0, lo, hi: n x 4 (amp, center, sigma, eta per peak); off: (start, lower, upper)
    p0, lo, hi, (off0, off_lo, off_hi) = auto_guesses(x, y, n, offsets, ref, ref_pos, prev)

    # The tail peak (last row) is fitted as a ratio r = tail amplitude / ref amplitude,
    # with 0 <= r <= tail_ratio, so it can never outgrow the reference line
    p0[-1, 0] = min(p0[-1, 0] / max(p0[ref, 0], 1e-12), 0.5 * tail_ratio)
    lo[-1, 0], hi[-1, 0] = 0.0, tail_ratio

    def model(x, *q):
        # curve_fit works on a flat list: q = (4 values per peak, row by row, then the offset)
        pk = np.array(q[:-1]).reshape(n, 4)
        pk[-1, 0] *= pk[ref, 0]                     # Tail: ratio -> amplitude
        y_fit = np.full_like(x, q[-1], dtype=float) # Offset
        for row in pk:
            y_fit += pseudo_voigt(x, *row)          # amp, center, sigma, eta
        return y_fit

    q, _ = curve_fit(model, x, y,
                     p0=np.r_[p0.ravel(), off0],
                     bounds=(np.r_[lo.ravel(), off_lo], np.r_[hi.ravel(), off_hi]),
                     maxfev=50000)

    peaks = np.array(q[:-1], dtype=float).reshape(n, 4)
    peaks[-1, 0] *= peaks[ref, 0] # Tail: ratio -> ordinary amplitude
    offset = float(q[-1])

    return peaks, offset