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

def noise_level(y):
    # Noise level: robust (MAD) spread of point-to-point differences, which peaks barely affect;
    # the sqrt(2) is because a difference of two noisy points has twice the variance
    d = np.diff(y)
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)

# ==========================================
# Peak Fitting
# ==========================================

def pseudo_voigt(x, amp, center, sigma, eta):
    g = np.exp(-(x - center)**2 / (2 * sigma**2)) # Gaussian
    gamma = sigma * np.sqrt(2 * np.log(2)) # Gaussian HWHM
    l = gamma**2 / ((x - center)**2 + gamma**2) # Lorentzian with Gaussian FWHM

    # Scale output between Gaussian and Lorentzian using factor eta
    return amp * (eta * l + (1 - eta) * g)

def auto_guesses(x, y, n, offsets, ref_pos):
    # Places the peak template on the raw data and returns starting values and bounds for n free peaks
    # Only used for a sample's first (highest power) fit, every lower power keeps these positions (see fit_locked)

    # ==========================================
    # Settings
    # ==========================================

    deg = 60 # Degree of Chebyshev polynomial fit

    min_snr = 3          # A feature must stand this many noise units above the baseline to count as a peak
    min_sep = 0.004      # Minimum peak separation (eV)
    search = 0.006       # How far the whole template may slide to line up with the data (eV), under half the ~10 meV A-peak spacing
    template_tol = 0.006 # How far a peak may stray from its template position (eV)

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

    noise = noise_level(y)

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
    # Placing the Template
    # ==========================================

    # Slide the whole template in 0.2 meV steps and keep the shift where its peaks sit closest to the detected
    # features; each peak's distance is capped at match_tol so one missing feature cannot dominate.
    # Shifts are tried smallest first, so a tie keeps the template where it is
    shifts = np.arange(-search, search + 1e-9, 0.0002)
    shifts = shifts[np.argsort(np.abs(shifts))]

    best_shift = 0.0
    best_cost = np.inf
    for shift in shifts:
        cost = 0.0
        for t in ref_pos + offsets + shift:
            if len(features) > 0:
                cost += min(np.min(np.abs(features - t)), match_tol)
            else:
                cost += match_tol
        if cost < best_cost:
            best_cost = cost
            best_shift = shift

    template = ref_pos + offsets + best_shift
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

def fit_free(x, y, n, offsets, ref_pos):
    # First fit of a sample (highest power): four free parameters for each peak (amp, center, sigma, eta)

    # p0, lo, hi: n x 4 (amp, center, sigma, eta per peak); off: (start, lower, upper)
    p0, lo, hi, (off0, off_lo, off_hi) = auto_guesses(x, y, n, offsets, ref_pos)

    def model(x, *q):
        # curve_fit works on a flat list: q = (4 values per peak, row by row, then the offset)
        pk = np.array(q[:-1]).reshape(n, 4)
        y_fit = np.full_like(x, q[-1], dtype=float) # Offset
        for row in pk:
            y_fit += pseudo_voigt(x, *row)          # amp, center, sigma, eta
        return y_fit

    q, _ = curve_fit(model, x, y,
                     p0=np.r_[p0.ravel(), off0],
                     bounds=(np.r_[lo.ravel(), off_lo], np.r_[hi.ravel(), off_hi]),
                     maxfev=50000)

    peaks = np.array(q[:-1], dtype=float).reshape(n, 4)
    offset = float(q[-1])

    return peaks, offset

def fit_locked(x, y, pattern, prev):
    # Fit at a lower power: every peak keeps its position from the sample's first fit (pattern) and the whole set
    # may only slide together by one common shift. Amplitudes, widths and eta stay free.
    # pattern: the first (highest power) fit, n x 4; prev: the previous (next higher) power's fit, n x 4

    # ==========================================
    # Settings
    # ==========================================

    max_shift    = 0.005            # Largest common shift away from the first fit's positions (eV). The data never needed
                                    # more than 4.8 meV, and 5 meV is half the ~10 meV peak spacing, so no peak can take its neighbour's place
    amp_start    = 0.7              # Start amplitudes below the curve height, since neighbours overlap
    sigma_bounds = (0.002, 0.009)   # Fit limits on sigma (eV), same as the first fit
    eta_bounds   = (0.0, 1.0)       # Fit limits on eta, same as the first fit

    n = pattern.shape[0]
    centres = pattern[:, 1]

    # ==========================================
    # Starting Values & Bounds
    # ==========================================

    # Start from the previous power's common shift
    shift0 = float(np.clip(prev[0, 1] - centres[0], -max_shift + 1e-12, max_shift - 1e-12))

    # Smoothed curve, baseline and height, as in auto_guesses
    y_s = savgol_filter(y, 5, 2)
    baseline = np.median(y[:20])
    top = y_s.max() - baseline

    p0, lo, hi = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    for j in range(n):
        # Amplitude from the curve height at the shifted center, width and eta from the previous power
        height = max(np.interp(centres[j] + shift0, x, y_s) - baseline, 0.01 * top)

        p0[j] = [amp_start * height, prev[j, 2], prev[j, 3]]
        lo[j] = [0.0,           sigma_bounds[0], eta_bounds[0]]
        hi[j] = [1.5 * y.max(), sigma_bounds[1], eta_bounds[1]]
    off_lo, off_hi = -0.05 * y.max(), 0.10 * y.max()

    # Keep every guess strictly inside its bounds (curve_fit rejects it otherwise)
    p0 = np.clip(p0, lo + 1e-12, hi - 1e-12)
    off0 = float(np.clip(baseline, off_lo + 1e-12, off_hi - 1e-12))

    # ==========================================
    # Fit
    # ==========================================

    def model(x, *q):
        # curve_fit works on a flat list: q = (amp, sigma, eta per peak, row by row, then the shift, then the offset)
        pk = np.array(q[:-2]).reshape(n, 3)
        shift = q[-2]
        y_fit = np.full_like(x, q[-1], dtype=float) # Offset
        for j in range(n):
            y_fit += pseudo_voigt(x, pk[j, 0], centres[j] + shift, pk[j, 1], pk[j, 2])
        return y_fit

    # x_scale='jac' lets the optimizer handle amplitudes (~1e5) and widths/shift (~1e-3) on equal footing;
    # without it about 1 fit in 10 stopped early in a bad spot
    q, _ = curve_fit(model, x, y,
                     p0=np.r_[p0.ravel(), shift0, off0],
                     bounds=(np.r_[lo.ravel(), -max_shift, off_lo], np.r_[hi.ravel(), max_shift, off_hi]),
                     maxfev=50000, x_scale='jac')

    pk = np.array(q[:-2], dtype=float).reshape(n, 3)
    shift = float(q[-2])
    offset = float(q[-1])

    # Back to the usual n x 4 layout (amp, center, sigma, eta)
    peaks = np.zeros((n, 4))
    for j in range(n):
        peaks[j] = [pk[j, 0], centres[j] + shift, pk[j, 1], pk[j, 2]]

    return peaks, offset
