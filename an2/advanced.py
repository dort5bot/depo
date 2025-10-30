# metrics/advanced.py
"""
Advanced quantitative metrics for MAPS framework.
Implements enhanced classical TA metrics, Kalman filter, wavelet transform,
Hilbert transform, fractal dimension, and entropy-based signal complexity measures.

Author: ysf-bot-framework
Version: 2025.2
Updated: 2025-10-28

advanced	Numpy	Matematiksel hesaplamalar için hızlı	Async + Process
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import pywt

try:
    import polars as pl
except ImportError:
    pl = None

# ================= Utility Functions =================

def _to_numpy(series):
    if isinstance(series, (pd.Series, pd.DataFrame)):
        return series.values.flatten()
    elif pl and isinstance(series, pl.Series):
        return series.to_numpy()
    return np.array(series)

def _to_output(series, arr):
    if isinstance(series, pd.Series):
        return pd.Series(arr, index=series.index)
    elif pl and isinstance(series, pl.Series):
        return pl.Series(series.name or "metric", arr)
    return arr

# ================= Kalman Filter =================

def Kalman_Filter_Trend(series, process_variance=1e-5, measurement_variance=0.01):
    """1D Kalman filter to smooth noisy signal."""
    data = _to_numpy(series)
    n_iter = len(data)
    xhat = np.zeros(n_iter)
    P = np.zeros(n_iter)
    K = np.zeros(n_iter)
    
    xhat[0] = data[0]
    P[0] = 1.0
    
    for k in range(1, n_iter):
        Pminus = P[k-1] + process_variance
        K[k] = Pminus / (Pminus + measurement_variance)
        xhat[k] = xhat[k-1] + K[k] * (data[k] - xhat[k-1])
        P[k] = (1 - K[k]) * Pminus
    
    return _to_output(series, xhat)

# ================= Wavelet Transform =================

def Wavelet_Transform(series, wavelet='db4', level=2):
    """DWT for denoising or feature extraction; returns reconstructed signal."""
    data = _to_numpy(series)
    coeffs = pywt.wavedec(data, wavelet, level=level)
    reconstructed = pywt.waverec(coeffs, wavelet)
    return _to_output(series, reconstructed[:len(data)])

# ================= Hilbert Transform =================

def Hilbert_Transform_Slope(series):
    """Computes instantaneous phase slope of a signal."""
    data = _to_numpy(series)
    analytic_signal = signal.hilbert(data)
    phase = np.unwrap(np.angle(analytic_signal))
    slope = np.gradient(phase)
    return _to_output(series, slope)

def Hilbert_Transform_Amplitude(series):
    """Computes instantaneous amplitude of a signal."""
    data = _to_numpy(series)
    analytic_signal = signal.hilbert(data)
    amplitude = np.abs(analytic_signal)
    return _to_output(series, amplitude)

# ================= Fractal Dimension Index (FDI) =================

def Fractal_Dimension_Index_FDI(series, k_max=100):
    """Higuchi method approximation for Fractal Dimension Index."""
    data = _to_numpy(series)
    N = len(data)
    L = np.zeros(k_max)
    
    for k in range(1, k_max):
        Lk = []
        for m in range(k):
            idxs = np.arange(m, N, k)
            if len(idxs) < 2:
                continue
            diffs = np.abs(np.diff(data[idxs]))
            Lm = (np.sum(diffs) * (N - 1) / (len(idxs) * k)) / k
            Lk.append(Lm)
        if Lk:
            L[k] = np.mean(Lk)
    
    k_vals = np.arange(1, k_max)
    mask = L[1:] > 0
    if np.sum(mask) < 2:
        return np.nan
    coeffs = np.polyfit(np.log(k_vals[mask]), np.log(L[1:][mask]), 1)
    return -coeffs[0]

# ================= Entropy Measures =================

def Shannon_Entropy(series, bins=30):
    """Shannon entropy of normalized histogram."""
    data = _to_numpy(series)
    hist, _ = np.histogram(data, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def Permutation_Entropy(series, order=3, delay=1):
    """Permutation entropy (Bandt & Pompe)."""
    x = _to_numpy(series)
    n = len(x)
    if n < order * delay:
        return np.nan
    perm_list = [tuple(np.argsort(x[i:i + order * delay:delay])) for i in range(n - order * delay)]
    counts = {}
    for p in perm_list:
        counts[p] = counts.get(p, 0) + 1
    probs = np.array(list(counts.values())) / len(perm_list)
    return -np.sum(probs * np.log2(probs + 1e-12))

def Sample_Entropy(series, m=2, r=None):
    """Sample entropy estimation."""
    x = _to_numpy(series)
    N = len(x)
    if r is None:
        r = 0.2 * np.std(x)
    def _phi(m):
        x_m = np.array([x[i:i+m] for i in range(N-m+1)])
        C = np.sum([np.sum(np.max(np.abs(x_m - xm), axis=1) <= r) - 1 for xm in x_m])
        return C / (N - m + 1)
    try:
        return -np.log(_phi(m+1) / _phi(m))
    except:
        return np.nan




# Tüm metrik fonksiyonları için template
def metric_template(series, *args, **kwargs):
    # 1. Input validation
    if series is None or len(series) == 0:
        return np.nan
    
    # 2. NaN check
    if isinstance(series, (pd.Series, pd.DataFrame)):
        if series.isna().all():
            return np.nan
        # Forward fill then backward fill
        series = series.ffill().bfill()
    
    # 3. Length check
    if len(series) < kwargs.get('min_periods', 2):
        return np.nan
    
    # 4. Try-except wrapper
    try:
        # Actual calculation
        result = calculate_metric(series, *args, **kwargs)
        return result
    except Exception as e:
        logger.warning(f"Metric calculation failed: {e}")
        return np.nan
        
        
# ================= Exported Functions =================

__all__ = [
    "Kalman_Filter_Trend",
    "Wavelet_Transform",
    "Hilbert_Transform_Slope",
    "Hilbert_Transform_Amplitude",
    "Fractal_Dimension_Index_FDI",
    "Shannon_Entropy",
    "Permutation_Entropy",
    "Sample_Entropy"
]
