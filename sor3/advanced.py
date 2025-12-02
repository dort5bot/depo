"""
analysis/metrics/advanced.py
date: 30.11.2025 19:25
Enhanced standard template
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional, Sequence, Tuple

# ==================== COLUMN GROUPS (module local opt.) ====================
COLUMN_GROUPS = {
    "ohlc": ["open", "high", "low", "close"],
    "close_only": ["close"],
    "ohlcv": ["open", "high", "low", "close", "volume"],
    "single_series": ["close"],  # For single series operations
    "dual_series": ["close", "close_secondary"]  # For operations requiring two series
}

# ==================== MODULE CONFIG ====================
_MODULE_CONFIG = {
    "data_model": "pandas",
    "execution_type": "sync",
    "category": "technical",

    # hangi kolon setine ihtiyaç var?
    "required_groups": {
        "kalman_filter_trend": "close_only",
        "wavelet_transform": "close_only",
        "hilbert_transform_amplitude": "close_only",
        "hilbert_transform_slope": "close_only",
        "fractal_dimension_index_fdi": "close_only",
        "shannon_entropy": "close_only",
        "permutation_entropy": "close_only",
        "sample_entropy": "close_only",
        "granger_causality": "dual_series",
        "phase_shift_index": "dual_series"
    },

    # opsiyonel
    "score_profile": {
        "kalman_filter_trend": {
            "method": "minmax",
            "range": [0, 1],
            "direction": "positive"
        },
        "wavelet_transform": {
            "method": "zscore",
            "direction": "positive"
        },
        "hilbert_transform_amplitude": {
            "method": "minmax",
            "range": [0, 1],
            "direction": "positive"
        },
        "hilbert_transform_slope": {
            "method": "zscore", 
            "direction": "both"
        },
        "fractal_dimension_index_fdi": {
            "method": "minmax",
            "range": [1, 2],
            "direction": "positive"
        },
        "shannon_entropy": {
            "method": "minmax",
            "range": [0, 10],
            "direction": "positive"
        },
        "permutation_entropy": {
            "method": "minmax",
            "range": [0, 2],
            "direction": "positive"
        },
        "sample_entropy": {
            "method": "minmax",
            "range": [0, 2],
            "direction": "positive"
        },
        "granger_causality": {
            "method": "binary",
            "direction": "positive"
        },
        "phase_shift_index": {
            "method": "minmax", 
            "range": [-np.pi, np.pi],
            "direction": "both"
        }
    }
}

# ==================== UTILITY FUNCTIONS ====================
_eps = 1e-12

def _to_numpy(x) -> np.ndarray:
    """Convert input to numpy array, handling various data types."""
    if isinstance(x, pd.Series):
        arr = x.values
    elif isinstance(x, pd.DataFrame):
        arr = x.iloc[:, 0].values if len(x.columns) > 0 else np.array([])
    else:
        arr = np.asarray(x, dtype=float)
    
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.ravel()
    return arr

def _mask_valid(arr: np.ndarray) -> np.ndarray:
    """Return boolean mask of finite values."""
    return np.isfinite(arr)

def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise safe division."""
    return a / (b + _eps)

def _create_lag_matrix(series: np.ndarray, lag: int) -> np.ndarray:
    """Create lag matrix for time series."""
    n = series.size
    if n <= lag:
        return np.array([]).reshape(0, lag)
    
    matrix = np.full((n - lag, lag), np.nan)
    for i in range(lag):
        matrix[:, i] = series[i:n - lag + i]
    
    return matrix

def _analytic_signal_via_fft(x: np.ndarray) -> np.ndarray:
    """Compute analytic signal using FFT (no scipy)."""
    x = _to_numpy(x)
    n = x.size
    if n == 0:
        return np.array([], dtype=complex)
    X = np.fft.fft(x)
    h = np.zeros(n, dtype=float)
    if n % 2 == 0:
        # even
        h[0] = 1.0
        h[n//2] = 1.0
        h[1:n//2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n+1)//2] = 2.0
    analytic = np.fft.ifft(X * h)
    return analytic

def _dwt_haar(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Single-level Haar DWT: returns (approx, detail). If odd length, last sample dropped."""
    n = signal.size
    if n < 2:
        return signal.copy(), np.array([], dtype=float)
    even = signal[0:(n // 2) * 2:2]
    odd = signal[1:(n // 2) * 2:2]
    approx = (even + odd) / np.sqrt(2.0)
    detail = (even - odd) / np.sqrt(2.0)
    return approx, detail

def _idwt_haar(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
    """Inverse single-level Haar DWT."""
    n = approx.size + detail.size
    if detail.size == 0:
        return approx.copy()
    out = np.empty(approx.size * 2, dtype=float)
    out[0::2] = (approx + detail) / np.sqrt(2.0)
    out[1::2] = (approx - detail) / np.sqrt(2.0)
    return out

def _extract_series(data: Union[pd.DataFrame, pd.Series, np.ndarray], column: str = None) -> np.ndarray:
    """Extract series from data based on column name or position."""
    if isinstance(data, pd.DataFrame):
        if column and column in data.columns:
            return data[column].values
        elif len(data.columns) > 0:
            return data.iloc[:, 0].values
        else:
            return np.array([])
    elif isinstance(data, pd.Series):
        return data.values
    else:
        return _to_numpy(data)

# ==================== PURE FUNCTIONS ====================

def kalman_filter_trend(data: Union[pd.DataFrame, pd.Series, np.ndarray], 
                       process_variance: float = 1e-5,
                       measurement_variance: float = 1e-2,
                       **kwargs) -> pd.Series:
    """
    1D Kalman filter (simple, robust).
    Returns smoothed series same length as input.
    """
    series = _extract_series(data, kwargs.get('column', 'close'))
    x = _to_numpy(series)
    n = x.size
    if n == 0:
        return pd.Series([], dtype=float)

    # initialize
    xhat = np.full(n, np.nan, dtype=float)
    P = np.zeros(n, dtype=float)

    # handle initial valid value
    mask = _mask_valid(x)
    if not np.any(mask):
        return pd.Series(np.full(n, np.nan))
    first_idx = np.argmax(mask)
    x0 = x[first_idx]

    xhat[first_idx] = x0
    P[first_idx] = 1.0

    # forward pass
    for t in range(first_idx + 1, n):
        if not mask[t]:
            xhat[t] = xhat[t - 1]
            P[t] = P[t - 1] + process_variance
            continue

        # predict
        Pminus = P[t - 1] + process_variance
        # update
        K = Pminus / (Pminus + measurement_variance + _eps)
        xhat[t] = xhat[t - 1] + K * (x[t] - xhat[t - 1])
        P[t] = (1.0 - K) * Pminus

    # backward fill for leading NaNs (if any before first_idx)
    if first_idx > 0:
        xhat[:first_idx] = xhat[first_idx]

    return pd.Series(xhat, index=getattr(data, 'index', None))

def wavelet_transform(data: Union[pd.DataFrame, pd.Series, np.ndarray], 
                     level: int = 1, **kwargs) -> pd.Series:
    """
    Simple Haar wavelet denoise-like transform using multilevel DWT + full reconstruction.
    Returns reconstructed signal same length as input (padding/truncation as needed).
    """
    series = _extract_series(data, kwargs.get('column', 'close'))
    x = _to_numpy(series)
    n0 = x.size
    if n0 == 0:
        return pd.Series([], dtype=float)

    # copy and pad to allow dyadic decomposition
    data_arr = x.copy()
    # if odd, drop last sample for stable haar decomposition; we'll pad back later
    drop_last = False
    if data_arr.size % 2 == 1:
        drop_last = True
        data_arr = data_arr[:-1]

    approx = data_arr
    details = []
    lvl = max(1, int(level))
    for _ in range(lvl):
        a, d = _dwt_haar(approx)
        details.append(d)
        approx = a
        if approx.size < 2:
            break

    # naive thresholding: zero out smallest-detail coefficients (light denoising)
    # compute global median absolute deviation across detail coeffs for threshold
    if details:
        all_details = np.concatenate([d for d in details if d.size > 0]) if any(d.size>0 for d in details) else np.array([])
        if all_details.size > 0:
            mad = np.median(np.abs(all_details - np.median(all_details)))
            thr = 3.0 * (mad + _eps)
            details = [np.where(np.abs(d) < thr, 0.0, d) for d in details]

    # reconstruct backward
    recon = approx
    for d in reversed(details):
        recon = _idwt_haar(recon, d)

    # if we dropped last sample, append it back (simple copy)
    if drop_last:
        recon = np.concatenate([recon, x[-1:]])

    # ensure same length
    if recon.size > n0:
        recon = recon[:n0]
    elif recon.size < n0:
        recon = np.concatenate([recon, np.full(n0 - recon.size, recon[-1] if recon.size>0 else 0.0)])

    return pd.Series(recon, index=getattr(data, 'index', None))

def hilbert_transform_amplitude(data: Union[pd.DataFrame, pd.Series, np.ndarray], **kwargs) -> pd.Series:
    """Instantaneous amplitude (envelope) from analytic signal."""
    series = _extract_series(data, kwargs.get('column', 'close'))
    x = _to_numpy(series)
    if x.size == 0:
        return pd.Series([], dtype=float)
    analytic = _analytic_signal_via_fft(x)
    return pd.Series(np.abs(analytic), index=getattr(data, 'index', None))

def hilbert_transform_slope(data: Union[pd.DataFrame, pd.Series, np.ndarray], **kwargs) -> pd.Series:
    """Instantaneous phase slope (derivative of unwrapped angle)."""
    series = _extract_series(data, kwargs.get('column', 'close'))
    x = _to_numpy(series)
    if x.size == 0:
        return pd.Series([], dtype=float)
    analytic = _analytic_signal_via_fft(x)
    phase = np.unwrap(np.angle(analytic))
    # slope: gradient of phase; preserve length by using np.gradient
    slope = np.gradient(phase)
    return pd.Series(slope, index=getattr(data, 'index', None))

def fractal_dimension_index_fdi(data: Union[pd.DataFrame, pd.Series, np.ndarray], 
                               k_max: int = 100, **kwargs) -> float:
    """
    Higuchi-like fractal dimension estimate.
    Returns scalar fractal dimension (float) or np.nan if insufficient data.
    """
    series = _extract_series(data, kwargs.get('column', 'close'))
    x = _to_numpy(series)
    N = x.size
    if N < 10:
        return float(np.nan)

    k_max = min(int(k_max), N // 2)
    Lk = np.zeros(k_max, dtype=float)

    for k in range(1, k_max + 1):
        Lm_sum = 0.0
        m_count = 0
        for m in range(k):
            idx = np.arange(m, N, k)
            if idx.size < 2:
                continue
            diffs = np.abs(np.diff(x[idx]))
            if diffs.size == 0:
                continue
            norm = (N - 1) / ( (idx.size) * k )
            Lm = (np.sum(diffs) * norm) / k
            Lm_sum += Lm
            m_count += 1
        if m_count > 0:
            Lk[k - 1] = Lm_sum / m_count
        else:
            Lk[k - 1] = np.nan

    valid = np.isfinite(Lk) & (Lk > 0)
    ks = np.arange(1, k_max + 1)[valid]
    Lk_valid = Lk[valid]
    if ks.size < 2:
        return float(np.nan)

    coeffs = np.polyfit(np.log(ks), np.log(Lk_valid), 1)
    # fractal dimension ~ -slope
    return float(-coeffs[0])

def shannon_entropy(data: Union[pd.DataFrame, pd.Series, np.ndarray], 
                   bins: int = 30, **kwargs) -> float:
    """Shannon entropy of distribution (bits)."""
    series = _extract_series(data, kwargs.get('column', 'close'))
    x = _to_numpy(series)
    if x.size == 0:
        return float(np.nan)
    hist, _ = np.histogram(x[~np.isnan(x)], bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return float(0.0)
    probs = hist / (np.sum(hist) + _eps)
    return float(-np.sum(probs * np.log2(probs + _eps)))

def permutation_entropy(data: Union[pd.DataFrame, pd.Series, np.ndarray], 
                       order: int = 3, delay: int = 1, **kwargs) -> float:
    """
    Permutation entropy (Bandt & Pompe). Returns value in bits.
    """
    series = _extract_series(data, kwargs.get('column', 'close'))
    x = _to_numpy(series)
    n = x.size
    if n < order * delay + 1:
        return float(np.nan)

    patterns = {}
    m = order
    for i in range(n - (m - 1) * delay):
        window = x[i : i + m * delay : delay]
        ranks = tuple(np.argsort(window).tolist())
        patterns[ranks] = patterns.get(ranks, 0) + 1

    counts = np.array(list(patterns.values()), dtype=float)
    probs = counts / (np.sum(counts) + _eps)
    return float(-np.sum(probs * np.log2(probs + _eps)))

def sample_entropy(data: Union[pd.DataFrame, pd.Series, np.ndarray], 
                  m: int = 2, r: Optional[float] = None, **kwargs) -> float:
    """
    Sample entropy approximation (unbiased-ish).
    Uses direct comparisons; O(N^2) in worst case.
    """
    series = _extract_series(data, kwargs.get('column', 'close'))
    x = _to_numpy(series)
    N = x.size
    if N < m + 2:
        return float(np.nan)
    if r is None:
        r = 0.2 * np.nanstd(x) + _eps
    # Build m-length templates
    def _count_similar(m_len: int) -> float:
        count = 0
        templates = N - m_len + 1
        for i in range(templates):
            xi = x[i:i + m_len]
            # compare to subsequent templates
            for j in range(i + 1, templates):
                xj = x[j:j + m_len]
                if np.max(np.abs(xi - xj)) <= r:
                    count += 1
        return float(count)

    try:
        B = _count_similar(m)
        A = _count_similar(m + 1)
        if B == 0:
            return float(np.nan)
        return float(-np.log((A + _eps) / (B + _eps)))
    except Exception:
        return float(np.nan)

def granger_causality(data: Union[pd.DataFrame, pd.Series, np.ndarray], 
                     max_lag: int = 5, significance_level: float = 0.05,
                     **kwargs) -> dict:
    """
    Granger causality test between two time series.
    Tests if series_y Granger-causes series_x.
    
    Expects data to have two columns or provides secondary_series in kwargs.
    
    Returns dict with:
    - f_statistic: F-test statistic
    - p_value: p-value of the test
    - significant: boolean indicating significance
    - best_lag: optimal lag according to AIC
    """
    # Extract primary and secondary series
    if isinstance(data, pd.DataFrame) and len(data.columns) >= 2:
        series_x = data.iloc[:, 0].values
        series_y = data.iloc[:, 1].values
    elif 'secondary_series' in kwargs:
        series_x = _extract_series(data, kwargs.get('column', 'close'))
        series_y = _extract_series(kwargs['secondary_series'], kwargs.get('secondary_column', 'close'))
    else:
        return {
            'f_statistic': float(np.nan),
            'p_value': float(np.nan),
            'significant': False,
            'best_lag': 0,
            'error': 'Insufficient data: need two series for Granger causality'
        }
    
    x = _to_numpy(series_x)
    y = _to_numpy(series_y)
    
    # Ensure same length and remove NaNs
    mask = _mask_valid(x) & _mask_valid(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    n = x_clean.size
    if n < max_lag + 10:  # Minimum sample size requirement
        return {
            'f_statistic': float(np.nan),
            'p_value': float(np.nan),
            'significant': False,
            'best_lag': 0,
            'error': 'Insufficient data after cleaning'
        }
    
    # Find optimal lag using AIC
    best_lag = 1
    best_aic = np.inf
    
    for lag in range(1, max_lag + 1):
        if n <= lag * 3:
            continue
            
        # Restricted model (only x's own lags)
        X_r = _create_lag_matrix(x_clean, lag)
        if X_r.shape[0] == 0:
            continue
            
        # Unrestricted model (x's lags + y's lags)
        X_ur = np.column_stack([
            _create_lag_matrix(x_clean, lag),
            _create_lag_matrix(y_clean, lag)
        ])
        
        # Remove rows with NaN
        valid_mask = ~(np.any(np.isnan(X_ur), axis=1) | np.isnan(x_clean[lag:]))
        if np.sum(valid_mask) < lag + 5:
            continue
            
        X_r_valid = X_r[valid_mask]
        X_ur_valid = X_ur[valid_mask]
        y_target = x_clean[lag:][valid_mask]
        
        try:
            # Fit models
            beta_r = np.linalg.lstsq(X_r_valid, y_target, rcond=None)[0]
            beta_ur = np.linalg.lstsq(X_ur_valid, y_target, rcond=None)[0]
            
            # Calculate residuals
            resid_r = y_target - X_r_valid @ beta_r
            resid_ur = y_target - X_ur_valid @ beta_ur
            
            # Calculate AIC
            k_r = lag + 1
            k_ur = 2 * lag + 1
            T = len(y_target)
            
            aic_r = T * np.log(np.var(resid_r)) + 2 * k_r
            aic_ur = T * np.log(np.var(resid_ur)) + 2 * k_ur
            
            if aic_ur < best_aic:
                best_aic = aic_ur
                best_lag = lag
                
        except np.linalg.LinAlgError:
            continue
    
    # Perform Granger test with best lag
    if best_lag == 0:
        return {
            'f_statistic': float(np.nan),
            'p_value': float(np.nan),
            'significant': False,
            'best_lag': 0,
            'error': 'No suitable lag found'
        }
    
    # Final test with best lag
    X_r = _create_lag_matrix(x_clean, best_lag)
    X_ur = np.column_stack([
        _create_lag_matrix(x_clean, best_lag),
        _create_lag_matrix(y_clean, best_lag)
    ])
    
    valid_mask = ~(np.any(np.isnan(X_ur), axis=1) | np.isnan(x_clean[best_lag:]))
    X_r_valid = X_r[valid_mask]
    X_ur_valid = X_ur[valid_mask]
    y_target = x_clean[best_lag:][valid_mask]
    
    try:
        # Fit final models
        beta_r = np.linalg.lstsq(X_r_valid, y_target, rcond=None)[0]
        beta_ur = np.linalg.lstsq(X_ur_valid, y_target, rcond=None)[0]
        
        # Calculate residuals
        resid_r = y_target - X_r_valid @ beta_r
        resid_ur = y_target - X_ur_valid @ beta_ur
        
        # Calculate F-statistic
        RSS_r = np.sum(resid_r ** 2)
        RSS_ur = np.sum(resid_ur ** 2)
        T = len(y_target)
        
        if RSS_ur < _eps or T <= 2 * best_lag + 1:
            f_statistic = 0.0
            p_value = 1.0
        else:
            f_statistic = ((RSS_r - RSS_ur) / best_lag) / (RSS_ur / (T - 2 * best_lag - 1))
            
            # Calculate p-value using F-distribution
            from scipy.stats import f
            p_value = 1 - f.cdf(f_statistic, best_lag, T - 2 * best_lag - 1)
        
        significant = p_value < significance_level
        
        return {
            'f_statistic': float(f_statistic),
            'p_value': float(p_value),
            'significant': significant,
            'best_lag': best_lag
        }
        
    except (np.linalg.LinAlgError, ValueError):
        return {
            'f_statistic': float(np.nan),
            'p_value': float(np.nan),
            'significant': False,
            'best_lag': best_lag,
            'error': 'Matrix calculation failed'
        }

def phase_shift_index(data: Union[pd.DataFrame, pd.Series, np.ndarray], 
                     method: str = "hilbert", **kwargs) -> float:
    """
    Calculate phase shift between two signals.
    
    Expects data to have two columns or provides secondary_series in kwargs.
    
    Parameters:
    - data: input data containing two series
    - method: 'hilbert' (using analytic signal) or 'fft' (using FFT phase)
    
    Returns phase shift in radians.
    """
    # Extract primary and secondary series
    if isinstance(data, pd.DataFrame) and len(data.columns) >= 2:
        series1 = data.iloc[:, 0].values
        series2 = data.iloc[:, 1].values
    elif 'secondary_series' in kwargs:
        series1 = _extract_series(data, kwargs.get('column', 'close'))
        series2 = _extract_series(kwargs['secondary_series'], kwargs.get('secondary_column', 'close'))
    else:
        return float(np.nan)
    
    x1 = _to_numpy(series1)
    x2 = _to_numpy(series2)
    
    # Ensure same length and remove NaNs
    min_len = min(len(x1), len(x2))
    x1 = x1[:min_len]
    x2 = x2[:min_len]
    
    mask = _mask_valid(x1) & _mask_valid(x2)
    x1_clean = x1[mask]
    x2_clean = x2[mask]
    
    n = x1_clean.size
    if n < 10:
        return float(np.nan)
    
    if method.lower() == "hilbert":
        # Using analytic signal method
        analytic1 = _analytic_signal_via_fft(x1_clean)
        analytic2 = _analytic_signal_via_fft(x2_clean)
        
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
    elif method.lower() == "fft":
        # Using FFT phase method
        fft1 = np.fft.fft(x1_clean)
        fft2 = np.fft.fft(x2_clean)
        
        # Get phase at dominant frequency
        dominant_idx = np.argmax(np.abs(fft1))
        phase1 = np.angle(fft1[dominant_idx])
        phase2 = np.angle(fft2[dominant_idx])
        
    else:
        raise ValueError("Method must be 'hilbert' or 'fft'")
    
    # Calculate phase difference
    phase_diff = phase1 - phase2
    
    # Unwrap phase differences to avoid 2π jumps
    phase_diff_unwrapped = np.unwrap(phase_diff)
    
    # Return mean phase shift
    return float(np.mean(phase_diff_unwrapped))

# ==================== REGISTRY ====================
_METRICS = {
    "kalman_filter_trend": kalman_filter_trend,
    "wavelet_transform": wavelet_transform,
    "hilbert_transform_amplitude": hilbert_transform_amplitude,
    "hilbert_transform_slope": hilbert_transform_slope,
    "fractal_dimension_index_fdi": fractal_dimension_index_fdi,
    "shannon_entropy": shannon_entropy,
    "permutation_entropy": permutation_entropy,
    "sample_entropy": sample_entropy,
    "granger_causality": granger_causality,
    "phase_shift_index": phase_shift_index
}

def get_metrics() -> List[str]:
    return list(_METRICS.keys())

def get_function(metric_name: str):
    return _METRICS.get(metric_name)

def get_module_config() -> Dict[str, Any]:
    return _MODULE_CONFIG.copy()

def get_column_groups() -> Dict[str, List[str]]:
    return COLUMN_GROUPS.copy()