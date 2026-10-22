"""Observation-only bidirectional complex AR in the STFT domain.

Independent Python CFAR-Burg adaptation: keep the shared 2-D detector mask,
predict each frequency row along time, then ISTFT. See legacy_sources.json.
"""
from __future__ import annotations

import numpy as np
from .methods import _analysis, _istft

METHODS = ('legacy_ar',)  # Retain the selector/deployment column identifier.
LEGACY_VERSION = 'observation_only_cfar_stft_bidirectional_burg_ar_v3'
AR_PARAMETERS = dict(max_ar_order=6, ar_train_len=64, ar_max_horizon=32,
                     min_train_frames=8)


def _segments(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def burg_ar(x, max_order=6, min_train_frames=8):
    """Complex Burg coefficients, stable reflections and BIC order selection.

    Fits contiguous CLEAN frames only. The complex AR convention is
    z[t] = dot(phi, z[t-1:t-p-1:-1]).
    """
    z = np.asarray(x, complex)
    m = z.size
    if m < min_train_frames:
        return None
    scale = float(np.sqrt(np.mean(abs(z) ** 2)))
    if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
        return None
    forward = z / scale
    backward = forward.copy()
    a = np.ones(1, complex)
    variance = 1.
    best, best_score = None, np.inf
    for order in range(1, min(int(max_order), m // 4) + 1):
        f = forward[order:]
        b = backward[order - 1:m - 1]
        denominator = float(np.sum(abs(f) ** 2 + abs(b) ** 2))
        if denominator <= 0:
            break
        reflection = -2 * np.sum(f * b.conj()) / denominator
        if abs(reflection) >= 1 - 1e-10:
            reflection *= (1 - 1e-10) / abs(reflection)
        variance *= 1 - abs(reflection) ** 2
        forward = np.r_[forward[:order], f + reflection * b]
        backward = np.r_[backward[:order], b + reflection.conjugate() * f]
        old = np.r_[a, 0j]
        a = old + reflection * old[::-1].conj()
        a[0] = 1.
        phi = -a[1:]
        score = np.log(max(variance, 1e-12)) + (order + 1) * np.log(m) / m
        if score < best_score and np.all(abs(np.roots(np.r_[1., -phi])) <= 1 + 1e-7):
            best, best_score = phi.copy(), score
    return best


def _predict(train, length, max_order, max_horizon, min_train_frames):
    predicted, valid = np.zeros(length, complex), np.zeros(length, bool)
    phi = burg_ar(train, max_order, min_train_frames)
    if phi is None:
        return predicted, valid
    horizon = min(length, train.size, int(max_horizon))
    history = train[-phi.size:].copy()
    amplitude_limit = max(float(np.max(abs(train))) * 4., np.finfo(float).tiny)
    for k in range(horizon):
        value = np.dot(phi, history[::-1])
        if not np.isfinite(value) or abs(value) > amplitude_limit:
            break
        predicted[k], valid[k] = value, True
        history = np.r_[history[1:], value]
    return predicted, valid


def _repair_spectrum(spectrum, tf_mask, *, max_ar_order=6, ar_train_len=64,
                     ar_max_horizon=32, min_train_frames=8):
    """Repair frequency-by-time coefficients and report actual support.

    Separate models use the original contiguous clean runs on either side.
    Filled values never enter another fit. Unsupported cells stay CFAR-Z zeros.
    Unflagged TF coefficients remain bit-exact here, but ISTFT overlap means
    individual unmasked ADC samples need not remain unchanged.
    """
    s = np.asarray(spectrum, complex)
    mask = np.asarray(tf_mask, bool)
    if s.ndim != 2 or mask.shape != s.shape or not np.isfinite(s).all():
        raise ValueError('spectrum and tf_mask must be finite matching 2-D arrays')
    for name, value in dict(max_ar_order=max_ar_order, ar_train_len=ar_train_len,
                            ar_max_horizon=ar_max_horizon,
                            min_train_frames=min_train_frames).items():
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    max_ar_order, ar_train_len, ar_max_horizon, min_train_frames = map(
        int, (max_ar_order, ar_train_len, ar_max_horizon, min_train_frames))
    if min_train_frames < 4 or ar_train_len < min_train_frames:
        raise ValueError('require ar_train_len >= min_train_frames >= 4')
    out = s.copy()
    out[mask] = 0.
    forward_count = backward_count = both_count = predicted_count = 0
    for row in np.flatnonzero(np.any(mask, axis=1)):
        clean_runs = _segments(~mask[row])
        for a, b in _segments(mask[row]):
            left = next(((lo, hi) for lo, hi in clean_runs if hi == a), None)
            right = next(((lo, hi) for lo, hi in clean_runs if lo == b), None)
            pf, vf = np.zeros(b-a, complex), np.zeros(b-a, bool)
            pb, vb = pf.copy(), vf.copy()
            if left:
                lo, hi = left
                pf, vf = _predict(s[row, max(lo, hi-ar_train_len):hi], b-a,
                                  max_ar_order, ar_max_horizon, min_train_frames)
            if right:
                lo, hi = right
                reverse, valid = _predict(s[row, lo:min(hi, lo+ar_train_len)][::-1], b-a,
                                          max_ar_order, ar_max_horizon, min_train_frames)
                pb, vb = reverse[::-1], valid[::-1]
            supported, both = vf | vb, vf & vb
            fused = np.zeros(b-a, complex)
            fused[vf & ~vb] = pf[vf & ~vb]
            fused[vb & ~vf] = pb[vb & ~vf]
            weight = (b-a-np.arange(b-a)) / (b-a+1.)
            fused[both] = weight[both]*pf[both] + (1-weight[both])*pb[both]
            out[row, a:b] = fused
            forward_count += int(vf.sum())
            backward_count += int(vb.sum())
            both_count += int(both.sum())
            predicted_count += int(supported.sum())
    masked_count = int(mask.sum())
    if not np.array_equal(out[~mask], s[~mask]) or not np.isfinite(out).all():
        raise AssertionError('TF repair changed clean coefficients or returned nonfinite values')
    return out, dict(masked_tf_cells=masked_count, predicted_tf_cells=predicted_count,
        forward_tf_cells=forward_count, backward_tf_cells=backward_count,
        bidirectional_tf_cells=both_count, zero_fallback_tf_cells=masked_count-predicted_count,
        status='no_op' if not masked_count else 'tf_bidirectional_fill' if predicted_count else 'tf_zero_fallback',
        domain='stft', prediction_axis='time_frames', independent_validation=False,
        limitation='local complex AR; unsupported cells zeroed; reconstruction is not guaranteed')


def _time_mask_to_tf(mask, times, fs, context, shape):
    """Exclude every frame overlapping a measured ADC erasure.

    Compatibility for controlled tests/observation time masks only. Default
    CFAR retains the 2-D mask and does not take this all-frequency union.
    """
    cumulative = np.r_[0, np.cumsum(np.asarray(mask, bool), dtype=np.int64)]
    center = np.rint(times*fs).astype(int)
    a = np.clip(center-context.window_samples//2, 0, context.length)
    b = np.clip(center+context.window_samples//2, 0, context.length)
    flagged = cumulative[b]-cumulative[a] > 0
    return np.broadcast_to(flagged, shape).copy()


def reconstruct(x, fs, max_beat_hz, method='ar', *, mask=None, tf_mask=None,
                envelope='blackman', chirp_duration_s=None, chirp_slope_hz_per_s=4e12,
                simulation_sample_rate_hz=2e9,
                max_ar_order=6, ar_train_len=64, ar_max_horizon=32,
                min_train_frames=8, return_diagnostics=False):
    """Observed CFAR/STFT -> per-frequency bidirectional complex AR -> ISTFT.

    Optional masks must come from an observation detector; known erasures are
    solely for controlled tests. tf_mask is frequency-by-time like _stft.
    mask also accepts a measured ADC mask for compatibility.
    AR training/prediction lengths are STFT FRAME counts. Public waveform
    keywords are accepted for dispatch compatibility; no Blackman envelope
    normalization is used in this TF implementation.
    """
    if method not in ('ar', 'legacy_ar'):
        raise ValueError('method must be ar or legacy_ar')
    if mask is not None and tf_mask is not None:
        raise ValueError('provide mask or tf_mask, not both')
    values, scale, times, spectrum, context, detected = _analysis(x, fs, max_beat_hz)
    if values.size < 16 or max_beat_hz >= fs/2:
        raise ValueError('require length >= 16 and 0 < max_beat_hz < fs/2')
    if envelope not in ('blackman', 'constant'):
        raise ValueError('envelope must be constant or blackman')
    supplied = tf_mask if tf_mask is not None else mask
    if supplied is None:
        excluded = detected
    else:
        supplied = np.asarray(supplied, bool)
        if supplied.shape == spectrum.shape:
            excluded = supplied
        elif tf_mask is None and supplied.shape == values.shape:
            excluded = _time_mask_to_tf(supplied, times, fs, context, spectrum.shape)
        else:
            raise ValueError('mask must match ADC samples or the 2-D STFT spectrum')
    repaired, diag = _repair_spectrum(spectrum, excluded, max_ar_order=max_ar_order,
        ar_train_len=ar_train_len, ar_max_horizon=ar_max_horizon,
        min_train_frames=min_train_frames)
    out = _istft(repaired, fs, context)*scale if excluded.any() else values.copy()
    if out.shape != values.shape or not np.isfinite(out).all():
        raise AssertionError('TF AR returned invalid ADC samples')
    diag.update(version=LEGACY_VERSION, method=method, stft_window_samples=context.window_samples,
        stft_hop_samples=context.hop_samples,
        parameters=dict(max_ar_order=max_ar_order, ar_train_len=ar_train_len,
                        ar_max_horizon=ar_max_horizon, min_train_frames=min_train_frames),
        future_chirps_used=False)
    return (out, diag) if return_diagnostics else out


def run_legacy_methods(x, fs, max_beat_hz, *, mask=None, **kwargs):
    """Return the retained {legacy_ar: complex_vector} selector column."""
    return {name: reconstruct(x, fs, max_beat_hz, method=name, mask=mask, **kwargs)
            for name in METHODS}
