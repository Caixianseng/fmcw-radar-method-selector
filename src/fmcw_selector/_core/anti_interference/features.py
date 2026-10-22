"""Observed ADC features and separate, offline-only interference descriptions.

The MLP receives ``extract_features`` only. ``describe_interference`` uses the
simulator's separated signals for labels and evaluation, never for inference.
"""
from __future__ import annotations

import numpy as np

FEATURE_VERSION = 'observed-chirp-v1'
FEATURE_NAMES = (
    'log10_mean_power', 'amplitude_q10_over_rms', 'amplitude_q50_over_rms',
    'amplitude_q90_over_rms', 'amplitude_q99_over_rms', 'amplitude_peak_over_rms',
    'power_kurtosis', 'amplitude_skewness', 'energy_top10_fraction',
    'energy_first_half_fraction', 'envelope_cv', 'spectrum_entropy',
    'spectrum_flatness', 'spectrum_top5_fraction', 'spectrum_effective_fraction',
    'spectrum_outside_target_band_fraction', 'spectrum_negative_fraction',
    'tf_entropy_mean', 'tf_entropy_q90', 'tf_occupancy_mean',
    'tf_occupancy_q90', 'tf_peak_fraction_mean', 'tf_outside_band_mean',
    'tf_outside_band_q90', 'tf_time_energy_entropy', 'tf_ridge_motion',
    'tf_ridge_linear_r2', 'proxy_coverage_fraction', 'proxy_longest_fraction',
    'proxy_burst_count', 'proxy_first_active_fraction', 'proxy_last_active_fraction',
)

LABEL_VERSION = 'signal-relative-effective-support-v1'
LABEL_RULES = {
    'envelope_smoothing_us': 0.25,
    'threshold_relative_clean_mean_power_db': -10.0,
    'threshold_relative_interference_peak_power_db': -30.0,
    'strength_db_boundaries': [-20.0, 0.0, 10.0],
    'continuous_min_coverage_fraction': 0.65,
    'definition': ('Effective filtered interference support, not an RF transmit '
                   'duty or an ideal LPF frequency mask. Threshold is the maximum '
                   'of clean mean power / 10 and smoothed interference peak / 1000.'),
}


def _validated_signal(x, fs):
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 1 or x.size < 8 or not np.isfinite(x).all():
        raise ValueError('A finite one-dimensional chirp with at least 8 samples is required')
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('A positive finite sample rate is required')
    return x


def _smooth_power(x, width):
    p = np.abs(x) ** 2
    width = max(1, min(int(width), p.size))
    # Correct the number of contributing samples at finite-record edges.
    kernel = np.ones(width)
    return np.convolve(p, kernel, mode='same') / np.convolve(np.ones(p.size), kernel, mode='same')


def _segments(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _entropy(p, axis=-1):
    p = np.asarray(p, float)
    total = p.sum(axis=axis, keepdims=True)
    q = np.divide(p, total, out=np.zeros_like(p), where=total > 0)
    return -(q * np.log(np.maximum(q, 1e-300))).sum(axis=axis) / max(np.log(p.shape[axis]), 1.0)


def extract_features(x, fs, max_beat_hz):
    """Return the fixed observed-only feature vector for one complex ADC chirp.

    ``max_beat_hz`` must come from the public receiver/operating-range setting,
    rather than from known target ranges. Detection-like features are explicitly
    named proxies: they do not claim that target modulation is interference.
    """
    x = _validated_signal(x, fs)
    if not np.isfinite(max_beat_hz) or not 0 < max_beat_hz <= fs / 2:
        raise ValueError('max_beat_hz must lie in (0, Nyquist]')
    n = x.size
    p = np.abs(x) ** 2
    mean = float(p.mean())
    rms = np.sqrt(max(mean, 1e-300))
    amp = np.abs(x) / rms
    sd = float(amp.std())
    skew = float(np.mean(((amp - amp.mean()) / sd) ** 3)) if sd > 1e-12 else 0.0
    ordered = np.sort(p)
    energy = max(float(p.sum()), 1e-300)
    smooth = _smooth_power(x, max(1, round(0.25e-6 * fs)))
    envelope_cv = float(smooth.std() / max(float(smooth.mean()), 1e-300))
    sp = np.abs(np.fft.fft(x * np.hanning(n))) ** 2
    total_sp = max(float(sp.sum()), 1e-300)
    frequencies = np.fft.fftfreq(n, 1.0 / fs)
    flatness = (float(np.exp(np.mean(np.log(np.maximum(sp / total_sp, 1e-300))))) * n
                if mean > 0 else 0.0)
    spectrum_effective = total_sp ** 2 / max(float(np.sum(sp ** 2)), 1e-300) / n if mean else 0.0
    width = min(256, n)
    hop = max(1, width // 4)
    starts = np.unique(np.r_[np.arange(0, n - width + 1, hop), n - width]).astype(int)
    chunks = np.stack([x[a:a + width] for a in starts]) * np.hanning(width)
    tf = np.abs(np.fft.fft(chunks, axis=1)) ** 2
    tf_sum = tf.sum(axis=1)
    tf_freq = np.fft.fftfreq(width, 1.0 / fs)
    entropy = _entropy(tf)
    peak = tf.max(axis=1)
    occupancy = (tf > peak[:, None] * 10 ** (-12 / 10)).mean(axis=1)
    tf_outside = np.divide(tf[:, np.abs(tf_freq) > max_beat_hz].sum(axis=1), tf_sum,
                           out=np.zeros_like(tf_sum), where=tf_sum > 0)
    tf_peak = np.divide(peak, tf_sum, out=np.zeros_like(tf_sum), where=tf_sum > 0)
    # Observed proxy: frequency breadth or energy beyond the public target band,
    # with a relative energy guard against nearly-zero tapered edges.
    flagged = ((occupancy > .08) | (tf_outside > .20)) & (tf_sum > .01 * max(float(tf_sum.max()), 1e-300))
    proxy = np.zeros(n, bool)
    for start, hit in zip(starts, flagged):
        if hit:
            proxy[start:start + width] = True
    first, end = _segments(proxy)
    longest = int((end - first).max()) if len(first) else 0
    ridge_motion = ridge_r2 = 0.0
    if len(starts) >= 3 and mean > 0:
        # Unwrap the signed FFT frequency for a ridge crossing the FFT boundary.
        ridge = np.unwrap(2 * np.pi * np.argmax(tf, axis=1) / width) / (2 * np.pi)
        times = starts / max(n - 1, 1)
        weight = tf_sum / max(float(tf_sum.max()), 1e-300)
        good = weight > .01
        if good.sum() >= 3:
            fit = np.polyfit(times[good], ridge[good], 1, w=np.sqrt(weight[good]))
            reference = np.average(ridge[good], weights=weight[good])
            ss_total = float(np.sum(weight[good] * (ridge[good] - reference) ** 2))
            ss_residual = float(np.sum(weight[good] * (ridge[good] - np.polyval(fit, times[good])) ** 2))
            ridge_r2 = float(np.clip(1 - ss_residual / ss_total, 0, 1)) if ss_total > 1e-15 else 0.0
            ridge_motion = float(abs(fit[0]) * ridge_r2)
    values = (
        np.log10(max(mean, 1e-300)), *np.quantile(amp, [.1, .5, .9, .99]), float(amp.max()),
        float(np.mean((p / max(mean, 1e-300)) ** 2)), skew,
        float(ordered[-max(1, int(np.ceil(.1 * n))):].sum() / energy),
        float(p[:n // 2].sum() / energy), envelope_cv, float(_entropy(sp)), flatness,
        float(np.sort(sp)[-max(1, int(np.ceil(.05 * n))):].sum() / total_sp), spectrum_effective,
        float(sp[np.abs(frequencies) > max_beat_hz].sum() / total_sp),
        float(sp[frequencies < 0].sum() / total_sp), float(entropy.mean()), float(np.quantile(entropy, .9)),
        float(occupancy.mean()), float(np.quantile(occupancy, .9)), float(tf_peak.mean()),
        float(tf_outside.mean()), float(np.quantile(tf_outside, .9)), float(_entropy(tf_sum)),
        ridge_motion, ridge_r2, float(proxy.mean()), longest / n, float(len(first)),
        float(first[0] / n) if len(first) else 0.0, float(end[-1] / n) if len(first) else 0.0,
    )
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError('Non-finite or inconsistent observed feature vector')
    return result


def describe_interference(clean, interf, source_filtered, fs):
    """Describe filtered truth for offline labels, separate from MLP features.

    Signal-relative and peak-relative thresholds replace a zero noise threshold.
    Source masks describe each effective contribution, not theoretical RF activity.
    This function's outputs must never be passed to ``extract_features`` or used as
    selector inputs at deployment.
    """
    clean = _validated_signal(clean, fs)
    interf = _validated_signal(interf, fs)
    if clean.shape != interf.shape:
        raise ValueError('clean and interference chirps must have matching shapes')
    sources = np.asarray(source_filtered, dtype=np.complex128)
    if sources.size == 0:
        sources = np.empty((0, clean.size), dtype=np.complex128)
    elif sources.ndim == 1:
        sources = sources[None, :]
    if sources.ndim != 2 or sources.shape[1] != clean.size or not np.isfinite(sources).all():
        raise ValueError('source_filtered must be finite with shape (sources, samples)')
    width = max(1, round(LABEL_RULES['envelope_smoothing_us'] * 1e-6 * fs))
    clean_power = float(np.mean(np.abs(clean) ** 2))
    interference_power = float(np.mean(np.abs(interf) ** 2))
    envelope = _smooth_power(interf, width)
    peak = float(envelope.max())
    threshold = max(clean_power * .1, peak * .001, 1e-300)
    mask = envelope > threshold
    first, end = _segments(mask)
    n = clean.size
    longest = int((end - first).max()) if len(first) else 0
    per_source = []
    source_masks = []
    for source in sources:
        source_envelope = _smooth_power(source, width)
        source_threshold = max(clean_power * .1, float(source_envelope.max()) * .001, 1e-300)
        source_mask = source_envelope > source_threshold
        source_masks.append(source_mask)
        left, right = _segments(source_mask)
        per_source.append(dict(power_w=float(np.mean(np.abs(source) ** 2)),
                               peak_amplitude=float(np.abs(source).max()),
                               duty_fraction=float(source_mask.mean()),
                               threshold_power_w=source_threshold,
                               intervals_us=[[float(a / fs * 1e6), float(b / fs * 1e6)] for a, b in zip(left, right)]))
    counts = np.sum(source_masks, axis=0) if len(source_masks) else np.zeros(n, int)
    effective_sources = sum(bool(m.any()) for m in source_masks)
    ratio_db = float(10 * np.log10(max(interference_power, 1e-300) / max(clean_power, 1e-300)))
    if interference_power == 0 or not mask.any():
        strength = 'negligible'
        temporal = 'none'
    else:
        strength = 'strong' if ratio_db >= 10 else 'moderate' if ratio_db >= 0 else 'weak' if ratio_db >= -20 else 'negligible'
        temporal = ('continuous' if mask.mean() >= .65 else 'fragmented' if len(first) >= 2
                    else 'single_burst' if mask.mean() <= .35 else 'partial')
    source_regime = 'multi_source' if effective_sources >= 2 else 'single_source' if effective_sources == 1 else 'none'
    tags = [strength, temporal, source_regime]
    return dict(label_version=LABEL_VERSION, label=':'.join(tags), tags=tags,
                strength=strength, temporal_shape=temporal, source_regime=source_regime,
                clean_power_w=clean_power, interference_power_w=interference_power,
                interference_to_signal_db=ratio_db, duty_fraction=float(mask.mean()),
                longest_fraction=longest / n, longest_us=longest / fs * 1e6,
                burst_count=int(len(first)), supplied_source_count=int(len(sources)),
                effective_source_count=int(effective_sources),
                max_simultaneous_sources=int(counts.max()),
                multisource_overlap_fraction=float(np.mean(counts >= 2)),
                threshold_power_w=threshold, threshold_rules=dict(LABEL_RULES),
                intervals_us=[[float(a / fs * 1e6), float(b / fs * 1e6)] for a, b in zip(first, end)],
                source_descriptions=per_source)
