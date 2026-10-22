"""Finite observation-only upgrades to the AR stage of C-AR.

Input is the already chirplet-cancelled residual. The detector is unchanged.
The public transmit/LO Blackman taper is removed from STFT-row amplitudes
while fitting/predicting, then reapplied. No simulated target is read.
Two predeclared modes are exposed; no automatic tuning uses test data.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from fmcw_selector._core.anti_interference.methods import _analysis, _istft
from fmcw_selector._core.anti_interference.legacy_methods import burg_ar, _segments

VERSION = "observed_envelope_burg_ar_validation_v1"
MODES = ("envelope", "validated")
PARAMETERS = {
    "envelope": dict(max_order=6, train_len=64, max_horizon=32,
                     min_train=8, taper_floor=0.03, validation=False),
    "validated": dict(max_order=6, train_len=64, max_horizon=64,
                      min_train=12, taper_floor=0.03, validation=True,
                      max_validation_nmse=0.35, validation_frames=6),
}


def _frame_envelope(times, fs, context, public):
    """Known Blackman TX times Blackman LO, Hann-weighted STFT amplitude.

    The target delay is deliberately unknown. Thus this is an approximate
    public-envelope compensation, not a target-specific amplitude model.
    """
    if public.get("envelope", "blackman") == "constant":
        return np.ones(len(times))
    n = context.length
    taper = np.maximum(np.blackman(n), 0.0) ** 2
    win = signal.get_window("hann", context.window_samples)
    offsets = np.arange(context.window_samples) - context.window_samples // 2
    indices = np.rint(np.asarray(times) * fs).astype(int)[:, None] + offsets
    valid = (indices >= 0) & (indices < n)
    samples = taper[np.clip(indices, 0, n - 1)] * valid
    return np.sum(samples * win[None, :], axis=1) / np.sum(win)


def _forecast(train, length, p):
    output = np.zeros(length, complex)
    valid = np.zeros(length, bool)
    diag = dict(validation_nmse=None, order=0, accepted=False)
    if len(train) < p["min_train"]:
        return output, valid, diag
    if p["validation"]:
        held = min(p["validation_frames"], max(2, len(train) // 5))
        fit = train[:-held]
        phi = burg_ar(fit, p["max_order"], 8)
        if phi is None:
            return output, valid, diag
        history = fit[-len(phi):].copy()
        check = []
        for _ in range(held):
            value = np.dot(phi, history[::-1])
            check.append(value)
            history = np.r_[history[1:], value]
        error = float(np.mean(abs(np.asarray(check) - train[-held:])**2)
                      / max(np.mean(abs(train[-held:])**2), 1e-30))
        diag["validation_nmse"] = error
        if not np.isfinite(error) or error > p["max_validation_nmse"]:
            return output, valid, diag
    phi = burg_ar(train, p["max_order"], 8)
    if phi is None:
        return output, valid, diag
    diag.update(order=int(len(phi)), accepted=True)
    history = train[-len(phi):].copy()
    limit = max(float(np.max(abs(train))) * 4.0, np.finfo(float).tiny)
    horizon = min(length, p["max_horizon"], len(train))
    for k in range(horizon):
        value = np.dot(phi, history[::-1])
        if not np.isfinite(value) or abs(value) > limit:
            break
        output[k], valid[k] = value, True
        history = np.r_[history[1:], value]
    return output, valid, diag


def _repair(spectrum, mask, frame_envelope, p):
    scale = np.maximum(frame_envelope, p["taper_floor"])
    source = spectrum / scale[None, :]
    # Only genuinely observed support participates in each independent fit.
    # Filled cells never become training for another gap.
    available = ~mask & (frame_envelope[None, :] >= p["taper_floor"])
    output = spectrum.copy()
    output[mask] = 0.0
    predicted = both_count = fit_count = reject_count = 0
    errors, orders = [], []
    for row in np.flatnonzero(np.any(mask, axis=1)):
        clean_runs = _segments(available[row])
        for a, b in _segments(mask[row]):
            width = b - a
            pf, pb = np.zeros(width, complex), np.zeros(width, complex)
            vf, vb = np.zeros(width, bool), np.zeros(width, bool)
            left = next(((lo, hi) for lo, hi in clean_runs if hi == a), None)
            right = next(((lo, hi) for lo, hi in clean_runs if lo == b), None)
            for direction, support in (("forward", left), ("backward", right)):
                if support is None:
                    continue
                lo, hi = support
                train = (source[row, max(lo, hi-p["train_len"]):hi]
                         if direction == "forward"
                         else source[row, lo:min(hi, lo+p["train_len"])][::-1])
                values, usable, info = _forecast(train, width, p)
                fit_count += 1
                reject_count += int(not info["accepted"])
                if info["validation_nmse"] is not None:
                    errors.append(info["validation_nmse"])
                if info["accepted"]:
                    orders.append(info["order"])
                if direction == "forward":
                    pf, vf = values, usable
                else:
                    pb, vb = values[::-1], usable[::-1]
            both = vf & vb
            fused = np.zeros(width, complex)
            fused[vf & ~vb] = pf[vf & ~vb]
            fused[vb & ~vf] = pb[vb & ~vf]
            weight = (width - np.arange(width)) / (width + 1.0)
            fused[both] = weight[both]*pf[both] + (1-weight[both])*pb[both]
            output[row, a:b] = fused * scale[a:b]
            predicted += int(np.sum(vf | vb))
            both_count += int(np.sum(both))
    assert np.array_equal(output[~mask], spectrum[~mask])
    assert np.isfinite(output).all()
    count = int(mask.sum())
    return output, dict(masked_tf_cells=count, predicted_tf_cells=predicted,
                       bidirectional_tf_cells=both_count,
                       zero_fallback_tf_cells=count-predicted,
                       forecast_fit_count=fit_count,
                       forecast_rejected_count=reject_count,
                       validation_nmse_mean=float(np.mean(errors)) if errors else None,
                       accepted_order_mean=float(np.mean(orders)) if orders else None)


def process(observed, public, mode="envelope", settings=None):
    """Return (ADC output, diagnostics) for one chirplet residual vector.

    No raw/chirplet stage is called here. Empty and exactly zero inputs are
    returned unchanged. Public receiver parameters are the only other inputs.
    """
    if settings is not None:
        if set(settings) != {"mode"}:
            raise ValueError("settings may contain the fixed mode only")
        mode = settings["mode"]
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    values = np.asarray(observed, dtype=np.complex128)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("observed must be one finite complex vector")
    fs = float(public["sample_rate_hz"])
    beat = float(public["max_beat_hz"])
    diag = dict(version=VERSION, mode=mode, parameters=dict(PARAMETERS[mode]),
                stage="ar_after_existing_chirplet", truth_inputs_used=False,
                future_chirps_used=False, detector_unchanged=True,
                envelope_model="public_Hann_weighted_Blackman_TX_times_LO",
                limitation="Approximate envelope; no guaranteed support under long interference")
    if not values.size or not np.any(values):
        diag.update(status="empty_or_zero_noop", masked_tf_cells=0,
                    predicted_tf_cells=0, zero_fallback_tf_cells=0)
        return values.copy(), diag
    original, amplitude, times, spectrum, context, mask = _analysis(values, fs, beat)
    if not np.any(mask):
        diag.update(status="detector_noop", masked_tf_cells=0,
                    predicted_tf_cells=0, zero_fallback_tf_cells=0)
        return original.copy(), diag
    envelope = _frame_envelope(times, fs, context, public)
    repaired, actual = _repair(spectrum, mask, envelope, PARAMETERS[mode])
    output = _istft(repaired, fs, context) * amplitude
    if output.shape != values.shape or not np.isfinite(output).all():
        raise AssertionError("AR upgrade produced invalid ADC output")
    diag.update(actual, status="envelope_ar_fill" if actual["predicted_tf_cells"]
                else "tf_zero_fallback")
    return output, diag
