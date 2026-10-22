"""Isolated observation-only robust slow-chirplet fit.

This is an engineering variant of the frozen Chirplet stage, not a new paper
reproduction. Short, unusually strong bursts receive less weight only while
estimating slow-chirplet parameters/amplitudes. Accepted physical chirplets are
subtracted from the original unweighted ADC. Nuisance tones are fitted jointly
and never subtracted. No truth, source state, labels, or future chirps are used.
"""
from __future__ import annotations
from pathlib import Path
import importlib.util
import json
import math
import time
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d, binary_dilation, label
from scipy.optimize import minimize_scalar

from . import chirplet_omp as old
VERSION = "robust_weighted_slow_chirplet_stage_20261019_v1"


def _weights(x, model):
    """Downweight only short normalized-envelope bursts; all constants fixed."""
    n = len(x)
    # Public Blackman transmit/LO product is known. The floor avoids amplifying
    # the ADC's taper endpoints. A short mean estimates power, not signed I/Q.
    envelope = np.maximum(model.common, 0.04)
    amp = np.sqrt(uniform_filter1d(abs(x / envelope)**2, size=17, mode="nearest"))
    width = min(n//2*2-1, 601)
    reference = median_filter(amp, size=width, mode="nearest")
    floor = 0.35 * float(np.quantile(amp, 0.75))
    raw = amp > 4.0*np.maximum(reference, max(floor, 1e-12))
    raw &= model.common > 0.04
    labels, count = label(raw)
    accepted = np.zeros(n, bool)
    for k in range(1, count+1):
        index = np.flatnonzero(labels == k)
        if 4 <= index.size <= int(0.18*n):
            accepted[index] = True
    accepted = binary_dilation(accepted, iterations=12)
    fraction = float(accepted.mean())
    if fraction > 0.45:
        return np.ones(n), dict(status="too_much_short_burst_mask", fraction=fraction)
    # Retain a finite weight to avoid making a short missing interval a new
    # source of spectral sidelobes during the parameter search.
    weights = np.where(accepted, 0.02, 1.0)
    return weights, dict(status="weighted_short_bursts" if accepted.any() else "no_short_bursts",
                         fraction=fraction, threshold_ratio=4.0, low_weight=0.02)


class _WeightedModel:
    def __init__(self, model, sqrt_weight):
        self.base = model
        self.weight = sqrt_weight
        self.p, self.n = model.p, model.n
        self.t, self.tc, self.center = model.t, model.tc, model.center
        self.common = model.common*sqrt_weight

    def atom(self, rate, frequency, exact=False):
        return self.base.atom(rate, frequency, exact=exact)*self.weight

    def tone(self, frequency):
        return self.base.tone(frequency)*self.weight


def process(observed, public, mode="robust_slow"):
    """Return pre-AR residual, diagnostics. Modes: baseline, robust_slow."""
    if mode == "baseline":
        return old.process(observed, public)
    if mode != "robust_slow":
        raise ValueError("Unknown Chirplet stage mode: "+str(mode))
    x = np.asarray(observed, complex)
    if x.ndim != 1 or x.size < 32 or not np.isfinite(x).all():
        raise ValueError("Expected finite one-dimensional complex ADC")
    power = float(np.vdot(x, x).real)
    if power <= 1e-300:
        return x.copy(), dict(version=VERSION, status="zero_input", truth_inputs_used=False)
    options = dict(old.DEFAULT_SETTINGS)
    p = old._public(public, x.size)
    base = old._Model(p, x.size)
    scale = math.sqrt(power/x.size)
    original = x/scale
    weights, weight_diag = _weights(original, base)
    if np.array_equal(weights, np.ones(x.size)):
        y, diag = old.process(x, public)
        diag.update(version=VERSION, base_version=old.VERSION, weight_diagnostic=weight_diag,
                    identical_to_frozen_stage=True)
        return y, diag
    model = _WeightedModel(base, np.sqrt(weights))
    weighted_original = original*model.weight
    nfft = 1 << int(np.ceil(np.log2(x.size*int(options["fft_oversampling"]))))
    residual = weighted_original.copy()
    atoms, full_atoms, parameters, entries, attempts = [], [], [], [], []
    for iteration in range(int(options["max_chirplets"])):
        _, rate, frequency = old._coarse(residual, model, options, nfft)
        rate, frequency, score, converged = old._refine(residual, model, rate, frequency, options, nfft)
        tone_frequency, _ = old._peak_frequency(residual, model, nfft)
        df = p["fs"]/nfft
        fit = minimize_scalar(lambda f: -old._power_projection(residual, model.tone(f)),
                              bounds=(tone_frequency-2*df, tone_frequency+2*df), method="bounded")
        tone_score = old._power_projection(residual, model.tone(float(fit.x)))
        ratio = score/max(tone_score, 1e-200)
        fraction = score/max(float(np.vdot(residual, residual).real), 1e-200)
        attempt = dict(iteration=iteration, estimated_rate_hz_per_s=rate,
                       estimated_center_frequency_hz=frequency,
                       chirp_tone_power_ratio=ratio, explained_fraction=fraction)
        attempts.append(attempt)
        if ratio < options["min_chirp_tone_power_ratio"] or fraction < options["min_explained_fraction"]:
            attempt["reason"] = "insufficient_chirplet_over_tone_evidence"
            break
        if any(abs(rate-r) < 2*options["coarse_step_hz_per_s"] and
               abs(frequency-f) < 2*p["fs"]/x.size for r, f in parameters):
            attempt["reason"] = "duplicate_atom"
            break
        atom = model.atom(rate, frequency)
        coeff = np.vdot(atom, residual)/max(float(np.vdot(atom, atom).real), 1e-200)
        tones, tone_frequencies = old._nuisance_tones(residual-coeff*atom, model, options, nfft)
        if options["joint_refinement"] and tones:
            rate, frequency, _, converged = old._refine(residual, model, rate, frequency, options, nfft, tones)
        atom = model.atom(rate, frequency, exact=True)
        full_atom = base.atom(rate, frequency, exact=True)
        if np.linalg.norm(atom) < 1e-12:
            attempt["reason"] = "out_of_receiver_band"
            break
        trial = atoms+[atom]
        matrix = np.column_stack(trial+tones)
        coeffs = np.linalg.lstsq(matrix, weighted_original, rcond=1e-10)[0]
        next_residual = weighted_original-np.column_stack(trial)@coeffs[:len(trial)]
        if (not np.isfinite(next_residual).all() or
            np.vdot(next_residual, next_residual).real > 1.2*np.vdot(residual, residual).real):
            attempt["reason"] = "unstable_joint_fit"
            break
        atoms, full_atoms = trial, full_atoms+[full_atom]
        parameters += [(rate, frequency)]
        residual = next_residual
        for j, entry in enumerate(entries):
            entry.update(amplitude_real=float(coeffs[j].real*scale),
                         amplitude_imag=float(coeffs[j].imag*scale))
        entries.append(dict(rate_hz_per_s=rate, center_frequency_hz=frequency,
                            amplitude_real=float(coeffs[len(atoms)-1].real*scale),
                            amplitude_imag=float(coeffs[len(atoms)-1].imag*scale),
                            nuisance_tone_frequencies_hz=tone_frequencies,
                            optimizer_converged=bool(converged)))
        attempt["reason"] = "cancelled"
    if atoms:
        # Subtraction uses full physical atoms; weight masks are never imposed
        # on the ADC output. Targets/nuisance fits remain in the residual.
        amplitudes = np.array([complex(q["amplitude_real"], q["amplitude_imag"]) for q in entries])
        y = x-np.column_stack(full_atoms)@amplitudes
    else:
        y = x.copy()
    diag = dict(version=VERSION, base_version=old.VERSION, mode=mode,
                status="cancelled" if atoms else "no_supported_chirplet",
                truth_inputs_used=False, weight_diagnostic=weight_diag,
                cancelled_chirplet_count=len(atoms), estimated_chirplets=entries,
                attempts=attempts, removed_power_fraction=float(np.vdot(x-y, x-y).real/power),
                full_unweighted_signal_subtraction=True, nuisance_tones_subtracted=False)
    assert y.shape == x.shape and np.isfinite(y).all()
    return y, diag


def synthetic_checks():
    from threadpoolctl import threadpool_limits
    public = dict(sample_rate_hz=50e6, chirp_duration_s=75e-6, chirp_slope_hz_per_s=4e12,
                  simulation_sample_rate_hz=2e9, transmit_bandwidth_hz=300e6,
                  max_beat_hz=20e6)
    p = old._public(public, 3750)
    m = old._Model(p, 3750)
    target = m.tone(3.15e6)+0.4*np.exp(0.8j)*m.tone(6.7e6)
    slow = 8*np.exp(0.4j)*m.atom(5.0e10, 4.0e6, exact=True)
    t = m.t
    fast = (30*np.exp(-.5*((t-30e-6)/1e-6)**2)*
            np.exp(2j*np.pi*(2e6*(t-30e-6)+.5*1e13*(t-30e-6)**2)))
    records = []
    with threadpool_limits(limits=1):
        for name, x, clean in [("two_clean_tones", target, target),
                              ("slow_plus_targets", target+slow, target),
                              ("slow_fast_plus_targets", target+slow+fast, target+fast)]:
            for mode in ("baseline", "robust_slow"):
                tick = time.perf_counter()
                y, d = process(x, public, mode)
                nmse = float(np.vdot(y-clean, y-clean).real/np.vdot(target, target).real)
                records.append(dict(case=name, mode=mode, nmse_relative_target=nmse,
                                    seconds=time.perf_counter()-tick, diagnostics=d))
    return dict(version=VERSION, truth_only_for_scoring=True, records=records)


if __name__ == "__main__":
    out = ROOT/"新实验_CAR改进验证_20261019"/"analysis"/"chirplet_synthetic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = synthetic_checks()
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k:v for k,v in r.items() if k!="diagnostics"} for r in result["records"]]))
