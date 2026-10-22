"""Wang--Ding--Yarovoy iterative matrix-pencil gap reconstruction.

Primary source: arXiv:2102.11557, Section III-C, equations (15)--(30),
DOI 10.1109/TMTT.2021.3090798. This independently written implementation
constructs a separate Hankel pencil per continuous observed segment, stacks
the pencils vertically, applies TWO truncated SVDs and the reduced pencil
eigenproblem (24), and fits complex coefficients at the true sample indices.
Missing samples are fused with unchanged observations and the pencil is
re-estimated, stopping when observed-segment error increases.

Explicit adaptations, not a claim of an exact paper reproduction:
* Generalized from two segments/one gap to multiple continuous segments.
* Fixed relative singular-value threshold (discussed by the paper for its
  measured data) and bounded rank instead of SAMOS, whose formula is not
  specified by that paper. No FFT-based pole or frequency estimation is used.
* Unit-circle projection and public target-band pole rejection.
* In fusion iterations, complex amplitudes are refitted to the ORIGINAL
  observed samples at their true indices. The paper instead describes an
  ordinary contiguous-data parameter refit after inserting the predictions.
* Blackman mode safely normalizes by the PUBLIC transmit/LO window product
  and fits a frequency-dependent delayed-window basis. The common-window
  division is approximate: target delay and FIR response violate the paper's
  constant-amplitude model. Constant mode directly tests the paper core.

Only x, an observation-derived mask and public receiver settings are read.
The output preserves every unmasked input sample bit for bit.
"""
from __future__ import annotations

import numpy as np
from numbers import Real
from scipy.linalg import svd

METHOD = 'matrix_pencil'
MP_VERSION = 'stacked_two_tsvd_pencil_fusion_blackman_v1'


def _segments(mask):
    e = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def _blackman(t, duration, simulation_fs):
    last = duration - 1. / simulation_fs
    phase = 2 * np.pi * np.clip(t, 0., last) / last
    return np.where((t >= 0.) & (t <= last),
                    .42 - .5 * np.cos(phase) + .08 * np.cos(2 * phase), 0.)


def stacked_hankel_pencil(sequence, segments, pencil_length):
    """Eq. (15)--(17); never create a sample adjacency across a gap.

    Each length-M segment contributes (M-L) x L shifted Hankel matrices.
    The absolute offset is intentionally irrelevant ONLY for estimating the
    shared poles. Amplitude fitting elsewhere retains absolute sample indices.
    """
    z = np.asarray(sequence, complex)
    length = int(pencil_length)
    if length < 2:
        raise ValueError('pencil_length must be at least 2')
    left, right, used = [], [], []
    for a, b in segments:
        if b - a <= length + 1:
            continue
        h = np.lib.stride_tricks.sliding_window_view(z[a:b], length + 1)
        left.append(h[:, :-1])
        right.append(h[:, 1:])
        used.append((int(a), int(b)))
    if not left:
        return np.empty((0, length), complex), np.empty((0, length), complex), used
    return np.vstack(left), np.vstack(right), used


def _poles(sequence, segments, pencil_length, max_rank, relative_threshold,
           fs, max_beat_hz, fixed_rank=None):
    eligible = [(a, b) for a, b in segments if b - a >= 16]
    if not eligible:
        return np.empty(0, complex), {}
    # A single extremely short clean piece does not shrink the useful pencils
    # of all other segments. Segments too short for this L are excluded only
    # from pole estimation; all eligible observations still fit amplitudes.
    longest = max(b - a for a, b in eligible)
    length = min(int(pencil_length), max(2, longest // 3))
    x0, x1, used = stacked_hankel_pencil(sequence, eligible, length)
    if x0.shape[0] < 4:
        return np.empty(0, complex), {}
    u0, s0, v0h = svd(x0, full_matrices=False, check_finite=False,
                      overwrite_a=True, lapack_driver='gesdd')
    u1, s1, v1h = svd(x1, full_matrices=False, check_finite=False,
                      overwrite_a=True, lapack_driver='gesdd')
    if not s0.size or s0[0] <= np.finfo(float).tiny:
        return np.empty(0, complex), {}
    count = int(np.count_nonzero(s0 >= s0[0] * relative_threshold))
    rank = min(int(fixed_rank) if fixed_rank is not None else count,
               int(max_rank), length - 1, x0.shape[0] - 1)
    if rank < 1:
        return np.empty(0, complex), {}
    # Eq. (24): BOTH matrices' own rank-r SVDs, including V1^H V0.
    core = ((u0[:, :rank].conj().T @ u1[:, :rank]) * s1[:rank][None, :])
    core = (core @ (v1h[:rank] @ v0h[:rank].conj().T)) / s0[:rank, None]
    poles = np.linalg.eigvals(core)
    radius = abs(poles)
    freq = np.angle(poles) * fs / (2 * np.pi)
    keep = np.isfinite(poles) & (radius > 0.) & (abs(radius - 1.) < .03)
    keep &= abs(freq) <= max_beat_hz
    poles = poles[keep] / radius[keep]
    # Collapse numerically duplicate poles before a potentially singular LS.
    order = np.argsort(np.angle(poles))
    poles = poles[order]
    if poles.size > 1:
        separation = abs(np.angle(poles[1:] / poles[:-1])) * fs / (2 * np.pi)
        poles = poles[np.r_[True, separation >= fs / len(sequence) * .05]]
    return poles, dict(rank=int(rank), accepted_poles=int(poles.size),
                       pencil_length=length, hankel_shape=list(x0.shape),
                       hankel_segments=used,
                       singular_values_relative=(s0[:max_rank + 3] / s0[0]).tolist(),
                       model_order_strategy='relative_singular_threshold_not_SAMOS',
                       pole_solver='two_TSVDs_equation_24')


def _basis(poles, indices, fs, envelope, duration, slope, simulation_fs):
    indices = np.asarray(indices, dtype=float)
    frequencies = np.angle(poles) * fs / (2 * np.pi)
    phase = np.exp(2j * np.pi * indices[:, None] / fs * frequencies[None, :])
    if envelope == 'constant':
        return phase
    t = indices / fs
    delays = np.maximum(frequencies, 0.) / slope
    amplitude = _blackman(t, duration, simulation_fs)[:, None] * _blackman(
        t[:, None] - delays[None, :], duration, simulation_fs)
    return amplitude * phase


def reconstruct(x, fs, max_beat_hz, mask=None, envelope='blackman',
                pencil_length=64, max_rank=12, max_iterations=4,
                relative_threshold=.01, chirp_duration_s=None,
                chirp_slope_hz_per_s=4e12, simulation_fs=2e9,
                normalization_floor=.02, return_diagnostics=False):
    """Return a complex observed-mask-only MP reconstruction.

    A supplied mask must be measured/observed, not a simulator truth mask.
    mask=None invokes this project's observation-only STFT/CFAR detector.
    Complexity per fusion pass is O(N L^2 + L^3), memory O(N L).
    Failure or insufficient clean observations leaves the input unchanged.
    """
    values = np.asarray(x, dtype=np.complex128)
    if values.ndim != 1 or values.size < 16 or not np.isfinite(values).all():
        raise ValueError('x must be finite complex 1D with length >=16')
    if not np.isfinite(fs) or fs <= 0 or not 0 < max_beat_hz < fs / 2:
        raise ValueError('require fs>0 and 0<max_beat_hz<fs/2')
    if envelope not in ('constant', 'blackman'):
        raise ValueError('envelope must be constant or blackman')
    if not 0 < relative_threshold < 1 or not 0 < normalization_floor < 1:
        raise ValueError('threshold and normalization_floor must lie in (0,1)')
    budgets = []
    for name, value, minimum in (('pencil_length', pencil_length, 3),
                                  ('max_rank', max_rank, 1),
                                  ('max_iterations', max_iterations, 1)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < minimum or int(value) != value):
            raise ValueError(f'{name} must be an integer >= {minimum}')
        budgets.append(int(value))
    pencil_length, max_rank, max_iterations = budgets
    if not np.isfinite(simulation_fs) or simulation_fs <= 0:
        raise ValueError('simulation_fs must be finite and positive')
    duration = values.size / fs if chirp_duration_s is None else float(chirp_duration_s)
    if not np.isfinite(duration) or duration <= 1. / simulation_fs:
        raise ValueError('public chirp duration must be finite and exceed one simulation sample')
    if not np.isfinite(chirp_slope_hz_per_s) or chirp_slope_hz_per_s <= 0:
        raise ValueError('public chirp slope must be finite and positive')
    if mask is None:
        from .methods import detect_interference
        mask = detect_interference(values, fs, max_beat_hz)
    mask = np.asarray(mask, bool)
    if mask.shape != values.shape:
        raise ValueError('mask must have x shape')
    output = values.copy()
    info = dict(method=METHOD, version=MP_VERSION, envelope=envelope,
                masked_samples=int(mask.sum()), written_samples=0,
                independent_validation=False, frequencies_initialized_by_fft=False,
                model_order_strategy='relative_singular_threshold_not_SAMOS')

    def finish(status):
        info['status'] = status
        return (output, info) if return_diagnostics else output

    if not mask.any():
        return finish('no_detected_mask')
    common = np.ones(values.size)
    if envelope == 'blackman':
        common = _blackman(np.arange(values.size) / fs, duration, simulation_fs) ** 2
    stable = common >= (normalization_floor if envelope == 'blackman' else 0.)
    observed = (~mask) & stable
    indices = np.flatnonzero(observed)
    if indices.size < max(48, 4 * int(max_rank)):
        return finish('insufficient_observations')
    scale = max(float(np.sqrt(np.mean(abs(values[observed]) ** 2))), np.finfo(float).tiny)
    normalized = np.zeros(values.size, complex)
    normalized[observed] = values[observed] / (common[observed] * scale)
    measured = values / scale
    segments = _segments(observed)
    info.update(observed_samples=int(indices.size),
                initial_observed_segments=[list(q) for q in segments])
    best_error, best_prediction, history = np.inf, None, []
    fixed_rank = None
    for iteration in range(int(max_iterations)):
        try:
            poles, diagnostics = _poles(normalized, segments, pencil_length,
                                        max_rank, relative_threshold, fs,
                                        max_beat_hz, fixed_rank)
            if not poles.size:
                break
            design = _basis(poles, indices, fs, envelope, duration,
                            chirp_slope_hz_per_s, simulation_fs)
            amplitudes, _, _, _ = np.linalg.lstsq(design, measured[indices], rcond=1e-8)
            predicted = _basis(poles, np.arange(values.size), fs, envelope,
                               duration, chirp_slope_hz_per_s, simulation_fs) @ amplitudes
            if not np.isfinite(predicted).all():
                break
            # Eq. (29), extended to any number of observed segments. Complex
            # residuals retain I/Q phase; they are evaluated before restoring.
            error = float(sum(np.linalg.norm(predicted[a:b] - measured[a:b])
                              for a, b in _segments(observed)))
            history.append(dict(iteration=iteration, observed_segment_error=error,
                                frequencies_hz=(np.angle(poles) * fs / (2 * np.pi)).tolist(),
                                **diagnostics))
            if error > best_error * (1 + 1e-10):
                break
            best_error, best_prediction = error, predicted.copy()
            info.update(diagnostics)
            info['frequencies_hz'] = history[-1]['frequencies_hz']
            info['complex_amplitudes'] = [[float(a.real * scale), float(a.imag * scale)]
                                          for a in amplitudes]
            fixed_rank = diagnostics['rank']
            # Eq. (28): fill missing samples, restore measured observations,
            # THEN re-estimate the ordinary continuous-data matrix pencil.
            normalized[stable] = predicted[stable] / common[stable]
            normalized[observed] = measured[observed] / common[observed]
            segments = _segments(stable)
            if error <= 1e-10 * max(float(np.linalg.norm(measured[observed])), 1e-30):
                break
        except (ValueError, np.linalg.LinAlgError):
            break
    info['fusion_history'] = history
    if best_prediction is None:
        return finish('no_supported_pencil_model')
    # Safeguard extreme extrapolation/model failure using ONLY observations.
    max_observed = float(np.max(abs(measured[observed])))
    if np.max(abs(best_prediction[mask])) > max(max_observed * 20., 1e-30):
        return finish('unstable_extrapolation')
    output[mask] = best_prediction[mask] * scale
    info.update(written_samples=int(mask.sum()), observed_segment_error=best_error,
                fusion_iterations=len(history),
                limitation='open-loop recovery; mask and constant/envelope model may be wrong')
    return finish('open_loop_fill')
