"""Offline output quality and resolved-reference component fidelity.

The primary SINR/NMSE implementation is imported unchanged from the previously
frozen experiment.  All added quantities are *offline diagnostics/labels*.
Clean targets, noise, and any truth record must never enter feature extraction,
method execution, or selector prediction.  Inference modules must not import
this module.

The added fit compares complex coefficients, rather than spectral magnitudes
alone.  A different-phase/incoherent nuisance at a target frequency therefore
does not count as faithful recovery.  A component exactly coherent with an
actual target is fundamentally indistinguishable from that target in one
channel; no claimed physical target/interference decomposition is made here.

Use ``prepare_reference`` once per chirp and reuse it for every candidate.
Definitions are fixed in ``definition()``; they must be frozen before holdout.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

from . import sinr_metrics as _legacy
score_one = _legacy.score_one
METRIC_NAMES = _legacy.METRIC_NAMES
METRIC_VERSION = _legacy.METRIC_VERSION

FIDELITY_VERSION = "resolved_reference_joint_complex_fidelity_v1"
REFERENCE_MIN_AMPLITUDE_FRACTION = 0.01  # -40 dB amplitude from strongest.
MINIMUM_SEPARATION_NATIVE_BINS = 4.0
WEAK_AMPLITUDE_RATIO = 10.0 ** (-6.0 / 20.0)
MAXIMUM_COMPONENTS = 12
CONDITION_LIMIT = 1.0e6
REFERENCE_FIT_RESIDUAL_LIMIT = 0.05
SEVERE_COMPLEX_RELATIVE_ERROR = 1.0
ATTENUATION_THRESHOLD_DB = -20.0


def definition() -> dict[str, Any]:
    """Serializable, predeclared definition; dB means amplitude dB if stated."""
    return dict(
        version=FIDELITY_VERSION,
        primary=_legacy.definition(),
        truth_scope="Offline scoring and supervision only; never inference",
        support="Local peaks of FFT(clean * symmetric Hann), >= -40 dB amplitude from strongest; full signed complex spectrum",
        initial_fft_oversampling=8,
        minimum_frequency_separation_native_bins=MINIMUM_SEPARATION_NATIVE_BINS,
        unresolved_policy="Within 4 native bins keep strongest reference spectral peak; report excluded count; these are resolved spectral groups, not physical target counts",
        maximum_components=MAXIMUM_COMPONENTS,
        continuous_refinement="Two coordinate passes, ±0.75 native bin around each starting peak; subtract other fitted clean components; fit only clean",
        atom="Public TX Blackman(t-f/K) times LO Blackman(t), with center-referenced complex sinusoid; rectangular envelope if explicitly requested or public taper parameters unavailable",
        fir_policy="Reference and output jointly projected onto identical basis; fitted clean residual reported, no physical transfer-function claim",
        coefficient_fit="Unweighted complex least squares across all ADC samples, all supported components fitted jointly; no output/reference gain or phase alignment",
        weak_definition="Clean fitted component amplitude <= -6 dB from strongest, after >= -40 dB support admission",
        amplitude_relative_error="abs(a_y-a_clean)/abs(a_clean), dimensionless complex error",
        phase_error="abs(angle(a_y * conj(a_clean))), radians, meaningful retention subset only if amplitude ratio > -20 dB",
        attenuation="20log10(abs(a_y)/abs(a_clean)), amplitude dB",
        severe_weak_distortion="Any weak support has complex relative coefficient error >=1 or amplitude ratio <=-20 dB; valid reference fit required",
        fit_validity=dict(condition_number_limit=CONDITION_LIMIT,
                          maximum_clean_fit_residual_fraction=REFERENCE_FIT_RESIDUAL_LIMIT),
        invalid_label_policy="risk label unavailable, not a successful recovery; expose fidelity_available=false and counts",
        reference_error_limit="Coherent interference identical to target cannot be distinguished; this is coefficient fidelity, not isolated target transfer gain",
        legacy_secondary="Existing N-point symmetric Hann peak rule >=1% strongest, distance3; exclude strongest, report magnitude attenuation separately",
        secondary_warning="Magnitude-only FFT-bin diagnostic can be filled by residual interference and is not target detection probability",
        no_physical_pd_or_false_alarm_claim=True,
    )


def _signal(a: Any, name: str) -> np.ndarray:
    out = np.asarray(a, dtype=np.complex128)
    if out.ndim != 1 or not out.size or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a nonempty finite complex chirp")
    return out


def _blackman(t: np.ndarray, duration: float, sim_fs: float) -> np.ndarray:
    last = (int(round(duration * sim_fs)) - 1) / sim_fs
    if last <= 0:
        raise ValueError("Taper duration contains fewer than two high-rate samples")
    q = t / last
    return np.where((q >= 0) & (q <= 1),
                    .42 - .5 * np.cos(2 * np.pi * q) + .08 * np.cos(4 * np.pi * q), 0.)


def _tone_factory(public: dict[str, Any], n: int):
    fs = float(public["sample_rate_hz"])
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Public sample_rate_hz must be finite and positive")
    t = np.arange(n) / fs
    tc = t - (n - 1) / (2 * fs)
    explicit = public.get("target_envelope")
    if explicit not in (None, "blackman_product", "rectangular"):
        raise ValueError("Unknown target_envelope")
    tapered = explicit == "blackman_product" or (explicit is None and all(
        key in public for key in ("chirp_duration_s", "chirp_slope_hz_per_s", "simulation_sample_rate_hz")))
    if tapered:
        duration = float(public["chirp_duration_s"])
        slope = float(public["chirp_slope_hz_per_s"])
        sim_fs = float(public["simulation_sample_rate_hz"])
        if not np.isfinite([duration, slope, sim_fs]).all() or min(duration, slope, sim_fs) <= 0:
            raise ValueError("Public waveform parameters must be positive and finite")
        lo = _blackman(t, duration, sim_fs)

        def atom(f):
            env = lo * _blackman(t - f / slope, duration, sim_fs)
            return env * np.exp(2j * np.pi * f * tc)
    else:
        def atom(f):
            return np.exp(2j * np.pi * f * tc)
    return atom, fs, "blackman_product" if tapered else "rectangular"


def _circular_distance(a: float, b: float, period: float) -> float:
    d = abs(a - b) % period
    return min(d, period - d)


@dataclass
class Reference:
    clean: np.ndarray
    public: dict[str, Any]
    frequencies_hz: np.ndarray
    design: np.ndarray
    projector: np.ndarray
    clean_coefficients: np.ndarray
    weak: np.ndarray
    condition: float
    clean_fit_residual_fraction: float
    excluded_unresolved_count: int
    excluded_cap_count: int
    envelope_mode: str
    secondary_bins: np.ndarray
    clean_fft: np.ndarray
    window: np.ndarray
    noise_power_w: float
    available: bool


def prepare_reference(clean, public, noise=None, truth_record=None) -> Reference:
    """Prepare offline clean-derived resolved support, once per ADC chirp.

    ``truth_record`` is accepted for the API contract but deliberately unused:
    support construction depends only on clean and public receiver parameters.
    Noise, if supplied, contributes only its scalar measured power.
    """
    c = _signal(clean, "clean")
    pc = float(np.mean(abs(c) ** 2))
    if pc <= 0:
        raise ValueError("A positive-energy clean reference is required")
    public = dict(public)
    atom, fs, mode = _tone_factory(public, len(c))
    n = len(c)
    df = fs / n
    window = np.hanning(n)
    clean_fft = np.fft.fft(c * window)
    peak = int(np.argmax(abs(clean_fft)))
    oldpeaks, _ = find_peaks(abs(clean_fft), height=abs(clean_fft[peak]) * .01, distance=3)
    secondary = oldpeaks[oldpeaks != peak]
    nfft = 1 << int(np.ceil(np.log2(max(32, n * 8))))
    amp = abs(np.fft.fft(c * window, nfft))
    frequencies = np.fft.fftfreq(nfft, 1 / fs)
    # Circular neighbours include DC and negative-Nyquist peaks correctly.
    local = (amp > np.roll(amp, 1)) & (amp >= np.roll(amp, -1))
    keep = local & (amp >= amp.max() * REFERENCE_MIN_AMPLITUDE_FRACTION)
    band = min(float(public.get("max_beat_hz", fs / 2)), fs / 2)
    keep &= abs(frequencies) <= band
    ordered = sorted(np.flatnonzero(keep), key=lambda j: (-amp[j], int(j)))
    selected = []
    unresolved = capped = 0
    for index in ordered:
        f = float(frequencies[index])
        if any(_circular_distance(f, other, fs) < MINIMUM_SEPARATION_NATIVE_BINS * df for other in selected):
            unresolved += 1
        elif len(selected) >= MAXIMUM_COMPONENTS:
            capped += 1
        else:
            selected.append(f)
    if not selected:
        selected = [float(frequencies[int(np.argmax(amp))])]
    starts = np.asarray(selected, dtype=float)
    freqs = starts.copy()
    a = np.column_stack([atom(f) for f in freqs])
    cnorm = float(np.linalg.norm(c))
    z = c / cnorm
    coeff = np.linalg.lstsq(a, z, rcond=1e-10)[0]
    # Jointly account for neighbouring admitted supports during refinement.
    for _ in range(2):
        for j in range(len(freqs)):
            residual = z - a @ coeff + a[:, j] * coeff[j]

            def objective(f):
                v = atom(f)
                den = max(float(np.vdot(v, v).real), 1e-300)
                return -float(abs(np.vdot(v, residual)) ** 2 / den)

            fit = minimize_scalar(objective,
                bounds=(starts[j] - .75 * df, starts[j] + .75 * df),
                method="bounded", options=dict(xatol=max(1e-4, df * 1e-6), maxiter=30))
            freqs[j] = float(fit.x)
            a[:, j] = atom(freqs[j])
            coeff = np.linalg.lstsq(a, z, rcond=1e-10)[0]
    norm = np.linalg.norm(a, axis=0)
    norm[norm <= 1e-150] = 1.
    design = a / norm
    condition = float(np.linalg.cond(design))
    if not np.isfinite(condition):
        condition = np.finfo(float).max
    projector = np.linalg.pinv(design, rcond=1 / CONDITION_LIMIT)
    clean_coefficients = projector @ c
    fit_error = float(np.mean(abs(c - design @ clean_coefficients) ** 2) / pc)
    amplitudes = abs(clean_coefficients)
    weak = amplitudes <= amplitudes.max() * WEAK_AMPLITUDE_RATIO
    noise_power = 0.
    if noise is not None:
        noise_array = _signal(noise, "noise")
        if noise_array.shape != c.shape:
            raise ValueError("noise and clean shapes differ")
        noise_power = float(np.mean(abs(noise_array) ** 2))
    available = condition < CONDITION_LIMIT and fit_error <= REFERENCE_FIT_RESIDUAL_LIMIT
    return Reference(c.copy(), public, freqs, design, projector, clean_coefficients,
                     weak, condition, fit_error, unresolved, capped, mode,
                     secondary, clean_fft, window, noise_power, bool(available))


def _mean(a: np.ndarray) -> float:
    return float(np.mean(a)) if a.size else 0.


def score_prepared(y, x, ref: Reference) -> dict[str, Any]:
    """Score one method after inference, sharing an offline reference basis."""
    y, x = _signal(y, "output"), _signal(x, "observed")
    if y.shape != ref.clean.shape or x.shape != y.shape:
        raise ValueError("All scored chirps must have matching length")
    result = score_one(y, ref.clean, x)
    ay = ref.projector @ y
    ac = ref.clean_coefficients
    floor = max(float(abs(ac).max()) * 1e-12, 1e-300)
    relative = abs(ay - ac) / np.maximum(abs(ac), floor)
    ratio = abs(ay) / np.maximum(abs(ac), floor)
    amplitude_db = 20 * np.log10(np.maximum(ratio, 1e-15))
    phase = abs(np.angle(ay * ac.conjugate()))
    phase_valid = amplitude_db > ATTENUATION_THRESHOLD_DB
    weak = ref.weak
    weak_severe = (relative >= SEVERE_COMPLEX_RELATIVE_ERROR) | (amplitude_db <= ATTENUATION_THRESHOLD_DB)
    clean_power = float(np.mean(abs(ref.clean) ** 2))
    residual = y - ref.design @ ay
    # Compatible historical diagnostic, explicitly magnitude-only.
    yf = np.fft.fft(y * ref.window)
    bins = ref.secondary_bins
    secondary_db = 20 * np.log10((abs(yf[bins]) + 1e-150) / (abs(ref.clean_fft[bins]) + 1e-150))
    result.update(
        fidelity_version=FIDELITY_VERSION,
        fidelity_available=ref.available,
        fidelity_risk_label_available=bool(ref.available and np.any(weak)),
        target_support_count=int(len(ac)), weak_support_count=int(weak.sum()),
        target_support_frequencies_hz=ref.frequencies_hz.tolist(),
        target_fit_condition_number=ref.condition,
        clean_reference_fit_residual_fraction=ref.clean_fit_residual_fraction,
        unresolved_reference_peak_count=ref.excluded_unresolved_count,
        capped_reference_peak_count=ref.excluded_cap_count,
        target_envelope_mode=ref.envelope_mode,
        target_joint_amp_rel_rmse=float(np.sqrt(_mean(relative ** 2))),
        target_joint_phase_mae_rad=_mean(phase[phase_valid]),
        target_phase_available_count=int(phase_valid.sum()),
        weak_joint_amp_rel_rmse=float(np.sqrt(_mean(relative[weak] ** 2))),
        weak_joint_complex_error_mean=_mean(relative[weak]),
        weak_joint_complex_error_max=float(relative[weak].max()) if weak.any() else 0.,
        weak_joint_phase_mae_rad=_mean(phase[weak & phase_valid]),
        weak_phase_available_count=int((weak & phase_valid).sum()),
        weak_attenuation_fraction_20db=_mean((amplitude_db[weak] <= -20).astype(float)),
        weak_attenuation_fraction_6db=_mean((amplitude_db[weak] <= -6).astype(float)),
        weak_complex_distortion_fraction=_mean(weak_severe[weak].astype(float)),
        weak_attenuation_count_20db=int(np.sum(amplitude_db[weak] <= -20)),
        weak_complex_distortion_count=int(np.sum(weak_severe[weak])),
        severe_weak_distortion=bool(ref.available and np.any(weak_severe[weak])),
        target_fit_residual_fraction=float(np.mean(abs(residual) ** 2) / clean_power),
        target_coefficient_relative_error=relative.tolist(),
        target_coefficient_amplitude_ratio_db=amplitude_db.tolist(),
        target_coefficient_phase_error_rad=phase.tolist(),
        target_coefficient_is_weak=weak.tolist(),
        secondary_reference_peak_count=int(len(bins)),
        secondary_reference_peak_attenuation_20db_count=int(np.sum(secondary_db <= -20)),
        secondary_reference_peak_attenuation_6db_count=int(np.sum(secondary_db <= -6)),
        secondary_reference_peak_amplitude_error_db=secondary_db.tolist(),
        secondary_reference_peaks=[int(len(bins)), int(np.sum(secondary_db <= -20))],
        measured_noise_power_w=ref.noise_power_w,
        metric_truth_scope="Offline scoring/supervision; not an inference feature",
    )
    # Explicit finiteness contract: invalid basis is flagged, never NaN-coded.
    for key, value in result.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite offline score: {key}")
    return result


def score_output(y, clean, x, public, truth_record=None, noise=None) -> dict[str, Any]:
    """Convenience path; prefer prepared references for method banks."""
    return score_prepared(y, x, prepare_reference(clean, public, noise, truth_record))
