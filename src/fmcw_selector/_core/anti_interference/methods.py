"""Observed-signal-only CFAR-Z and CFAR-AC for complex FMCW ADC data.

Independent implementation of Wang, IEEE T-ITS 23(8), 2022,
doi:10.1109/TITS.2021.3111514; arXiv:2101.01257, Algorithm 1 / Eq. (7).
The mitigation steps follow that work; detection is a declared local OS-CFAR
adaptation for multiple crossing sources, with public Blackman-LO variance
compensation and frequency support filtering. See sources.json.
No clean signal, interference components, source clocks, or truth masks enter
these algorithms. A TF detection mask is used after ADC acquisition; it does
not replace the simulated receiver's low-pass filter.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage, optimize, signal


DETECTOR_VERSION = "os_cfar_blackman_lo_frequency_support_v2"
METHOD_PARAMETERS = {
    "detector_version": DETECTOR_VERSION,
    "background_estimator": "order_statistic",
    "order_statistic_rank_fraction": 0.5,
    "threshold_calibration": "independent_exponential_OS_product_equation",
    "detection_lo_taper": "blackman",
    "variance_compensation": "Hann_squared_weighted_Blackman_LO_squared",
    "variance_floor": 1e-9,
    "minimum_frequency_run_cells": 4,
    "time_mask_projection": "detected_frame_center_hop_cells",
    "window": "hann_periodic",
    "window_samples": 256,
    "hop_samples": 32,
    "training_cells_each_side": 16,
    "guard_cells_each_side": 4,
    "probability_false_alarm": 1e-4,
    "dilation_radius_cells": 2,
    "stft_boundary": "zeros",
    "stft_padded": True,
}


@dataclass(frozen=True)
class _STFTContext:
    length: int
    window_samples: int
    hop_samples: int


def _validate(x: np.ndarray, fs: float, max_beat_hz: float) -> np.ndarray:
    values = np.asarray(x, dtype=np.complex128)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("x must be a finite one-dimensional complex signal")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive")
    if (not np.isfinite(max_beat_hz) or max_beat_hz <= 0
            or max_beat_hz > fs / 2):
        raise ValueError("max_beat_hz must be in (0, fs/2]")
    return values


def _stft(x: np.ndarray, fs: float):
    """Two-sided STFT; retained context allows exact length restoration."""
    n = len(x)
    if n == 0:
        return np.empty(0), np.empty(0), np.empty((0, 0), complex), _STFTContext(0, 8, 1)
    window_n = min(256, max(8, 2 ** int(np.floor(np.log2(n)))))
    hop = min(32, max(1, window_n // 8))
    values = np.pad(x, (0, max(0, window_n - n)))
    frequencies, times, spectrum = signal.stft(
        values, fs=fs, window="hann", nperseg=window_n,
        noverlap=window_n-hop, nfft=window_n, return_onesided=False,
        boundary="zeros", padded=True,
    )
    return frequencies, times, spectrum, _STFTContext(n, window_n, hop)


def _istft(spectrum: np.ndarray, fs: float, context: _STFTContext) -> np.ndarray:
    if context.length == 0:
        return np.empty(0, dtype=complex)
    _, restored = signal.istft(
        spectrum, fs=fs, window="hann", nperseg=context.window_samples,
        noverlap=context.window_samples-context.hop_samples,
        nfft=context.window_samples, input_onesided=False, boundary=True,
    )
    return np.asarray(restored[:context.length], dtype=np.complex128)


@lru_cache(maxsize=256)
def _os_threshold_factor(count: int, rank: int, pfa: float) -> float:
    """Solve Pfa = product(i/(i+alpha)), i=N-k+1..N.

    This calibration assumes independent exponential powers, not correlated
    STFT coefficients. The configured Pfa is nominal; empirical errors are
    checked separately. Rank is one-based and edges use their actual count.
    """
    if not 1 <= rank <= count or not 0 < pfa < 1:
        raise ValueError("OS-CFAR requires 1 <= rank <= count and 0 < pfa < 1")
    indices = np.arange(count-rank+1, count+1, dtype=float)
    target = np.log(pfa)

    def objective(alpha):
        return -np.log1p(alpha/indices).sum()-target

    high = 1.
    while objective(high) > 0:
        high *= 2.
    return float(optimize.brentq(objective, 0., high))


def _lo_variance_profile(times, context: _STFTContext, fs: float) -> np.ndarray:
    """Public Blackman-LO noise variance under each finite Hann window.

    The receiver mixes RF noise with a Blackman-tapered LO. Use the known
    taper and window support, never noise/target truth or measured SNR. This
    is an approximate variance trend after the receiver FIR, not NF fitting.
    It changes detection statistics only; the acquired spectrum stays intact.
    """
    if context.length == 0:
        return np.ones(len(times))
    window = signal.get_window("hann", context.window_samples)
    lo = np.maximum(np.blackman(context.length), 0.)
    centers = np.rint(np.asarray(times)*fs).astype(int)
    offsets = np.arange(context.window_samples)-context.window_samples//2
    indices = centers[:, None]+offsets[None, :]
    valid = (indices >= 0) & (indices < context.length)
    power = lo[np.clip(indices, 0, context.length-1)]**2*valid
    return (power*window[None, :]**2).sum(axis=1)/np.sum(window**2)


def _cfar_mask(spectrum: np.ndarray, *, variance_profile=None) -> np.ndarray:
    """OS-CFAR along time in every frequency row; no time/Nyquist wrap.

    A uniform profile is available for stationary-noise statistics. The project
    acquisition path supplies its declared public Blackman-LO profile.
    Require a four-bin frequency run before dilation: isolated noise outliers
    must not become a full ADC-window erasure. Parallel/narrowband interference
    can remain ambiguous with targets and is not guaranteed to be detected.
    """
    if spectrum.size == 0:
        return np.zeros(spectrum.shape, dtype=bool)
    power = np.abs(spectrum) ** 2
    if variance_profile is not None:
        variance_profile = np.asarray(variance_profile, float)
        if (variance_profile.shape != (power.shape[1],)
                or not np.isfinite(variance_profile).all()
                or np.any(variance_profile < 0)):
            raise ValueError("variance_profile must contain one nonnegative value per frame")
        power = power/np.maximum(variance_profile[None, :], METHOD_PARAMETERS["variance_floor"])
    n_time = power.shape[1]
    cell = np.arange(n_time)
    training = METHOD_PARAMETERS["training_cells_each_side"]
    guard = METHOD_PARAMETERS["guard_cells_each_side"]
    offsets = np.r_[np.arange(-guard-training, -guard),
                    np.arange(guard+1, guard+training+1)]
    indices = cell[:, None]+offsets[None, :]
    valid = (indices >= 0) & (indices < n_time)
    count = valid.sum(axis=1)
    ranks = np.maximum(1, np.ceil(METHOD_PARAMETERS["order_statistic_rank_fraction"]*count).astype(int))
    references = power[:, np.clip(indices, 0, n_time-1)]
    references = np.where(valid[None, :, :], references, np.inf)
    ordered = np.sort(references, axis=2)
    background = ordered[:, cell, ranks-1]
    pfa = METHOD_PARAMETERS["probability_false_alarm"]
    alpha = np.asarray([_os_threshold_factor(max(int(n), 1), int(k), pfa)
                        for n, k in zip(count, ranks)])
    # Numerical floor avoids false detections in zero/roundoff-only TF rows.
    floor = np.max(power, axis=1, keepdims=True)*1e-12 + np.max(power)*1e-14
    detected = (power > background*alpha[None, :]+floor) & (count[None, :] >= 4)
    # Interference can occupy any acquired ADC frequency, including outside
    # the expected target beat band. Detect all two-sided STFT frequency rows.
    radius = METHOD_PARAMETERS["dilation_radius_cells"]
    q = np.arange(-radius, radius+1)
    disk = q[:, None]**2+q[None, :]**2 <= radius**2
    # fftshift puts bins on the signed frequency axis and joins the DC
    # neighborhood across the original FFT seam. No wrap at +/-Nyquist.
    centered = np.fft.fftshift(detected, axes=0)
    centered = ndimage.binary_opening(centered, structure=np.ones(
        (METHOD_PARAMETERS["minimum_frequency_run_cells"], 1), dtype=bool))
    centered = ndimage.binary_dilation(centered, structure=disk)
    return np.fft.ifftshift(centered, axes=0)


def _analysis(x: np.ndarray, fs: float, max_beat_hz: float):
    values = _validate(x, fs, max_beat_hz)
    scale = float(np.max(np.abs(values))) if values.size else 0.
    if not np.isfinite(scale):
        raise ValueError("x magnitude exceeds floating-point range")
    normalized = values/scale if scale else values.copy()
    _, times, spectrum, context = _stft(normalized, fs)
    variance = _lo_variance_profile(times, context, fs)
    mask = _cfar_mask(spectrum, variance_profile=variance)
    return values, scale, times, spectrum, context, mask


def run_methods(x: np.ndarray, fs: float, max_beat_hz: float) -> dict[str, np.ndarray]:
    """Return none, cfar_zero, cfar_ac; each output has x's length and IQ.

    The two mitigation methods share one STFT and detection pass. CFAR-Z
    removes flagged TF coefficients. CFAR-AC keeps their observed phases and
    replaces magnitudes by the mean of unflagged magnitudes in the same row.
    If a complete frequency row is flagged, no uncontaminated estimate is
    available, so that row is zeroed rather than filled with NaNs.
    max_beat_hz validates the public target-band prior for interface
    compatibility; it does not restrict interference detection to that band.
    """
    values, scale, _, spectrum, context, mask = _analysis(x, fs, max_beat_hz)
    if not np.any(mask):
        return {name: values.copy() for name in ("none", "cfar_zero", "cfar_ac")}
    zeroed = spectrum.copy()
    zeroed[mask] = 0.
    magnitudes = np.abs(spectrum)
    unflagged_count = (~mask).sum(axis=1)
    mean_amplitude = (magnitudes*(~mask)).sum(axis=1) / np.maximum(unflagged_count, 1)
    corrected = spectrum.copy()
    phase = np.divide(spectrum, magnitudes, out=np.zeros_like(spectrum), where=magnitudes > 0)
    corrected[mask] = (mean_amplitude[:, None]*phase)[mask]
    return {
        "none": values.copy(),
        "cfar_zero": _istft(zeroed, fs, context)*scale,
        "cfar_ac": _istft(corrected, fs, context)*scale,
    }


def detect_interference(x: np.ndarray, fs: float, max_beat_hz: float) -> np.ndarray:
    """Local time estimate from detected TF frame centers, solely from x.

    Each flagged frame marks its central hop cell (Voronoi time bin), not its
    entire Hann window. Existing TF dilation expands neighboring frames before
    this projection. Full-window projection made multi-source erasures overlap
    and removed useful observations from the matrix pencil. This estimator can
    miss TF/window tails; it is not an LPF gate or a truth-presence mask.
    """
    values, _, times, _, context, mask = _analysis(x, fs, max_beat_hz)
    support = np.zeros(len(values)+1, dtype=np.int64)
    if values.size and mask.size:
        centers = np.rint(times[np.any(mask, axis=0)]*fs).astype(int)
        left = context.hop_samples//2
        starts = np.clip(centers-left, 0, len(values))
        ends = np.clip(centers+context.hop_samples-left, 0, len(values))
        np.add.at(support, starts, 1)
        np.add.at(support, ends, -1)
    return np.cumsum(support[:-1]) > 0
