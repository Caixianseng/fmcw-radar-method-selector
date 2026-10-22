"""Offline reference-error SINR, explicitly following Wang et al. (2022), Eq. 9.

This is a fixed-reference output quality measure, not a physical decomposition
of a nonlinear algorithm's output into separately processed signal and noise.
The denominator includes retained interference, noise, and target distortion.
No gain/phase alignment, projection, mean removal, or extra time window is used
for the primary SINR. The relative 1e-12 denominator guard caps exact recovery
at 120 dB. Auxiliary range-spectrum metrics use the existing symmetric Hann.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

METRIC_VERSION = 'reference_error_sinr_wang2022_eq9_v1'
LOSS_NAME = '-reference_error_sinr_db'
RELATIVE_POWER_GUARD = 1e-12
METRIC_NAMES = (
    'nmse', 'loss_db', 'sinr_db', 'input_sinr_db', 'sinr_gain_db',
    'strongest_target_bin_amplitude_error_db', 'range_spectrum_nmse',
)
PAPER_SOURCE = ('https://pure.tudelft.nl/ws/portalfiles/portal/154907760/'
                'CFAR_Based_Interference_Mitigation_for_FMCW_Automotive_Radar_Systems.pdf')


def definition():
    return dict(metric_version=METRIC_VERSION, loss_name=LOSS_NAME,
        primary_name='Reference-error SINR following Wang et al. (2022), Eq. (9)',
        formula_db='10*log10(mean(abs(clean)**2)/(mean(abs(output-clean)**2)+1e-12*mean(abs(clean)**2)))',
        denominator='Actual output minus the fixed noise-free target reference; includes remaining interference, noise and target distortion',
        alignment='No amplitude/phase fitting or scale alignment',
        domain='Complex ADC waveform, all 3750 samples, common physical scale',
        primary_window='None added for primary SINR',
        numerical_guard_relative_to_target_power=RELATIVE_POWER_GUARD,
        exact_recovery_ceiling_db=120., physical_component_sinr=False,
        reference='Wang (2022), CFAR-Based Interference Mitigation for FMCW Automotive Radar Systems, Eq. (9)',
        reference_url=PAPER_SOURCE,
        relation_to_nmse='SINR_db=-10*log10(NMSE+1e-12)',
        averaging='Arithmetic mean of per-chirp dB SINR; linear NMSE reported separately')


def _validate(output, clean, observed):
    y, c, x = (np.asarray(a, dtype=np.complex128) for a in (output, clean, observed))
    if y.ndim != 1 or not y.size or y.shape != c.shape or y.shape != x.shape:
        raise ValueError('SINR inputs must be matching, nonempty, one-dimensional chirps')
    if not all(np.isfinite(a).all() for a in (y, c, x)):
        raise ValueError('SINR inputs must be finite')
    pc = float(np.mean(np.abs(c)**2))
    if not np.isfinite(pc) or pc <= 0:
        raise ValueError('Cannot score zero or invalid target-reference energy')
    return y, c, x, pc


def score_one(output, clean, observed):
    """Score one actual method output after observed-only inference completes."""
    y, c, x, pc = _validate(output, clean, observed)
    pe = float(np.mean(np.abs(y-c)**2))
    p_input_error = float(np.mean(np.abs(x-c)**2))
    nmse = pe/pc
    input_nmse = p_input_error/pc
    loss = float(10*np.log10(nmse+RELATIVE_POWER_GUARD))
    input_loss = float(10*np.log10(input_nmse+RELATIVE_POWER_GUARD))
    win = np.hanning(len(c))
    cf, yf = np.fft.fft(c*win), np.fft.fft(y*win)
    peak = int(np.argmax(np.abs(cf)))
    amp_error = float(20*np.log10((abs(yf[peak])+1e-150)/(abs(cf[peak])+1e-150)))
    spec_nmse = float(np.mean((np.abs(yf)-np.abs(cf))**2)
        / max(float(np.mean(np.abs(cf)**2)), 1e-300))
    target_peaks, _ = find_peaks(abs(cf), height=abs(cf[peak])*.01, distance=3)
    target_bins = np.zeros(len(c), dtype=bool)
    for p in target_peaks:
        target_bins[max(0, p-3):min(len(c), p+4)] = True
    anomalies, _ = find_peaks(abs(yf), height=max(float(abs(cf[peak])*.1), 1e-150), distance=3)
    result = dict(nmse=nmse, loss_db=loss, sinr_db=-loss,
        input_sinr_db=-input_loss, sinr_gain_db=input_loss-loss,
        strongest_target_bin_amplitude_error_db=amp_error, range_spectrum_nmse=spec_nmse,
        peak_amplitude_error_db=amp_error,
        peak_phase_error_deg=float(np.angle(yf[peak]*cf[peak].conjugate())*180/np.pi),
        spectrum_anomaly_count=int(np.sum(~target_bins[anomalies])),
        unchanged_input=bool(np.array_equal(y, x)), target_power_w=pc,
        error_power_w=pe, input_error_power_w=p_input_error,
        metric_version=METRIC_VERSION, truth_used_for_scoring_only=True)
    if not all(np.isfinite(result[key]) for key in METRIC_NAMES):
        raise ValueError('SINR score is not finite')
    return result


def score_bank(outputs, clean, observed):
    """Return method-major numerical rows in the frozen METRIC_NAMES order."""
    ys = np.asarray(outputs, dtype=np.complex128)
    if ys.ndim != 2:
        raise ValueError('A method by sample waveform bank is required')
    return np.asarray([[r[key] for key in METRIC_NAMES]
                       for r in (score_one(y, clean, observed) for y in ys)], dtype=float)


def metric_records(outputs, clean, observed, method_names):
    ys = np.asarray(outputs, dtype=np.complex128)
    if len(ys) != len(method_names):
        raise ValueError('Method names and output waveforms differ')
    return {name: score_one(y, clean, observed) for name, y in zip(method_names, ys)}


def self_check():
    """Meaningful metric invariants: zero output, exact recovery, phase and units."""
    t = np.arange(64)
    c = np.exp(2j*np.pi*7*t/64) + .3*np.exp(2j*np.pi*11*t/64)
    nuisance = .2*np.exp(2j*np.pi*17*t/64)
    x = c+nuisance
    raw = score_one(x, c, x)
    perfect = score_one(c, c, x)
    zero = score_one(np.zeros_like(c), c, x)
    phase = score_one(-c, c, x)
    scaled_units = score_one(x*1e-7, c*1e-7, x*1e-7)
    np.testing.assert_allclose(raw['sinr_db'], raw['input_sinr_db'], atol=1e-12)
    np.testing.assert_allclose(raw['sinr_gain_db'], 0., atol=1e-12)
    np.testing.assert_allclose(perfect['sinr_db'], 120., atol=1e-12)
    np.testing.assert_allclose(zero['sinr_db'], -10*np.log10(1+1e-12), atol=1e-12)
    np.testing.assert_allclose(phase['sinr_db'], -10*np.log10(4+1e-12), atol=1e-12)
    np.testing.assert_allclose(scaled_units['sinr_db'], raw['sinr_db'], atol=1e-12)
    assert zero['nmse'] == 1 and phase['nmse'] == 4
    try:
        score_one(c, np.zeros_like(c), x)
    except ValueError:
        pass
    else:
        raise AssertionError('Zero target must be rejected')
    return dict(passed=True, checks=['raw_equals_input_and_zero_gain',
        'exact_recovery_has_explicit_120dB_guard', 'zero_output_is_0dB_not_infinite',
        'phase_inversion_is_penalized', 'physical_unit_scaling_invariant',
        'zero_target_reference_rejected'])
