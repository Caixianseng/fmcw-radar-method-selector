"""Observation-only slow-chirplet OMP and complex least-squares cancellation.

Paper inspiration: Correas-Serrano & Gonzalez-Huici, 'Sparse Reconstruction
of Chirplets for Automotive FMCW Radar Interference Mitigation', ICMIM 2019,
DOI 10.1109/ICMIM.2019.8726758, https://arxiv.org/abs/2106.05594 .
This is an independently written adaptation,
NOT a reproduction of its reduced two-parameter rectangular chirplet basis.
We restore arbitrary frequency and non-zero curvature; the public TX/LO
Blackman windows and receiver FIR are included. Parameters are inferred from
the ADC only. No source ID, true clock/slope/range, clean signal or label enters.

Model assumption: the slow interferer is an up-chirp with the same centered
transmit bandwidth as the victim. Ki=Kv-k, Ti=B/Ki, d=f_start/Ki. Doppler and
carrier offset are absorbed into frequency, so d is an *estimated envelope
shift*, not a measured physical clock. Coarse-to-fine dechirp FFT searches a
continuous generic curvature interval, not the six experiment truth slopes.
Tone-vs-chirplet evidence and nuisance tones fitted from residual observations
protect targets. Only estimated chirplet(s) are subtracted; nuisance target
fits are never emitted as a reconstructed signal. This standalone algorithm
uses only chirplet estimation and subtraction, with no reconstruction fallback.
"""
from __future__ import annotations

from functools import lru_cache
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import find_peaks, firwin, kaiserord, freqz, fftconvolve

VERSION = 'standalone_observed_continuous_chirplet_omp_blackman_fir_v1'
DEFAULT_SETTINGS = dict(
    max_rate_hz_per_s=8e10,
    min_rate_hz_per_s=1e9, coarse_step_hz_per_s=1e9,
    fft_oversampling=4, min_chirp_tone_power_ratio=1.35,
    min_explained_fraction=0.03, max_chirplets=2,
    max_nuisance_tones=8, tone_relative_power=0.0025,
    exact_final_fir=True, joint_refinement=True,
)


def _blackman(t, duration, sim_fs):
    last = (int(round(duration * sim_fs)) - 1) / sim_fs
    q = np.asarray(t) / last
    return np.where((q >= 0.) & (q <= 1.),
                    .42 - .5*np.cos(2*np.pi*q) + .08*np.cos(4*np.pi*q), 0.)


@lru_cache(maxsize=8)
def _fir_design(sim_fs, cutoff, stop_edge, attenuation):
    count, beta = kaiserord(attenuation, (stop_edge-cutoff)/(sim_fs/2))
    return firwin(max(3, int(count) | 1), (cutoff+stop_edge)/2,
                  window=('kaiser', beta), fs=sim_fs)


@lru_cache(maxsize=8)
def _fir_response(taps_tuple, sim_fs):
    taps = np.asarray(taps_tuple)
    w, h = freqz(taps, worN=131072, fs=sim_fs)
    real = np.real(h*np.exp(2j*np.pi*w*((len(taps)-1)/2)/sim_fs))
    return w, real


def _public(public, n):
    fs = float(public['sample_rate_hz'])
    duration = float(public.get('chirp_duration_s', n/fs))
    slope = float(public['chirp_slope_hz_per_s'])
    sim_fs = float(public.get('simulation_sample_rate_hz', fs))
    bandwidth = float(public.get('transmit_bandwidth_hz', slope*duration))
    if not all(np.isfinite([fs, duration, slope, sim_fs, bandwidth])):
        raise ValueError('Public receiver parameters must be finite')
    if fs <= 0 or duration <= 0 or slope <= 0 or bandwidth <= 0:
        raise ValueError('This adaptation requires a public up-chirp receiver')
    if abs(n/fs-duration) > 1.5/fs:
        raise ValueError('ADC length and public chirp duration disagree')
    q = sim_fs/fs
    if q < 1 or abs(q-round(q)) > 1e-7:
        raise ValueError('High-rate receiver sample rate must be an ADC multiple')
    taps = np.asarray(public.get('fir_taps', _fir_design(
        sim_fs, float(public.get('lpf_cutoff_hz', 20e6)),
        float(public.get('lpf_stop_edge_hz', 23.5e6)),
        float(public.get('lpf_stop_atten_db', 40.)))), float)
    if taps.ndim != 1 or not taps.size or not np.isfinite(taps).all():
        raise ValueError('Public FIR taps must be a finite 1D array')
    # Group-delay compensation below corresponds to the original receiver.
    if taps.size % 2 != 1 or not np.allclose(taps, taps[::-1], rtol=1e-11, atol=1e-15):
        raise ValueError('This adaptation expects an odd symmetric public FIR')
    return dict(fs=fs, duration=duration, slope=slope, sim_fs=sim_fs,
                bandwidth=bandwidth, q=int(round(q)), taps=taps,
                maxbeat=float(public.get('max_beat_hz', fs*.4)))


class _Model:
    def __init__(self, p, n):
        self.p, self.n = p, n
        self.t = np.arange(n)/p['fs']
        self.center = (n-1)/(2*p['fs'])
        self.tc = self.t-self.center
        self.lo = _blackman(self.t, p['duration'], p['sim_fs'])
        self.common = self.lo*self.lo
        self.rf_f, self.rf_h = _fir_response(tuple(p['taps']), p['sim_fs'])
        self.high_t = None

    def atom(self, rate, center_frequency, exact=False):
        ki = self.p['slope']-rate
        if ki <= 0:
            return np.zeros(self.n, complex)
        ti = self.p['bandwidth']/ki
        start_frequency = center_frequency-rate*self.center
        shift = start_frequency/ki
        t = self.t
        if exact:
            if self.high_t is None:
                self.high_t = np.arange(self.n*self.p['q'])/self.p['sim_fs']
                self.high_lo = _blackman(self.high_t, self.p['duration'], self.p['sim_fs'])
            t = self.high_t
            envelope = self.high_lo*_blackman(t-shift, ti, self.p['sim_fs'])
        else:
            envelope = self.lo*_blackman(t-shift, ti, self.p['sim_fs'])
            envelope *= np.interp(np.abs(start_frequency+rate*t), self.rf_f,
                                  self.rf_h, left=self.rf_h[0], right=0.)
        tc = t-self.center
        atom = envelope*np.exp(2j*np.pi*(center_frequency*tc+.5*rate*tc*tc))
        if exact:
            return fftconvolve(atom, self.p['taps'], mode='same')[::self.p['q']]
        return atom

    def tone(self, f):
        # Delay is an observation-derived window approximation, not true range.
        delay = f/self.p['slope']
        envelope = self.lo*_blackman(self.t-delay, self.p['duration'], self.p['sim_fs'])
        return envelope*np.exp(2j*np.pi*f*self.tc)


def _power_projection(x, atom):
    den = float(np.vdot(atom, atom).real)
    return 0. if den < 1e-200 else float(abs(np.vdot(atom, x))**2/den)


def _projection(x, atoms):
    if not atoms:
        return x.copy(), np.empty(0, complex)
    a = np.column_stack(atoms)
    coeff = np.linalg.lstsq(a, x, rcond=1e-10)[0]
    return x-a@coeff, coeff


def _peak_frequency(x, model, nfft, rate=0.):
    z = np.fft.fft(x*model.common*np.exp(-1j*np.pi*rate*model.tc**2), n=nfft)
    frequencies = np.fft.fftfreq(nfft, 1/model.p['fs'])
    keep = np.abs(frequencies) <= model.p['fs']*.47
    power = np.where(keep, np.abs(z)**2, -1.)
    index = int(np.argmax(power))
    return float(frequencies[index]), float(power[index]/max(np.dot(model.common, model.common), 1e-200))


def _coarse(x, model, settings, nfft):
    step = float(settings['coarse_step_hz_per_s'])
    maximum, minimum = float(settings['max_rate_hz_per_s']), float(settings['min_rate_hz_per_s'])
    rates = np.r_[np.arange(-maximum, -minimum+step*.1, step),
                  np.arange(minimum, maximum+step*.1, step)]
    best = (-1., 0., 0.)
    for rate in rates:
        frequency, score = _peak_frequency(x, model, nfft, rate)
        if score > best[0]:
            best = score, float(rate), frequency
    return best


def _refine(x, model, rate, frequency, settings, nfft, nuisances=()):
    step = float(settings['coarse_step_hz_per_s'])
    max_rate, min_rate = float(settings['max_rate_hz_per_s']), float(settings['min_rate_hz_per_s'])
    low, high = max(-max_rate, rate-2*step), min(max_rate, rate+2*step)
    if rate < 0:
        high = min(-min_rate, high)
    else:
        low = max(min_rate, low)
    df = model.p['fs']/nfft
    # Eliminate observed nuisance tone amplitudes before optimizing the chirplet.
    q = None
    residual = x
    if nuisances:
        q = np.linalg.qr(np.column_stack(nuisances), mode='reduced')[0]
        residual = x-q@(q.conj().T@x)

    def atom_projected(v):
        atom = model.atom(v[0]*1e10, v[1]*1e6)
        if q is not None:
            atom = atom-q@(q.conj().T@atom)
        return atom

    def objective(v):
        return -_power_projection(residual, atom_projected(v))

    fit = minimize(objective, [rate/1e10, frequency/1e6], method='Powell',
                   bounds=[(low/1e10, high/1e10), ((frequency-3*df)/1e6, (frequency+3*df)/1e6)],
                   options=dict(xtol=1e-8, ftol=1e-10, maxiter=60))
    kr, fc = float(fit.x[0]*1e10), float(fit.x[1]*1e6)
    return kr, fc, _power_projection(residual, atom_projected(fit.x)), bool(fit.success)


def _nuisance_tones(x, model, settings, nfft):
    z = np.fft.fft(x*model.common, n=nfft)
    frequencies = np.fft.fftfreq(nfft, 1/model.p['fs'])
    powers = np.abs(z)**2
    powers[np.abs(frequencies) > model.p['maxbeat']] = 0.
    peaks, _ = find_peaks(powers, distance=max(1, nfft//model.n))
    peaks = sorted(peaks, key=lambda i: -powers[i])
    if not peaks:
        return [], []
    peaks = [i for i in peaks if powers[i] >= powers[peaks[0]]*settings['tone_relative_power']]
    selected = peaks[:int(settings['max_nuisance_tones'])]
    fs = [float(frequencies[i]) for i in selected]
    df = model.p['fs']/nfft
    # Coordinate refinement suppresses neighbouring tones during each fit.
    for _ in range(2):
        for i, frequency in enumerate(fs):
            others = [model.tone(f) for j, f in enumerate(fs) if j != i]
            residual, _ = _projection(x, others)
            fit = minimize_scalar(lambda f: -_power_projection(residual, model.tone(f)),
                                  bounds=(frequency-2*df, frequency+2*df),
                                  method='bounded', options=dict(xatol=1e-3, maxiter=50))
            fs[i] = float(fit.x)
    return [model.tone(f) for f in fs], fs


def process(observed, public, settings=None):
    """Return cancelled ADC chirp and JSON-serializable observed diagnostics."""
    x = np.asarray(observed, np.complex128)
    if x.ndim != 1 or x.size < 32 or not np.isfinite(x).all():
        raise ValueError('A finite complex 1D ADC chirp of at least 32 samples is required')
    options = {**DEFAULT_SETTINGS, **(settings or {})}
    unknown = set(options)-set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError('Unknown chirplet settings: '+', '.join(sorted(unknown)))
    if not 0 < options['min_rate_hz_per_s'] < options['max_rate_hz_per_s']:
        raise ValueError('Curvature search excludes zero and needs ordered positive limits')
    if options['coarse_step_hz_per_s'] <= 0 or not 1 <= options['max_chirplets'] <= 6:
        raise ValueError('Invalid rate step or maximum chirplet count')
    p = _public(public, x.size)
    diag = dict(version=VERSION, status='no_supported_chirplet', settings=options.copy(),
                model_assumption='same_centered_bandwidth_upchirp_blackman',
                truth_inputs_used=False, estimated_chirplets=[], attempts=[])
    power = float(np.vdot(x, x).real)
    if power <= 1e-300:
        diag['status'] = 'zero_input'
        return x.copy(), diag
    # Scale prevents optimizers mistaking physical sqrt(W) amplitudes for zero.
    scale = math.sqrt(power/x.size)
    original = x/scale
    model = _Model(p, x.size)
    nfft = 1 << int(np.ceil(np.log2(x.size*int(options['fft_oversampling']))))
    residual = original.copy()
    atoms, parameters = [], []
    for iteration in range(int(options['max_chirplets'])):
        _, rate, frequency = _coarse(residual, model, options, nfft)
        rate, frequency, chirp_score, converged = _refine(residual, model, rate, frequency, options, nfft)
        tone_frequency, _ = _peak_frequency(residual, model, nfft)
        df = p['fs']/nfft
        tone_fit = minimize_scalar(lambda f: -_power_projection(residual, model.tone(f)),
                                  bounds=(tone_frequency-2*df, tone_frequency+2*df), method='bounded')
        tone_score = _power_projection(residual, model.tone(float(tone_fit.x)))
        ratio = chirp_score/max(tone_score, 1e-200)
        fraction = chirp_score/max(float(np.vdot(residual, residual).real), 1e-200)
        attempt = dict(iteration=iteration, estimated_rate_hz_per_s=rate,
                       estimated_center_frequency_hz=frequency, chirp_tone_power_ratio=ratio,
                       explained_fraction=fraction, optimizer_converged=converged)
        diag['attempts'].append(attempt)
        if ratio < options['min_chirp_tone_power_ratio'] or fraction < options['min_explained_fraction']:
            attempt['reason'] = 'insufficient_chirplet_over_tone_evidence'
            break
        if any(abs(rate-r) < 2*options['coarse_step_hz_per_s'] and
               abs(frequency-f) < 2*p['fs']/x.size for r, f in parameters):
            attempt['reason'] = 'duplicate_atom'
            break
        atom = model.atom(rate, frequency)
        rough_coeff = np.vdot(atom, residual)/max(float(np.vdot(atom, atom).real), 1e-200)
        tone_input = residual-rough_coeff*atom
        tones, tone_frequencies = _nuisance_tones(tone_input, model, options, nfft)
        if options['joint_refinement'] and tones:
            rate, frequency, _, converged = _refine(residual, model, rate, frequency, options, nfft, tones)
        atom = model.atom(rate, frequency, exact=bool(options['exact_final_fir']))
        if np.linalg.norm(atom) < 1e-12:
            attempt['reason'] = 'out_of_receiver_band'
            break
        trial_atoms = atoms+[atom]
        design = trial_atoms+tones
        design_matrix = np.column_stack(design)
        coefficients = np.linalg.lstsq(design_matrix, original, rcond=1e-10)[0]
        cancellation = np.column_stack(trial_atoms)@coefficients[:len(trial_atoms)]
        next_residual = original-cancellation
        if not np.isfinite(next_residual).all() or np.vdot(next_residual, next_residual).real > 1.2*np.vdot(residual, residual).real:
            attempt['reason'] = 'unstable_joint_fit'
            break
        atoms, parameters = trial_atoms, parameters+[(rate, frequency)]
        residual = next_residual
        # Joint OMP refits earlier coefficients too; diagnostics report the
        # coefficients actually used in the final cumulative subtraction.
        for j, previous in enumerate(diag['estimated_chirplets']):
            previous['amplitude_real'] = float(coefficients[j].real*scale)
            previous['amplitude_imag'] = float(coefficients[j].imag*scale)
        ki = p['slope']-rate
        entry = dict(rate_hz_per_s=rate, center_frequency_hz=frequency,
                     start_frequency_hz=frequency-rate*model.center,
                     estimated_source_sweep_s=p['bandwidth']/ki,
                     estimated_envelope_shift_s=(frequency-rate*model.center)/ki,
                     amplitude_real=float(coefficients[len(atoms)-1].real*scale),
                     amplitude_imag=float(coefficients[len(atoms)-1].imag*scale),
                     nuisance_tone_frequencies_hz=tone_frequencies,
                     exact_final_fir=bool(options['exact_final_fir']), optimizer_converged=converged)
        diag['estimated_chirplets'].append(entry)
        attempt['reason'] = 'cancelled'
    y = residual*scale if atoms else x.copy()
    diag['status'] = 'cancelled' if atoms else 'no_supported_chirplet'
    diag['cancelled_chirplet_count'] = len(atoms)
    diag['removed_power_fraction'] = float(np.vdot(x-y, x-y).real/max(power, 1e-300))
    return np.asarray(y, np.complex128), diag
