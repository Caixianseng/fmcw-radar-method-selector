"""Cheap observation-only curvature evidence; not a recovery algorithm.

Compare matched Fourier concentration after exp(-j*pi*kappa*t_c**2)
with the same zero-curvature projection. Public Blackman-squared is only
an approximate target envelope; delayed interfering tapers/FIR differ.
No total-phase unwrapping, continuous optimizer, cancellation, candidate
execution, source clock, clean target, interference component or label.
Grid and descriptors were fixed before development-pilot evaluation.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np

VERSION = 'observed48_plus4_coarse_curvature_evidence_v1'
EXTRA_FEATURE_NAMES = (
    'coarse_nonzero_curvature_over_victim_slope',
    'coarse_nonzero_vs_tone_log10_peak_power_ratio',
    'coarse_nonzero_projection_energy_fraction',
    'coarse_nonzero_four_block_complex_amplitude_consistency',
)
PARAMETERS = dict(max_rate_fraction=.02, coarse_step_fraction=.0005,
                  fine_step_fraction=.000125, fft_minimum=4096,
                  public_envelope='Blackman_squared', block_count=4,
                  candidate_recovery_executed=False,
                  unavailable_value=0., maximum_fft_calls=91)
EXTRA_FEATURE_DEFINITIONS = (
    'Best nonzero curvature / positive public victim slope; [-.02,.02]. Not an identified source slope.',
    'log10(best nonzero matched peak power / zero-curvature matched peak power); clipped [-12,12].',
    'Best nonzero matched projection energy / observed energy; [0,1].',
    'Squared coherent sum of four block correlations divided by independent block-fit energy; [0,1]. Long stationary amplitude approaches1; partial dwell, multiple components or phase mismatch reduce it.',
)


def _validate(observed, public):
    x=np.asarray(observed,np.complex128)
    fs=float(public['sample_rate_hz'])
    slope=float(public['chirp_slope_hz_per_s'])
    band=float(public.get('lpf_cutoff_hz',public['max_beat_hz']))
    if x.ndim!=1 or len(x)<32 or not np.isfinite(x).all():
        raise ValueError('Finite complex one-dimensional ADC with at least32 samples required')
    if not np.isfinite([fs,slope,band]).all() or fs<=0 or slope<=0 or not 0<band<fs/2:
        raise ValueError('Finite positive public fs/slope and band below Nyquist required')
    return x,fs,slope,band


@lru_cache(maxsize=8)
def _grid(n,fs,slope,band):
    tc=(np.arange(n)-(n-1)/2)/fs
    # Generate the public taper at ADC timestamps using the actual public
    # waveform sample convention when sim_fs is absent: end at(n-1)/fs.
    g=np.blackman(n)**2
    nfft=max(4096,1<<int(np.ceil(np.log2(n))))
    frequencies=np.fft.fftfreq(nfft,1/fs)
    valid=np.flatnonzero(np.abs(frequencies)<=band)
    rates=np.arange(-40,41,dtype=float)*(.0005*slope)
    phases=np.exp(-1j*np.pi*rates[:,None]*tc[None,:]**2)
    return tc,g,nfft,frequencies,valid,rates,phases


def near_slope_descriptors(observed,public):
    """Return four fixed descriptors and observed-only fit diagnostics.

    At most81 coarse+9 fine+1 diagnostic evaluations. FFT calls are
    independent and batched, no waveform is fitted/subtracted or emitted.
    Endpoints, grid, band and number of blocks never use development truth.
    """
    x,fs,slope,band=_validate(observed,public)
    tc,g,nfft,freq,valid,rates,phases=_grid(len(x),fs,slope,band)
    energy=float(np.vdot(x,x).real)
    if energy<=np.finfo(float).tiny:
        return np.zeros(4),dict(available=False,reason='zero_observed_energy',fft_calls=0,
            observed_only=True,truth_inputs_used=False,candidate_methods_executed=False)
    # Physical amplitude scaling cancels; normalizing avoids under/overflow.
    x=x/np.sqrt(energy)
    weighted=x*g
    spectra=np.fft.fft(phases*weighted[None,:],n=nfft,axis=1)
    power=np.abs(spectra[:,valid])**2
    bins=np.argmax(power,axis=1)
    peaks=power[np.arange(len(rates)),bins]
    zero=int(np.flatnonzero(rates==0)[0])
    tone=float(peaks[zero])
    nonzero=np.flatnonzero(rates!=0)
    coarse=int(nonzero[np.argmax(peaks[nonzero])])
    step=.000125*slope
    fine=rates[coarse]+np.arange(-4,5)*step
    fine=fine[(np.abs(fine)<=.02*slope+1e-6)&(np.abs(fine)>=step*.5)]
    fine_spectra=np.fft.fft(np.exp(-1j*np.pi*fine[:,None]*tc[None,:]**2)*weighted[None,:],n=nfft,axis=1)
    fp=np.abs(fine_spectra[:,valid])**2
    fb=np.argmax(fp,axis=1)
    fv=fp[np.arange(len(fine)),fb]
    winner=int(np.argmax(fv))
    kappa=float(fine[winner]);frequency=float(freq[valid[fb[winner]]]);peak=float(fv[winner])
    # Fine grid includes coarse rate, so result cannot worsen the coarse scan.
    template=g*np.exp(2j*np.pi*(frequency*tc+.5*kappa*tc**2))
    template_energy=float(np.vdot(template,template).real)
    projection=peak/template_energy
    correlations=[];block_energies=[];block_fractions=[]
    for indices in np.array_split(np.arange(len(x)),4):
        z=np.vdot(template[indices],x[indices])
        et=float(np.vdot(template[indices],template[indices]).real)
        ex=float(np.vdot(x[indices],x[indices]).real)
        correlations.append(z);block_energies.append(et)
        block_fractions.append(float(abs(z)**2/max(et*ex,np.finfo(float).tiny)))
    independent=sum(abs(z)**2/et for z,et in zip(correlations,block_energies) if et>0)
    consistency=abs(sum(correlations))**2/max(template_energy*independent,np.finfo(float).tiny)
    values=np.array([kappa/slope,np.log10(max(peak,1e-300)/max(tone,1e-300)),projection,consistency],float)
    values[1]=np.clip(values[1],-12,12);values[2:]=np.clip(values[2:],0,1)
    assert np.isfinite(values).all()
    return values,dict(available=True,estimated_rate_hz_per_s=kappa,
        estimated_center_frequency_hz=frequency,zero_curvature_peak_power=tone,
        best_nonzero_peak_power=peak,projection_fraction=float(values[2]),
        four_block_projection_fractions=block_fractions,
        four_block_complex_amplitude_consistency=float(values[3]),
        fft_calls=len(rates)+len(fine),fft_length=nfft,
        maximum_curvature_hz_per_s=.02*slope,
        minimum_nonzero_curvature_hz_per_s=step,
        approximate_public_envelope=True,actual_source_slope_identified=False,
        observed_only=True,truth_inputs_used=False,candidate_methods_executed=False,
        total_multicomponent_phase_unwrapped=False)


def extract_features_near_slope(observed,public):
    from fmcw_selector._core.features_subspace import extract_features_subspace
    prefix=extract_features_subspace(observed,public)
    extra,_=near_slope_descriptors(observed,public)
    return np.concatenate((prefix,extra))


def extract_features(observed,public):
    return extract_features_near_slope(observed,public)


# Importing the prefix requires the experiment's normal source_snapshot
# path initialization (study_common); descriptor-only use is standalone.
try:
    from fmcw_selector._core.features_subspace import FEATURE_NAMES48
    FEATURE_NAMES=tuple(FEATURE_NAMES48)+EXTRA_FEATURE_NAMES
    FEATURE_NAMES52=FEATURE_NAMES
except ModuleNotFoundError:
    FEATURE_NAMES52=None
    FEATURE_NAMES=None
