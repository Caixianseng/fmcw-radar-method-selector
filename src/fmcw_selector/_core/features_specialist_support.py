"""Original52 plus four observed JSS initial-target-support probes.

This explicitly incurs the initial JSS curvature-search and single-atom
projection cost. It stops BEFORE joint robust fitting, target optimization
or output synthesis; no method output, actual status or truth is an input.
The model is still an MLP candidate selector, never a feature hard router.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
from fmcw_selector._core.features_near_slope import FEATURE_NAMES52,extract_features_near_slope
from fmcw_selector._core.jss_settings import DEFAULT_CONFIG
from fmcw_selector._core.chirplet_omp import (_public,_Model,_coarse,_refine,_peak_frequency,
                          _power_projection,DEFAULT_SETTINGS)

VERSION='observed52_plus4_jss_initial_support_v1'
EXTRA_FEATURE_NAMES=(
    'jss_probe_residual_target_band_log10_peak_to_median_power',
    'jss_probe_evidence_gated_target_proposal_count_fraction',
    'jss_probe_single_chirp_projection_residual_energy_fraction',
    'jss_probe_leading_target_four_block_amplitude_consistency',
)
FEATURE_NAMES56=tuple(FEATURE_NAMES52)+EXTRA_FEATURE_NAMES
FEATURE_NAMES=FEATURE_NAMES56
PARAMETERS=dict(initial_configuration=dict(DEFAULT_CONFIG),blocks=4,
    maximum_peak_log10_ratio=30.,target_proposal_floor_median_multiplier=20.,
    initial_target_proposal_reused=True,initial_curvature_search_executed=True,
    joint_robust_fit_executed=False,reconstruction_synthesized=False,
    actual_candidate_status_input=False,unavailable_value=0.)


def specialist_support_descriptors(observed,public):
    """Four descriptors and observed-only pre-reconstruction diagnostics.

    Descriptor2 is zero when the *observed initial evidence* is unsupported,
    even if the residual spectrum contains tones. No actual process/status
    is called or read. Descriptors1/3/4 retain their independent definitions.
    """
    x=np.asarray(observed,np.complex128)
    if x.ndim!=1 or len(x)<32 or not np.isfinite(x).all():
        raise ValueError('Finite complex one-dimensional ADC required')
    cfg=PARAMETERS['initial_configuration']
    p=_public(public,len(x));model=_Model(p,len(x))
    scale=float(np.sqrt(np.mean(abs(x)**2)))
    common=dict(observed_only=True,truth_inputs_used=False,actual_method_status_used=False,
        initial_curvature_search_executed=scale>0,joint_robust_fit_executed=False,
        target_frequency_refinement_executed=False,reconstruction_synthesized=False,
        hard_method_route=False,initial_configuration=dict(cfg))
    if scale==0:return np.zeros(4),dict(common,available=False,reason='zero_observed_energy',first_evidence_accepted=False,target_proposal_count=0)
    y=x/scale;nfft=1<<int(np.ceil(np.log2(len(x)*cfg['fft_oversampling'])))
    search={**DEFAULT_SETTINGS,**{k:cfg[k] for k in ('max_rate_hz_per_s','min_rate_hz_per_s','coarse_step_hz_per_s')}}
    _,rate,freq=_coarse(y,model,search,nfft)
    rate,freq,chirp_score,converged=_refine(y,model,rate,freq,search,nfft)
    tonef,_=_peak_frequency(y,model,nfft);df=p['fs']/nfft
    fit=minimize_scalar(lambda f:-_power_projection(y,model.tone(f)),
        bounds=(tonef-3*df,tonef+3*df),method='bounded')
    ratio=chirp_score/max(-float(fit.fun),1e-100)
    explained=chirp_score/max(float(np.vdot(y,y).real),1e-100)
    accepted=bool(ratio>=cfg['min_chirp_tone_power_ratio'] and explained>=cfg['min_explained_fraction'])
    atom=model.atom(rate,freq,exact=cfg['exact_final_fir'])
    coefficient=np.vdot(atom,y)/max(float(np.vdot(atom,atom).real),1e-100)
    residual=y-atom*coefficient
    spectrum=abs(np.fft.fft(residual*model.common,nfft))**2
    frequencies=np.fft.fftfreq(nfft,1/p['fs']);band=np.abs(frequencies)<=p['maxbeat']
    spectrum[~band]=0
    maximum=float(spectrum.max());median=float(np.median(spectrum[band]))
    logratio=0. if maximum<=1e-100 else float(np.clip(np.log10(maximum/max(median,1e-100)),0,30))
    peaks,_=find_peaks(spectrum,distance=max(2,nfft//len(y)))
    floor=max(median*20,maximum*cfg['target_peak_relative_power'])
    peaks=sorted((j for j in peaks if spectrum[j]>=floor),key=lambda j:-spectrum[j])[:cfg['maximum_target_tones']]
    proposed=[float(frequencies[j]) for j in peaks]
    gated_count=len(proposed) if accepted else 0
    residual_fraction=float(np.vdot(residual,residual).real/np.vdot(y,y).real)
    consistency=0.
    if proposed:
        target=model.tone(proposed[0]);correlations=[];energies=[]
        for indices in np.array_split(np.arange(len(y)),4):
            correlations.append(np.vdot(target[indices],residual[indices]))
            energies.append(float(np.vdot(target[indices],target[indices]).real))
        et=float(np.vdot(target,target).real)
        independent=sum(abs(z)**2/e for z,e in zip(correlations,energies) if e>0)
        consistency=float(abs(sum(correlations))**2/max(et*independent,1e-100))
    values=np.array([logratio,gated_count/cfg['maximum_target_tones'],
                     np.clip(residual_fraction,0,1),np.clip(consistency,0,1)],float)
    assert values.shape==(4,) and np.isfinite(values).all()
    return values,dict(common,available=True,first_evidence_accepted=accepted,
        initial_chirp_tone_power_ratio=float(ratio),initial_explained_fraction=float(explained),
        estimated_rate_hz_per_s=float(rate),estimated_center_frequency_hz=float(freq),
        initial_optimizer_converged=bool(converged),target_proposal_count=gated_count,
        ungated_residual_target_proposal_count=len(proposed),proposed_target_frequencies_hz=proposed,
        residual_target_band_peak_power=maximum,residual_target_band_median_power=median,
        residual_energy_fraction=float(values[2]),leading_tone_four_block_consistency=float(values[3]),
        no_supported_target_mathematical_precondition=bool(accepted and not proposed),
        no_actual_candidate_execution=True)


def extract_features_specialist_support(observed,public):
    return np.concatenate((extract_features_near_slope(observed,public),
                           specialist_support_descriptors(observed,public)[0]))


extract_features=extract_features_specialist_support
