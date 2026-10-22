"""Observed-block validated matrix-pencil gap reconstruction.

Engineering adaptation of the project's Wang--Ding--Yarovoy implementation.
Only observed x and public receiver settings are read. Rank is selected by
predicting contiguous observed blocks excluded from BOTH pencil estimation
and complex-amplitude fitting. The selected rank is then refitted using all
available observations. No target truth, scene identity, seed or CAR is read.
Unmasked samples are restored bit for bit.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import svd

SNAPSHOT = Path(__file__).resolve().parent
from fmcw_selector._core.anti_interference import matrix_pencil as base
from fmcw_selector._core.anti_interference.methods import detect_interference

VERSION = 'mp_contiguous_observed_validation_v1'
CONFIGS = [
    dict(name='legacy', mode='legacy'),
    dict(name='validated', mode='validated', pencil_length=64, max_rank=18,
         validation_error_limit=0.50, fallback='legacy', max_iterations=4),
    dict(name='validated_strict', mode='validated', pencil_length=64, max_rank=18,
         validation_error_limit=0.15, fallback='unchanged', max_iterations=4),
]


def _native(value):
    if isinstance(value,dict):return {str(k):_native(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_native(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def dependencies():
    paths = [Path(base.__file__), SNAPSHOT/'anti_interference'/'methods.py']
    return [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]


def _settings(public):
    return dict(fs=float(public['sample_rate_hz']),
                max_beat_hz=float(public['max_beat_hz']),
                chirp_duration_s=float(public.get('chirp_duration_s', 75e-6)),
                chirp_slope_hz_per_s=float(public.get('chirp_slope_hz_per_s', 4e12)),
                simulation_fs=float(public.get('simulation_sample_rate_hz', 2e9)))


def _split_blocks(observed):
    """Two nonadjacent contiguous blocks; absolute indices are never joined."""
    segments = sorted(base._segments(observed), key=lambda q: q[1]-q[0], reverse=True)
    held = np.zeros(len(observed), bool)
    if not segments:
        return observed.copy(), held, []
    longest = segments[0][1]-segments[0][0]
    blocks = []
    # Prefer separately located segments. A lone long segment supplies two
    # blocks, each preserving substantial training support on both sides.
    if len(segments) > 1 and segments[1][1]-segments[1][0] >= 160:
        candidates = [(segments[0], .50), (segments[1], .50)]
    else:
        candidates = [(segments[0], .33), (segments[0], .67)]
    for (a,b), fraction in candidates:
        if b-a < 192:
            continue
        width = min(96, max(32, (b-a)//8))
        center = int(a+fraction*(b-a))
        left, right = max(a+32, center-width//2), min(b-32, center+(width+1)//2)
        if right-left < 24 or held[left:right].any():
            continue
        held[left:right] = True
        blocks.append([int(left), int(right)])
    train = observed & ~held
    return train, held, blocks


def _pencil_candidates(sequence, train, config, settings):
    """One pair of TSVDs, multiple ranks; every Hankel row is training-only."""
    segments = [q for q in base._segments(train) if q[1]-q[0]>=16]
    if not segments:
        return [], {}
    length = min(int(config['pencil_length']), max(2,max(b-a for a,b in segments)//3))
    h0,h1,used = base.stacked_hankel_pencil(sequence, segments, length)
    if h0.shape[0]<4:
        return [], {}
    u0,s0,v0 = svd(h0, full_matrices=False, check_finite=False, lapack_driver='gesdd')
    u1,s1,v1 = svd(h1, full_matrices=False, check_finite=False, lapack_driver='gesdd')
    if not len(s0) or s0[0]<=np.finfo(float).tiny:
        return [], {}
    nominal = max(1,int(np.sum(s0 >= .01*s0[0])))
    weak = max(nominal,int(np.sum(s0 >= .002*s0[0])))
    cap=min(int(config['max_rank']),length-1,h0.shape[0]-1)
    ranks=sorted(set(min(cap,max(1,r)) for r in [nominal,nominal+2,nominal+4,weak]))
    out=[]
    for rank in ranks:
        core=((u0[:,:rank].conj().T@u1[:,:rank])*s1[:rank][None,:])
        core=core@(v1[:rank]@v0[:rank].conj().T)/s0[:rank,None]
        poles=np.linalg.eigvals(core); radius=abs(poles)
        frequencies=np.angle(poles)*settings['fs']/(2*np.pi)
        keep=np.isfinite(poles)&(radius>0)&(abs(radius-1)<.03)&(abs(frequencies)<=settings['max_beat_hz'])
        poles=poles[keep]/radius[keep]
        poles=poles[np.argsort(np.angle(poles))]
        if len(poles)>1:
            separation=abs(np.angle(poles[1:]/poles[:-1]))*settings['fs']/(2*np.pi)
            poles=poles[np.r_[True,separation>=settings['fs']/len(sequence)*.05]]
        if len(poles): out.append((rank,poles))
    return out, dict(training_hankel_segments=[list(q) for q in used],
                     training_hankel_shape=list(h0.shape), pencil_length=length,
                     candidate_ranks=ranks, nominal_rank=nominal,
                     singular_values_relative=(s0[:min(len(s0),cap+2)]/s0[0]).tolist())


def _design(poles, indices, settings, envelope='blackman'):
    return base._basis(poles, indices, settings['fs'], envelope,
                       settings['chirp_duration_s'], settings['chirp_slope_hz_per_s'],
                       settings['simulation_fs'])


def _refit(values, observed, stable, common, scale, mask, rank, settings, config, envelope):
    indices=np.flatnonzero(observed)
    normalized=np.zeros(len(values),complex)
    normalized[observed]=values[observed]/(common[observed]*scale)
    measured=values/scale
    segments=base._segments(observed)
    best_error=np.inf; best=None; history=[]; final={}
    initial_frequency=None
    for iteration in range(int(config.get('max_iterations',4))):
        poles,diagnostic=base._poles(normalized,segments,config['pencil_length'],
                                    max(rank,1),.000001,settings['fs'],settings['max_beat_hz'],fixed_rank=rank)
        if not len(poles): break
        amplitudes=np.linalg.lstsq(_design(poles,indices,settings,envelope),measured[indices],rcond=1e-8)[0]
        prediction=_design(poles,np.arange(len(values)),settings,envelope)@amplitudes
        if not np.isfinite(prediction).all(): break
        error=float(sum(np.linalg.norm(prediction[a:b]-measured[a:b]) for a,b in base._segments(observed)))
        frequencies=np.angle(poles)*settings['fs']/(2*np.pi)
        if initial_frequency is None: initial_frequency=frequencies
        history.append(dict(iteration=iteration, observed_segment_error=error,
                            frequencies_hz=frequencies.tolist(),**diagnostic))
        if error>best_error*(1+1e-10):break
        best_error=error;best=prediction.copy();final=dict(diagnostic)
        final.update(frequencies_hz=frequencies.tolist(),complex_amplitudes=[[float(a.real*scale),float(a.imag*scale)] for a in amplitudes])
        normalized[stable]=prediction[stable]/common[stable]
        normalized[observed]=measured[observed]/common[observed]
        segments=base._segments(stable)
        if error<=1e-10*max(np.linalg.norm(measured[observed]),1e-30):break
    if best is None:return None,dict(fusion_history=history,status='no_supported_pencil_model')
    maximum=float(np.max(abs(measured[observed])))
    ratio=float(np.max(abs(best[mask]))/max(maximum,1e-30))
    current=np.asarray(final['frequencies_hz'])
    drift=float(np.median(np.min(abs(current[:,None]-initial_frequency[None,:]),axis=1))/(settings['fs']/len(values))) if len(current) else 100.
    final.update(fusion_history=history,fusion_iterations=len(history),observed_segment_error=best_error,
                 extrapolation_peak_ratio=ratio,pole_fusion_shift_bins=drift)
    if ratio>20:return None,dict(final,status='unstable_extrapolation')
    return best*scale,dict(final,status='validated_gap_fill')


def process(x, public, config=None, return_diagnostics=True):
    started=time.perf_counter()
    config=dict(CONFIGS[1] if config is None else (next(c for c in CONFIGS if c['name']==config) if isinstance(config,str) else config))
    settings=_settings(public);values=np.asarray(x,dtype=np.complex128)
    if values.ndim!=1 or len(values)<16 or not np.isfinite(values).all():raise ValueError('finite complex 1D x required')
    envelope=str(config.get('envelope','blackman'))
    mask=np.asarray(config.get('observed_mask',detect_interference(values,settings['fs'],settings['max_beat_hz'])),bool)
    if mask.shape!=values.shape:raise ValueError('mask shape mismatch')
    info=dict(method='matrix_pencil',version=VERSION,config_name=config['name'],
              independent_validation=True,truth_used=False,masked_samples=int(mask.sum()),
              written_samples=0,validation_nmse=-1.,validation_worst_block_nmse=-1.,
              extrapolation_peak_ratio=-1.,pole_fusion_shift_bins=-1.,rank_fraction=-1.)
    def finish(output,status):
        output=np.asarray(output,dtype=np.complex128)
        output[~mask]=values[~mask]
        info.update(status=status,runtime_s=time.perf_counter()-started)
        return (output,_native(info)) if return_diagnostics else output
    def fallback(reason):
        info['validation_fallback_reason']=reason
        if config.get('fallback')=='unchanged':return finish(values.copy(),'validation_rejected_unchanged')
        y,d=base.reconstruct(values,mask=mask,envelope=envelope,return_diagnostics=True,**settings)
        info['legacy_diagnostics']=d;info['written_samples']=d.get('written_samples',0)
        info['legacy_status']=d['status']
        return finish(y,'validation_unsupported_legacy' if d.get('written_samples',0) else d['status'])
    if config.get('mode')=='legacy':
        y,d=base.reconstruct(values,mask=mask,envelope=envelope,return_diagnostics=True,**settings)
        info.update(d);info['independent_validation']=False
        return finish(y,d['status'])
    if not mask.any():return finish(values.copy(),'no_detected_mask')
    common=np.ones(len(values)) if envelope=='constant' else base._blackman(np.arange(len(values))/settings['fs'],settings['chirp_duration_s'],settings['simulation_fs'])**2
    stable=common>= (0. if envelope=='constant' else .02)
    observed=~mask&stable
    if observed.sum()<72:return fallback('insufficient_observations')
    train,held,blocks=_split_blocks(observed)
    info.update(validation_blocks=blocks,training_samples=int(train.sum()),validation_samples=int(held.sum()),observed_samples=int(observed.sum()),
                observed_segments=[list(q) for q in base._segments(observed)])
    if held.sum()<48 or train.sum()<72:return fallback('no_independent_contiguous_validation_blocks')
    scale=max(float(np.sqrt(np.mean(abs(values[train])**2))),np.finfo(float).tiny)
    sequence=np.zeros(len(values),complex);sequence[train]=values[train]/(common[train]*scale)
    try:
        candidates,diagnostic=_pencil_candidates(sequence,train,config,settings)
        info.update(diagnostic);scores=[]
        training=np.flatnonzero(train);validation=np.flatnonzero(held)
        for rank,poles in candidates:
            design=_design(poles,training,settings,envelope)
            amplitude=np.linalg.lstsq(design,values[training]/scale,rcond=1e-8)[0]
            prediction=_design(poles,validation,settings,envelope)@amplitude*scale
            error=float(np.sum(abs(prediction-values[validation])**2)/max(np.sum(abs(values[validation])**2),1e-300))
            block_errors=[]
            for a,b in blocks:
                active=(validation>=a)&(validation<b)
                block_errors.append(float(np.sum(abs(prediction[active]-values[validation[active]])**2)/max(np.sum(abs(values[validation[active]])**2),1e-300)))
            scores.append(dict(rank=rank,accepted_poles=len(poles),nmse=error,block_nmse=block_errors))
        info['validation_candidates']=scores
        if not scores:return fallback('no_training_poles')
        # Smallest rank within 1% relative validation error of the minimum.
        # This permits genuine weak tones only when independent prediction
        # improves; no truth-derived order/clean-mask information is used.
        minimum=min(s['nmse'] for s in scores)
        selected=min((s for s in scores if s['nmse']<=minimum*1.01+1e-10),key=lambda q:q['rank'])
        info.update(selected_validation_rank=selected['rank'],validation_nmse=selected['nmse'],
                    validation_worst_block_nmse=max(selected['block_nmse']),
                    rank_fraction=selected['rank']/config['max_rank'])
        if not np.isfinite(minimum) or minimum>config['validation_error_limit']:return fallback('prediction_error_exceeds_limit')
        prediction,fit=_refit(values,observed,stable,common,scale,mask,selected['rank'],settings,config,envelope)
        info.update(fit)
        if prediction is None:return fallback(fit['status'])
        output=values.copy();output[mask]=prediction[mask]
        info['written_samples']=int(mask.sum())
        return finish(output,'validated_gap_fill')
    except (np.linalg.LinAlgError,ValueError,FloatingPointError) as exc:
        info['exception_type']=type(exc).__name__
        return fallback('numerical_failure')


RISK_FEATURE_NAMES=('validation_nmse','validation_worst_block_nmse',
                    'extrapolation_peak_ratio','pole_fusion_shift_bins','rank_fraction')


def risk_features(diagnostics):
    """Finite observation-derived diagnostics; -1 explicitly marks unavailable."""
    return np.array([np.clip(float(diagnostics.get(k,-1.)),-1.,1e3) for k in RISK_FEATURE_NAMES],float)
