"""Observation-only JSS output-fidelity variants, with one shared final fit.

The original deterministic joint-fit and evidence-gated support augmentation
are copied locally without algorithmic changes. Only output composition differs:
target-only synthesis, slow-interference subtraction, or robust residual retention.
No target truth, clock truth, AR, Matrix Pencil, or other restoration is used.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key]='1'
import sys
from pathlib import Path
import time
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize, minimize_scalar
from fmcw_selector._core.chirplet_omp import (_public,_Model,_coarse,_refine,_peak_frequency,
                          _power_projection,DEFAULT_SETTINGS)

BASE_VERSION='joint_stationary_slowchirp_robust_variable_projection_pilot_v1'
DEFAULT_CONFIG=dict(max_rate_hz_per_s=8e10,min_rate_hz_per_s=1e9,
    coarse_step_hz_per_s=2e9,fft_oversampling=4,
    min_chirp_tone_power_ratio=1.35,min_explained_fraction=.03,
    maximum_target_tones=10,maximum_outer_iterations=3,
    robust_iterations=8,huber_multiplier=2.5,
    target_peak_relative_power=.0005,exact_final_fir=True)

def _robust_fit(y,atoms,cfg,initial_weights=None):
    a=np.column_stack(atoms)
    weights=np.ones(len(y)) if initial_weights is None else initial_weights.copy()
    coefficients=np.zeros(a.shape[1],complex)
    for _ in range(cfg['robust_iterations']):
        root=np.sqrt(weights)
        coefficients=np.linalg.lstsq(a*root[:,None],y*root,rcond=1e-10)[0]
        error=y-a@coefficients
        scale=max(float(np.median(abs(error)))/np.sqrt(np.log(2)),1e-12)
        weights=1/np.sqrt(1+(abs(error)/(cfg['huber_multiplier']*scale))**2)
    return coefficients,weights,y-a@coefficients

def _find_targets(residual,model,nfft,cfg):
    spec=abs(np.fft.fft(residual*model.common,nfft))**2
    f=np.fft.fftfreq(nfft,1/model.p['fs'])
    spec[abs(f)>model.p['maxbeat']]=0
    peaks,_=find_peaks(spec,distance=max(2,nfft//len(residual)))
    if not len(peaks):return []
    floor=max(float(np.median(spec[abs(f)<=model.p['maxbeat']]))*20,
              float(spec.max())*cfg['target_peak_relative_power'])
    peaks=sorted((i for i in peaks if spec[i]>=floor),key=lambda i:-spec[i])
    return [float(f[i]) for i in peaks[:cfg['maximum_target_tones']]]

def _basefit(observed,public,config=None,return_diagnostics=True):
    start=time.perf_counter();x=np.asarray(observed,np.complex128)
    if x.ndim!=1 or len(x)<32 or not np.isfinite(x).all():
        raise ValueError('Finite complex observation required')
    cfg={**DEFAULT_CONFIG,**(config or {})}
    if set(cfg)!=set(DEFAULT_CONFIG):raise ValueError('Unknown prototype setting')
    p=_public(public,len(x));model=_Model(p,len(x))
    scale=float(np.sqrt(np.mean(abs(x)**2)))
    diag=dict(version=BASE_VERSION,status='no_supported_near_slope',truth_inputs_used=False,
        method_calls=[],configuration=cfg,targets=[],runtime_s=0.,
        paper_relationship='Independent deterministic joint-fit adaptation; not VEM/SBL reproduction')
    if scale==0:
        diag.update(status='zero_input',runtime_s=time.perf_counter()-start)
        return (x.copy(),diag) if return_diagnostics else x.copy()
    y=x/scale;nfft=1<<int(np.ceil(np.log2(len(x)*cfg['fft_oversampling'])))
    search={**DEFAULT_SETTINGS,**{k:cfg[k] for k in ('max_rate_hz_per_s','min_rate_hz_per_s','coarse_step_hz_per_s')}}
    _,rate,freq=_coarse(y,model,search,nfft)
    rate,freq,chirp_score,converged=_refine(y,model,rate,freq,search,nfft)
    tonef,_=_peak_frequency(y,model,nfft)
    df=p['fs']/nfft
    tonefit=minimize_scalar(lambda f:-_power_projection(y,model.tone(f)),
                           bounds=(tonef-3*df,tonef+3*df),method='bounded')
    tonescore=-float(tonefit.fun)
    ratio=chirp_score/max(tonescore,1e-100)
    explained=chirp_score/max(float(np.vdot(y,y).real),1e-100)
    diag.update(estimated_rate_hz_per_s=rate,estimated_center_frequency_hz=freq,
                chirp_tone_power_ratio=ratio,initial_explained_fraction=explained)
    if ratio<cfg['min_chirp_tone_power_ratio'] or explained<cfg['min_explained_fraction']:
        diag['runtime_s']=time.perf_counter()-start
        return (x.copy(),diag) if return_diagnostics else x.copy()
    atom=model.atom(rate,freq,exact=cfg['exact_final_fir'])
    c=np.vdot(atom,y)/max(float(np.vdot(atom,atom).real),1e-100)
    targets=_find_targets(y-atom*c,model,nfft,cfg)
    if not targets:
        diag.update(status='no_supported_target',runtime_s=time.perf_counter()-start)
        return (x.copy(),diag) if return_diagnostics else x.copy()
    weights=np.ones(len(y));fits=[]
    for iteration in range(cfg['maximum_outer_iterations']):
        # Fit every target jointly with the interference: never infer target
        # amplitude from a residual already biased by interference subtraction.
        atoms=[model.atom(rate,freq)]+[model.tone(f) for f in targets]
        coeff,weights,error=_robust_fit(y,atoms,cfg,weights)
        root=np.sqrt(weights);tone_matrix=np.column_stack(atoms[1:])
        q=np.linalg.qr(tone_matrix*root[:,None],mode='reduced')[0]
        yr=y*root-q@(q.conj().T@(y*root))
        def objective(v):
            a=model.atom(v[0]*1e10,v[1]*1e6)*root
            a=a-q@(q.conj().T@a)
            return -_power_projection(yr,a)
        half=cfg['coarse_step_hz_per_s']*2
        low=max(-cfg['max_rate_hz_per_s'],rate-half)
        high=min(cfg['max_rate_hz_per_s'],rate+half)
        if rate<0:high=min(-cfg['min_rate_hz_per_s'],high)
        else:low=max(cfg['min_rate_hz_per_s'],low)
        fit=minimize(objective,[rate/1e10,freq/1e6],method='Powell',
            bounds=[(low/1e10,high/1e10),((freq-3*df)/1e6,(freq+3*df)/1e6)],
            options=dict(xtol=1e-7,ftol=1e-9,maxiter=35))
        rate,freq=float(fit.x[0]*1e10),float(fit.x[1]*1e6)
        # Continuous target frequencies, with all other source components
        # projected out, instead of choosing a fixed Fourier-bin support.
        for j,f in enumerate(targets):
            other=[model.atom(rate,freq)]+[model.tone(g) for k,g in enumerate(targets) if k!=j]
            qa=np.linalg.qr(np.column_stack(other)*root[:,None],mode='reduced')[0]
            ry=y*root-qa@(qa.conj().T@(y*root))
            def target_objective(g):
                a=model.tone(g)*root;a-=qa@(qa.conj().T@a)
                return -_power_projection(ry,a)
            ff=minimize_scalar(target_objective,bounds=(f-3*df,f+3*df),method='bounded',
                               options=dict(xatol=.01,maxiter=40))
            targets[j]=float(ff.x)
        fits.append(dict(iteration=iteration,rate=rate,frequency=freq,
                         mean_weight=float(weights.mean()),optimizer_converged=bool(fit.success)))
    atoms=[model.atom(rate,freq,exact=cfg['exact_final_fir'])]+[model.tone(f) for f in targets]
    coeff,weights,error=_robust_fit(y,atoms,cfg,weights)
    # The target dictionary is synthesized in physical units without oracle
    # gain fitting. Short ordinary bursts remain in residual, not emitted.
    recovered=np.column_stack(atoms[1:])@coeff[1:]*scale
    diag.update(status='joint_target_reconstruction',estimated_rate_hz_per_s=rate,
        estimated_center_frequency_hz=freq,targets=[dict(frequency_hz=f,amplitude=float(abs(c)*scale))
            for f,c in zip(targets,coeff[1:])],iterations=fits,
        weighted_residual_fraction=float(np.sum(weights*abs(error)**2)/np.sum(weights*abs(y)**2)),
        runtime_s=time.perf_counter()-start,observation_scale=scale)
    diag['_shared_fit']=dict(model=model,atoms=atoms,coeff=coeff,weights=weights,error=error,scale=scale)
    if not np.isfinite(recovered).all():raise FloatingPointError('Nonfinite joint recovery')
    return (recovered,diag) if return_diagnostics else recovered

SUPPORT_VERSION='research_jss_residual_crossvalidated_support_v2a'
CONFIG=dict(max_additions=3,maximum_tones=10,proposal_limit=8,
    residual_floor_multiplier=20.,relative_residual_peak_power=.0005,
    minimum_block_coherence=.90,minimum_block_amplitude_ratio=.25,
    minimum_total_gain_fraction=.01,minimum_each_block_gain_fraction=.001,
    minimum_atom_weight_fraction=.05)

def _fit(observed,public,return_diagnostics=True):
    started=time.perf_counter();x=np.asarray(observed,np.complex128)
    output,diag=_basefit(x,public)
    diag=dict(diag);diag['v1_status']=diag['status'];diag['version']=SUPPORT_VERSION
    diag['support_augmentation']=dict(configuration=CONFIG,attempts=[],accepted=[])
    if diag['status']!='joint_target_reconstruction':
        diag['runtime_s']=time.perf_counter()-started
        return (output,diag) if return_diagnostics else output
    scale=float(diag['observation_scale']);y=x/scale
    p=_public(public,len(y));m=_Model(p,len(y));nfft=1<<int(np.ceil(np.log2(len(y)*4)))
    df=p['fs']/nfft;rate=float(diag['estimated_rate_hz_per_s']);freq=float(diag['estimated_center_frequency_hz'])
    targets=[float(t['frequency_hz']) for t in diag['targets']]
    cfg=dict(DEFAULT_CONFIG);atoms=[m.atom(rate,freq,exact=True)]+[m.tone(f) for f in targets]
    coeff,weights,error=_robust_fit(y,atoms,cfg)
    original_count=len(targets)
    # A Blackman product has negligible information at its endpoints.  Define
    # blocks using public envelope support, never signal truth.
    active=np.flatnonzero(m.common>=.05*m.common.max())
    blocks=[b for b in np.array_split(active,4) if len(b)]
    for addition in range(CONFIG['max_additions']):
        if len(targets)>=CONFIG['maximum_tones']:break
        root=np.sqrt(weights);A=np.column_stack(atoms);Q=np.linalg.qr(A*root[:,None],mode='reduced')[0]
        ry=y*root-Q@(Q.conj().T@(y*root))
        residual=error*weights
        spec=abs(np.fft.fft(residual*m.common,nfft))**2
        freqs=np.fft.fftfreq(nfft,1/p['fs']);band=abs(freqs)<=p['maxbeat'];spec[~band]=0
        floor=max(float(np.median(spec[band]))*CONFIG['residual_floor_multiplier'],float(spec.max())*CONFIG['relative_residual_peak_power'])
        peaks,_=find_peaks(spec,distance=max(2,nfft//len(y)))
        peaks=sorted([i for i in peaks if spec[i]>=floor and all(abs(freqs[i]-f)>2*p['fs']/len(y) for f in targets)],key=lambda i:-spec[i])[:CONFIG['proposal_limit']]
        accepted=False
        for index in peaks:
            center=float(freqs[index])
            def obj(f):
                a=m.tone(f)*root;a=a-Q@(Q.conj().T@a)
                return -_power_projection(ry,a)
            opt=minimize_scalar(obj,bounds=(center-2*df,center+2*df),method='bounded',options=dict(xatol=.05,maxiter=35))
            f=float(opt.x);candidate=m.tone(f);aug=np.column_stack(atoms+[candidate])
            # Four-fold blocked predictive comparison.  Each omitted block is
            # not used to estimate either model's complex coefficients.
            gains=[];block_coeff=[]
            for test in blocks:
                train=np.ones(len(y),bool);train[test]=False
                wa=root[train]
                c0=np.linalg.lstsq(A[train]*wa[:,None],y[train]*wa,rcond=1e-10)[0]
                c1=np.linalg.lstsq(aug[train]*wa[:,None],y[train]*wa,rcond=1e-10)[0]
                before=float(np.sum(weights[test]*abs(y[test]-A[test]@c0)**2))
                after=float(np.sum(weights[test]*abs(y[test]-aug[test]@c1)**2))
                gains.append((before-after)/max(before,1e-100));block_coeff.append(c1[-1])
            bc=np.asarray(block_coeff);coherence=float(abs(bc.sum())/max(abs(bc).sum(),1e-100))
            amp_ratio=float(abs(bc).min()/max(abs(bc).max(),1e-100))
            c1,w1,e1=_robust_fit(y,atoms+[candidate],cfg,weights)
            before=float(np.sum(weights*abs(error)**2));after=float(np.sum(weights*abs(e1)**2))
            gain=(before-after)/max(before,1e-100)
            weight_fraction=float(np.sum(weights*abs(candidate)**2)/max(np.sum(abs(candidate)**2),1e-100))
            passed=bool(coherence>=CONFIG['minimum_block_coherence'] and amp_ratio>=CONFIG['minimum_block_amplitude_ratio'] and min(gains)>=CONFIG['minimum_each_block_gain_fraction'] and gain>=CONFIG['minimum_total_gain_fraction'] and weight_fraction>=CONFIG['minimum_atom_weight_fraction'])
            rec=dict(frequency_hz=f,coherence=coherence,amplitude_ratio=amp_ratio,block_predictive_gains=gains,total_gain_fraction=gain,weighted_atom_fraction=weight_fraction,accepted=passed)
            diag['support_augmentation']['attempts'].append(rec)
            if passed:
                targets.append(f);atoms.append(candidate);coeff,weights,error=c1,w1,e1
                diag['support_augmentation']['accepted'].append(rec);accepted=True;break
        if not accepted:break
    if len(targets)>original_count:
        output=np.column_stack(atoms[1:])@coeff[1:]*scale
        diag['_shared_fit']=dict(model=m,atoms=atoms,coeff=coeff,weights=weights,error=error,scale=scale)
        diag['status']='joint_target_reconstruction_support_augmented'
        diag['targets']=[dict(frequency_hz=f,amplitude=float(abs(c)*scale)) for f,c in zip(targets,coeff[1:])]
    diag['runtime_s']=time.perf_counter()-started
    return (output,diag) if return_diagnostics else output


VERSION='jss_sharedfit_residual_retention_v1'
CONFIG_IDS=('target_only','subtract_slow','residual_stable','residual10','residual25')
RISK_NAMES=('jss_atom_coherence','jss_residual_outlier_energy_fraction',
            'jss_target_block_instability','jss_fit_condition_log10')

def _risk(fit):
    """Four observation-only descriptors; no true number/frequency of targets."""
    A=np.column_stack(fit['atoms']);c=fit['coeff'];w=fit['weights'];e=fit['error']
    norms=np.maximum(np.linalg.norm(A,axis=0),1e-100);An=A/norms
    coherence=float(np.max(abs(An[:,0].conj()@An[:,1:]))) if A.shape[1]>1 else 0.
    energy=abs(e)**2;outlier=float(energy[w<.5].sum()/max(energy.sum(),1e-100))
    singular=np.linalg.svd(An*np.sqrt(w[:,None]),compute_uv=False)
    condition=float(np.log10(max(singular[0]/max(singular[-1],1e-12),1.)))
    model=fit['model'];active=np.flatnonzero(model.common>=.05*model.common.max())
    block_coeff=[]
    y=A@c+e
    for b in np.array_split(active,4):
        if len(b)<A.shape[1]+4:continue
        root=np.sqrt(w[b]);cb=np.linalg.lstsq(A[b]*root[:,None],y[b]*root,rcond=1e-10)[0]
        block_coeff.append(cb[1:])
    if len(block_coeff)>=2 and A.shape[1]>1:
        bc=np.asarray(block_coeff);mid=np.median(bc.real,axis=0)+1j*np.median(bc.imag,axis=0)
        instability=float(np.mean(abs(bc-mid[None,:])**2)/max(float(np.mean(abs(c[1:])**2)),1e-100))
    else:instability=-1.
    return dict(zip(RISK_NAMES,(coherence,outlier,min(instability,1e6),min(condition,12.))))

def variants(observed,public):
    """Return all output variants from exactly one JSS fit.

    All unsupported inputs preserve the old output exactly. Residual-stable
    retains unmodelled signal using squared robust weights from the joint fit;
    it suppresses burst outliers without discarding low-amplitude weak tones.
    This is a bounded output-composition variant, not a new support search.
    """
    started=time.perf_counter();x=np.asarray(observed,np.complex128)
    target,base=_fit(x,public);fit=base.pop('_shared_fit',None)
    result={}
    supported=fit is not None and base['status'].startswith('joint_target_reconstruction')
    risk=_risk(fit) if supported else dict(zip(RISK_NAMES,(-1.,-1.,-1.,-1.)))
    if supported:
        slow=np.asarray(fit['atoms'][0])*fit['coeff'][0]*fit['scale']
        residual=x-slow-target
        retention=np.asarray(fit['weights'])**2
        outputs=dict(target_only=target,subtract_slow=x-slow,
                     residual_stable=target+retention*residual,
                     residual10=target+.10*retention*residual,
                     residual25=target+.25*retention*residual)
        fraction=float(np.sum(abs(retention*residual)**2)/max(float(np.sum(abs(x)**2)),1e-100))
    else:
        outputs={name:target.copy() for name in CONFIG_IDS};fraction=0.
    for name,out in outputs.items():
        d=dict(base);d.update(version=VERSION,variant=name,supported=bool(supported),
            output_composition=dict(target_only='joint_target_synthesis',subtract_slow='observation_minus_joint_slow_component',residual_stable='joint_targets_plus_squared_robust_weighted_residual',residual10='joint_targets_plus_0.10_squared_robust_weighted_residual',residual25='joint_targets_plus_0.25_squared_robust_weighted_residual')[name],
            residual_retention_gain=dict(target_only=0.,subtract_slow=1.,residual_stable=1.,residual10=.10,residual25=.25)[name],
            risk_features=dict(risk),retained_robust_residual_energy_fraction=fraction,
            shared_fit_count=1,additional_method_calls=[],truth_inputs_used=False,
            runtime_s=time.perf_counter()-started)
        if out.shape!=x.shape or not np.isfinite(out).all():raise FloatingPointError('Invalid JSS output')
        result[name]=(np.asarray(out,np.complex128),d)
    return result

def process(observed,public,config=None,return_diagnostics=True):
    name='target_only' if config is None else (config if isinstance(config,str) else config.get('variant','target_only'))
    if name not in CONFIG_IDS:raise ValueError('Unknown JSS variant: '+str(name))
    y,d=variants(observed,public)[name]
    return (y,d) if return_diagnostics else y
