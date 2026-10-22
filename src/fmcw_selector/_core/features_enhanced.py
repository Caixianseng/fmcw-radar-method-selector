"""Observed32 plus12support/coherence/multiple-ridge descriptors.

These descriptors are hypotheses for a development ablation, not evidence that
the earlier48-feature risk policy was effective. No actual recovery, pole/SVD
estimate, method status, target truth, source clock or selector prediction is
accepted. A four-coefficient local linear prediction check uses withheld
observed samples in one unmasked segment; it never fills a detected gap.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from scipy.signal import find_peaks
ROOT=Path(__file__).resolve().parent
from fmcw_selector._core.anti_interference.features import extract_features as _base_features,FEATURE_NAMES as BASE_FEATURE_NAMES
from fmcw_selector._core.anti_interference.methods import _analysis
from fmcw_selector._core.anti_interference.matrix_pencil import _blackman

FEATURE_VERSION='observed32_plus12_support_coherence_multiridge_v1'
EXTRA_FEATURE_NAMES=(
 'observed_os_adc_mask_fraction','observed_os_mask_energy_fraction',
 'stable_unmasked_fraction','longest_stable_unmasked_fraction','initial_hankel_rows_fraction',
 'unmasked_support_quarter_entropy','stable_unmasked_lag1_coherence',
 'local_ar4_observed_holdout_log1p_nmse','local_ar4_observed_holdout_available',
 'public_band_tf_local_peak_count_mean','public_band_stationary_peak_energy_fraction',
 'public_band_secondary_global_peak_power_ratio',
)
FEATURE_NAMES=tuple(BASE_FEATURE_NAMES)+EXTRA_FEATURE_NAMES
FEATURE_GROUPS=dict(support=EXTRA_FEATURE_NAMES[:6],coherence=EXTRA_FEATURE_NAMES[6:9],multiple_ridges=EXTRA_FEATURE_NAMES[9:])
EXTRA_FEATURE_DEFINITIONS=(
 ('observed_os_adc_mask_fraction','Marked ADC fraction from the unchanged observation detector','[0,1]','0 if no samples marked'),
 ('observed_os_mask_energy_fraction','Observed ADC energy inside that mask divided by total observed energy','[0,1]','0 for zero signal'),
 ('stable_unmasked_fraction','Unmasked samples / public Blackman-squared>=.02 samples','[0,1]','0 if none'),
 ('longest_stable_unmasked_fraction','Longest stable unmasked run / full chirp length','[0,1]','0 if none'),
 ('initial_hankel_rows_fraction','Initial MP necessary geometric row count / full chirp length; no SVD or poles','[0,1]','0 without eligible segment'),
 ('unmasked_support_quarter_entropy','Entropy of stable unmasked counts across four equal time quarters / log4','[0,1]','0 with no support'),
 ('stable_unmasked_lag1_coherence','Absolute normalized adjacent-sample inner product after public-window division','[0,1]','0 without energetic adjacent support'),
 ('local_ar4_observed_holdout_log1p_nmse','log1p normalized one-step residual on withheld observed samples; ridge LS order4, no gap filling','[0,log(1+1e6)]','0 if unavailable; interpret with next flag'),
 ('local_ar4_observed_holdout_available','Enough observed segment>=32 and nonzero holdout energy for the cheap local check','{0,1}','0 if unavailable or numerical solve fails'),
 ('public_band_tf_local_peak_count_mean','Mean local maxima above existing -12dB relative-power convention in valid observed frames and public abs-frequency band','[0,frequency_bins/2]','0 without valid frames'),
 ('public_band_stationary_peak_energy_fraction','In-band observed TF energy weighted by time persistence of local maxima within one frequency bin','[0,1]','0 without in-band energy'),
 ('public_band_secondary_global_peak_power_ratio','Second/first observed global Hann spectral local-peak power in public band, separation>=4bins','[0,1]','0 without two peaks'),
)
PARAMETERS=dict(stable_window_squared_floor=.02,mp_minimum_segment_samples=16,mp_maximum_initial_pencil_length=64,
 support_quarters=4,local_ar_order=4,local_ar_minimum_segment_samples=32,local_ar_maximum_segment_samples=256,
 local_ar_ridge_relative=1e-6,local_ar_train_fraction=2/3,local_ar_nmse_numerical_cap=1e6,
 local_peak_relative_power_db=-12.,valid_frame_relative_energy=.01,peak_persistence_frequency_radius=1,
 global_peak_minimum_separation_bins=4)

def segments(mask):
 change=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(np.int8))
 return [(int(a),int(b)) for a,b in zip(np.flatnonzero(change==1),np.flatnonzero(change==-1))]

def support_geometry(mask,fs):
 n=len(mask);common=_blackman(np.arange(n)/fs,n/fs,2e9)**2
 stable=common>=PARAMETERS['stable_window_squared_floor'];observed=stable&~mask;runs=segments(observed)
 eligible=[(a,b) for a,b in runs if b-a>=16];longest=max((b-a for a,b in runs),default=0)
 length=min(64,max(2,max((b-a for a,b in eligible),default=0)//3)) if eligible else 0
 rows=sum(b-a-length for a,b in eligible if b-a>length+1)
 counts=np.array([part.sum() for part in np.array_split(observed,4)],float);total=counts.sum()
 q=counts/total if total else counts;entropy=float(-np.sum(q[q>0]*np.log(q[q>0]))/np.log(4))
 return dict(common=common,stable=stable,observed=observed,runs=runs,longest=longest,pencil_length=length,hankel_rows=rows,quarter_entropy=entropy)

def observed_ar_holdout(values,runs):
 """Cheap one-step predictability check; no recursive gap reconstruction."""
 if not runs:return 0.,False,{}
 a,b=max(runs,key=lambda ab:ab[1]-ab[0]);count=b-a
 if count<32:return 0.,False,dict(segment_samples=count)
 if count>256:
  a+=(count-256)//2;b=a+256
 y=values[a:b];p=4;split=int(len(y)*2/3)
 design=np.stack([y[p-lag:len(y)-lag] for lag in range(1,p+1)],axis=1);target=y[p:]
 train_count=split-p;xx,yy=design[:train_count],target[:train_count];test_x,test_y=design[train_count:],target[train_count:]
 energy=float(np.mean(abs(test_y)**2)) if len(test_y) else 0.
 if energy<=1e-28:return 0.,False,dict(segment_samples=len(y),train_rows=len(xx),holdout_rows=len(test_y))
 gram=xx.conj().T@xx;ridge=1e-6*float(np.trace(gram).real)/p
 try:coef=np.linalg.solve(gram+np.eye(p)*max(ridge,np.finfo(float).tiny),xx.conj().T@yy)
 except np.linalg.LinAlgError:return 0.,False,dict(segment_samples=len(y),linear_solve_failed=True)
 error=float(np.mean(abs(test_x@coef-test_y)**2)/energy)
 result=float(np.log1p(min(max(error,0.),1e6)))
 return result,True,dict(segment_samples=len(y),segment_start=a,train_rows=len(xx),holdout_rows=len(test_y),
  one_step_observed_validation_nmse=error,order=4,recursive_gap_prediction=False)

def multiple_ridges(values,spectrum,fs,band):
 power=abs(spectrum)**2;energy=power.sum(0);valid=energy>.01*max(float(energy.max()),1e-300)
 frequencies=np.fft.fftfreq(power.shape[0],1/fs);inside=abs(frequencies)<=band
 peaks=(power>np.roll(power,1,axis=0))&(power>=np.roll(power,-1,axis=0))
 peaks&=power>=power.max(0)[None,:]*10**(-12/10);peaks&=inside[:,None]
 count=float(peaks[:,valid].sum(0).mean()) if valid.any() else 0.
 expanded=peaks|np.roll(peaks,1,axis=0)|np.roll(peaks,-1,axis=0)
 persistence=expanded[:,valid].mean(1) if valid.any() else np.zeros(power.shape[0])
 total=float(power[inside][:,valid].sum())
 stationary=float(np.sum(power[inside][:,valid]*persistence[inside,None])/total) if total else 0.
 global_power=abs(np.fft.fft(values*np.hanning(len(values))))**2
 p,_=find_peaks(global_power,distance=4);p=p[abs(np.fft.fftfreq(len(values),1/fs)[p])<=band]
 heights=np.sort(global_power[p])[::-1];secondary=float(heights[1]/heights[0]) if len(heights)>1 and heights[0]>0 else 0.
 return count,stationary,secondary,dict(valid_tf_frames=int(valid.sum()),observed_only=True,peak_count_is_not_target_count=True)

def extract_features_enhanced(observed,public,*,return_diagnostics=False):
 x=np.asarray(observed,dtype=np.complex128);fs=float(public['sample_rate_hz']);band=float(public['max_beat_hz'])
 if x.ndim!=1 or len(x)<32 or not np.isfinite(x).all() or not 0<band<fs/2:raise ValueError('Require finite complexADC and publicfs/band')
 base=_base_features(x,fs,band)
 _,scale,times,spectrum,context,tf_mask=_analysis(x,fs,band)
 support=np.zeros(len(x)+1,np.int64);centers=np.rint(times[np.any(tf_mask,axis=0)]*fs).astype(int);left=context.hop_samples//2
 np.add.at(support,np.clip(centers-left,0,len(x)),1);np.add.at(support,np.clip(centers+context.hop_samples-left,0,len(x)),-1)
 mask=np.cumsum(support[:-1])>0;g=support_geometry(mask,fs);norm=x/scale if scale else x.copy();power=abs(norm)**2
 energy=float(power.sum());mask_energy=float(power[mask].sum()/energy) if energy else 0.
 normalized=np.divide(norm,g['common'],out=np.zeros(len(x),complex),where=g['observed'])
 adjacent=g['observed'][:-1]&g['observed'][1:];a,b=normalized[:-1][adjacent],normalized[1:][adjacent]
 denom=float(np.sqrt(np.sum(abs(a)**2)*np.sum(abs(b)**2)));coherence=float(np.clip(abs(np.vdot(a,b))/denom,0,1)) if denom else 0.
 ar,available,ar_diag=observed_ar_holdout(normalized,g['runs']);peaks,stationary,secondary,tf_diag=multiple_ridges(norm,spectrum,fs,band)
 extra=(float(mask.mean()),mask_energy,float(g['observed'].sum()/max(int(g['stable'].sum()),1)),g['longest']/len(x),
  g['hankel_rows']/len(x),g['quarter_entropy'],coherence,ar,float(available),peaks,stationary,secondary)
 result=np.r_[base,np.asarray(extra,float)]
 if result.shape!=(44,) or not np.isfinite(result).all():raise ValueError('Invalid enhanced feature vector')
 if not return_diagnostics:return result
 return result,dict(version=FEATURE_VERSION,feature_dimension=44,base_dimension=32,parameters=PARAMETERS,
  observed_only=True,truth_or_source_clock_used=False,actual_candidate_state_used=False,candidate_reconstruction_executed=False,
  support_stable_samples=int(g['stable'].sum()),support_unmasked_samples=int(g['observed'].sum()),
  support_longest_samples=g['longest'],support_segments=g['runs'],initial_hankel_rows=g['hankel_rows'],
  initial_pencil_length=g['pencil_length'],os_tf_mask_fraction=float(tf_mask.mean()),local_observed_validation=ar_diag,
  multiple_ridge_diagnostics=tf_diag,necessary_geometry_does_not_guarantee_recovery=True)

extract_features=extract_features_enhanced
extract=extract_features_enhanced
