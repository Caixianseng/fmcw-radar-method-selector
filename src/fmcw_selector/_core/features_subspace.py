"""Original44 plus four small observed-Hankel subspace descriptors.

For K distinct complex exponentials a Hankel matrix has rank at most K.
This is the structural assumption underlying the existing Matrix Pencil
method (Wang/Ding/Yarovoy2021, Eq15--24); these features test that assumption
on OBSERVED contiguous pieces only. They do not estimate poles, fit target
amplitudes, extrapolate a missing sample or execute a mitigation candidate.

Fixed dimensions are L=16 (17 adjacent samples), at most256 chronological
rows, four leading energies/subspaces and gaps within the first8ranks.
Unavailable descriptors are explicitly -1, outside their physical [0,1]
range. Public-window compensation remains approximate under FIR/delay.
"""
from __future__ import annotations
import numpy as np
from fmcw_selector._core.features_enhanced import (FEATURE_NAMES as FEATURE_NAMES44,
                               extract_features_enhanced, support_geometry)
from fmcw_selector._core.anti_interference.methods import _analysis

FEATURE_VERSION='observed44_plus4_small_hankel_subspace_v1'
EXTRA_FEATURE_NAMES=(
    'small_hankel_normalized_entropy_effective_rank',
    'small_hankel_top4_energy_fraction',
    'small_hankel_largest_relative_gap_ranks1to8',
    'small_hankel_temporal_top4_subspace_disagreement',
)
FEATURE_NAMES48=tuple(FEATURE_NAMES44)+EXTRA_FEATURE_NAMES
FEATURE_NAMES=FEATURE_NAMES48
PARAMETERS=dict(hankel_shift_length=16,hankel_columns=17,maximum_rows=256,
                leading_energy_rank=4,largest_gap_maximum_rank=8,
                temporal_subspace_rank=4,unavailable_value=-1.,
                missing_samples_filled=False,candidate_methods_executed=False)
EXTRA_FEATURE_DEFINITIONS=(
 (EXTRA_FEATURE_NAMES[0],'exp(entropy of squared singular spectrum)/17; lower indicates fewer dominant exponential components','[1/17,1]; -1 unavailable'),
 (EXTRA_FEATURE_NAMES[1],'Sum of first4 squared singular values / total observed Hankel energy','[0,1]; -1 unavailable'),
 (EXTRA_FEATURE_NAMES[2],'Largest1-s[k+1]/s[k] for k=1..8, ignoring exactly zero denominators','[0,1]; -1 unavailable'),
 (EXTRA_FEATURE_NAMES[3],'1-||V_early^H V_late||F²/4 for leading4 right singular subspaces of chronological half rows','[0,1]; -1 unavailable including numerical rank<4'),
)


def observed_hankel_matrix(observed,public):
    """Small stack; never create adjacency across an observed-mask gap."""
    x=np.asarray(observed,dtype=np.complex128)
    fs=float(public['sample_rate_hz']);band=float(public['max_beat_hz'])
    if x.ndim!=1 or len(x)<32 or not np.isfinite(x).all() or not 0<band<fs/2:
        raise ValueError('Finite complex observation and public fs/band required')
    _,scale,times,_,context,tf_mask=_analysis(x,fs,band)
    counts=np.zeros(len(x)+1,np.int64)
    centers=np.rint(times[np.any(tf_mask,axis=0)]*fs).astype(int)
    left=context.hop_samples//2
    np.add.at(counts,np.clip(centers-left,0,len(x)),1)
    np.add.at(counts,np.clip(centers+context.hop_samples-left,0,len(x)),-1)
    mask=np.cumsum(counts[:-1])>0
    geometry=support_geometry(mask,fs)
    starts=np.concatenate([np.arange(a,b-16,dtype=int) for a,b in geometry['runs'] if b-a>=17]) if any(b-a>=17 for a,b in geometry['runs']) else np.empty(0,int)
    possible=len(starts)
    if possible>256:starts=starts[np.linspace(0,possible-1,256,dtype=int)]
    norm=x/scale if scale else x.copy()
    compensated=np.divide(norm,geometry['common'],out=np.zeros(len(x),complex),where=geometry['observed'])
    matrix=compensated[starts[:,None]+np.arange(17)[None,:]]
    diag=dict(rows_available=possible,rows_used=len(starts),columns=17,
              source_start_indices=starts.tolist(),support_unmasked_samples=int(geometry['observed'].sum()),
              support_longest_samples=geometry['longest'],public_window_compensation=True,
              no_adjacency_across_missing_gap=True,observed_only=True,
              truth_inputs_used=False,actual_candidate_state_used=False,
              candidate_reconstruction_executed=False,poles_estimated=False,missing_samples_filled=False)
    return matrix,diag


def descriptors_from_hankel(matrix):
    """Only three bounded small SVDs; no pencil eigenproblem or recovery."""
    h=np.asarray(matrix,np.complex128)
    if h.ndim!=2 or h.shape[1]!=17 or not np.isfinite(h).all():
        raise ValueError('Require finite observed Hankel rows with17columns')
    values=np.full(4,-1.,float)
    diag=dict(available=[False]*4)
    if len(h)<2 or not np.any(h):return values,diag
    singular=np.linalg.svd(h,full_matrices=False,compute_uv=False)
    energy=singular**2;total=float(energy.sum())
    if total<=np.finfo(float).tiny:return values,diag
    p=energy/total
    entropy=-float(np.sum(p[p>0]*np.log(p[p>0])))
    values[0]=float(np.clip(np.exp(entropy)/17,0,1))
    values[1]=float(np.clip(p[:4].sum(),0,1))
    upper=min(8,len(singular)-1)
    ratios=np.divide(singular[1:upper+1],singular[:upper],out=np.ones(upper),where=singular[:upper]>0)
    if upper:values[2]=float(np.clip(np.max(1-ratios),0,1))
    diag.update(available=[True,True,bool(upper),False],singular_values_relative=(singular/singular[0]).tolist(),
                normalized_rank_denominator=17)
    split=len(h)//2
    if min(split,len(h)-split)>=4:
        _,a,vha=np.linalg.svd(h[:split],full_matrices=False)
        _,b,vhb=np.linalg.svd(h[split:],full_matrices=False)
        tol_a=np.finfo(float).eps*max(h[:split].shape)*a[0]
        tol_b=np.finfo(float).eps*max(h[split:].shape)*b[0]
        rank_a=int((a>tol_a).sum());rank_b=int((b>tol_b).sum())
        diag.update(temporal_numerical_rank_early=rank_a,temporal_numerical_rank_late=rank_b,
                    temporal_rows_early=split,temporal_rows_late=len(h)-split)
        if min(rank_a,rank_b)>=4:
            overlap=float(np.linalg.norm(vha[:4]@vhb[:4].conj().T,'fro')**2/4)
            values[3]=float(np.clip(1-overlap,0,1));diag['available'][3]=True
    return values,diag


def extract_subspace_only(observed,public,*,return_diagnostics=False):
    h,geometry=observed_hankel_matrix(observed,public)
    values,diagnostics=descriptors_from_hankel(h)
    if not np.isfinite(values).all():raise ValueError('Nonfinite subspace descriptors')
    diagnostics.update(geometry,feature_version=FEATURE_VERSION,parameters=PARAMETERS)
    return (values,diagnostics) if return_diagnostics else values


def extract_features_subspace(observed,public,*,return_diagnostics=False):
    original=extract_features_enhanced(observed,public)
    extra,diagnostics=extract_subspace_only(observed,public,return_diagnostics=True)
    result=np.r_[original,extra]
    if result.shape!=(48,) or not np.isfinite(result).all():raise ValueError('Invalid48feature vector')
    if return_diagnostics:return result,diagnostics
    return result


extract_features=extract_features_subspace
extract=extract_features_subspace
