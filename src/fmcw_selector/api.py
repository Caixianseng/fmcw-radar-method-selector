"""Inference receives only observed complex ADC and public receiver parameters."""
from functools import lru_cache
from pathlib import Path
import hashlib, json, time
import joblib
import numpy as np

METHODS = ('MP','C-AR','JSS')
FAILED_MP = ('insufficient_observations','no_supported_pencil_model','unstable_extrapolation')
DATA = Path(__file__).resolve().parent/'data'


@lru_cache(maxsize=1)
def _default_model():
    path=DATA/'selector.joblib'
    metadata=json.loads((DATA/'metadata.json').read_text(encoding='utf-8'))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['model_sha256']:
        raise RuntimeError('Frozen selector checksum mismatch')
    return joblib.load(path)


def load_model(path=None):
    """Load trusted model artifacts only; default model is checksum-verified."""
    return _default_model() if path is None else joblib.load(path)


def extract_features(observed, public):
    from ._core.compact import extract
    return extract(observed,public)


def predict(features, model=None):
    model=load_model() if model is None else model
    x=np.asarray(features,float)
    if x.ndim==1:x=x[None,:]
    if x.ndim!=2 or x.shape[1]!=32 or not np.isfinite(x).all():
        raise ValueError('Expected finite (N,32) observation features')
    out=model['y_scaler'].inverse_transform(np.asarray(model['net'].predict(
        model['x_scaler'].transform(x[:,model['columns']]))).reshape(len(x),9))
    catastrophic=np.clip(out[:,3:6],0,1);weak=np.clip(out[:,6:9],0,1)
    penalties=model['penalties']
    cost=out[:,:3]+penalties[0]*catastrophic+penalties[1]*weak
    return dict(negative_sinr=out[:,:3],catastrophe=catastrophic,
                weak_distortion=weak,cost=cost,choice=cost.argmin(1),
                calibrated_probabilities=False)


def execute(method, observed, public):
    from ._core import mp_improved,jss_improved,chirplet_observed_upgrade,ar_observed_upgrade
    if method=='MP':return mp_improved.process(observed,public,'validated')
    if method=='JSS':return jss_improved.process(observed,public,'residual10')
    if method=='C-AR':
        residual,cd=chirplet_observed_upgrade.process(observed,public,mode='robust_slow')
        y,ad=ar_observed_upgrade.process(residual,public,mode='validated')
        return y,dict(variant='robust_validated',chirplet=cd,ar=ad,truth_inputs_used=False)
    raise ValueError('Method must be MP, C-AR or JSS')


def _failed(diag):return diag.get('legacy_status',diag.get('status')) in FAILED_MP


def process(observed, public, model=None):
    """Select first, then execute one action; explicit MP failure falls back to C-AR."""
    x=np.asarray(observed,complex)
    start=time.perf_counter();features=extract_features(x,public);p=predict(features,model)
    requested=int(p['choice'][0]);selected=requested
    y,d=execute(METHODS[requested],x,public)
    if requested==0 and _failed(d):
        y,cd=execute('C-AR',x,public);selected=1;d=dict(mp_failure=d,fallback=cd)
    if y.shape!=x.shape or not np.isfinite(y).all():raise RuntimeError('Invalid recovered ADC')
    return y,dict(method=METHODS[selected],requested_method=METHODS[requested],
          selected_index=selected,features=features,**p,diagnostics=d,
          truth_inputs_used=False,total_s=time.perf_counter()-start)


def process_all(observed, public, model=None):
    """Offline comparison; selection is made before any clean reference is read."""
    features=extract_features(observed,public);p=predict(features,model)
    outputs=[];details=[]
    for method in METHODS:
        tic=time.perf_counter();y,d=execute(method,observed,public)
        outputs.append(y);details.append(dict(method=method,seconds=time.perf_counter()-tic,diagnostics=d))
    requested=int(p['choice'][0]);choice=1 if requested==0 and _failed(details[0]['diagnostics']) else requested
    return np.stack(outputs),dict(features=features,prediction=p,choice=choice,
                   requested=requested,method=METHODS[choice],methods=details,
                   mp_failed=_failed(details[0]['diagnostics']),truth_inputs_used=False)
