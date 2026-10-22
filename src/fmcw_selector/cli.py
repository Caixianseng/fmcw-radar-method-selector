"""Generate ADC, run frozen inference, evaluate a library, or fit an offline model."""
from pathlib import Path
from dataclasses import asdict,is_dataclass
import argparse,json
import numpy as np
from threadpoolctl import threadpool_limits
from .api import METHODS,load_model,process_all


def native(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if is_dataclass(x):return asdict(x)
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)


def write_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=native,allow_nan=False),encoding='utf-8')


def seeds(value):
    if ':' in value:
        a,b=value.split(':');return list(range(int(a),int(b)+1))
    return [int(x) for x in value.split(',')]


def evaluate(signals,public,rows,model,destination):
    from ._core.metrics import prepare_reference,score_prepared
    results=[];choices=[];features=[];score_rows=[];metric_rows=[];valid=[];failures=[];diagnostics=[]
    for i,x in enumerate(signals['observed']):
        outputs,d=process_all(x,public,model)
        # Only now access clean/noise; all inference decisions are already complete.
        ref=prepare_reference(signals['clean'][i],public,signals['noise'][i])
        ss=[score_prepared(y,x,ref) for y in outputs]
        ins=score_prepared(x,x,ref)
        score_rows.append([dict(method='原始接收',**ins)]+[dict(method=name,**v) for name,v in zip(METHODS,ss)])
        metric_rows.append([[v['nmse'],-v['sinr_db'],v['sinr_db'],float(v['severe_weak_distortion'])] for v in ss])
        results.append(outputs);choices.append(d['choice']);features.append(d['features']);valid.append(ref.available);failures.append(d['mp_failed']);diagnostics.append(d)
    outputs=np.stack(results);choices=np.asarray(choices);metrics=np.asarray(metric_rows)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/'processed.npz',outputs=outputs,features=np.stack(features),
        choices=choices,metrics=metrics,reference_valid=valid,mp_failed=failures,
        groups=np.array([r['seed'] for r in rows]),method_names=np.array(METHODS))
    write_json(destination/'scores.json',score_rows);write_json(destination/'diagnostics.json',diagnostics)
    actual=metrics[np.arange(len(choices)),choices]
    summary={name:dict(mean_sinr_db=float(metrics[:,i,2].mean()),mean_nmse=float(metrics[:,i,0].mean())) for i,name in enumerate(METHODS)}
    summary['MLP']=dict(mean_sinr_db=float(actual[:,2].mean()),mean_nmse=float(actual[:,0].mean()),choice_counts={n:int(np.sum(choices==i)) for i,n in enumerate(METHODS)})
    write_json(destination/'summary.json',summary)
    from .report import render
    render(destination/'report.html',signals,public,rows,outputs,choices,score_rows)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo',help='Run all three bundled paper ADC examples')
    demo.add_argument('--out',default='outputs/demo');demo.add_argument('--examples',default='examples');demo.add_argument('--model')
    sim=sub.add_parser('simulate',help='Generate independent-clock traffic ADC and compare methods')
    sim.add_argument('--seed',type=int,default=97028);sim.add_argument('--windows',type=float,nargs='+',default=[.01,20.,39.]);sim.add_argument('--chirps',type=int,default=8);sim.add_argument('--geometry',action='store_true');sim.add_argument('--out',default='outputs/simulation');sim.add_argument('--model')
    dat=sub.add_parser('dataset',help='Generate observation features and offline three-action labels')
    dat.add_argument('--seeds',required=True,type=seeds);dat.add_argument('--windows',type=float,nargs='+',default=[.01,20.,39.]);dat.add_argument('--chirps',type=int,default=8);dat.add_argument('--out',default='datasets/development.npz')
    fit=sub.add_parser('train',help='Fit on independent development sessions; keep holdout external')
    fit.add_argument('--dataset',required=True);fit.add_argument('--out',default='outputs/trained_selector.joblib')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command=='train':
            from .training import train
            result=train(args.dataset,args.out);write_json(Path(args.out).with_suffix('.json'),result);print(json.dumps(result));return
        if args.command=='demo':
            folder=Path(args.examples);manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            values={k:[] for k in ('observed','clean','interference','noise')};rows=[];public=None
            for rec in manifest:
                with np.load(folder/rec['file'],allow_pickle=False) as z:
                    p=json.loads(str(z['public_parameters'].item()))
                    if public is not None and p!=public:raise ValueError('Demo public parameter mismatch')
                    public=p
                    for k in values:values[k].append(z[k].copy())
                old=rec['scene'];rows.append(dict(seed=rec['seed'],window_s=old['window_start_s'],chirp=old['chirp_index'],target_paths=old['target_paths'],configured_extreme_event=old['configured_extreme_event'],scene=old['scene_frame']))
            result=evaluate({k:np.stack(v) for k,v in values.items()},public,rows,load_model(args.model),args.out)
        elif args.command=='simulate':
            from .scene import generate
            values,public,rows=generate(args.seed,args.windows,args.chirps,args.geometry)
            dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'adc.npz',**values,public_parameters=np.array(json.dumps(public)))
            write_json(dest/'scene.json',rows)
            result=evaluate(values,public,rows,load_model(args.model),dest)
        else:
            from .scene import generate
            from ._core.metrics import prepare_reference,score_prepared
            features=[];metrics=[];groups=[];valid=[];failed=[]
            for seed in args.seeds:
                values,public,rows=generate(seed,args.windows,args.chirps)
                for i,x in enumerate(values['observed']):
                    yy,diag=process_all(x,public)
                    ref=prepare_reference(values['clean'][i],public,values['noise'][i])
                    ss=[score_prepared(y,x,ref) for y in yy]
                    features.append(diag['features']);groups.append(seed);valid.append(ref.available);failed.append(diag['mp_failed'])
                    metrics.append([[s['nmse'],-s['sinr_db'],s['sinr_db'],float(s['severe_weak_distortion'])] for s in ss])
                print('completed session',seed,flush=True)
            Path(args.out).parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(args.out,features=features,metrics=metrics,groups=groups,reference_valid=valid,mp_failed=failed)
            result=dict(dataset=args.out,sessions=len(args.seeds),chirps=len(features))
        print(json.dumps(result,ensure_ascii=False,default=native,indent=2))


if __name__=='__main__':main()
