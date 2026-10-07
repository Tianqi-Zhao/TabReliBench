"""Run with python -m conformal --help."""
import argparse
import pickle
from pathlib import Path
from evaluation.store import ArtifactStore
from .registry import METHODS, default_methods
from .pipeline import ConformalPipeline, load_bundle
from .metrics import ConformalMetricsCalculator


def main():
    p=argparse.ArgumentParser(description='Conformal prediction from saved model outputs and benchmark metrics')
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare',help='Fit TFM on proper training and freeze train/cal/test predictive grids')
    source=prep.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset-id',type=int);source.add_argument('--split',type=Path);prep.add_argument('--seed',type=int,default=0)
    prep.add_argument('--base-model',choices=['tabpfnv2','tabpfnv2.5','tabpfnv3','tabiclv2'],required=True)
    prep.add_argument('--output',type=Path,required=True);prep.add_argument('--n-estimators',type=int,default=8)
    prep.add_argument('--max-n',type=int,default=10000);prep.add_argument('--reference',type=Path)
    inputs=sub.add_parser('prepare-predictions',help='Split held-out predictions into auxiliary/calibration/test; no base refit')
    inputs.add_argument('--input',type=Path,required=True)
    inputs.add_argument('--output',type=Path,required=True)
    inputs.add_argument('--n-estimators',type=int,help='Required for TFM predictions; not applicable to BART')
    run=sub.add_parser('run',help='Run selected postprocessors and evaluate saved intervals')
    run.add_argument('--resume',action='store_true')
    run.add_argument('--input',type=Path,required=True);run.add_argument('--output',type=Path,required=True)
    run.add_argument('--methods',nargs='+',choices=METHODS)
    run.add_argument('--alphas',nargs='+',type=float,default=[.05,.1,.15,.2]);run.add_argument('--device',default='cpu')
    ev=sub.add_parser('evaluate',help='Recompute metrics from a normalized prediction or tree')
    ev.add_argument('--bundle-root',type=Path,help='Portable root containing splits/ for migrated records');ev.add_argument('--input',type=Path,required=True);ev.add_argument('--output',type=Path,required=True)
    summ=sub.add_parser('summarize',help='Paired method-minus-Vanilla tables, datasets equally weighted')
    summ.add_argument('--input',type=Path,required=True);summ.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare':
        from .prepare import prepare
        print(prepare(args.dataset_id,args.seed,args.base_model,args.output,args.n_estimators,args.max_n,args.reference,args.split))
    elif args.command=='prepare-predictions':
        from .predictions import prepare_predictions
        print(prepare_predictions(args.input,args.output,args.n_estimators))
    elif args.command=='run':
        paths=sorted(args.input.rglob('*.pkl')) if args.input.is_dir() else [args.input]
        if not paths:raise ValueError('No input bundles')
        from .experiment import ConformalExperiment
        for path in paths:
            experiment=ConformalExperiment.from_prepared(path,args.output)
            for method in args.methods or default_methods(experiment.bundle['base_model']):
                print(experiment.run(method,args.alphas,device=args.device,resume=args.resume))
    elif args.command=='summarize':
        from .summarize import summarize
        print(summarize(args.input,args.output))
    else:
        paths=sorted(args.input.rglob('*.pkl')) if args.input.is_dir() else [args.input]
        if not paths:raise ValueError('No predictions')
        count=0
        for path in paths:
            with path.open('rb') as f:record=pickle.load(f)
            if record.get('schema')!='conformal_prediction_v1':continue
            if 'split_ref' in record:
                if args.bundle_root is None:raise ValueError('Migrated records require --bundle-root')
                split_path=(args.bundle_root/record['split_ref']).resolve()
                split_path.relative_to(args.bundle_root.resolve())
                with split_path.open('rb') as f:split=pickle.load(f)
                record.update(X_test=split['X_test'],y_test=split['y_test'],feature_names=list(split['X_test'].columns))
            result=ConformalMetricsCalculator(sorted(record['intervals'])).compute_for_record(record)
            relative=path.relative_to(args.input) if args.input.is_dir() else Path(path.name)
            dest=args.output/relative.with_name(relative.stem+'_metrics.pkl')
            if dest.exists():raise FileExistsError(dest)
            ArtifactStore._atomic_pickle(dest,result);count+=1
        if not count:raise ValueError('No conformal_prediction_v1 records found')
        print(f'Evaluated {count} records')

if __name__=='__main__':main()
