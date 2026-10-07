"""Export locally generated conformal metrics for common-panel analysis."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np

METHODS = {'Vanilla','SCP-ABS','SCP-NORM','SCP-CQR-standard','SCP-PIT',
           'CalLCP-standard','RLCP-standard','SLCP-standard','PCP','RCP-standard'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=[];samples={}
    def sha(x):return hashlib.sha256(x).hexdigest()
    for root in args.results:
        for path in sorted((root/'regression/metrics').glob('*_metrics.pkl')):
            raw=path.read_bytes();r=pickle.loads(raw)
            if r['method'] not in METHODS:continue
            key=r['base_model']+'/'+r['method']
            if key not in samples:
                pred=root/'regression/predictions'/path.name.replace('_metrics.pkl','.pkl')
                with pred.open('rb') as f:record=pickle.load(f)
                samples[key]={'params':record['params']}
            rows.append(dict(type='metric',source_file=str(path),source_sha256=sha(raw),
                **{k:r.get(k) for k in ['dataset_id','seed','base_model','protocol','provenance','data_usage','n_test']},
                method=r['method'],source_method=r['method'],rcp_semantics='a_fit_c_calibrate',
                y_sha256=sha(np.asarray(r['y_test']).tobytes()),
                y_eval_std=float(np.std(r['y_test'][len(r['y_test'])//2:])),
                alpha_free=r['alpha_free']['per_dataset'],metrics={str(a):s['per_dataset'] for a,s in r['alpha_dependent'].items()}))
    if not rows:raise ValueError('No conformal metrics found')
    repo=Path(__file__).resolve().parents[1]
    ids=[int(x) for x in (repo/'dataset_ids_regression.txt').read_text().splitlines() if x.strip() and not x.startswith('#')]
    header=dict(type='metadata',root=[str(r) for r in args.results],samples=samples,
                exported_at=datetime.now(timezone.utc).isoformat(),manifest=dict(datasets=ids,seeds=list(range(5)),rcp_naming={}))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    with gzip.open(args.output,'wt') as f:
        for r in [header,*rows,dict(type='complete',metric_files=len(rows))]:f.write(json.dumps(r,default=default)+'\n')

if __name__=='__main__':main()
