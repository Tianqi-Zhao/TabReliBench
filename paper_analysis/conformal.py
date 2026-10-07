"""Validate and aggregate the paper conformal panel, without rendering figures."""
from pathlib import Path
import argparse, gzip, hashlib, itertools, json
import numpy as np
import pandas as pd

MODELS=['tabpfnv3.5','tabpfnv3','tabpfnv2.5','tabpfnv2','tabiclv2','causilo','tabdpt1.3','bart']
LABELS=['TabPFNv3.5','TabPFNv3','TabPFNv2.5','TabPFNv2','TabICLv2','Causilo','TabDPT1.3','BART']
METHODS=['Vanilla','SCP-ABS','SCP-NORM','SCP-CQR','SCP-PIT','CalLCP','RLCP','SLCP','PCP','RCP']
ALPHAS=[.05,.10,.15,.20]
METRICS=['marginal_coverage','worst_slab_coverage','cov_abs_dev','wsc_abs_dev','avg_width_norm','infinite_interval_rate','empty_set_rate','interval_score','pinball_mean']
def dump(path,value): path.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def aggregate(df,keys,metrics=METRICS):
    return df.groupby(keys,sort=True)[metrics].agg(lambda x:np.mean(x.to_numpy())).reset_index()
def analyze():
    for sub in ['tables','data']: (ROOT/sub).mkdir(parents=True,exist_ok=True)
    with gzip.open(INPUT,'rt') as f: raw=[json.loads(l) for l in f]
    header,done=raw[0],raw[-1]; records=raw[1:-1]
    assert done['type']=='complete' and len(records)==done['metric_files']
    manifest=header['manifest'];datasets=manifest['datasets']
    expected=set(itertools.product(datasets,manifest['seeds'],MODELS,METHODS))
    inventory=[];rows=[];identities={};points={};usage=[]
    for r in records:
        source_method=r.get('source_method',r['method']).removesuffix('-standard')
        if source_method == 'RCP' and r.get('rcp_semantics') != 'a_fit_c_calibrate':
            continue
        m=r['method'].removesuffix('-standard')
        if m == 'RCP+': m='RCP'
        key=(r['dataset_id'],r['seed'],r['base_model'],m)
        assert key in expected
        inventory.append(dict(zip(['dataset_id','seed','base_model','method'],key))|{k:r[k] for k in ['source_file','source_sha256']})
        assert set(map(float,r['metrics']))==set(ALPHAS)
        assert r['protocol']=='cached_holdout_equal_budget50_test50_v1'
        u=r['data_usage'];p=r['provenance'];pair=key[:2]
        identity=(r['y_sha256'],u['test_rows_sha256'],u['available_rows_sha256'],r['n_test'])
        assert pair not in identities or identities[pair]==identity, ('test identity',key)
        identities[pair]=identity
        pm=tuple(r['alpha_free'][k] for k in ['r2','rmse','mae'])
        assert key[:3] not in points or np.allclose(points[key[:3]],pm,rtol=0,atol=0,equal_nan=True)
        points[key[:3]]=pm
        assert p['base_refitted'] is False
        if key[2]!='bart': assert p['n_estimators']==8
        assert u['test_count']==r['n_test']
        n=u['available_label_count']
        if m in ['SCP-ABS','SCP-NORM','SCP-CQR','SCP-PIT','CalLCP','RLCP']:
            assert u['calibration_count']==n and u['auxiliary_fit_count']==0
        elif m in ['RCP','SLCP','PCP']:
            assert u['calibration_count']+u['auxiliary_fit_count']==n
        else: assert u['calibration_count']==u['auxiliary_fit_count']==0
        for a,v in r['metrics'].items():
            a=float(a);v=dict(v);v.setdefault('empty_set_rate',0.)
            assert np.isclose(v['cov_abs_dev'],abs(v['marginal_coverage']-(1-a)))
            assert np.isclose(v['wsc_abs_dev'],abs(v['worst_slab_coverage']-(1-a)),equal_nan=True)
            assert np.isclose(v['avg_width_norm'],v['avg_length']/max(r['y_eval_std'],1e-10))
            assert 0<=v['infinite_interval_rate']<=1 and 0<=v['empty_set_rate']<=1
            if v['empty_set_rate']>0: assert np.isnan(v['interval_score']) and np.isnan(v['pinball_mean'])
            else: assert np.isclose(v['interval_score']*a/4,v['pinball_mean'])
            rows.append(dict(zip(['dataset_id','seed','base_model','method'],key))|dict(alpha=a,y_eval_std=r['y_eval_std'],**{k:v[k] for k in METRICS}))
        usage.append(dict(key=key,provenance=p,data_usage=u))
    inv=pd.DataFrame(inventory); assert not inv.duplicated(['dataset_id','seed','base_model','method']).any()
    present=set(map(tuple,inv[['dataset_id','seed','base_model','method']].to_numpy()))
    missing=sorted(expected-present)
    df=pd.DataFrame(rows)
    invalid=set(map(tuple,df.loc[~np.isfinite(df.worst_slab_coverage),['dataset_id','seed']].to_numpy()))
    excluded=invalid|{x[:2] for x in missing}
    df['common_panel']=[(d,s) not in excluded for d,s in zip(df.dataset_id,df.seed)]
    common=df[df.common_panel];means=aggregate(common,['dataset_id','base_model','method','alpha'])
    summary=aggregate(means,['base_model','method','alpha'])
    nondeg=common[common.y_eval_std>1e-10]
    nondeg_summary=aggregate(aggregate(nondeg,['dataset_id','base_model','method','alpha']),['base_model','method','alpha'])
    vanilla_width=means[means.method.eq('Vanilla')][['dataset_id','base_model','alpha','avg_width_norm']]
    width_changes=means.merge(vanilla_width,on=['dataset_id','base_model','alpha'],suffixes=('', '_vanilla'),validate='many_to_one')
    width_changes['width_change_percent']=100*(width_changes.avg_width_norm/width_changes.avg_width_norm_vanilla-1)
    width_summary=width_changes.groupby(['base_model','method','alpha'],as_index=False).agg(
        median_width_change_percent=('width_change_percent','median'),n_datasets=('dataset_id','nunique'))
    width_summary.to_csv(ROOT/'tables/relative_width_changes.csv',index=False)
    ds=sorted(means.dataset_id.unique());draws=np.random.default_rng(20260930).integers(0,len(ds),(10000,len(ds)))
    contrasts=[]; mean_ci=[]
    for model,a,metric in itertools.product(MODELS,ALPHAS,['cov_abs_dev','wsc_abs_dev']):
        w=means[(means.base_model==model)&(means.alpha==a)].pivot(index='dataset_id',columns='method',values=metric).reindex(ds)
        for method in METHODS:
            x=w[method].to_numpy(); delta=x-w.Vanilla.to_numpy();assert np.isfinite(delta).all()
            lo,hi=np.quantile(delta[draws].mean(axis=1),[.025,.975])
            contrasts.append(dict(base_model=model,method=method,alpha=a,metric=metric,mean_delta=delta.mean(),ci_low=lo,ci_high=hi))
            lo,hi=np.quantile(x[draws].mean(axis=1),[.025,.975])
            mean_ci.append(dict(base_model=model,method=method,alpha=a,metric=metric,mean=x.mean(),ci_low=lo,ci_high=hi))
    c=pd.DataFrame(contrasts)
    status=common.assign(undefined_loss=common.interval_score.isna(),infinite_loss=np.isinf(common.interval_score)).groupby(['base_model','method','alpha']).agg(trials=('seed','size'),undefined_loss_trials=('undefined_loss','sum'),infinite_loss_trials=('infinite_loss','sum')).reset_index()
    for name,frame in [('source_inventory',inv),('split_metrics',df),('dataset_means',means),('dataset_equal_summary',summary),('nondegenerate_width_summary',nondeg_summary),('paired_error_bootstrap',c),('mean_error_bootstrap',pd.DataFrame(mean_ci)),('loss_status',status),('missing_configurations',pd.DataFrame(missing,columns=['dataset_id','seed','base_model','method']))]:
        frame.to_csv(ROOT/'tables'/f'{name}.csv',index=False)
    for k,sample in header['samples'].items():
        if k.endswith('-standard'):
            for p in sample['params'].values():
                assert p['score']=='quantile_standard' and p['empty_set_policy']=='exact_sublevel_set_v1'
        if k.endswith('/PCP'):
            for p in sample['params'].values(): assert 'official' in p['implementation']
    report=dict(source_root=header['root'],exported_at=header['exported_at'],source_sha256=hashlib.sha256((INPUT).read_bytes()).hexdigest(),source_files=len(inventory),raw_source_files=len(records),expected_files=len(expected),missing=missing,undefined_wsc_pairs=sorted(invalid),excluded_pairs=sorted(excluded),datasets=int(common.dataset_id.nunique()),dataset_seed_pairs=len(common[['dataset_id','seed']].drop_duplicates()),zero_sd_pairs=sorted(set(map(tuple,common.loc[common.y_eval_std<=1e-10,['dataset_id','seed']].to_numpy()))),nondegenerate_dataset_seed_pairs=len(nondeg[['dataset_id','seed']].drop_duplicates()),test_identity_matched=True,point_predictions_unchanged=True,allocation_checked=True,metric_arithmetic_checked=True,imputed_records=0,rcp_source_label=('RCP' if 'rcp_naming' in manifest else 'RCP+'),rcp_output_label='RCP',rcp_archive=manifest.get('rcp_naming',{}).get('archive'),bootstrap_replicates=10000,bootstrap_seed=20260930,parameter_verification='one prediction sample per method and kind; not full-array reevaluation')
    dump(ROOT/'validation.json',report);dump(ROOT/'data/source_documents.json',header)
    print(json.dumps(report,default=str,indent=2))
    return summary,c,nondeg_summary


def main():
    global ROOT, INPUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();ROOT=args.output;INPUT=args.input
    analyze()

if __name__=='__main__':main()
