"""Spearman associations and per-metric top-five unions used in the paper."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .components import AUXILIARY, feature_groups

SPECS={'classification':{'Accuracy':'accuracy','Brier':'brier_score','Confidence ECE':'confidence_ece_em','Classwise ECE':'classwise_ece_em'},'regression':{'R2':'r2','Normalized CRPS':'crps_normalized','PIT-KS':'pit_ks_stat','Coverage error':'alpha_dependent.0.1.per_dataset.cov_abs_dev','WSC error':'alpha_dependent.0.1.per_dataset.wsc_abs_dev','Total error':'alpha_dependent.0.1.per_dataset.total_abs_dev'}}
ORDER=['tabpfnv3.5','tabpfnv3','tabpfnv2.5','tabpfnv2','tabiclv2','tabicl','tabdpt1.3','tabdpt','causilo','limix2','tabfm','realmlp_hpo','realmlp','xgboost_quantile_hpo','xgboost_quantile','bart']
K=5
def bh(p):
 p=np.asarray(p);ix=np.argsort(p);q=np.empty(len(p));q[ix]=np.minimum(1,np.minimum.accumulate((p[ix]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);return q

def correlations():
 metrics=pd.read_csv(INPUT/'dataset_metrics.tsv',sep='\t');raw=pd.read_csv(INPUT/'dataset_feature_means.tsv',sep='\t');out=[]
 for task,spec in SPECS.items():
  f=raw[raw.task.eq(task)].pivot(index='dataset_id',columns='feature',values='mean_across_five_splits');f['context_to_feature_ratio']=f.n_train/f.n_features.replace(0,np.nan)
  for metric,key in spec.items():
   key=key if key.startswith('alpha_') else 'alpha_free.per_dataset.'+key
   w=metrics[metrics.task.eq(task)&metrics.metric.eq(key)].pivot(index='dataset_id',columns='model',values='mean_across_five_splits');models=[x for x in ORDER if x in w];w=w[models];assert np.isfinite(w).all().all()
   for tier,target,y in [('pooled','all_models',w.median(axis=1))]+[('individual',x,w[x]) for x in models]:
    for feature in FEATURES[task]:
     if feature in AUXILIARY:continue
     z=pd.concat([f[feature],y],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=p=np.nan;status='ok'
     if len(z)<(30 if task=='classification' else 20):status='insufficient_support'
     elif z.iloc[:,0].nunique()<3:status='feature_has_fewer_than_3_values'
     elif z.iloc[:,1].nunique()<2:status='constant_outcome'
     else:rho,p=spearmanr(z.iloc[:,0],z.iloc[:,1])
     out.append(dict(task=task,metric=metric,tier=tier,target=target,feature=feature,n=len(z),rho=rho,p=p,status=status))
 c=pd.DataFrame(out);c['q_family']=np.nan
 for _,g in c[c.status.eq('ok')].groupby(['task','metric','tier']):c.loc[g.index,'q_family']=bh(g.p)
 c.to_csv(T/'section5_topk_correlations.csv',index=False)
 return c

def union(g,cols,colkey):
 rows=[]
 for col in cols:
  z=g[g[colkey].eq(col)&g.status.eq('ok')].assign(strength=lambda x:x.rho.abs()).sort_values(['strength','feature'],ascending=[False,True]).head(K)
  for row in z.itertuples():
   selection.append(dict(task=row.task,metric=row.metric,tier=row.tier,target=row.target,feature=row.feature,rho=row.rho,n=row.n))
   if row.feature not in rows:rows.append(row.feature)
 return rows


def main():
 global INPUT, T, FEATURES, selection
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--tables',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 args=p.parse_args();INPUT=args.tables;T=args.output;T.mkdir(parents=True,exist_ok=True)
 FEATURES={task:groups['+Aux'] for task,groups in feature_groups().items()}
 selection=[];c=correlations()
 for task,spec in SPECS.items():
  z=c[c.task.eq(task)&c.tier.eq('pooled')];union(z,list(spec),'metric')
  for metric in spec:
   z=c[c.task.eq(task)&c.tier.eq('individual')&c.metric.eq(metric)]
   union(z,[m for m in ORDER if m in z.target.unique()],'target')
 pd.DataFrame(selection).to_csv(T/'section5_topk_selection.csv',index=False)

if __name__=='__main__': main()
