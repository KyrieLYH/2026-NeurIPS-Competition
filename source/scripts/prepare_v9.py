"""Freeze new benchmark confirmation and sample thin tables before joining text."""
from pathlib import Path
import sys,json,time
import joblib,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predai.data_v2 import balanced_sample
from predai.paths import RAW_DATA
from scripts.finetune_current import attach_features
from scripts.improve_v3 import build_episodes,materialize

SOURCE=ROOT/'experiments/empirical-v7-20261028'
OUT=ROOT/'experiments/train-v9-20261029'
FRESH=['mmlu','appworld','devbench','liveaopsbench','helm_bbq','kormedmcqa','wmt_mqm','egoschema']
BRANCHES=['small','large80','large160','large_ridge1000','portable_hgb','portable_extra']

def candidates():
    result={name:{name:1.} for name in BRANCHES}
    for branch in ['large80','large160']:
        result['small_half_'+branch]={'small':.5,branch:.5}
    for anchor in ['small','large80']:
        for branch in ['portable_hgb','portable_extra']:
            for w in [.25,.5]:result[f'{anchor}_{branch}_{w:g}']={anchor:1-w,branch:w}
    return result

def main():
    OUT.mkdir(exist_ok=True)
    plan={'date':'2026-10-29','fresh_confirmation':FRESH,'branches':BRANCHES,'candidates':candidates(),
        'large_rows':500000,'large_seed':0,'adaptation':{'method':'beta','tau':2.,'retain':.5,'shared_weight':.5},
        'development':'All 33 previously opened benchmarks; original five reported separately.',
        'selection':'Official equal-episode ALC, equally summarized across own-only/one-peer scenarios; baseline small explicitly included; neither scenario can get worse.',
        'fresh_confirmation':'Read for scoring only after one candidate is frozen. If baseline wins, leave fresh scoring unopened.',
        'promotion':'Development improvement in both scenarios, fresh improvement in both scenarios; cluster confidence intervals reported, not treated as a score guarantee.',
        'dependencies':'Existing numpy/scipy/sklearn/joblib/threadpoolctl only. CPU, no installation or GPU.',
        'export':'No automatic upload; no old ZIP overwritten.'}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:p.write_text(json.dumps(plan,indent=2))
    if (OUT/'prepared.json').exists():print('PREPARATION_REUSED',flush=True);return
    started=time.perf_counter()
    small_path=OUT/'sample_small.parquet'
    if small_path.exists():small=pd.read_parquet(small_path)
    else:
        frame=pd.read_parquet(SOURCE/'train_sample.parquet')
        small=frame[~frame.benchmark_id.isin(FRESH)].copy();del frame
        small.to_parquet(small_path,index=False)
    allowed=sorted(small.benchmark_id.unique())
    dev=joblib.load(SOURCE/'development_predictions.joblib')+joblib.load(SOURCE/'confirmation_predictions.joblib')
    development=sorted({r['bid'] for r in dev})
    assert not set(allowed)&set(development+FRESH)
    large_path=OUT/'sample_large.parquet'
    if not large_path.exists():
        print('LOAD_SLIM_FOR_SAMPLING',len(allowed),'benchmarks',flush=True)
        slim=pd.read_parquet(ROOT/'experiments/improve-v4/slim.parquet',filters=[('benchmark_id','in',allowed)])
        sample=balanced_sample(slim,500000,seed=0).copy();del slim
        sample.to_parquet(OUT/'sample_large_ids.parquet',index=False)
        print('SAMPLED',len(sample),flush=True)
        large=attach_features(RAW_DATA,sample);del sample
        large.to_parquet(large_path,index=False);del large
    confirm_path=OUT/'confirmation.joblib'
    if not confirm_path.exists():
        fresh=pd.read_parquet(ROOT/'experiments/improve-v4/slim.parquet',filters=[('benchmark_id','in',FRESH)])
        episodes=build_episodes(fresh,seed=901029,cap=60);picked=[]
        for bid in FRESH:
            group=[e for e in episodes if e['benchmark_id']==bid]
            indices=sorted(np.random.default_rng(901029).choice(len(group),size=min(16,len(group)),replace=False))
            picked.extend(group[j] for j in indices)
        del fresh,episodes
        ready=materialize(picked);joblib.dump(ready,confirm_path,compress=3)
    else:ready=joblib.load(confirm_path)
    meta={'small_rows':len(small),'large_rows':500000,'training_benchmarks':allowed,
        'development_benchmarks':development,'development_episodes':len(dev),
        'confirmation_counts':{b:sum(e['benchmark_id']==b for e in ready) for b in FRESH},
        'confirmation_episodes':len(ready),'seconds':time.perf_counter()-started}
    (OUT/'prepared.json').write_text(json.dumps(meta,indent=2));print('PREPARED',json.dumps(meta),flush=True)

if __name__=='__main__':main()
