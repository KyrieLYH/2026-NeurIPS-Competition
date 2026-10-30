"""Second, fresh benchmark confirmation after the v5 data-only change failed."""
from pathlib import Path
from collections import defaultdict
import sys,json,hashlib,time,zipfile,tempfile,subprocess
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from predai.ensemble_v2 import Ensemble,pair_weights,BUDGETS,ALC_WEIGHTS
from predai.data_v2 import to_pair,balanced_sample
from predai.adaptive_v3 import beta_predict
from predai.shared_v3 import subject_key,hierarchical_predict
from predai.reliable_v6 import ReliableEnsemble,reliable_predict
from predai.paths import RAW_DATA
from scripts.finetune_current import attach_features
from scripts.improve_v3 import build_episodes,materialize

PREV=ROOT/'experiments/robust-v5-20261028'
OUT=ROOT/'experiments/reliable-v6-20261028'
FRESH=['bertaqa','clasheval','emoji_attack','igakuqa','legal_rag_bench',
       'medhelm_pubmed_qa','mind2web','software_world_models','wikihow_agent']
PRIORS=[(.15,.25,.6),(1/3,1/3,1/3),(.6,.3,.1),(.8,.15,.05)]
PRED_CACHE={}
POOL_CACHE={}
PREDICTOR_CLASS=ReliableEnsemble
PREDICTOR_MODULE='reliable_v6'
EXTRA_FILES=[]
ARCHIVE_NAME='submission_v6_reliable.zip'
CONFIG_ENRICH=lambda config:config

def prepare():
    OUT.mkdir(exist_ok=True)
    plan={'fresh_confirmation':FRESH,'sd':[1.,2.,3.],'slope_priors':PRIORS,
          'contractions':[0.,.5],'shared_weights':[0.,.25,.5],
          'objective':'Equal benchmark mean ALC, averaged over own-only and one-companion evidence.',
          'development':'Original five plus previously opened eleven, all excluded from the new base fit.',
          'caveat':'V5 confirmation is now development, not independent evidence for this round.',
          'confirmation_rule':'Fresh benchmark episode and benchmark means must improve with and without peer evidence.'}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(plan))
    else:p.write_text(json.dumps(plan,indent=2))
    if (OUT/'prepared.json').exists():return
    frame=pd.read_parquet(PREV/'legacy_sample.parquet')
    frame=frame[~frame.benchmark_id.isin(FRESH)].copy()
    frame.to_parquet(OUT/'train_sample.parquet',index=False)
    dev=joblib.load(PREV/'development.joblib')+joblib.load(PREV/'confirmation.joblib')
    joblib.dump(dev,OUT/'development.joblib',compress=3)
    slim=pd.read_parquet(ROOT/'experiments/improve-v4/slim.parquet',filters=[('benchmark_id','in',FRESH)])
    eps=build_episodes(slim,seed=601028,cap=60);picked=[]
    for bid in FRESH:
        candidates=[e for e in eps if e['benchmark_id']==bid]
        rng=np.random.default_rng(601028)
        indices=sorted(rng.choice(len(candidates),size=min(16,len(candidates)),replace=False))
        picked.extend(candidates[k] for k in indices)
    fresh=materialize(picked)
    joblib.dump(fresh,OUT/'confirmation.joblib',compress=3)
    meta={'train_rows':len(frame),'development_episodes':len(dev),'confirmation_episodes':len(fresh),
          'fresh_counts':{b:sum(e['benchmark_id']==b for e in fresh) for b in FRESH}}
    (OUT/'prepared.json').write_text(json.dumps(meta,indent=2));print('PREPARED',meta,flush=True)

def train():
    path=OUT/'base.joblib'
    if path.exists():return joblib.load(path)
    frame=pd.read_parquet(OUT/'train_sample.parquet')
    plan5=json.loads((PREV/'plan.json').read_text())
    assert not set(frame.benchmark_id)&set(FRESH+plan5['development']+plan5['confirmation'])
    pairs=[to_pair(row) for row in frame.to_dict('records')]
    print('FIT_FRESH_BASE',len(pairs),flush=True);start=time.perf_counter()
    with threadpool_limits(limits=12):
        model=Ensemble(kappa=2.,n_jobs=12).fit(pairs,frame.label.to_numpy(),pair_weights(frame))
    joblib.dump(model,path,compress=3);print('FRESH_BASE_SAVED',time.perf_counter()-start,flush=True)
    return model

def pack(model,phase):
    path=OUT/(phase+'_predictions.joblib')
    if path.exists():return joblib.load(path)
    eps=joblib.load(OUT/(phase+'.joblib'));rows=[]
    for j,e in enumerate(eps):
        rows.append({'bid':e['benchmark_id'],'sid':e['subject_id'],
            'key':subject_key(e['target_pairs'][0][0]),'keys':[subject_key(p[0]) for p in e['evidence_pairs']],
            'y':e['target_y'],'ey':e['evidence_y'],
            't':model.base_many(e['target_pairs']),'e':model.base_many(e['evidence_pairs'])})
        if (j+1)%80==0:print('PREDICT',phase,j+1,'/',len(eps),flush=True)
    joblib.dump(rows,path,compress=3);return rows

def contexts(rows):
    bybid=defaultdict(list)
    for r in rows:bybid[r['bid']].append(r)
    output=[]
    for r in rows:
        others=[x for x in bybid[r['bid']] if x['sid']!=r['sid']]
        digest=hashlib.sha256(f"context|0|{r['bid']}|{r['sid']}".encode()).digest()[:8]
        peer=others[int(np.random.default_rng(int.from_bytes(digest,'little')).integers(len(others)))] if others else None
        output.append((r,peer))
    return output

def summary(records):
    groups=defaultdict(list)
    for row,curve in records:groups[row['bid']].append(curve)
    per={b:{'episodes':len(curves),'brier':np.mean(curves,axis=0).tolist(),
         'alc':float(np.mean(np.asarray(curves)@ALC_WEIGHTS))} for b,curves in groups.items()}
    return {'alc':float(np.mean([ALC_WEIGHTS@curve for row,curve in records])),
         'benchmark_mean_alc':float(np.mean([r['alc'] for r in per.values()])),
         'brier':np.mean([curve for row,curve in records],axis=0).tolist(),'benchmarks':per}

def score(rows,cfg,method='reliable'):
    own_records=[];peer_records=[]
    for ri,(r,peer) in enumerate(contexts(rows)):
        c=cfg.get('contraction',0.);t=.5+(1-c)*(r['t']-.5);eb=.5+(1-c)*(r['e']-.5)
        ob=.5+(1-c)*(peer['e']-.5) if peer else []
        own_curve=[];peer_curve=[]
        for n in BUDGETS:
            own_eb,own_y=eb[:n],r['ey'][:n]
            # Own-only evidence has exactly the current pair's revealed labels.
            if method=='beta':p=beta_predict(t,own_eb,own_y,tau=2.,retain=.5)
            else:
                key=(id(rows),r['bid'],r['sid'],ri,n,cfg['sd'],tuple(cfg['slope_prior']),c)
                if key not in PRED_CACHE:PRED_CACHE[key]=reliable_predict(t,own_eb,own_y,cfg['sd'],cfg['slope_prior'])
                p=PRED_CACHE[key]
            own_curve.append(float(np.mean((p-r['y'])**2)))
            q=p
            if n and peer:
                if peer['key']==r['key']:
                    if method=='beta':q=beta_predict(t,np.r_[own_eb,ob[:n]],np.r_[own_y,peer['ey'][:n]],tau=2.,retain=.5)
                    else:q=reliable_predict(t,np.r_[own_eb,ob[:n]],np.r_[own_y,peer['ey'][:n]],cfg['sd'],cfg['slope_prior'])
                elif cfg.get('shared_weight',.5):
                    key=(id(rows),r['bid'],r['sid'],ri,n,c)
                    if key not in POOL_CACHE:POOL_CACHE[key]=hierarchical_predict(t,np.r_[eb[:n],ob[:n]],np.r_[r['ey'][:n],peer['ey'][:n]],r['keys'][:n]+peer['keys'][:n],r['key'])
                    pool=POOL_CACHE[key]
                    w=cfg.get('shared_weight',.5);q=(1-w)*p+w*pool
            peer_curve.append(float(np.mean((q-r['y'])**2)))
        own_records.append((r,own_curve));peer_records.append((r,peer_curve))
    return {'own_only':summary(own_records),'with_peer':summary(peer_records)}

def select(model):
    path=OUT/'selection.json'
    if path.exists():return json.loads(path.read_text())
    rows=pack(model,'development');baseline=score(rows,{'contraction':0.,'shared_weight':.5},'beta')
    ranking=[]
    for sd in [1.,2.,3.]:
        for prior in PRIORS:
            for contraction in [0.,.5]:
                for sw in [0.,.25,.5]:
                    cfg={'sd':sd,'slope_prior':prior,'contraction':contraction,'shared_weight':sw,'boost_weight':0.}
                    report=score(rows,cfg)
                    objective=.5*(report['own_only']['benchmark_mean_alc']+report['with_peer']['benchmark_mean_alc'])
                    ranking.append({'config':cfg,'objective':objective,'report':report})
        print('SEARCH_SD_COMPLETE',sd,flush=True)
    ranking.sort(key=lambda x:x['objective']);selected={'candidate':ranking[0],'baseline':baseline,'top':ranking[:8]}
    path.write_text(json.dumps(selected,indent=2));print('FROZEN_SELECTION',json.dumps(ranking[0]),flush=True)
    return selected

def confirm(model,selection):
    path=OUT/'confirmation_report.json'
    if path.exists():return json.loads(path.read_text())
    rows=pack(model,'confirmation')
    baseline=score(rows,{'contraction':0.,'shared_weight':.5},'beta')
    candidate=score(rows,selection['candidate']['config'])
    approved=all(candidate[scenario][metric]<baseline[scenario][metric]
         for scenario in ['own_only','with_peer'] for metric in ['alc','benchmark_mean_alc'])
    bids=sorted(baseline['own_only']['benchmarks'])
    diffs=np.array([candidate['own_only']['benchmarks'][b]['alc']-baseline['own_only']['benchmarks'][b]['alc'] for b in bids])
    rng=np.random.default_rng(601028);boot=np.mean(rng.choice(diffs,size=(5000,len(diffs)),replace=True),axis=1)
    result={'approved':bool(approved),'baseline':baseline,'candidate':candidate,
         'paired_benchmark_ci_95':np.quantile(boot,[.025,.975]).tolist(),
         'benchmarks_improved':int((diffs<0).sum()),'benchmark_count':len(bids),
         'note':'Fresh benchmarks excluded from base training and this round parameter search. Prior v5 confirmation is development only.'}
    path.write_text(json.dumps(result,indent=2));print('FRESH_CONFIRMATION',json.dumps(result),flush=True)
    return result

def export(selection,confirmation):
    if not confirmation['approved']:print('REJECTED_NO_PACKAGE',flush=True);return
    archive=OUT/ARCHIVE_NAME
    if archive.exists():print('PACKAGE_EXISTS',archive,flush=True);return
    artifact=OUT/'final_ensemble.joblib'
    if not artifact.exists():
        cache=OUT/'final_sample.parquet'
        if cache.exists():frame=pd.read_parquet(cache)
        else:
            slim=pd.read_parquet(ROOT/'experiments/improve-v4/slim.parquet')
            sample=balanced_sample(slim,150_000,seed=0);del slim
            frame=attach_features(RAW_DATA,sample);frame.to_parquet(cache,index=False)
        pairs=[to_pair(row) for row in frame.to_dict('records')]
        print('FINAL_FIT',len(pairs),flush=True)
        with threadpool_limits(limits=12):base=Ensemble(kappa=2.,n_jobs=12).fit(pairs,frame.label.to_numpy(),pair_weights(frame))
        fitted=PREDICTOR_CLASS(base,None,CONFIG_ENRICH(selection['candidate']['config']))
        joblib.dump(fitted,artifact,compress=3)
        (OUT/'final_training.json').write_text(json.dumps({'rows':len(frame),'benchmarks':sorted(frame.benchmark_id.unique()),'config':fitted.config},indent=2))
    entry='''from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import joblib
from threadpoolctl import threadpool_limits
import predai.reliable_v6
_MODEL=None
def predict(input: list[dict],labeled: list | None=None)->float:
    global _MODEL
    with threadpool_limits(limits=1):
        if _MODEL is None:_MODEL=joblib.load(Path(__file__).resolve().parent/'artifacts'/'ensemble.joblib')
        return float(_MODEL.predict_many([input],labeled)[0])
'''
    import numpy,scipy,sklearn,threadpoolctl
    entry=entry.replace('import predai.reliable_v6','import predai.'+PREDICTOR_MODULE)
    req={'numpy':numpy.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'joblib':joblib.__version__,'threadpoolctl':threadpoolctl.__version__}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        z.writestr('model.py',entry);z.write(artifact,'artifacts/ensemble.joblib')
        for name in ['__init__.py','features.py','ensemble_v2.py','adaptive_v3.py','shared_v3.py','compact_boost.py','robust_v5.py','reliable_v6.py']+EXTRA_FILES:
            z.write(ROOT/'predai'/name,'predai/'+name)
        z.write(OUT/'final_training.json','artifacts/training.json')
        z.write(OUT/'confirmation_report.json','artifacts/confirmation.json')
        z.writestr('requirements.txt',''.join(f'{k}=={v}\n' for k,v in req.items()))
        z.writestr('README.txt','Offline sklearn predictor with Bayesian calibration and evidence-dependent ranking reliability. Random platform acquisition; no runtime network calls.\n')
    print('EXPORTED',archive,archive.stat().st_size,flush=True)
    from scripts.check_v5 import CHECK
    with tempfile.TemporaryDirectory(prefix='verify-',dir=OUT) as tmp:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None;z.extractall(tmp)
        proc=subprocess.run([sys.executable,'-I','-c',CHECK,tmp],cwd=tmp,capture_output=True,text=True)
        if proc.returncode:print(proc.stderr,flush=True);raise RuntimeError('Package check failed')
        (OUT/'package_check.json').write_text(proc.stdout);print(proc.stdout,flush=True)

def main():
    prepare();model=train()
    with threadpool_limits(limits=1):
        selection=select(model);confirmation=confirm(model,selection)
    export(selection,confirmation)
    print('V6_COMPLETE',flush=True)

if __name__=='__main__':main()
