"""Group-held-out calibration; original five development tasks never fit parameters."""
from pathlib import Path
import sys,json,time,copy,io,zipfile,tempfile,subprocess
import numpy as np,pandas as pd,joblib
from scipy.optimize import minimize
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import evaluate_v9 as scoring
from scripts.improve_v3 import DEV,build_episodes,materialize
from scripts.finetune_current import attach_features
from predai.data_v2 import to_pair
from predai.ensemble_v2 import Ensemble,pair_weights,ALC_WEIGHTS,BUDGETS
from predai.paths import RAW_DATA
from predai.calibrated_v10 import apply_calibration,CalibratedEnsemble

V9=ROOT/'experiments/train-v9-20261029'
OLD=ROOT/'experiments/empirical-v7-20261028'
OUT=ROOT/'experiments/calibrate-v10-20261029'
FRESH=['ceval','mathconstraint','jailbreakbench','helm_bold','swe_smith','medhelm_med_qa']
SPECS=[{'kind':kind,'regularization':reg,'cold_only':cold} for kind in ['intercept','platt'] for reg in [.001,.01] for cold in [True,False]]

def development():
    return joblib.load(OLD/'development.joblib')+joblib.load(OLD/'confirmation.joblib')+joblib.load(V9/'confirmation.joblib')

def prepare():
    OUT.mkdir(exist_ok=True)
    plan={'fresh':FRESH,'specs':SPECS,'original_development_never_fit':DEV,
        'meta_training':'Previously opened 36 non-original benchmark tasks; leave-one-benchmark-out calibration for their development scores.',
        'objective':'Official equal-episode ALC; both evidence scenarios protected; original beta2 baseline included.',
        'calibration_loss':'Cold-start response Brier with equal episode weights; global intercept or Platt slope/intercept, regularized towards identity.',
        'confirmation':'One frozen candidate only. No tuning on fresh labels. Default random acquisition unchanged.'}
    path=OUT/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    if (OUT/'prepared.json').exists():return
    frame=pd.read_parquet(V9/'sample_small.parquet');frame=frame[~frame.benchmark_id.isin(FRESH)].copy()
    frame.to_parquet(OUT/'train_sample.parquet',index=False)
    eps=development();assert not set(frame.benchmark_id)&({e['benchmark_id'] for e in eps}|set(FRESH))
    slim=pd.read_parquet(ROOT/'experiments/improve-v4/slim.parquet',filters=[('benchmark_id','in',FRESH)])
    candidates=build_episodes(slim,seed=1001029,cap=60);picked=[]
    for bid in FRESH:
        group=[e for e in candidates if e['benchmark_id']==bid]
        ids=sorted(np.random.default_rng(1001029).choice(len(group),size=min(12,len(group)),replace=False))
        picked.extend(group[j] for j in ids)
    ready=materialize(picked);joblib.dump(ready,OUT/'confirmation.joblib',compress=3)
    meta={'base_train_rows':len(frame),'development_episodes':len(eps),'development_benchmarks':len({e['benchmark_id'] for e in eps}),
         'confirmation_episodes':len(ready),'confirmation_counts':{b:sum(e['benchmark_id']==b for e in ready) for b in FRESH}}
    (OUT/'prepared.json').write_text(json.dumps(meta,indent=2));print('PREPARED',meta,flush=True)

def train():
    path=OUT/'base.joblib'
    if path.exists():return joblib.load(path)
    frame=pd.read_parquet(OUT/'train_sample.parquet');pairs=[to_pair(r) for r in frame.to_dict('records')]
    print('FIT_BASE',len(frame),flush=True)
    with threadpool_limits(limits=12):base=Ensemble(kappa=2.,n_jobs=12).fit(pairs,frame.label.to_numpy(),pair_weights(frame))
    joblib.dump(base,path,compress=3);print('BASE_SAVED',flush=True);return base

def fit_calibration(rows,spec):
    z=np.concatenate([logit(np.clip(r['models']['small']['t'],.001,.999)) for r in rows])
    y=np.concatenate([r['y'] for r in rows]);w=np.concatenate([np.full(len(r['y']),1/len(r['y'])/len(rows)) for r in rows])
    reg=spec['regularization'];platt=spec['kind']=='platt'
    def f(theta):
        a,b=theta if platt else (1.,theta[0])
        raw=expit(a*z+b);p=np.clip(raw,.001,.999)
        value=np.sum(w*(p-y)**2)+reg*((a-1)**2+b*b)
        g=2*w*(p-y)*raw*(1-raw)*((raw>.001)&(raw<.999))
        grad=np.array([np.sum(g*z)+2*reg*(a-1),np.sum(g)+2*reg*b]) if platt else np.array([np.sum(g)+2*reg*b])
        return value,grad
    start=np.array([1.,0.]) if platt else np.array([0.])
    fit=minimize(f,start,jac=True,bounds=[(.1,3.),(-3.,3.)] if platt else [(-3.,3.)],method='L-BFGS-B',options={'maxiter':100,'ftol':1e-12})
    assert fit.success,fit.message
    a,b=fit.x if platt else (1.,fit.x[0])
    return {'a':float(a),'b':float(b),'loss':float(fit.fun),'meta_episodes':len(rows)}

def calibrated_rows(rows,parameters):
    out=[]
    for r in rows:
        p=parameters[r['bid']] if 'a' not in parameters else parameters
        new=dict(r);pred=r['models']['small']
        new['models']={'small':{'t':apply_calibration(pred['t'],p['a'],p['b']),'e':apply_calibration(pred['e'],p['a'],p['b'])}}
        out.append(new)
    return out

def curves(rows,raw,spec,parameters):
    if not spec:return raw.copy()
    transformed=calibrated_rows(rows,parameters)
    if not spec['cold_only']:return scoring.score(transformed,{'small':1.})
    result=raw.copy()
    for j,r in enumerate(transformed):
        error=np.mean((r['models']['small']['t']-r['y'])**2);result[:,j,0]=error
    return result

def runtime_check(base,rows,eps,spec,params,c):
    from scripts.run_reliable_v6 import contexts
    lookup={(e['benchmark_id'],e['subject_id']):e for e in eps};groups=contexts(rows)
    maximum=0.;calls=0
    for j in sorted(set([0,len(rows)//2,len(rows)-1])):
        r,peer=groups[j];e=lookup[(r['bid'],r['sid'])];other=lookup[(peer['bid'],peer['sid'])] if peer else None
        p=params[r['bid']] if 'a' not in params else params
        wrapped=CalibratedEnsemble(base,p['a'],p['b'],spec['cold_only'])
        for bi,n in enumerate(BUDGETS):
            for si in range(2):
                labeled=list(zip(e['evidence_pairs'][:n],e['evidence_y'][:n]))
                if si and other:labeled+=list(zip(other['evidence_pairs'][:n],other['evidence_y'][:n]))
                pred=wrapped.predict_many(e['target_pairs'],labeled)
                err=abs(float(np.mean((pred-e['target_y'])**2))-c[si,j,bi])
                assert err<1e-10,(j,n,si,err)
                maximum=max(maximum,err);calls+=1
    print('RUNTIME_CHECK_PASS',calls,maximum,flush=True)
    return {'status':'PASS','checks':calls,'maximum_error':maximum}

def evaluate(base):
    scoring.OUT=OUT;scoring.development=development
    models={'small':base};rows=scoring.pack(models,'development');raw=scoring.score(rows,{'small':1.})
    baseline=scoring.summary(raw,rows);meta=[r for r in rows if r['bid'] not in DEV]
    bids=sorted({r['bid'] for r in rows});results=[];params_by_spec={};all_curves={}
    for i,spec in enumerate(SPECS):
        p=OUT/f'parameters_{i}.json'
        if p.exists():parameters=json.loads(p.read_text())
        else:
            default=fit_calibration(meta,spec);parameters={}
            for bid in bids:
                parameters[bid]=default if bid in DEV else fit_calibration([r for r in meta if r['bid']!=bid],spec)
            p.write_text(json.dumps(parameters,indent=2))
        params_by_spec[i]=parameters
        c=curves(rows,raw,spec,parameters);report=scoring.summary(c,rows);all_curves[i]=c
        results.append({'index':i,'spec':spec,'report':report,'objective':float(scoring.primary(report).mean())})
        print('DEVELOPMENT',i,spec,scoring.primary(report).tolist(),flush=True)
    reference=scoring.primary(baseline)
    eligible=[r for r in results if np.all(scoring.primary(r['report'])<=reference+1e-12) and r['objective']<reference.mean()-.0002]
    if not eligible:
        (OUT/'result.json').write_text(json.dumps({'selected':'baseline','confirmation_opened':False,'baseline':baseline,'all_development':results},indent=2))
        print('BASELINE_RETAINED',flush=True);return
    winner=min(eligible,key=lambda x:x['objective']);index=winner['index'];spec=winner['spec']
    final_parameters=fit_calibration(meta,spec)
    selected={'selected':winner,'baseline':baseline,'all_development':results,'fresh_parameters':final_parameters,
         'meta_benchmarks':sorted({r['bid'] for r in meta}),'original_five_never_fit':DEV}
    (OUT/'selection.json').write_text(json.dumps(selected,indent=2));print('FROZEN',spec,final_parameters,flush=True)
    check=runtime_check(base,rows,development(),spec,params_by_spec[index],all_curves[index]);(OUT/'development_runtime_check.json').write_text(json.dumps(check,indent=2))
    fresh=scoring.pack(models,'confirmation');fresh_raw=scoring.score(fresh,{'small':1.})
    picked=curves(fresh,fresh_raw,spec,final_parameters);old=scoring.summary(fresh_raw,fresh);new=scoring.summary(picked,fresh)
    ci=scoring.intervals((picked-fresh_raw)@ALC_WEIGHTS,fresh)
    passed=bool(np.all(scoring.primary(new)<scoring.primary(old)))
    result={'selected':winner,'confirmation_opened':True,'baseline':old,'candidate':new,'parameters':final_parameters,
         'paired_ci95':ci,'point_improvement_both':passed,'strong_evidence':bool(passed and ci['scenario_average'][1]<0)}
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    check=runtime_check(base,fresh,joblib.load(OUT/'confirmation.joblib'),spec,final_parameters,picked);(OUT/'confirmation_runtime_check.json').write_text(json.dumps(check,indent=2))
    print('CONFIRMATION',json.dumps({'baseline':scoring.primary(old).tolist(),'candidate':scoring.primary(new).tolist(),'pass':passed,'ci':ci}),flush=True)
    if passed:export(selected,result,rows+fresh)

def export(selection,result,all_rows):
    spec=selection['selected']['spec'];parameters=fit_calibration(all_rows,spec)
    source=ROOT/'experiments/finetune-current/submission_v3_current.zip'
    with zipfile.ZipFile(source) as z:full=joblib.load(io.BytesIO(z.read('artifacts/ensemble.joblib'))).base.legacy
    final=CalibratedEnsemble(full,parameters['a'],parameters['b'],spec['cold_only'])
    joblib.dump(final,OUT/'final_ensemble.joblib',compress=3)
    metadata={'spec':spec,'parameters':parameters,'base':'Original V3 full-public 150k fit, reused unchanged',
         'calibration':'Refit global calibration on all available out-of-benchmark predictions after confirmation, including original five and fresh six.',
         'confirmation':result}
    (OUT/'final_training.json').write_text(json.dumps(metadata,indent=2))
    entry='''from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import joblib
from threadpoolctl import threadpool_limits
import predai.calibrated_v10
_MODEL=None
def predict(input: list[dict],labeled: list | None=None)->float:
    global _MODEL
    with threadpool_limits(limits=1):
        if _MODEL is None:_MODEL=joblib.load(Path(__file__).resolve().parent/'artifacts'/'ensemble.joblib')
        return float(_MODEL.predict_many([input],labeled)[0])
'''
    import numpy,scipy,sklearn,threadpoolctl
    versions={'numpy':numpy.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'joblib':joblib.__version__,'threadpoolctl':threadpoolctl.__version__}
    path=OUT/'submission_v10_calibrated.zip'
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        z.writestr('model.py',entry);z.write(OUT/'final_ensemble.joblib','artifacts/ensemble.joblib');z.write(OUT/'final_training.json','artifacts/training.json')
        for name in ['__init__.py','features.py','ensemble_v2.py','adaptive_v3.py','shared_v3.py','calibrated_v10.py']:
            z.write(ROOT/'predai'/name,'predai/'+name)
        z.writestr('requirements.txt',''.join(f'{k}=={v}\n' for k,v in versions.items()))
    from scripts.check_v5 import CHECK
    with tempfile.TemporaryDirectory(prefix='verify-',dir=OUT) as tmp:
        with zipfile.ZipFile(path) as z:assert z.testzip() is None;z.extractall(tmp)
        proc=subprocess.run([sys.executable,'-I','-c',CHECK,tmp],cwd=tmp,text=True,capture_output=True)
        if proc.returncode:print(proc.stderr,flush=True);raise RuntimeError('ZIP verification failed')
        (OUT/'package_check.json').write_text(proc.stdout);print(proc.stdout,flush=True)
    print('EXPORTED',path,path.stat().st_size,flush=True)

def main():
    prepare();base=train()
    with threadpool_limits(limits=1):evaluate(base)
    print('V10_COMPLETE',flush=True)

if __name__=='__main__':main()
