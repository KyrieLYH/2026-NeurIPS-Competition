"""Select base predictors by official ALC, then open one fresh confirmation."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np,joblib
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_v9 import OUT,SOURCE,BRANCHES,candidates as original_candidates
from scripts.run_reliable_v6 import contexts
from predai.ensemble_v2 import BUDGETS,ALC_WEIGHTS
from predai.adaptive_v3 import beta_predict,AdaptiveEnsemble
from predai.shared_v3 import subject_key,hierarchical_predict,SharedEnsemble
from predai.portable_v9 import FrozenBlend

SCENARIOS=['own_only','with_peer']
ALL_BRANCHES=BRANCHES+['portable_ability']

def candidates():
    choices=original_candidates();choices['portable_ability']={'portable_ability':1.}
    for anchor in ['small','large80']:
        for w in [.25,.5]:choices[f'{anchor}_portable_ability_{w:g}']={anchor:1-w,'portable_ability':w}
    return choices

def development():
    return joblib.load(SOURCE/'development.joblib')+joblib.load(SOURCE/'confirmation.joblib')

def distinct_predict(model,pairs):
    lookup={};unique=[];inverse=[]
    for p in pairs:
        key=json.dumps(p,sort_keys=True,ensure_ascii=False)
        if key not in lookup:lookup[key]=len(unique);unique.append(p)
        inverse.append(lookup[key])
    return np.asarray(model.base_many(unique))[inverse]

def pack(models,phase):
    path=OUT/(phase+'_predictions.joblib')
    if path.exists():
        payload=joblib.load(path)
        assert set(payload['models'])==set(models),'Prediction cache model set changed'
        return payload['rows']
    eps=development() if phase=='development' else joblib.load(OUT/'confirmation.joblib')
    rows=[]
    for j,e in enumerate(eps):
        r={'bid':e['benchmark_id'],'sid':e['subject_id'],'key':subject_key(e['target_pairs'][0][0]),
           'keys':[subject_key(p[0]) for p in e['evidence_pairs']], 'y':e['target_y'],'ey':e['evidence_y'], 'models':{}}
        pairs=e['target_pairs']+e['evidence_pairs'];n=len(e['target_pairs'])
        for name,model in models.items():
            p=distinct_predict(model,pairs);r['models'][name]={'t':p[:n],'e':p[n:]}
        rows.append(r)
        if (j+1)%80==0:print('PREDICT',phase,j+1,'/',len(eps),flush=True)
    joblib.dump({'models':list(models),'rows':rows},path,compress=3)
    return rows

def score(rows,weights):
    curves=np.empty((2,len(rows),len(BUDGETS)))
    for j,(r,peer) in enumerate(contexts(rows)):
        target=sum(w*r['models'][name]['t'] for name,w in weights.items())
        eb=sum(w*r['models'][name]['e'] for name,w in weights.items())
        ob=sum(w*peer['models'][name]['e'] for name,w in weights.items()) if peer else []
        for bi,n in enumerate(BUDGETS):
            own=beta_predict(target,eb[:n],r['ey'][:n],tau=2.,retain=.5)
            shared=own
            if n and peer:
                if peer['key']==r['key']:
                    shared=beta_predict(target,np.r_[eb[:n],ob[:n]],np.r_[r['ey'][:n],peer['ey'][:n]],tau=2.,retain=.5)
                else:
                    pool=hierarchical_predict(target,np.r_[eb[:n],ob[:n]],np.r_[r['ey'][:n],peer['ey'][:n]],r['keys'][:n]+peer['keys'][:n],r['key'])
                    shared=.5*own+.5*pool
            curves[0,j,bi]=np.mean((np.clip(own,.001,.999)-r['y'])**2)
            curves[1,j,bi]=np.mean((np.clip(shared,.001,.999)-r['y'])**2)
    return curves

def summary(curves,rows):
    output={};alcs=curves@ALC_WEIGHTS
    for si,scenario in enumerate(SCENARIOS):
        groups={b:np.array([j for j,r in enumerate(rows) if r['bid']==b]) for b in sorted({r['bid'] for r in rows})}
        per={b:{'episodes':len(ids),'alc':float(np.mean(alcs[si,ids])),
                'brier':np.mean(curves[si,ids],axis=0).tolist()} for b,ids in groups.items()}
        output[scenario]={'alc':float(np.mean(alcs[si])),'brier':np.mean(curves[si],axis=0).tolist(),
           'benchmark_mean_alc':float(np.mean([v['alc'] for v in per.values()])),'benchmarks':per}
    return output

def primary(report):return np.array([report[s]['alc'] for s in SCENARIOS])

def intervals(delta,rows):
    # Keep official equal-episode weights while sampling complete benchmark clusters.
    groups=[np.array([j for j,r in enumerate(rows) if r['bid']==b]) for b in sorted({r['bid'] for r in rows})]
    sizes=np.array([len(ids) for ids in groups])
    totals=np.array([delta[:,ids].sum(axis=1) for ids in groups])
    draws=np.random.default_rng(901029).integers(len(groups),size=(5000,len(groups)))
    bootstrap=totals[draws].sum(axis=1)/sizes[draws].sum(axis=1)[:,None]
    return {**{s:np.quantile(bootstrap[:,i],[.025,.975]).tolist() for i,s in enumerate(SCENARIOS)},
            'scenario_average':np.quantile(bootstrap.mean(axis=1),[.025,.975]).tolist()}

def runtime_parity(rows,models,weights,curves,phase,suffix=''):
    eps=development() if phase=='development' else joblib.load(OUT/'confirmation.joblib')
    bykey={(e['benchmark_id'],e['subject_id']):e for e in eps}
    base=FrozenBlend([models[n] for n in weights],[weights[n] for n in weights])
    wrapped=SharedEnsemble(AdaptiveEnsemble(base,None,{'boost_weight':0.,'method':'beta','tau':2.,'retain':.5}),.5)
    maximum=0.;calls=0;groups=contexts(rows)
    for ri in sorted(set([0,len(rows)//3,len(rows)//2,len(rows)-1])):
        r,peer=groups[ri];e=bykey[(r['bid'],r['sid'])]
        other=bykey[(peer['bid'],peer['sid'])] if peer else None
        for bi,n in enumerate(BUDGETS):
            for si in range(2):
                labeled=list(zip(e['evidence_pairs'][:n],e['evidence_y'][:n]))
                if si and other:labeled+=list(zip(other['evidence_pairs'][:n],other['evidence_y'][:n]))
                p=wrapped.predict_many(e['target_pairs'],labeled)
                actual=float(np.mean((p-e['target_y'])**2));err=abs(actual-curves[si,ri,bi])
                assert err<1e-10,(phase,ri,n,si,err)
                maximum=max(maximum,err);calls+=1
    result={'status':'PASS','checks':calls,'maximum_brier_error':maximum}
    (OUT/(phase+suffix+'_runtime_parity.json')).write_text(json.dumps(result,indent=2));print('RUNTIME_PARITY',phase+suffix,result,flush=True)

def main():
    assert all((OUT/(name+'.joblib')).exists() for name in ALL_BRANCHES)
    with threadpool_limits(limits=1):
        models={name:joblib.load(OUT/(name+'.joblib')) for name in ALL_BRANCHES}
        rows=pack(models,'development');choices=candidates();reports={};curves={}
        for name,weights in choices.items():
            path=OUT/('curves_'+name+'.npz')
            if path.exists():c=np.load(path)['curves']
            else:c=score(rows,weights);np.savez_compressed(path,curves=c)
            curves[name]=c;reports[name]=summary(c,rows)
            print('DEVELOPMENT',name,primary(reports[name]).tolist(),flush=True)
        baseline=primary(reports['small']);eligible=['small']
        for name,rep in reports.items():
            values=primary(rep)
            if name!='small' and np.all(values<=baseline+1e-12) and np.mean(values)<np.mean(baseline)-.0002:eligible.append(name)
        winner=min(eligible,key=lambda name:(float(np.mean(primary(reports[name]))),name!='small'))
        selection={'selected':winner,'weights':choices[winner],'baseline':reports['small'],
            'selected_report':reports[winner],'all_development':reports,'selection_metric':'Official equal-episode ALC; both scenarios protected; baseline fallback.',
            'development_delta_ci':intervals((curves[winner]-curves['small'])@ALC_WEIGHTS,rows)}
        selection_path=OUT/'selection.json'
        if selection_path.exists():assert json.loads(selection_path.read_text())==selection
        else:selection_path.write_text(json.dumps(selection,indent=2))
        if winner!='small':runtime_parity(rows,models,choices['small'],curves['small'],'development','_baseline')
        runtime_parity(rows,models,choices[winner],curves[winner],'development')
        print('FROZEN_CHOICE',winner,choices[winner],flush=True)
        if winner=='small':
            (OUT/'result.json').write_text(json.dumps({'baseline_selected':True,'confirmation_opened':False,
                'message':'No candidate passed development guard. Fresh confirmation scores remain unopened.'},indent=2))
            print('BASELINE_RETAINED_FRESH_UNOPENED',flush=True);return
        needed=set(choices[winner])|{'small'};fresh_models={k:models[k] for k in models if k in needed}
        fresh=pack(fresh_models,'confirmation');ref=score(fresh,{'small':1.});picked=score(fresh,choices[winner])
        reference=summary(ref,fresh);new=summary(picked,fresh)
        ci=intervals((picked-ref)@ALC_WEIGHTS,fresh)
        passed=bool(np.all(primary(new)<primary(reference)))
        result={'baseline_selected':False,'confirmation_opened':True,'selected':winner,'weights':choices[winner],
            'baseline':reference,'candidate':new,'paired_ci95':ci,'point_improvement_both':passed,
            'strong_evidence':bool(passed and ci['scenario_average'][1]<0),
            'caveat':'A single frozen candidate on fresh benchmarks. Confidence intervals and model selection uncertainty still matter.'}
        (OUT/'result.json').write_text(json.dumps(result,indent=2))
        runtime_parity(fresh,fresh_models,choices[winner],picked,'confirmation')
        print('FRESH_RESULT',json.dumps({k:v for k,v in result.items() if k not in ['baseline','candidate']}),flush=True)
    print('V9_EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':main()
