"""Refit only the chosen intercept on bundled PUBLIC out-of-task predictions.

The objective/optimizer are copied from source/scripts/run_cal_v10.py:
fit_calibration; this convenience entry does not repeat candidate selection.
"""
from pathlib import Path
import argparse,json
import joblib,numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit

ROOT=Path(__file__).resolve().parents[1]

def fit(rows):
    z=np.concatenate([logit(np.clip(r['models']['small']['t'],.001,.999)) for r in rows])
    y=np.concatenate([r['y'] for r in rows])
    w=np.concatenate([np.full(len(r['y']),1/len(r['y'])/len(rows)) for r in rows])
    def objective(theta):
        b=theta[0];raw=expit(z+b);p=np.clip(raw,.001,.999)
        value=np.sum(w*(p-y)**2)+.001*b*b
        g=2*w*(p-y)*raw*(1-raw)*((raw>.001)&(raw<.999))
        return value,np.array([np.sum(g)+.002*b])
    result=minimize(objective,np.array([0.]),jac=True,bounds=[(-3.,3.)],method='L-BFGS-B',options={'maxiter':100,'ftol':1e-12})
    if not result.success:raise RuntimeError(result.message)
    return {'a':1.,'b':float(result.x[0]),'loss':float(result.fun),'meta_episodes':len(rows)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache-dir',type=Path,default=ROOT/'calibration_cache')
    ap.add_argument('--expected-metadata',type=Path,default=ROOT/'runtime/artifacts/training.json')
    ap.add_argument('--output',type=Path,help='Optional NEW JSON output; existing files are never overwritten')
    args=ap.parse_args();rows=[]
    for phase in ['development','confirmation']:
        payload=joblib.load(args.cache_dir/(phase+'_predictions.joblib'))
        if payload['models']!=['small']:raise ValueError('Wrong prediction cache model set')
        rows.extend(payload['rows'])
    params=fit(rows);expected=json.loads(args.expected_metadata.read_text(encoding='utf-8'))['parameters']
    if params['meta_episodes']!=758 or abs(params['b']-expected['b'])>1e-10:raise AssertionError((params,expected))
    report={'status':'PASS','parameters':params,'public_benchmarks':len({r['bid'] for r in rows}),
            'warning':'Refit training objective, not independent test performance. Original five/fresh six are included in final refit.'}
    content=json.dumps(report,indent=2,ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as f:f.write(content+'\n')
    print(content)

if __name__=='__main__':main()
