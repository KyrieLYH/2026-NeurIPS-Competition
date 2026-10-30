"""Hierarchical task + subject offsets using only supplied same-benchmark labels."""
from __future__ import annotations
import json
import numpy as np
from scipy.special import expit,logit
from predai.ensemble_v2 import visible_subject
from predai.adaptive_v3 import AdaptiveEnsemble,adapt


def subject_key(subject):return json.dumps(visible_subject(subject),sort_keys=True)


def hierarchical_predict(target, evidence_base, labels, subjects, target_subject,
                         task_precision=.5,subject_precision=2.):
    """Penalized logistic task/subject random effects; no private ID features.

    Precisions are fixed before development/confirmation, not tuned on feedback.
    Subject deviations absorb configuration differences rather than pooling raw outcomes.
    """
    eb=np.clip(np.asarray(evidence_base),.001,.999);ys=np.asarray(labels,float)
    keys=sorted(set(subjects)|{target_subject});positions={k:j+1 for j,k in enumerate(keys)}
    x=np.zeros((len(ys),len(keys)+1));x[:,0]=1
    for j,key in enumerate(subjects):x[j,positions[key]]=1
    penalty=np.full(len(keys)+1,subject_precision);penalty[0]=task_precision
    theta=np.zeros(len(keys)+1);z=logit(eb)
    for _ in range(25):
        p=expit(z+x@theta);w=p*(1-p)
        grad=x.T@(p-ys)+penalty*theta
        h=x.T@(w[:,None]*x)+np.diag(penalty)
        step=np.linalg.solve(h,grad);theta-=np.clip(step,-2,2)
        if np.max(np.abs(step))<1e-8:break
    delta=theta[0]+theta[positions[target_subject]]
    return expit(logit(np.clip(target,.001,.999))+delta)


class SharedEnsemble:
    def __init__(self,base,shared_weight=.5):
        self.base=base;self.shared_weight=float(shared_weight)

    def predict_many(self,pairs,labeled=None):
        independent=np.asarray(self.base.predict_many(pairs,labeled));out=independent.copy()
        groups={}
        for j,p in enumerate(pairs):
            key=(subject_key(p[0]),p[1].get('benchmark_id',''))
            groups.setdefault(key,[]).append(j)
        for (sk,bid),ids in groups.items():
            ev=[(p,float(y)) for p,y in (labeled or []) if bid and p[1].get('benchmark_id')==bid]
            if not any(subject_key(p[0])!=sk for p,y in ev):continue
            ep=[p for p,y in ev];ys=[y for p,y in ev]
            shared=hierarchical_predict(self.base.base_many([pairs[j] for j in ids]),
                self.base.base_many(ep),ys,[subject_key(p[0]) for p,y in ev],sk)
            out[ids]=(1-self.shared_weight)*independent[ids]+self.shared_weight*shared
        return [float(p) for p in np.clip(out,.001,.999)]
