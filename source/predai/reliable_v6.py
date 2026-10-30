"""Infer task calibration AND trust in item ranking from acquired labels."""
import numpy as np
from scipy.special import expit,logit,logsumexp
from predai.robust_v5 import RobustEnsemble
from predai.shared_v3 import subject_key,hierarchical_predict
from predai.features import _s

def reliable_predict(target,eb,labels,sd=2.,prior=(.6,.3,.1)):
    target=np.clip(np.asarray(target,float),.001,.999)
    eb=np.clip(np.asarray(eb,float),.001,.999)
    labels=np.asarray(labels,float)
    delta=np.linspace(-8,8,81)
    slopes=np.array([0.,.5,1.])
    a=np.repeat(slopes,len(delta));b=np.tile(delta,len(slopes))
    lp=-.5*(b/sd)**2+np.repeat(np.log(prior),len(delta))
    if len(labels):
        z=a[:,None]*logit(eb)[None,:]+b[:,None]
        lp+=np.sum(labels[None,:]*z-np.logaddexp(0,z),axis=1)
    weights=np.exp(lp-logsumexp(lp))
    return np.concatenate([weights@expit(a[:,None]*logit(chunk)[None,:]+b[:,None])
        for chunk in np.array_split(target,max(1,(len(target)+255)//256))])

class ReliableEnsemble(RobustEnsemble):
    def predict_many(self,pairs,labeled=None):
        out=np.empty(len(pairs),float);groups={}
        for j,(s,it) in enumerate(pairs):
            groups.setdefault((subject_key(s),_s(it.get('benchmark_id'))),[]).append(j)
        for (sk,bid),ids in groups.items():
            ev=[(p,float(y)) for p,y in (labeled or []) if bid and _s(p[1].get('benchmark_id'))==bid]
            own=[(p,y) for p,y in ev if subject_key(p[0])==sk]
            p=self.base_many([pairs[j] for j in ids])
            eb=self.base_many([pair for pair,y in own]);ys=[y for pair,y in own]
            independent=reliable_predict(p,eb,ys,self.config['sd'],self.config['slope_prior'])
            w=self.config.get('shared_weight',0.)
            if w and any(subject_key(pair[0])!=sk for pair,y in ev):
                pooled=hierarchical_predict(p,self.base_many([pair for pair,y in ev]),
                    [y for pair,y in ev],[subject_key(pair[0]) for pair,y in ev],sk)
                independent=(1-w)*independent+w*pooled
            out[ids]=independent
        return [float(p) for p in np.clip(out,.001,.999)]
