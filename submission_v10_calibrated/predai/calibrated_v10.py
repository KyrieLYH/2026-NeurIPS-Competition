import numpy as np
from scipy.special import expit,logit
from predai.adaptive_v3 import AdaptiveEnsemble
from predai.shared_v3 import SharedEnsemble

def apply_calibration(p,a,b):
    return np.clip(expit(a*logit(np.clip(np.asarray(p,float),.001,.999))+b),.001,.999)

class CalibratedBase:
    def __init__(self,base,a,b):self.base=base;self.a=float(a);self.b=float(b)
    def base_many(self,pairs):
        return apply_calibration(self.base.base_many(pairs),self.a,self.b) if pairs else np.array([])

class CalibratedEnsemble:
    def __init__(self,base,a=1.,b=0.,cold_only=True):
        self.a=float(a);self.b=float(b);self.cold_only=bool(cold_only)
        chosen=base if cold_only else CalibratedBase(base,a,b)
        self.wrapped=SharedEnsemble(AdaptiveEnsemble(chosen,None,{'method':'beta','tau':2.,'retain':.5,'boost_weight':0.}),.5)
    def predict_many(self,pairs,labeled=None):
        out=np.asarray(self.wrapped.predict_many(pairs,labeled))
        if self.cold_only:
            known={p[1].get('benchmark_id') for p,y in (labeled or [])}
            ids=[j for j,p in enumerate(pairs) if not p[1].get('benchmark_id') or p[1].get('benchmark_id') not in known]
            if ids:out[ids]=apply_calibration(out[ids],self.a,self.b)
        return [float(p) for p in np.clip(out,.001,.999)]
