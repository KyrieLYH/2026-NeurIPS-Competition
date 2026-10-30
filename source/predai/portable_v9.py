"""Portable numeric features and frozen convex base-model blends."""
import math,re
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from predai.robust_v5 import numeric_row,clean_pair,HAND_COUNT
from predai.compact_boost import FAMILIES,VARIANTS
from predai.features import _s

PROVIDERS=('openai','anthropic','google','meta','mistral','alibaba','deepseek',
           'microsoft','cohere','amazon','xai','nvidia','zhipu','baidu')
HARNESSES=('openhands','swe','react','agentless','browser','aider')

def portable_row(pair):
    subject,item=clean_pair(pair)
    base=numeric_row([subject,item]);tail=HAND_COUNT+len(FAMILIES)+len(VARIANTS)
    # Access date and literal name length can become dataset-specific shortcuts.
    base=np.delete(base,[tail+6,tail+10])
    name=_s(subject.get('normalized_name')).lower()
    provider=_s(subject.get('provider')).lower();harness=_s(subject.get('harness')).lower()
    no_size=re.sub(r'\d+(?:\.\d+)?\s*[bB]\b','',name)
    versions=[float(x) for x in re.findall(r'\d+(?:\.\d+)?',no_size)]
    versions=[x for x in versions if 0<=x<=50]
    ver=versions[0] if versions else 0.
    text=_s(item.get('item_content'))[:12000]
    extra=[math.log1p(ver),float(bool(versions)),float(bool(subject.get('release_date'))),
        float(any(x in name for x in ('vision','multimodal','-vl','audio'))),
        float(bool(re.search(r'\bo[134]\b',name))),
        float(bool(re.search(r'\b(select|from|where|sql)\b',text,re.I))),
        float(bool(re.search(r'\b(multiple choice|choose the|which of|options:)\b',text,re.I))),
        math.log1p(text.count('?')),math.log1p(text.count('```'))]
    extra.extend(float(p in provider) for p in PROVIDERS)
    extra.extend(float(h in harness) for h in HARNESSES)
    return np.r_[base,np.asarray(extra,dtype=np.float32)].astype(np.float32)

def portable_matrix(pairs,progress=False):
    rows=[]
    for j,p in enumerate(pairs):
        rows.append(portable_row(p))
        if progress and (j+1)%50000==0:print('PORTABLE_FEATURES',j+1,flush=True)
    return np.vstack(rows) if rows else np.empty((0,99),dtype=np.float32)

class PortableBase:
    def __init__(self,kind='hgb'):
        self.kind=kind
    def fit_matrix(self,x,y,w):
        if self.kind=='hgb':
            self.model=HistGradientBoostingRegressor(loss='squared_error',max_iter=240,
                max_leaf_nodes=31,min_samples_leaf=50,l2_regularization=10.,learning_rate=.05,
                early_stopping=False,random_state=901029)
        else:
            self.model=ExtraTreesRegressor(n_estimators=128,max_depth=16,min_samples_leaf=40,
                max_features=.75,n_jobs=12,random_state=901029)
        self.model.fit(x,y,sample_weight=w)
        if hasattr(self.model,'n_jobs'):self.model.n_jobs=1
        return self
    def base_many(self,pairs):
        if not pairs:return np.array([])
        return np.clip(self.model.predict(portable_matrix(pairs)),.001,.999)

class FrozenBlend:
    def __init__(self,models,weights):
        self.models=list(models);self.weights=np.asarray(weights,float)
        assert np.all(self.weights>=0) and abs(self.weights.sum()-1)<1e-10
    def base_many(self,pairs):
        if not pairs:return np.array([])
        return np.clip(sum(w*m.base_many(pairs) for m,w in zip(self.models,self.weights)),.001,.999)
