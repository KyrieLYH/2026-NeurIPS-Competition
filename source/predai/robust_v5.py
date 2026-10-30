"""Conservative text regression with identical train/inference preprocessing."""
from __future__ import annotations
import json,math,re
from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from predai.compact_boost import numeric_row as original_numeric_row, FAMILIES, VARIANTS, _date_num
from predai.ensemble_v2 import features, visible_subject, safe_extension
from predai.adaptive_v3 import beta_predict
from predai.shared_v3 import hierarchical_predict, subject_key
from predai.features import _s,item_handcrafted

HAND_COUNT=len(item_handcrafted({}))

@lru_cache(maxsize=8192)
def _item_numeric(content,attrs,interactors):
    # No labels or model identity are cached; identical items share preprocessing.
    item={'item_content':content,'item_features':attrs,'interactors':interactors}
    return tuple(original_numeric_row([{},item]))

def numeric_row(pair):
    subject,item=pair
    row=np.array(_item_numeric(_s(item.get('item_content')),
        _s(item.get('item_features')),_s(item.get('interactors'))),dtype=np.float32)
    name=_s(subject.get('normalized_name') or subject.get('display_name')).lower()
    effort=_s(subject.get('reasoning_effort')).lower()
    family_end=HAND_COUNT+len(FAMILIES);tail=family_end+len(VARIANTS)
    row[HAND_COUNT:family_end]=[float(token in name) for token in FAMILIES]
    row[family_end:tail]=[float(token in name) for token in VARIANTS]
    size=re.search(r'(\d+(?:\.\d+)?)\s*[bB]\b',name)
    row[tail]=float(any(token in effort for token in ('high','max')))
    row[tail+1]=float(any(token in effort for token in ('low','min','none','minimal')))
    row[tail+5]=_date_num(subject.get('release_date'))
    row[tail+6]=_date_num(subject.get('access_date'))
    row[tail+7]=math.log1p(float(size.group(1))) if size else 0.
    row[tail+8]=float(size is not None)
    row[tail+9]=float(bool(effort))
    row[tail+10]=math.log1p(len(name))/5.
    return row

ALLOWED_ITEM = {'lang','website','paper','competition','problem_idx','prompt_source',
    'suite','context_length','task','category','context_condition','question_polarity',
    'database_id','difficulty','prompt_condition','capability'}

def clean_pair(pair):
    s, it = pair
    return [visible_subject(s), {
        'item_content': _s(it.get('item_content'))[:12000],
        'item_features': safe_extension(it.get('item_features'), ALLOWED_ITEM),
        'interactors': _s(it.get('interactors')),
        'benchmark_id': _s(it.get('benchmark_id'))}]

class RegularizedTextBoost:
    def fit(self, pairs, labels, sample_weight):
        pairs = [clean_pair(p) for p in pairs]
        self.hasher = HashingVectorizer(n_features=2**15, ngram_range=(1,2),dtype=np.float32)
        self.svd = TruncatedSVD(n_components=32,n_iter=3,random_state=20261028)
        text = self.hasher.transform([features(p)[1] for p in pairs])
        latent = self.svd.fit_transform(text).astype(np.float32)
        dense = np.vstack([numeric_row(p) for p in pairs])
        self.tree = HistGradientBoostingRegressor(loss='squared_error',
            max_iter=180,max_leaf_nodes=15,min_samples_leaf=80,
            l2_regularization=30.,learning_rate=.05,early_stopping=False,
            random_state=20261028)
        self.tree.fit(np.hstack([dense,latent]),labels,sample_weight=sample_weight)
        return self

    def base_many(self,pairs):
        if not pairs:return np.array([],dtype=float)
        pairs = [clean_pair(p) for p in pairs]
        latent = self.svd.transform(self.hasher.transform([features(p)[1] for p in pairs])).astype(np.float32)
        dense = np.vstack([numeric_row(p) for p in pairs])
        return np.clip(self.tree.predict(np.hstack([dense,latent])),.01,.99)

def adapt_v5(target, evidence_base, labels, config, pooled=None):
    p=beta_predict(target,evidence_base,labels,tau=config.get('tau',2.),retain=config.get('retain',.5))
    if pooled is not None:
        w=config.get('shared_weight',.5)
        p=(1-w)*p+w*pooled
    return np.clip(p,.001,.999)

class RobustEnsemble:
    def __init__(self,legacy,boost=None,config=None):
        self.legacy=legacy;self.boost=boost;self.config=dict(config or {})

    def base_many(self,pairs):
        if not pairs:return np.array([],dtype=float)
        pairs=[clean_pair(p) for p in pairs]
        weight=self.config.get('boost_weight',0.)
        if self.config.get('contraction',0.)==1.:
            return np.full(len(pairs),.5)
        if weight==1. and self.boost is not None:
            p=self.boost.base_many(pairs)
        else:
            p=self.legacy.base_many(pairs)
        if 0<weight<1 and self.boost is not None:
            p=(1-weight)*p+weight*self.boost.base_many(pairs)
        c=self.config.get('contraction',0.)
        return np.clip(.5+(1-c)*(p-.5),.001,.999)

    def predict_many(self,pairs,labeled=None):
        out=np.empty(len(pairs),float);groups={}
        for j,(s,it) in enumerate(pairs):
            groups.setdefault((subject_key(s),_s(it.get('benchmark_id'))),[]).append(j)
        for (sk,bid),ids in groups.items():
            targets=[pairs[j] for j in ids]
            ev=[(p,float(y)) for p,y in (labeled or []) if bid and _s(p[1].get('benchmark_id'))==bid]
            own=[(p,y) for p,y in ev if subject_key(p[0])==sk]
            p=self.base_many(targets)
            eb=self.base_many([pair for pair,y in own]);ys=np.array([y for pair,y in own])
            pooled=None
            if self.config.get('shared_weight',.5) and any(subject_key(pair[0])!=sk for pair,y in ev):
                pooled=hierarchical_predict(p,self.base_many([pair for pair,y in ev]),
                    [y for pair,y in ev],[subject_key(pair[0]) for pair,y in ev],sk)
            out[ids]=adapt_v5(p,eb,ys,self.config,pooled)
        return [float(p) for p in np.clip(out,.001,.999)]
