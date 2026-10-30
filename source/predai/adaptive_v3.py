"""Bayesian task calibration and local evidence; all state is offline or call-local."""
from __future__ import annotations
import json
import re
import numpy as np
from scipy.special import expit, logit, logsumexp
from predai.ensemble_v2 import features, visible_subject, safe_extension, offset
from predai.features import _s


def attributes(item):
    text=safe_extension(item.get('item_features',''), {
        'lang','website','paper','competition','problem_idx','prompt_source','suite',
        'context_length','task','category','context_condition','question_polarity',
        'database_id','difficulty','prompt_condition','capability'})
    return dict(part.split('=',1) for part in text.split(';') if '=' in part)


def enhanced_features(pair):
    row,_=features(pair)
    s,it=pair; name=_s(s.get('normalized_name')).lower()
    for token in ('llama','qwen','gpt','claude','deepseek','gemini','mistral','mixtral',
                  'command','phi','yi','wizard','secalign','unifiedqa','t5'):
        row['family_'+token]=float(token in name)
    for token in ('instruct','chat','base','mini','nano','pro','flash','sonnet','opus','thinking'):
        row['variant_'+token]=float(token in name)
    row['reasoning_high']=float(any(t in _s(s.get('reasoning_effort')).lower() for t in ('high','max')))
    text=_s(it.get('item_content'))[:12000]
    row['cjk_fraction']=sum('\u4e00'<=c<='\u9fff' for c in text)/max(len(text),1)
    row['choice_count']=min(10,len(re.findall(r'(?:^|\n)\s*[A-E][.):]',text)))
    at=attributes(it)
    c=re.match(r'([\d.]+)\s*([km]?)',at.get('context_length',''),re.I)
    row['context_log_length']=float(np.log1p(float(c[1])*{'':1,'k':1024,'m':1048576}[c[2].lower()])) if c else 0.
    for field in ('release_date','access_date'):
        match=re.search(r'\b(20\d\d)(?:-(\d\d))?',_s(s.get(field)))
        row[field+'_numeric']=(int(match[1])-2020)+(int(match[2] or 1)-1)/12 if match else 0.
    # Drop long exact extension strings from the dense branch; reviewed values remain textual in v2.
    row.pop('subject_subject_features_extra',None)
    return row


class BoostedBase:
    """Brier-trained gradient boosting over visible structured + latent text features."""
    def fit(self, pairs, labels, sample_weight):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.dv=DictVectorizer(sparse=False,dtype=np.float32)
        x=self.dv.fit_transform([enhanced_features(p) for p in pairs])
        self.hasher=HashingVectorizer(n_features=2**15,ngram_range=(1,2),dtype=np.float32)
        texts=[features(p)[1] for p in pairs]
        self.svd=TruncatedSVD(n_components=24,n_iter=3,random_state=2026)
        z=self.svd.fit_transform(self.hasher.transform(texts)).astype(np.float32)
        self.tree=HistGradientBoostingRegressor(loss='squared_error',max_iter=140,
            max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=20.,
            learning_rate=.06,early_stopping=False,random_state=2026)
        self.tree.fit(np.column_stack([x,z]),labels,sample_weight=sample_weight)
        return self

    def base_many(self,pairs):
        if not pairs:return np.array([])
        x=self.dv.transform([enhanced_features(p) for p in pairs])
        z=self.svd.transform(self.hasher.transform([features(p)[1] for p in pairs])).astype(np.float32)
        return np.clip(self.tree.predict(np.column_stack([x,z])),.002,.998)


def bayes_predict(target, evidence_base, labels, sd=2., slope_mix=False):
    """Integrate a task intercept on a fixed grid, rather than plug in its mode.

    Center prior predictive at the base logit after optional slope shrinkage.
    Broad task uncertainty is intentional even with zero revealed labels.
    """
    target=np.clip(np.asarray(target,float),.001,.999)
    eb=np.clip(np.asarray(evidence_base,float),.001,.999)
    ys=np.asarray(labels,float)
    if len(eb)!=len(ys) or not np.all(np.isfinite(ys)) or np.any((ys<0)|(ys>1)):
        raise ValueError('Invalid revealed labels')
    if not np.isfinite(sd) or sd<=0:raise ValueError('sd must be positive')
    delta=np.linspace(-8,8,129)
    slopes=np.array([0.,.5,1.]) if slope_mix else np.array([1.])
    prior_slope=np.array([.15,.25,.6]) if slope_mix else np.array([1.])
    a=np.repeat(slopes,len(delta)); b=np.tile(delta,len(slopes))
    lp=-.5*(b/sd)**2+np.repeat(np.log(prior_slope),len(delta))
    if len(ys):
        logits=a[:,None]*logit(eb)[None,:]+b[:,None]
        lp+=np.sum(ys[None,:]*logits-np.logaddexp(0,logits),axis=1)
    weights=np.exp(lp-logsumexp(lp))
    # Chunking bounds memory for large evaluation response sets.
    return np.concatenate([weights@expit(a[:,None]*logit(chunk)[None,:]+b[:,None])
                           for chunk in np.array_split(target,max(1,(len(target)+511)//512))]) if len(target) else np.array([])


def beta_predict(target,evidence_base,labels,tau=2.,retain=.5):
    """Pseudo-observation update with base risk centered on acquisition predictions."""
    if not len(labels):return np.asarray(target)
    prior=float(np.mean(evidence_base))
    mean=(tau*prior+np.sum(labels))/(tau+len(labels))
    # Shift each item relative to the acquisition mean, avoiding evaluation-pool statistics.
    shifted=expit(logit(np.clip(target,.001,.999))-logit(np.clip(prior,.001,.999))+logit(np.clip(mean,.001,.999)))
    return retain*shifted+(1-retain)*mean


def local_similarity(pairs,evidence_pairs):
    from sklearn.feature_extraction.text import HashingVectorizer
    hv=HashingVectorizer(n_features=2**14,ngram_range=(1,2),alternate_sign=False)
    texts=[_s(p[1].get('item_content'))[:12000] for p in pairs]
    et=[_s(p[1].get('item_content'))[:12000] for p in evidence_pairs]
    sim=(hv.transform(texts)@hv.transform(et).T).toarray()
    ta=[attributes(p[1]) for p in pairs]; ea=[attributes(p[1]) for p in evidence_pairs]
    for j,e in enumerate(ea):
        for i,t in enumerate(ta):
            comparable=[k for k in ('task','context_length','category','difficulty','suite','lang','context_condition') if k in t and k in e]
            if comparable:sim[i,j]=.5*sim[i,j]+.5*np.mean([t[k]==e[k] for k in comparable])
    return sim


def local_predict(target,evidence_base,labels,pairs,evidence_pairs,kappa=2.,strength=3.,kernel=None):
    """Kernel-weighted likelihood using only columns for allowed evidence."""
    if not len(labels):return np.asarray(target)
    sim=local_similarity(pairs,evidence_pairs) if kernel is None else np.asarray(kernel)
    if sim.shape!=(len(target),len(labels)):raise ValueError('Kernel/evidence shape mismatch')
    w=.25+.75*np.clip(sim,0,1)**2
    z=logit(np.clip(evidence_base,.001,.999));delta=np.full(len(target),offset(np.asarray(evidence_base),np.asarray(labels),kappa))
    # Normalize to preserve an effective evidence size bounded by the actual count.
    for _ in range(20):
        q=expit(z[None,:]+delta[:,None])
        grad=np.sum(w*(q-np.asarray(labels)),axis=1)+strength*delta/4
        h=np.sum(w*q*(1-q),axis=1)+strength/4
        step=np.clip(grad/h,-2,2);delta-=step
        if np.max(np.abs(step))<1e-7:break
    return expit(logit(np.clip(target,.001,.999))+delta)


DEFAULT_CONFIG={'boost_weight':0.,'method':'bayes','sd':2.,'slope_mix':False}


def adapt(config,target,eb,ys,pairs=None,evidence_pairs=None,kernel=None):
    method=config['method']
    if not len(ys) and method not in ('bayes','bayes_slope'):
        return np.asarray(target)
    if method=='map':
        return expit(logit(np.clip(target,.001,.999))+offset(np.asarray(eb),np.asarray(ys),config.get('kappa',2.)))
    if method in ('bayes','bayes_slope'):
        return bayes_predict(target,eb,ys,config.get('sd',2.),method=='bayes_slope')
    if method=='beta':return beta_predict(target,eb,ys,config.get('tau',2.),config.get('retain',.5))
    if method=='local':return local_predict(target,eb,ys,pairs,evidence_pairs,strength=config.get('strength',3.),kernel=kernel)
    raise ValueError('Unknown adaptation method')


class AdaptiveEnsemble:
    def __init__(self,legacy,boost=None,config=None):
        self.legacy=legacy;self.boost=boost;self.config=dict(config or DEFAULT_CONFIG)

    def base_many(self,pairs):
        p=self.legacy.base_many(pairs);w=self.config.get('boost_weight',0.)
        if w and self.boost is not None:p=(1-w)*p+w*self.boost.base_many(pairs)
        return np.clip(p,.001,.999)

    def predict_many(self,pairs,labeled=None):
        groups={};out=np.empty(len(pairs),float)
        for j,p in enumerate(pairs):
            key=json.dumps([visible_subject(p[0]),_s(p[1].get('benchmark_id'))],sort_keys=True)
            groups.setdefault(key,[]).append(j)
        for ids in groups.values():
            ps=[pairs[j] for j in ids];subject,item=ps[0]
            # Normalize exactly the eight public subject fields; ignore extraneous keys.
            ev=[(p,float(y)) for p,y in (labeled or [])
                if item.get('benchmark_id') and p[1].get('benchmark_id')==item.get('benchmark_id')
                and visible_subject(p[0])==visible_subject(subject)]
            ep=[p for p,y in ev];ys=np.array([y for p,y in ev])
            p=adapt(self.config,self.base_many(ps),self.base_many(ep) if ep else [],ys,ps,ep)
            out[ids]=p
        return [float(p) for p in np.clip(out,.001,.999)]
