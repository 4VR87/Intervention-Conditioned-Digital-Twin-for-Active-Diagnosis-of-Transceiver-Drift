"""Executed synthetic study. No measured data or full wave propagation is implied."""
from pathlib import Path
import json, platform, sys, time
import numpy as np
import pandas as pd
from scipy.special import erfc
from scipy.stats import beta
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
import sklearn, scipy, joblib

ROOT=Path(__file__).resolve().parents[1]
LABELS=['Healthy','Transceiver','Link','Combined']
ACTIONS=np.array([[0,0,0],[0,.06,0],[0,-.06,0],[0,0,.75],[0,0,-.75],[1,0,0],[-1,0,0]],float)
ACTION_NAMES=['Passive','Bias +0.06 rad','Bias −0.06 rad','Detuning +0.75 GHz','Detuning −0.75 GHz','Power +1 dB','Power −1 dB']
SIZES={'train':4000,'policy':1200,'calibration':1000,'test':2000,'stress':1000}
DOMAINS={'train':11,'policy':22,'calibration':33,'test':44,'stress':55}
SEED=261029

def rng(domain,stream=0):
    return np.random.default_rng(np.random.SeedSequence([SEED,int(domain),int(stream)]))

def cohort(name,n=None,noise=1.,shift=False):
    """Twin coefficients estimated with error; episode-independent complete splits."""
    n=SIZES[name] if n is None else n;r=rng(DOMAINS[name],1)
    y=r.integers(0,4,n);tx=(y==1)|(y==3);link=(y==2)|(y==3)
    latent=np.column_stack([r.uniform(.006,.016,n),r.uniform(.0004,.0015,n),
        r.uniform(-.02,.02,n),r.uniform(.0003,.001,n),r.uniform(-.4,.4,n),
        r.uniform(.008,.016,n),r.uniform(.35,.55,n),r.uniform(-11,-9,n)])
    # A, eta, healthy IQ bias, healthy phase error, filter offset, filter coefficient,
    # IQ coefficient and nominal received power. Units are documented separately.
    delta=np.column_stack([tx*r.choice([-1,1],n)*r.uniform(.06,.19,n),
        tx*r.uniform(.001,.006,n),tx*r.uniform(0,.7,n),
        link*r.uniform(.5,3,n),link*r.choice([-1,1],n)*r.uniform(.8,3,n)])
    est=latent.copy();est[:,0:2]*=np.exp(r.normal(0,.10,(n,2)))
    est[:,2]+=r.normal(0,.008,n);est[:,3]*=np.exp(r.normal(0,.10,n))
    est[:,4]+=r.normal(0,.10,n);est[:,5:7]*=np.exp(r.normal(0,.08,(n,2)))
    est[:,7]+=r.normal(0,.10,n)
    if shift:latent[:,0:2]*=1.25;latent[:,5]*=1.25
    def means(par,current):
        p=10**(ACTIONS[None,:,0]/10)
        bias=par[:,2,None]+ACTIONS[None,:,1]+(delta[:,0,None] if current else 0)
        phase=par[:,3,None]+(delta[:,1,None] if current else 0)
        detune=par[:,4,None]-ACTIONS[None,:,2]+(delta[:,4,None] if current else 0)
        loss=delta[:,2,None]+delta[:,3,None] if current else np.zeros((n,1))
        # GN-inspired equivalent additive-noise terms; not an identified GN fiber model.
        eps=par[:,0,None]*10**(loss/10)/p+par[:,1,None]*p*p
        eps+=par[:,6,None]*np.sin(bias)**2+phase+par[:,5,None]*(detune/5)**2
        power=par[:,7,None]+ACTIONS[None,:,0]-loss-.08*(detune/5)**2
        return np.stack([-10*np.log10(eps),power],axis=-1),eps
    truth,evm2=means(latent,True);twin,_=means(est,False)
    z=rng(DOMAINS[name],2).normal(size=(n,7,2))
    # Shared episode offset models monitoring calibration uncertainty.
    shared=rng(DOMAINS[name],3).normal(size=(n,1,2))*np.array([.10,.08])
    observed=truth+noise*(z*np.array([.25,.18])+shared)
    residual=observed-twin
    return {'id':np.arange(n)+DOMAINS[name]*100000,'y':y,'latent':latent,
            'delta':delta,'twin':twin,'truth':truth,'observed':observed,
            'residual':residual,'evm2':evm2}

def features(d,a=0,raw=False):
    z=d['observed'] if raw else d['residual'];base=z[:,0]
    if a==0:return base
    if a=='all':return np.concatenate([base,z[:,1:].reshape(len(base),-1),
                                     (z[:,1:]-base[:,None]).reshape(len(base),-1)],axis=1)
    return np.column_stack([base,z[:,a],z[:,a]-base])

def clf(seed=SEED):
    return ExtraTreesClassifier(n_estimators=240,min_samples_leaf=8,max_features=1.,
        n_jobs=2,random_state=seed)

def train(ds,n=None,seed=SEED,raw=False):
    t,p=ds['train'],ds['policy'];n=len(t['y']) if n is None else n
    models={};gains=np.zeros((len(p['y']),6))
    for a in [0,1,2,3,4,5,6,'all']:
        m=clf(seed);m.fit(features(t,a,raw)[:n],t['y'][:n]);models[a]=m
    pp=models[0].predict_proba(features(p,0,raw));base=-np.log(np.maximum(pp[np.arange(len(pp)),p['y']],1e-9))
    value=[]
    for a in range(1,7):
        prob=models[a].predict_proba(features(p,a,raw))
        gains[:,a-1]=base+np.log(np.maximum(prob[np.arange(len(prob)),p['y']],1e-9))
        reg=ExtraTreesRegressor(n_estimators=180,min_samples_leaf=24,max_features=1.,n_jobs=2,random_state=seed+a)
        reg.fit(features(p,0,raw),gains[:,a-1]);value.append(reg)
    fixed=int(np.argmax(gains.mean(axis=0)))+1
    logistic=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,C=1.))
    logistic.fit(features(t,0,raw)[:n],t['y'][:n])
    return {'classifiers':models,'value':value,'fixed':fixed,'gains':gains,'logistic':logistic,'raw':raw}

def predict(model,d,policy):
    raw=model['raw'];m=model['classifiers'];n=len(d['y'])
    if policy=='Passive logistic':return model['logistic'].predict_proba(features(d,0,raw)),np.zeros(n,int)
    if policy=='Passive ML':return m[0].predict_proba(features(d,0,raw)),np.zeros(n,int)
    if policy=='All probes':return m['all'].predict_proba(features(d,'all',raw)),np.full(n,7,int)
    if policy=='Active ML':
        # Only passive telemetry enters action selection.
        values=np.column_stack([g.predict(features(d,0,raw)) for g in model['value']]);a=1+values.argmax(axis=1)
    elif policy=='Fixed probe':a=np.full(n,model['fixed'],int)
    elif policy=='Random probe':a=np.random.default_rng(np.random.SeedSequence([SEED,991,int(d['id'][0])])).integers(1,7,n)
    else:raise ValueError(policy)
    out=np.empty((n,4))
    for j in range(1,7):
        idx=np.where(a==j)[0]
        if len(idx):out[idx]=m[j].predict_proba(features(d,j,raw)[idx])
    return out,a

def cp(k,n,alpha=.05):
    if n==0:return [None,None]
    return [float(beta.ppf(alpha/2,k,n-k+1)) if k else 0.,
            float(beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.]

def threshold(p,y,alpha=.1):
    s=1-p[np.arange(len(y)),y];k=int(np.ceil((len(y)+1)*(1-alpha)))
    return float(np.sort(s)[k-1]) if k<=len(s) else 1.

def evaluate(p,y,q):
    pred=p.argmax(axis=1);sets=1-p<=q+1e-14;size=sets.sum(axis=1);single=size==1
    covered=sets[np.arange(len(y)),y];err=(pred!=y)&single
    return {'n':len(y),'correct':int((pred==y).sum()),'accuracy':float(np.mean(pred==y)),
        'accuracy_ci':cp(int((pred==y).sum()),len(y)),'log_loss':float(log_loss(y,p,labels=np.arange(4))),
        'coverage':float(covered.mean()),'coverage_count':int(covered.sum()),'coverage_ci':cp(int(covered.sum()),len(y)),
        'singleton_count':int(single.sum()),'singleton_rate':float(single.mean()),
        'singleton_errors':int(err.sum()),'singleton_error':float(err.sum()/single.sum()) if single.any() else None,
        'singleton_error_ci':cp(int(err.sum()),int(single.sum())),
        'mean_set_size':float(size.mean()),'empty_count':int((size==0).sum()),'q':q,
        'confusion':confusion_matrix(y,pred,labels=np.arange(4)).tolist()}

def paired_boot(x,y):
    delta=x.astype(float)-y.astype(float);r=rng(900,1)
    values=np.array([delta[r.integers(0,len(delta),len(delta))].mean() for _ in range(5000)])
    return {'difference':float(delta.mean()),'ci95':np.quantile(values,[.025,.975]).tolist(),
        'positive_discordances':int(((x==1)&(y==0)).sum()),'negative_discordances':int(((x==0)&(y==1)).sum())}

def run():
    tic=time.time();ds={s:cohort(s,shift=s=='stress') for s in SIZES}
    for name,d in ds.items():np.savez_compressed(ROOT/f'data/{name}.npz',**d)
    model=train(ds);joblib.dump(model,ROOT/'data/fitted_models.joblib',compress=3)
    results={'scope':'Executed synthetic equivalent-noise coherent-link study','classes':LABELS,
             'fixed_probe':ACTION_NAMES[model['fixed']],'fixed_probe_index':model['fixed'],
             'policy_mean_gains':model['gains'].mean(axis=0).tolist(),'populations':{s:np.bincount(d['y'],minlength=4).tolist() for s,d in ds.items()},'policies':{},'paired':{}}
    policies=['Passive logistic','Passive ML','Random probe','Fixed probe','Active ML','All probes'];saved={}
    for policy in policies:
        pc,ac=predict(model,ds['calibration'],policy);pt,at=predict(model,ds['test'],policy)
        ps,ass=predict(model,ds['stress'],policy);q=threshold(pc,ds['calibration']['y'])
        results['policies'][policy]={'nominal':evaluate(pt,ds['test']['y'],q),'stress':evaluate(ps,ds['stress']['y'],q)}
        saved[policy]=(pc,pt,at)
        for pop,p,a in [('calibration',pc,ac),('test',pt,at),('stress',ps,ass)]:
            dd=pd.DataFrame({'episode_id':ds[pop]['id'],'true_class':ds[pop]['y'],'predicted_class':p.argmax(axis=1),'action':a})
            for j in range(4):dd[f'probability_{j}']=p[:,j]
            dd['set_size']=(1-p<=q+1e-14).sum(axis=1);dd['covered']=1-p[np.arange(len(p)),ds[pop]['y']]<=q+1e-14
            dd.to_csv(ROOT/f'data/{pop}_{policy.lower().replace(" ","_")}.csv',index=False)
    truth=ds['test']['y'];active=saved['Active ML'][1].argmax(axis=1)==truth
    for policy in policies:
        if policy!='Active ML':results['paired'][policy]=paired_boot(active,saved[policy][1].argmax(axis=1)==truth)
    # Target error sensitivity, no retuning of frozen models.
    curves=[]
    for alpha in [.01,.025,.05,.1,.15,.2]:
        for policy in ['Passive ML','Fixed probe','Active ML','All probes']:
            pc,pt,_=saved[policy];z=evaluate(pt,truth,threshold(pc,ds['calibration']['y'],alpha))
            curves.append({'alpha':alpha,'policy':policy,**{k:v for k,v in z.items() if not isinstance(v,list)}})
    pd.DataFrame(curves).to_csv(ROOT/'data/coverage_sensitivity.csv',index=False)
    sizes=[]
    for n in [500,1000,2000,4000]:
        mm=train(ds,n) if n!=4000 else model
        for policy in ['Passive ML','Fixed probe','Active ML','All probes']:
            pt,at=predict(mm,ds['test'],policy);sizes.append({'n_training':n,'policy':policy,'accuracy':accuracy_score(truth,pt.argmax(axis=1))})
    pd.DataFrame(sizes).to_csv(ROOT/'data/training_sensitivity.csv',index=False)
    rows=[]
    for noise in [.5,1.,2.]:
        dc=cohort('calibration',noise=noise);dt=cohort('test',noise=noise)
        for policy in ['Passive ML','Fixed probe','Active ML','All probes']:
            pc,ac=predict(model,dc,policy);pt,at=predict(model,dt,policy);q=threshold(pc,dc['y']);ev=evaluate(pt,dt['y'],q)
            rows.append({'noise_multiplier':noise,'policy':policy,**{k:v for k,v in ev.items() if not isinstance(v,list)}})
    pd.DataFrame(rows).to_csv(ROOT/'data/noise_sensitivity.csv',index=False)
    seeds=[]
    for seed in [SEED,SEED+1,SEED+2]:
        mm=model if seed==SEED else train(ds,seed=seed)
        for policy in ['Passive ML','Fixed probe','Active ML']:
            pp,aa=predict(mm,ds['test'],policy);seeds.append({'seed':seed,'policy':policy,'accuracy':accuracy_score(truth,pp.argmax(axis=1))})
    pd.DataFrame(seeds).to_csv(ROOT/'data/seed_sensitivity.csv',index=False)
    # Residual-free twin ablation, same split/budget/hyperparameters.
    raw=train(ds,raw=True);ab={}
    for policy in ['Passive ML','Fixed probe','Active ML']:
        pc,_=predict(raw,ds['calibration'],policy);pt,_=predict(raw,ds['test'],policy)
        ab[policy]=evaluate(pt,truth,threshold(pc,ds['calibration']['y']))
    results['raw_telemetry_ablation']=ab
    results['active_action_counts']=np.bincount(saved['Active ML'][2],minlength=7).tolist()
    # AWGN BER is an approximation; no invented counted BER.
    snr=10**(ds['test']['truth'][:,0,0]/10);approx_ber=.375*erfc(np.sqrt(snr/10))
    np.savez_compressed(ROOT/'data/nominal_link_quality.npz',snr_db=ds['test']['truth'][:,0,0],gray16qam_awgn_ber=approx_ber,labels=truth)
    results['elapsed_seconds']=time.time()-tic
    (ROOT/'data/results.json').write_text(json.dumps(results,indent=2))
    (ROOT/'ENVIRONMENT.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},indent=2))
    print(json.dumps({'fixed':results['fixed_probe'],'nominal':{k:v['nominal'] for k,v in results['policies'].items()},'paired':results['paired'],'seconds':results['elapsed_seconds']},indent=2),flush=True)

if __name__=='__main__':run()
