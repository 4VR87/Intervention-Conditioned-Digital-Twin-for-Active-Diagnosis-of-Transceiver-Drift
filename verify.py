"""Independent checks of retained synthetic data, inference and moment calculations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from scipy.special import erfc
from study import cohort,SIZES,ACTIONS,cp,threshold,evaluate,predict,rng
ROOT=Path(__file__).resolve().parents[1]

def run():
    report={};ids=[]
    for name,n in SIZES.items():
        d=np.load(ROOT/f'data/{name}.npz');ids.extend(d['id'].tolist())
        repeat=cohort(name,shift=name=='stress')
        for key in d.files:np.testing.assert_array_equal(d[key],repeat[key])
        assert d['observed'].shape==(n,7,2) and np.isfinite(d['observed']).all()
        assert np.all(d['evm2']>0)
    assert len(ids)==len(set(ids));report['disjoint_complete_episodes']=len(ids)
    m=joblib.load(ROOT/'data/fitted_models.joblib');r=json.loads((ROOT/'data/results.json').read_text())
    replay=0
    for policy in r['policies']:
        c=dict(np.load(ROOT/'data/calibration.npz'));d=dict(np.load(ROOT/'data/test.npz'))
        pc,ac=predict(m,c,policy);pt,at=predict(m,d,policy);q=threshold(pc,c['y'])
        check=evaluate(pt,d['y'],q);assert check==r['policies'][policy]['nominal']
        csv=pd.read_csv(ROOT/f'data/test_{policy.lower().replace(" ","_")}.csv')
        np.testing.assert_allclose(csv[[f'probability_{k}' for k in range(4)]],pt,atol=2e-15)
        assert (csv.action.values==at).all();replay+=len(pt)
    report['nominal_policy_decisions_replayed']=replay
    # Test-time action selection must be invariant to unobserved candidate views and labels.
    d=dict(np.load(ROOT/'data/test.npz'));p,a=predict(m,d,'Active ML')
    altered={k:v.copy() for k,v in d.items()};altered['y']=(altered['y']+1)%4
    altered['observed'][:,1:]=1e6;altered['residual'][:,1:]=1e6
    pp,aa=predict(m,altered,'Active ML');np.testing.assert_array_equal(a,aa)
    report['active_selection_cannot_inspect_unqueried_views_or_labels']='passed'
    # Recompute means from scalar equations, independent of vectorized cohort closure.
    maxerr=0
    for name in SIZES:
        d=np.load(ROOT/f'data/{name}.npz')
        for i in range(min(50,len(d['y']))):
            A,eta,b0,ph0,f0,cF,cB,R0=d['latent'][i];db,dph,lT,lF,df=d['delta'][i]
            for j,(pdB,ub,uf) in enumerate(ACTIONS):
                P=10**(pdB/10);e=A*10**((lT+lF)/10)/P+eta*P**2+cB*np.sin(b0+db+ub)**2+ph0+dph+cF*((f0+df-uf)/5)**2
                y=np.array([-10*np.log10(e),R0+pdB-lT-lF-.08*((f0+df-uf)/5)**2])
                maxerr=max(maxerr,float(np.max(abs(y-d['truth'][i,j]))))
    assert maxerr<1e-11;report['scalar_vs_vector_mean_max_abs_error']=maxerr
    # Independent complex-symbol check: EVM and Gray 16-QAM AWGN BER.
    levels=np.array([-3,-1,1,3]);gray=np.array([[0,0],[0,1],[1,1],[1,0]])
    checks=[];N=500000
    for k,snrdb in enumerate([12.,14.,16.,18.,20.]):
        rr=rng(1001,k);ix=rr.integers(0,4,(N,2));s=(levels[ix[:,0]]+1j*levels[ix[:,1]])/np.sqrt(10)
        eps=10**(-snrdb/10);n=(rr.normal(size=N)+1j*rr.normal(size=N))*np.sqrt(eps/2);z=s+n
        dec=np.column_stack([np.argmin(abs(np.sqrt(10)*z.real[:,None]-levels),axis=1),np.argmin(abs(np.sqrt(10)*z.imag[:,None]-levels),axis=1)])
        biterr=int((gray[dec]!=gray[ix]).sum());evm=float(np.mean(abs(z-s)**2));ber=biterr/(N*4)
        checks.append({'snr_db':snrdb,'symbols':N,'bits':4*N,'bit_errors':biterr,'measured_ber':ber,'awgn_approximation':float(.375*erfc(np.sqrt(10**(snrdb/10)/10))),'measured_evm2':evm,'target_evm2':eps,'ber_ci95':cp(biterr,4*N)})
        assert abs(evm/eps-1)<.01
        # Illustrative constellation samples only, retained explicitly.
        if snrdb==16:np.savez_compressed(ROOT/'data/symbol_constellation.npz',transmitted=s[:3000],received=z[:3000])
    pd.DataFrame(checks).to_csv(ROOT/'data/symbol_validation.csv',index=False)
    report['symbol_check']={'total_symbols':len(checks)*N,'max_relative_evm_error':max(abs(c['measured_evm2']/c['target_evm2']-1) for c in checks),'scope':'Equivalent additive-noise symbols; no receiver DSP, propagation or phase-recovery validation.'}
    # Exact alias: scalar loss before or after a noiseless gain has the same product.
    rr=rng(2001,0);loss=rr.uniform(.5,3,1000);P=10**(ACTIONS[:,0]/10)
    tx=P[None,:]*10**(-loss[:,None]/10);fiber=10**(-loss[:,None]/10)*P[None,:]
    assert np.array_equal(tx,fiber)
    # Paired opposite labels imply 50% for every forced deterministic binary decision.
    obs=np.repeat(tx,2,axis=0);lab=np.tile([0,1],1000)
    pred=np.repeat((loss>1.5).astype(int),2);assert np.mean(pred==lab)==.5
    np.savez_compressed(ROOT/'data/nonidentifiable_control.npz',observed_power=obs,label=lab,pair_id=np.repeat(np.arange(1000),2))
    report['nonidentifiable_control']={'pairs':1000,'episodes':2000,'max_observation_discrepancy':0.,'forced_binary_accuracy':.5,'interpretation':'Exact property of a restricted multiplicative-gain submodel; not a test of field fault identifiability.'}
    report['status']='passed';(ROOT/'audit/NUMERICAL_VALIDATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':run()
