"""Tune the baselines with the fine integration step (0.5 ms) on synthetic motions from the training distribution.
   * best constant voltage (P-BEST)                       -- randomised plants
   * clipped-optimal LQG at 500 Hz: Kalman-filter q, then LQR weights (w_a, r)
       CO-NOM : weights minimising J_T on the nominal plant
       CO-ROB : weights minimising J_T on the randomised plants
"""
import numpy as np, json
from simcore import *
from env import *
bld=Building('six'); N=64
env=MREnv(bld,N,ctrl_dt=0.01,nsub=20)
ag,pga=synth_motions(N,bld.T_train,env.dt,np.random.default_rng(2024),bld.time_scale,bld.pga_range)
par=sample_params(bld,N,np.random.default_rng(2025)); nom=sample_params(bld,N,np.random.default_rng(2025),nominal=True)
KS=('JT','J1','J2','J3','J4')
res={'passive':{},'passive_nom':{},'kf500':{},'co_nom':{},'co_rob':{}}; unc=None; uncn=None
for fr in (0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0):
    J=rollout(env,Passive(fr),ag,par,noise_seed=1,unc=unc); unc=J['unc_cache']
    res['passive'][str(fr)]=[float(J[k].mean()) for k in KS]
    J=rollout(env,Passive(fr),ag,nom,noise_seed=1,unc=uncn); uncn=J['unc_cache']
    res['passive_nom'][str(fr)]=[float(J[k].mean()) for k in KS]
    print('P',fr,'rand',np.round(res['passive'][str(fr)],3),'nominal',np.round(res['passive_nom'][str(fr)],3),flush=True)
res['best_passive']=min(res['passive'],key=lambda k:res['passive'][k][0])
json.dump(res,open('../results/baselines.json','w'),indent=1)
# ---- Kalman filter at 500 Hz
for q in (0.2,2.0,20.0):
    e=MREnv(bld,N,ctrl_dt=0.002,nsub=4,kf_q=q); e.reset(ag,par,noise_seed=1,unc=unc); num=np.zeros(12); den=np.zeros(12); done=False; c=Passive(0.3)
    while not done:
        _,done=e.step_voltage(c.act_voltage(e)); num+=((e.xhat-e.S[:,:12])**2).sum(0); den+=(e.S[:,:12]**2).sum(0)
    res['kf500'][str(q)]=float(np.sqrt(num/den).mean()); print('KF500 q',q,'rel err %.3f'%res['kf500'][str(q)],flush=True)
qb=float(min(res['kf500'],key=lambda k:res['kf500'][k])); res['kf_q500']=qb
envf=MREnv(bld,N,ctrl_dt=0.002,nsub=4,kf_q=qb)
for wa in (0.0,0.3,1.0,3.0):
    for r in (1e-3,1e-2,1e-1):
        Kg=envf.nom.lqr_energy(bld,1.0,wa,r)
        J=rollout(envf,ClippedOptimal(Kg),ag,nom,noise_seed=1,unc=uncn); res['co_nom'][f'{wa}_{r}']=[float(J[k].mean()) for k in KS]
        J=rollout(envf,ClippedOptimal(Kg),ag,par,noise_seed=1,unc=unc); res['co_rob'][f'{wa}_{r}']=[float(J[k].mean()) for k in KS]
        print('CO',wa,r,'nominal',np.round(res['co_nom'][f'{wa}_{r}'],3),'rand',np.round(res['co_rob'][f'{wa}_{r}'],3),flush=True)
        json.dump(res,open('../results/baselines.json','w'),indent=1)
res['best_co_nom']=min(res['co_nom'],key=lambda k:res['co_nom'][k][0]); res['best_co_rob']=min(res['co_rob'],key=lambda k:res['co_rob'][k][0])
print('best',res['best_passive'],res['best_co_nom'],res['best_co_rob'])
json.dump(res,open('../results/baselines.json','w'),indent=1)
print('TUNE_DONE')
