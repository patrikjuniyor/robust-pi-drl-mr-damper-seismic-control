"""Extend the clipped-optimal grid where the optimum was found on the edge (w_a = 0, smaller r)."""
import numpy as np, json
from simcore import *
from env import *
bld=Building('six'); N=64
res=json.load(open('../results/baselines.json'))
envf=MREnv(bld,N,ctrl_dt=0.002,nsub=4,kf_q=res['kf_q500'])
ag,pga=synth_motions(N,bld.T_train,envf.dt,np.random.default_rng(2024),bld.time_scale,bld.pga_range)
par=sample_params(bld,N,np.random.default_rng(2025)); nom=sample_params(bld,N,np.random.default_rng(2025),nominal=True)
KS=('JT','J1','J2','J3','J4'); unc=None; uncn=None
# first extension: (0.0,1e-4),(0.0,3e-4),(0.0,3e-3); second extension: (0.0,3e-5),(0.0,1e-5),(0.3,1e-4)
for wa,r in ((0.0,1e-4),(0.0,3e-4),(0.0,3e-3),(0.0,3e-5),(0.0,1e-5),(0.3,1e-4)):
    Kg=envf.nom.lqr_energy(bld,1.0,wa,r)
    J=rollout(envf,ClippedOptimal(Kg),ag,nom,noise_seed=1,unc=uncn); uncn=J['unc_cache']; res['co_nom'][f'{wa}_{r}']=[float(J[k].mean()) for k in KS]
    J=rollout(envf,ClippedOptimal(Kg),ag,par,noise_seed=1,unc=unc); unc=J['unc_cache']; res['co_rob'][f'{wa}_{r}']=[float(J[k].mean()) for k in KS]
    print('CO',wa,r,'nominal',np.round(res['co_nom'][f'{wa}_{r}'],3),'rand',np.round(res['co_rob'][f'{wa}_{r}'],3),flush=True)
res['best_co_nom']=min(res['co_nom'],key=lambda k:res['co_nom'][k][0]); res['best_co_rob']=min(res['co_rob'],key=lambda k:res['co_rob'][k][0])
print('best',res['best_passive'],res['best_co_nom'],res['best_co_rob'])
json.dump(res,open('../results/baselines.json','w'),indent=1)
