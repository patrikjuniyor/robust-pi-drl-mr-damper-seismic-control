"""Non-learning voltage baselines tuned on the randomised training distribution (128 synthetic motions, 1 ms step):
   P-BEST  : best common constant voltage
   P-BEST2 : best pair of constant voltages (storey 1, storey 2)
   AS      : amplitude-scheduled passive heuristic, gains (g1, g2)"""
import numpy as np, json, itertools
from simcore import *
from env import *
bld=Building('six'); N=128
env=MREnv(bld,N,ctrl_dt=0.01,nsub=10)
ag,pga=synth_motions(N,bld.T_train,env.dt,np.random.default_rng(4048),bld.time_scale,bld.pga_range)
par=sample_params(bld,N,np.random.default_rng(4049))
unc=None
KS=('JT','J1','J2','J3','J4')
out={'passive2':{},'as':{}}
g1s=(0.0,0.1,0.2,0.3,0.45,0.65,1.0); g2s=(0.0,0.1,0.2,0.3,0.45)
pairs=list(itertools.product(g1s,g2s))+[(0.65,0.65),(1.0,1.0)]
for f1,f2 in pairs:
    J=rollout(env,Passive([f1,f2]),ag,par,noise_seed=1,unc=unc); unc=J['unc_cache']
    out['passive2'][f'{f1}_{f2}']=[float(J[k].mean()) for k in KS]
eq={f:out['passive2'][f'{f}_{f}'] for f in g1s}
out['passive_eq']={str(f):v for f,v in eq.items()}
out['best_passive']=str(min(eq,key=lambda f:eq[f][0]))
k=min(out['passive2'],key=lambda k:out['passive2'][k][0]); out['best_passive2']=[float(x) for x in k.split('_')]
for f in g1s: print('P equal',f,np.round(eq[f],3))
print('best passive2',k,np.round(out['passive2'][k],3),flush=True)
for g1,g2 in itertools.product((1.0,2.0,4.0,8.0),(0.5,1.0,2.0,4.0)):
    J=rollout(env,AmpSched([g1,g2]),ag,par,noise_seed=1,unc=unc)
    out['as'][f'{g1}_{g2}']=[float(J[k].mean()) for k in KS]
k=min(out['as'],key=lambda k:out['as'][k][0]); out['best_as']=[float(x) for x in k.split('_')]
for kk in sorted(out['as'],key=lambda k:out['as'][k][0])[:5]: print('  AS',kk,np.round(out['as'][kk],3))
r2=json.load(open('../results/baselines.json'))
if 'passive_setA' not in r2: r2['passive_setA']=r2['passive']; r2['best_passive_setA']=r2['best_passive']
r2.update(out); r2['tune_N']=N
json.dump(r2,open('../results/baselines.json','w'),indent=1)
print('best_passive',out['best_passive'],'best_passive2',out['best_passive2'],'best_as',out['best_as'])
