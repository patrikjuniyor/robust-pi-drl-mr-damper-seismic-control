"""Simulator verification: reproduce Table 2 of Dyke et al. (1996) (3-storey model, El Centro NS x5 time-compressed)."""
import numpy as np, json
from simcore import *
from env import *
bld=Building('dyke3'); recs=load_records(bld.time_scale)
env=MREnv(bld,1,ctrl_dt=0.002,nsub=4)
dt_r,acc=recs['El Centro 1940']; ag=resample(dt_r,acc,env.dt)[None]*G
p=sample_params(bld,1,np.random.default_rng(0),nominal=True)
out={}
def grab(nm):
    m=env.met; out[nm]=dict(x=(m['pk_x'][0]*100).round(3).tolist(),d=(m['pk_d'][0]*100).round(3).tolist(),a=(m['pk_a'][0]*100).round(0).tolist(),f=float(m['pk_f'][0].round(0)[0]))
rollout(env,Passive(0.0),ag,p); grab('Passive-off')
u=env.unc; out['Uncontrolled']=dict(x=(u['pk_x'][0]*100).round(3).tolist(),d=(u['pk_d'][0]*100).round(3).tolist(),a=(u['pk_a'][0]*100).round(0).tolist(),f=None)
rollout(env,Passive(1.0),ag,p); grab('Passive-on')
best=None
for r in (1e-17,1e-15,1e-14,1e-13,1e-12):
    rollout(env,ClippedOptimal(env.nom.lqr_dyke(r=r)),ag,p); grab('Clipped-optimal r=%g'%r)
out['reference']={'Uncontrolled':dict(x=[0.538,0.820,0.962],d=[0.538,0.319,0.201],a=[856,1030,1400],f=None),
 'Passive-off':dict(x=[0.211,0.357,0.455],d=[0.211,0.153,0.103],a=[420,480,717],f=258),
 'Passive-on':dict(x=[0.076,0.196,0.306],d=[0.076,0.158,0.110],a=[281,494,767],f=979),
 'Clipped-optimal':dict(x=[0.114,0.185,0.212],d=[0.114,0.090,0.101],a=[696,739,703],f=941)}
json.dump(out,open('../results/validation_dyke.json','w'),indent=1)
for k,v in out.items():
    if k!='reference': print(f"{k:26s}",v['x'],v['d'],v['a'],v['f'])
