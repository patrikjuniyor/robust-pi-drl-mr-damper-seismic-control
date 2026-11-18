"""Integration-accuracy check for the six-storey model on randomised plants (training distribution).
   Reference: 0.1 ms step.  Tested: training step (2 ms, adaptive micro-steps), evaluation step (1 ms),
   and the 0.5 ms step used with the 500 Hz clipped-optimal controller."""
import numpy as np, json, time, os, glob
from simcore import *
from env import *
from ppo import load_actor, make_policy_fn
bld=Building('six'); N=32
base=MREnv(bld,N)                                   # 100 Hz, 2 ms
ag0,pga=synth_motions(N,bld.T_train,base.dt,np.random.default_rng(5),bld.time_scale,bld.pga_range)
t0=np.arange(ag0.shape[1])*base.dt
p=sample_params(bld,N,np.random.default_rng(6))
def motions(dt):
    k=int(round(base.dt/dt)); t=np.arange((ag0.shape[1]-1)*k+1)*dt
    return np.array([np.interp(t,t0,a) for a in ag0])
b=json.load(open('../results/baselines.json'))
wa,r=[float(x) for x in b['best_co_nom'].split('_')]
pol=sorted(glob.glob('../runs/V_pi_dr_s0.pt'))
cases=[('Passive, V = 0',('p',0.0)),('Passive, V = Vmax',('p',1.0))]
if pol: cases.append(('PI-DRL policy (seed 0)',('d',pol[0])))
out={}
def run(kind,arg,ctrl_dt,nsub,noise):
    e=MREnv(bld,N,ctrl_dt=ctrl_dt,nsub=nsub,kf_q=(b['kf_q500'] if kind=='c' else None))
    pp=dict(p); 
    if not noise: pp['noise']=np.zeros(N)
    ag=motions(e.dt)
    if kind=='p': rollout(e,Passive(arg),ag,pp,noise_seed=1)
    elif kind=='c': rollout(e,ClippedOptimal(e.nom.lqr_energy(bld,1.0,wa,r)),ag,pp,noise_seed=1)
    else:
        a,args=load_actor(arg); rollout(e,None,ag,pp,noise_seed=1,policy_fn=make_policy_fn(a),obs_mode=args['obs'],K=args['K'])
    return e.met['pk_d'].max(1).copy(),e.met['pk_a'].max(1).copy()
def cmp(x,ref):
    d=np.abs(x[0]-ref[0])/ref[0]; a=np.abs(x[1]-ref[1])/ref[1]
    return dict(drift_med=float(np.median(d)),drift_max=float(d.max()),acc_med=float(np.median(a)),acc_max=float(a.max()))
for nm,(kind,arg) in cases:
    ref=run(kind,arg,0.01,100,False)
    out[nm]={'2 ms (training)':cmp(run(kind,arg,0.01,5,False),ref),'1 ms (evaluation)':cmp(run(kind,arg,0.01,10,False),ref)}
    print(nm,out[nm],flush=True)
ref=run('c',None,0.002,20,False)
out['Clipped-optimal, 500 Hz']={'0.5 ms (evaluation)':cmp(run('c',None,0.002,4,False),ref),'0.25 ms':cmp(run('c',None,0.002,8,False),ref)}
print('CO',out['Clipped-optimal, 500 Hz'])
json.dump(out,open('../results/integration_check.json','w'),indent=1)
