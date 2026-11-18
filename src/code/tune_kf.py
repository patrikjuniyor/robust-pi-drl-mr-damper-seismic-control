"""Choose the process-noise intensity of the nominal-model Kalman filter (state-estimation error under randomised plants)."""
import numpy as np, json
from simcore import *
from env import *
bld=Building('six'); N=48
e0=MREnv(bld,N)
ag,pga=synth_motions(N,bld.T_train,e0.dt,np.random.default_rng(11),bld.time_scale,bld.pga_range); p=sample_params(bld,N,np.random.default_rng(12))
out={}
for q in (0.02,0.2,2.0,20.0,200.0):
    for ri in (1.0,30.0):
        env=MREnv(bld,N,kf_q=q,kf_rinfl=ri); errs=[]
        for fr in (0.1,1.0):
            env.reset(ag,p,noise_seed=1); num=np.zeros(12); den=np.zeros(12); done=False; c=Passive(fr)
            while not done:
                _,done=env.step_voltage(c.act_voltage(env)); num+=((env.xhat-env.S[:,:12])**2).sum(0); den+=(env.S[:,:12]**2).sum(0)
            errs.append(np.sqrt(num/den))
        out[f'{q}_{ri}']=[float(x.mean()) for x in errs]+[float(errs[0][:6].mean()),float(errs[0][6:].mean())]
        print('q',q,'rinfl',ri,'rel. RMS state error: V=0.1Vmax %.3f, V=Vmax %.3f (displ %.3f vel %.3f)'%tuple(out[f'{q}_{ri}']),flush=True)
json.dump(out,open('../results/kf_tuning.json','w'),indent=1)
