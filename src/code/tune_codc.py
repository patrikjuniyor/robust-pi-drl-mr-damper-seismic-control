"""Tune the delay-compensated clipped-optimal controller (CO-DC) with the recipe used for CO-ROB:
same 64 randomised plants and motions (rng 2025 / 2024), 500 Hz loop, selection on the mean J_T of the randomised set.
Grid: control weight r x design delay (10 or 20 ms = 5 or 10 steps).  The oracle variant (true delay of every plant known)
is evaluated as a reference only; it is not realisable."""
import numpy as np, json, sys
from simcore import *
from env import *
bld = Building('six'); N = 64
res = json.load(open('../results/baselines.json'))
envf = MREnv(bld, N, ctrl_dt=0.002, nsub=4, kf_q=res['kf_q500'])
ag, pga = synth_motions(N, bld.T_train, envf.dt, np.random.default_rng(2024), bld.time_scale, bld.pga_range)
par = sample_params(bld, N, np.random.default_rng(2025))
KS = ('JT', 'J1', 'J2', 'J3', 'J4'); unc = None
out = res.get('co_dc', {})
for r in (3e-4, 1e-4, 1e-3, 3e-5, 3e-3):
    Kg = envf.nom.lqr_energy(bld, 1.0, 0.0, r)
    for d in ((5, 10, None) if r == 3e-4 else (5, 10)):       # oracle delay: one reference point only
        key = f'{r}_{d}'
        if key in out: continue
        J = rollout(envf, ClippedOptimalDC(Kg, d), ag, par, noise_seed=1, unc=unc); unc = J['unc_cache']
        out[key] = [float(J[k].mean()) for k in KS]
        print('CO-DC r', r, 'design delay', d, np.round(out[key], 3), flush=True)
        res['co_dc'] = out; json.dump(res, open('../results/baselines.json', 'w'), indent=1)
real = {k: v for k, v in out.items() if not k.endswith('None')}
res['best_co_dc'] = min(real, key=lambda k: real[k][0])
print('best realisable', res['best_co_dc'], np.round(real[res['best_co_dc']], 3))
json.dump(res, open('../results/baselines.json', 'w'), indent=1)
print('TUNE_DONE')
