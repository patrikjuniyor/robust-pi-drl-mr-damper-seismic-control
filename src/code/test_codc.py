"""Development test of the delay-compensated clipped-optimal controller on the nominal plant with known delays."""
import numpy as np, json, time, sys
import evaluate as ev
from env import MREnv, ClippedOptimal, ClippedOptimalDC, rollout
from simcore import G
bld = ev.bld
B = json.load(open('../results/baselines.json'))
rec, pga, p, meta = ev.make_set("sw_delay")
sel = np.isin(meta["val"], [0, 1, 2])
rec, pga = rec[sel], pga[sel]; p = {k: v[sel] for k, v in p.items()}; val = meta["val"][sel]
env = MREnv(bld, len(rec), ctrl_dt=0.002, nsub=4, kf_q=float(B["kf_q500"]))
ag = ev.unit_records(env.dt)[rec] * (pga * G)[:, None]
unc = None
for r in [float(x) for x in sys.argv[1:]] or [3e-5]:
    K = env.nom.lqr_energy(bld, 1.0, 0.0, r)
    for name, c in (("CO", ClippedOptimal(K)), ("DC-or", ClippedOptimalDC(K, None)), ("DC-or fhold", ClippedOptimalDC(K, None, 0, 1)), ("DC-5", ClippedOptimalDC(K, 5)), ("DC-5 fhold", ClippedOptimalDC(K, 5, 0, 1)), ("DC-10", ClippedOptimalDC(K, 10)), ("DC-10 fhold", ClippedOptimalDC(K, 10, 0, 1))):
        t = time.time(); J = rollout(env, c, ag, p, noise_seed=7, unc=unc); unc = J["unc_cache"]
        print(f"r={r:g} {name:10s}", " | ".join(f"d={int(10*d)}ms JT {J['JT'][val==d].mean():.3f} J1 {J['J1'][val==d].mean():.3f} J2 {J['J2'][val==d].mean():.3f}" for d in (0, 1, 2)), f"({time.time()-t:.0f}s)", flush=True)
