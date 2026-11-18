"""Diagnostic for the delay-compensated clipped-optimal controller: how good is the d-step-ahead prediction?
Nominal plant, PGA 0.07 g, 11 records, true delay 10 or 20 ms, predictor given the true delay (best case).
Writes ../results/codc_diag.json."""
import numpy as np, json
import evaluate as ev
from env import MREnv, ClippedOptimalDC
from simcore import G
bld = ev.bld; B = json.load(open('../results/baselines.json'))
r = float(B["best_co_dc"].split("_")[0])
out = {"r": r}
for dunit in (1, 2):
    rec, pga, p, meta = ev.make_set("sw_delay"); sel = meta["val"] == dunit
    rec, pga = rec[sel], pga[sel]; p = {k: v[sel] for k, v in p.items()}
    env = MREnv(bld, len(rec), ctrl_dt=0.002, nsub=4, kf_q=float(B["kf_q500"]))
    ag = ev.unit_records(env.dt)[rec] * (pga * G)[:, None]
    c = ClippedOptimalDC(env.nom.lqr_energy(bld, 1.0, 0.0, r), None)
    env.reset(ag, p, noise_seed=7); c.reset(env)
    D = int(c.D[0]); FP, FA, XP, XA, XH = [], [], [], [], []
    done = False
    while not done:
        v = c.act_voltage(env)
        FP.append(c.scr[1].copy()); XP.append(c.Sp[:, :12].copy())
        FA.append(env.m_f.copy()); XA.append(env.S[:, :12].copy()); XH.append(env.xhat.copy())
        _, done = env.step_voltage(v)
    FP, FA, XP, XA, XH = [np.array(a) for a in (FP, FA, XP, XA, XH)]
    n = len(FP) - D
    # strong-motion part only: samples where the true force exceeds 10 % of its peak
    fa = FA[D:D + n]; fp = FP[:n]; fh = FA[:n]; w = np.abs(fa) > 0.1 * np.abs(FA).max()
    dv = lambda X: np.stack([X[..., 6], X[..., 7] - X[..., 6]], -1)
    rel = lambda a, b: float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))
    out[f"{10 * dunit}ms"] = dict(
        steps=D,
        force_sign_pred=float((np.sign(fp) == np.sign(fa))[w].mean()), force_sign_hold=float((np.sign(fh) == np.sign(fa))[w].mean()),
        force_relerr_pred=rel(fp[w], fa[w]), force_relerr_hold=rel(fh[w], fa[w]),
        kf_vel_relerr=rel(XH[..., 6:], XA[..., 6:]), kf_disp_relerr=rel(XH[..., :6], XA[..., :6]),
        pred_vel_relerr=rel(XP[:n][..., 6:], XA[D:D + n][..., 6:]), pred_disp_relerr=rel(XP[:n][..., :6], XA[D:D + n][..., :6]),
        kf_dvel_sign=float((np.sign(dv(XH)) == np.sign(dv(XA))).mean()))
    print(out[f"{10 * dunit}ms"])
json.dump(out, open('../results/codc_diag.json', 'w'), indent=1)
