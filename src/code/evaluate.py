"""evaluate.py -- uncertainty-aware evaluation of all controllers on recorded ground motions (fine integration).
   python evaluate.py --part 0 --nparts 2 [--only P-,CO] [--sets nominal,mc_in]
   Passive and DRL controllers: 100 Hz control period, 1 ms integration step.
   Clipped-optimal LQG:         500 Hz control period, 0.5 ms integration step.
"""
import argparse, glob, json, os, sys, time
import numpy as np
from simcore import Building, load_records, resample, sample_params, G
from env import MREnv, Passive, AmpSched, ClippedOptimal, ClippedOptimalDC, rollout
from ppo import load_actor, make_policy_fn

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); RUNS = os.path.join(HERE, "..", "runs")
TMAX = 40.0
CHUNK = 88
bld = Building("six")
RECS = load_records(1.0)
RNAMES = list(RECS.keys())
NR = len(RNAMES)
RATES = {"slow": dict(ctrl_dt=0.01, nsub=10), "fast": dict(ctrl_dt=0.002, nsub=4)}
_unit = {}


def unit_records(dt):
    """Unit-PGA records resampled (linear interpolation) at the integration step dt."""
    key = round(dt, 9)
    if key not in _unit:
        out = []
        for nm in RNAMES:
            dt_r, acc = RECS[nm]
            a = resample(dt_r, acc, dt, T=TMAX)
            out.append(a / np.abs(a).max())
        _unit[key] = np.array(out)
    return _unit[key]


def make_set(kind):
    """Scenario definition: record index, PGA (g), plant/sensor/actuator parameters, meta data."""
    if kind == "nominal":
        levels = np.array([0.035, 0.07, 0.14])
        rec = np.tile(np.arange(NR), len(levels)); pga = np.repeat(levels, NR)
        p = sample_params(bld, len(rec), np.random.default_rng(0), nominal=True)
    elif kind in ("mc_in", "mc_ood"):
        rng = np.random.default_rng(31 if kind == "mc_in" else 32)
        ns = 30
        rec = np.tile(np.arange(NR), ns)
        pga = np.exp(rng.uniform(np.log(bld.pga_range[0]), np.log(bld.pga_range[1]), len(rec)))
        p = sample_params(bld, len(rec), rng, level=1.0 if kind == "mc_in" else 1.6)
    elif kind == "frag":
        rng = np.random.default_rng(33)
        levels = np.array([0.015, 0.03, 0.05, 0.075, 0.10, 0.14, 0.20, 0.25])
        ns = 4
        rec = np.tile(np.arange(NR), ns * len(levels)); pga = np.repeat(levels, NR * ns)
        p = sample_params(bld, len(rec), rng, level=1.0)
    elif kind.startswith("sw_"):
        vals = dict(sw_stiff=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], sw_noise=[0, 0.5, 1, 2, 3, 4],
                    sw_delay=[0, 1, 2, 3, 4, 5], sw_cap=[0.4, 0.6, 0.8, 1.0, 1.2])[kind]
        rec = np.tile(np.arange(NR), len(vals)); pga = np.full(len(rec), 0.07)
        v = np.repeat(np.array(vals, dtype=float), NR)
        p = sample_params(bld, len(rec), np.random.default_rng(0), nominal=True)
        if kind == "sw_stiff":
            p["sk"] = np.repeat(v[:, None], bld.n, axis=1)
        elif kind == "sw_noise":
            p["noise"] = v
        elif kind == "sw_delay":
            p["delay"] = v.astype(np.int64)
        elif kind == "sw_cap":
            p["sf"] = np.repeat(v[:, None], bld.nd, axis=1)
        return rec, pga, p, dict(rec=rec, pga=pga, val=v)
    else:
        raise ValueError(kind)
    meta = dict(rec=rec, pga=pga, sk=p["sk"], sm=p["sm"], sc=p["sc"], sf=p["sf"], noise=p["noise"], delay=p["delay"])
    return rec, pga, p, meta


def controllers():
    """(name, kind, payload): kind in {passive, co, drl}."""
    b = json.load(open(os.path.join(RES, "baselines.json")))
    cl = [("P-OFF", "passive", 0.0), ("P-ON", "passive", 1.0), ("P-BEST", "passive", float(b["best_passive"]))]
    for f in (0.1, 0.2, 0.3, 0.45, 0.65):                       # constant-voltage sweep (reference only)
        if abs(f - float(b["best_passive"])) > 1e-9:
            cl.append((f"P-{f:g}", "passive", f))
    if "best_passive2" in b:
        cl.append(("P-BEST2", "passive", b["best_passive2"]))
    if "best_as" in b:
        cl.append(("AS", "as", b["best_as"]))
    for nm, key in (("CO-NOM", "best_co_nom"), ("CO-ROB", "best_co_rob")):
        wa, r = [float(x) for x in b[key].split("_")]
        cl.append((nm, "co", (wa, r, float(b["kf_q500"]))))
    if "best_co_dc" in b:                                       # delay-compensated clipped-optimal (design delay in steps)
        r, d = b["best_co_dc"].split("_")
        cl.append(("CO-DC", "codc", (float(r), int(d), float(b["kf_q500"]))))
        cl.append(("CO-DCo", "codc", (float(r), None, float(b["kf_q500"]))))     # oracle delay: reference only
    for f in sorted(glob.glob(os.path.join(RUNS, "*_s[0-9].pt"))):
        if os.path.exists(f[:-3] + ".done"):                    # finished runs only (a run in progress also writes .pt)
            cl.append((os.path.basename(f)[:-3], "drl", f))
    return cl


MAIN_SETS = ["nominal", "mc_in", "mc_ood", "frag", "sw_stiff", "sw_noise", "sw_delay", "sw_cap"]
ABL_SETS = ["nominal", "mc_in", "mc_ood"]
KEYS = ("J1", "J2", "J3", "J4", "J5", "JE", "JT", "vmean", "pk_d", "pk_a", "unc_pk_d", "unc_pk_a")


def run(name, ckind, payload, kind, cache, force=False):
    out = os.path.join(RES, f"{kind}__{name}.npz")
    if os.path.exists(out) and not force:
        return
    rec, pga, p, meta = make_set(kind)
    N = len(rec)
    rate = "fast" if ckind in ("co", "codc") else "slow"
    t = time.time()
    acc = {k: [] for k in KEYS + ("pk_d_all", "pk_a_all", "unc_pk_d_all", "unc_pk_a_all")}
    actor = args = None
    if ckind == "drl":
        actor, args = load_actor(payload)
    for c0 in range(0, N, CHUNK):
        sl = slice(c0, min(c0 + CHUNK, N)); n = sl.stop - sl.start
        pc = {k: v[sl] for k, v in p.items()}
        kw = dict(RATES[rate])
        if ckind in ("co", "codc"):
            kw["kf_q"] = payload[2]
        if ckind == "drl":
            kw["act_mode"] = args["act"]
        env = MREnv(bld, n, **kw)
        ag = unit_records(env.dt)[rec[sl]] * (pga[sl] * G)[:, None]
        ck = (kind, rate, c0)
        if ckind == "passive":
            J = rollout(env, Passive(payload), ag, pc, noise_seed=7 + c0, unc=cache.get(ck))
        elif ckind == "as":
            J = rollout(env, AmpSched(payload), ag, pc, noise_seed=7 + c0, unc=cache.get(ck))
        elif ckind == "co":
            J = rollout(env, ClippedOptimal(env.nom.lqr_energy(bld, 1.0, payload[0], payload[1])), ag, pc, noise_seed=7 + c0,
                        unc=cache.get(ck))
        elif ckind == "codc":
            J = rollout(env, ClippedOptimalDC(env.nom.lqr_energy(bld, 1.0, 0.0, payload[0]), payload[1]), ag, pc,
                        noise_seed=7 + c0, unc=cache.get(ck))
        else:
            J = rollout(env, None, ag, pc, noise_seed=7 + c0, policy_fn=make_policy_fn(actor), obs_mode=args["obs"],
                        K=args["K"], unc=cache.get(ck))
        cache[ck] = J["unc_cache"]
        for k in KEYS:
            acc[k].append(J[k])
        acc["pk_d_all"].append(env.met["pk_d"].copy()); acc["pk_a_all"].append(env.met["pk_a"].copy())
        acc["unc_pk_d_all"].append(env.unc["pk_d"].copy()); acc["unc_pk_a_all"].append(env.unc["pk_a"].copy())
        del env, ag
    R = {k: np.concatenate(v) for k, v in acc.items()}
    np.savez(out, **R)
    mf = os.path.join(RES, f"meta__{kind}.npz")
    if not os.path.exists(mf):
        np.savez(mf, **meta)
    print(f"{kind:9s} {name:14s} N={N:4d} JT={R['JT'].mean():.3f} J1={R['J1'].mean():.3f} J2={R['J2'].mean():.3f} "
          f"J3={R['J3'].mean():.3f} J4={R['J4'].mean():.3f} ({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--sets", default=""); ap.add_argument("--only", default="")
    ap.add_argument("--main", default="V_pi_dr")
    a = ap.parse_args()
    cl = controllers()
    if a.only:
        cl = [c for c in cl if any(c[0].startswith(o) for o in a.only.split(","))]
    jobs = []
    for nm, ck, pl in cl:
        is_main = ((ck != "drl") or nm.startswith(a.main)) and not nm.startswith("P-0.") and nm != "CO-DCo"
        sets = a.sets.split(",") if a.sets else (MAIN_SETS if is_main else (["sw_delay", "mc_in"] if nm == "CO-DCo" else ABL_SETS))
        jobs += [(nm, ck, pl, s) for s in sets]
    jobs.sort(key=lambda j: MAIN_SETS.index(j[3]))
    cache = {}
    for nm, ck, pl, s in jobs[a.part::a.nparts]:
        run(nm, ck, pl, s, cache)
    print("EVAL_DONE", a.part)
