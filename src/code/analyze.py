"""analyze.py -- statistics, tables (JSON) and figures for the paper.   python analyze.py --main V_pi_dr"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import optimize, stats as sst

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); RUNS = os.path.join(HERE, "..", "runs"); FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200, "savefig.bbox": "tight",
                     "savefig.dpi": 450, "savefig.transparent": False, "savefig.facecolor": "white",
                     "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
ap = argparse.ArgumentParser(); ap.add_argument("--main", default="V_pi_dr"); ap.add_argument("--skip_traj", type=int, default=0)
A = ap.parse_args()
MAIN = A.main
ALL_SETS = ("nominal", "mc_in", "mc_ood", "frag", "sw_stiff", "sw_noise", "sw_delay", "sw_cap")


def seeds_of(key, kinds):
    """Seeds whose training has finished (.done marker) and that have been evaluated on every set in `kinds`."""
    return [i for i in range(10) if os.path.exists(os.path.join(RUNS, f"{key}_s{i}.done"))
            and all(os.path.exists(os.path.join(RES, f"{k}__{key}_s{i}.npz")) for k in kinds)]


SEEDS = {"PI-DRL (proposed)": (MAIN, seeds_of(MAIN, ALL_SETS)),
         "No domain randomisation": ("V_pi_nom", seeds_of("V_pi_nom", ALL_SETS[:3])),
         "No physics-based observer (raw history)": ("V_raw_dr", seeds_of("V_raw_dr", ALL_SETS[:3])),
         "Symmetric critic (no privileged information)": ("V_pi_sym", seeds_of("V_pi_sym", ALL_SETS[:3]))}
ABL = {g: [f"{k}_s{i}" for i in sd] for g, (k, sd) in SEEDS.items()}
GROUPS = {"P-OFF": ["P-OFF"], "P-ON": ["P-ON"], "P-BEST": ["P-BEST"], "P-BEST2": ["P-BEST2"], "AS": ["AS"],
          "CO-NOM": ["CO-NOM"], "CO-ROB": ["CO-ROB"], "CO-DC": ["CO-DC"], "PI-DRL": ABL["PI-DRL (proposed)"]}
BASES = ("P-OFF", "P-ON", "P-BEST", "P-BEST2", "AS", "CO-NOM", "CO-ROB", "CO-DC")
COL = {"UNC": "k", "P-OFF": "#7f7f7f", "P-ON": "#8c564b", "P-BEST": "#2ca02c", "P-BEST2": "#bcbd22", "AS": "#ff7f0e",
       "CO-NOM": "#1f77b4", "CO-ROB": "#17becf", "CO-DC": "#9467bd", "PI-DRL": "#d62728"}
print("seeds:", {g: sd for g, (k, sd) in SEEDS.items()})
JS = ("J1", "J2", "J3", "J4", "JT")
rng = np.random.default_rng(0)
T = {"seeds": {g: sd for g, (k, sd) in SEEDS.items()}}


def load(kind, name):
    f = os.path.join(RES, f"{kind}__{name}.npz")
    return dict(np.load(f)) if os.path.exists(f) else None


def meta(kind):
    return dict(np.load(os.path.join(RES, f"meta__{kind}.npz")))


def st(x):
    x = np.asarray(x); q = np.quantile(x, 0.9)
    return dict(mean=float(x.mean()), median=float(np.median(x)), p95=float(np.quantile(x, 0.95)), cvar90=float(x[x >= q].mean()))


def boot_ci(x, B=2000):
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def group_stats(kind, names):
    """Statistics per seed -> mean and std across seeds; plus seed-averaged per-scenario arrays."""
    D = [load(kind, n) for n in names]
    D = [d for d in D if d is not None]
    if not D:
        return None, None
    out = {}
    for j in JS + ("J5", "vmean"):
        per = [st(d[j]) for d in D]
        out[j] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        out[j]["seed_std"] = float(np.std([p["mean"] for p in per], ddof=1)) if len(per) > 1 else 0.0     # sample s.d.
        out[j]["nseeds"] = len(per)
        out[j]["ci"] = boot_ci(np.mean([d[j] for d in D], axis=0))
    avg = {j: np.mean([d[j] for d in D], axis=0) for j in JS + ("vmean", "pk_d", "pk_a", "unc_pk_d", "unc_pk_a")}
    avg["all"] = D
    return out, avg


# ------------------------------------------------------------------ tables: nominal, MC, ablation, paired
T["nominal"] = {}
mn = meta("nominal")
for g, names in GROUPS.items():
    _, avg = group_stats("nominal", names)
    if avg is None: continue
    T["nominal"][g] = {f"{lvl:g}": {j: float(avg[j][mn["pga"] == lvl].mean()) for j in JS} for lvl in np.unique(mn["pga"])}
for kind in ("mc_in", "mc_ood"):
    T[kind] = {}; T[kind + "_abl"] = {}; T[kind + "_paired"] = {}
    AV = {}
    for g, names in GROUPS.items():
        s, avg = group_stats(kind, names)
        if s is not None:
            T[kind][g] = s; AV[g] = avg
    for g, names in ABL.items():
        s, _ = group_stats(kind, names)
        if s is not None:
            T[kind + "_abl"][g] = {j: dict(mean=s[j]["mean"], seed_std=s[j]["seed_std"], cvar90=s[j]["cvar90"], nseeds=s[j]["nseeds"]) for j in JS}
    if "PI-DRL" in AV:
        for b in BASES:
            if b not in AV: continue
            T[kind + "_paired"][b] = {}
            for j in JS:
                d = AV["PI-DRL"][j] - AV[b][j]
                T[kind + "_paired"][b][j] = dict(diff=float(d.mean()), ci=boot_ci(d), rel=float(d.mean() / AV[b][j].mean()),
                                                 win=float((d < 0).mean()),
                                                 p=float(sst.wilcoxon(d).pvalue))
    T[kind + "_N"] = int(len(meta(kind)["pga"]))
    T[kind + "_seeds"] = {}
    for i in SEEDS["PI-DRL (proposed)"][1]:
        d = load(kind, f"{MAIN}_s{i}")
        if d is None: continue
        T[kind + "_seeds"][f"seed {i}"] = {j: float(d[j].mean()) for j in JS}
        for b in BASES:
            db = load(kind, b)
            if db is not None:
                T[kind + "_seeds"][f"seed {i}"][f"dJT_vs_{b}"] = float((d["JT"] - db["JT"]).mean())
                T[kind + "_seeds"][f"seed {i}"][f"ci_vs_{b}"] = boot_ci(d["JT"] - db["JT"])
    # seed-level inference: the training seed is the unit; one-sample t-test on the per-seed mean differences
    T[kind + "_seedtest"] = {}
    for b in BASES:
        dd = np.array([v[f"dJT_vs_{b}"] for v in T[kind + "_seeds"].values() if f"dJT_vs_{b}" in v])
        if len(dd) < 2: continue
        tt_ = sst.ttest_1samp(dd, 0.0); half = sst.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
        T[kind + "_seedtest"][b] = dict(n=int(len(dd)), mean=float(dd.mean()), sd=float(dd.std(ddof=1)), p=float(tt_.pvalue),
                                        ci=[float(dd.mean() - half), float(dd.mean() + half)], better=int((dd < 0).sum()),
                                        sig_better=int(sum(v[f"ci_vs_{b}"][1] < 0 for v in T[kind + "_seeds"].values())),
                                        sig_worse=int(sum(v[f"ci_vs_{b}"][0] > 0 for v in T[kind + "_seeds"].values())),
                                        diffs=dd.tolist())
    T[kind + "_psweep"] = {}
    for f_, nm_ in ((0.0, "P-OFF"), (0.1, "P-0.1"), (0.2, "P-0.2"), (0.3, "P-0.3"), (0.45, "P-0.45"), (0.65, "P-0.65"), (1.0, "P-ON")):
        d = load(kind, nm_)
        if d is None: d = load(kind, "P-BEST")
        if d is not None:
            T[kind + "_psweep"][str(f_)] = {j: float(d[j].mean()) for j in JS}

# ------------------------------------------------------------------ Fig: learning curves
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
cols = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
T["train"] = {}
for (g, names), c in zip(ABL.items(), cols):
    L = [json.load(open(os.path.join(RUNS, n + ".json"))) for n in names if os.path.exists(os.path.join(RUNS, n + ".json"))]
    if not L: continue
    m = min(len(l["val_cost"]) for l in L)
    x = np.array(L[0]["steps"])[np.array(L[0]["val_iter"][:m])] / 1e6
    v = np.array([l["val_cost"][:m] for l in L])
    ax[0].plot(x, v.mean(0), color=c, label=f"{g} (n = {len(L)})", lw=1.3); ax[0].fill_between(x, v.min(0), v.max(0), color=c, alpha=0.15)
    j1 = np.array([l["val_J1"][:m] for l in L]); j2 = np.array([l["val_J2"][:m] for l in L])
    ax[1].plot(x, j1.mean(0), color=c, lw=1.3); ax[1].plot(x, j2.mean(0), color=c, lw=1.3, ls="--")
    T["train"][g] = dict(final_val=float(v[:, -1].mean()), best_val=float(v.min(1).mean()), hours=float(np.mean([l["time"][-1] for l in L]) / 3600),
                         decisions=float(L[0]["steps"][-1]), nseeds=len(L),
                         best_val_sd=float(v.min(1).std(ddof=1)) if len(L) > 1 else 0.0,
                         val_at={str(it): float(np.mean([l["val_cost"][l["val_iter"].index(it - 1)] for l in L]))
                                 for it in (100, 200, 300, 400) if all((it - 1) in l["val_iter"] for l in L)})
RUNS100 = os.path.join(HERE, "..", "runs_b100"); RES100 = os.path.join(RES, "b100")
L0 = [json.load(open(os.path.join(RUNS100, n + ".json"))) for n in ABL["PI-DRL (proposed)"]
      if os.path.exists(os.path.join(RUNS100, n + ".json"))]
if L0:
    m = min(len(l["val_cost"]) for l in L0)
    x = np.array(L0[0]["steps"])[np.array(L0[0]["val_iter"][:m])] / 1e6
    v = np.array([l["val_cost"][:m] for l in L0])
    ax[0].plot(x, v.mean(0), color="k", lw=1.0, ls=":", label="PI-DRL, earlier 100-iteration budget")
ax[0].set_xlabel("agent decisions (millions)"); ax[0].set_ylabel(r"validation objective $J_T$")
ax[1].set_xlabel("agent decisions (millions)"); ax[1].set_ylabel(r"validation $J_1$ (solid), $J_2$ (dashed)")
ax[0].legend(fontsize=6.2, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_learning.png")); plt.close(fig)

# ------------------------------------------------------------------ Fig: CDFs (MC in / OOD)
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.4), sharey=True)
for r, kind in enumerate(("mc_in", "mc_ood")):
    for c, j in enumerate(("J1", "J2", "JT")):
        ax = axs[r, c]
        for g, names in GROUPS.items():
            D = [load(kind, n) for n in names]; D = [d for d in D if d is not None]
            if not D: continue
            x = np.sort(np.concatenate([d[j] for d in D]))
            ax.plot(x, np.linspace(0, 1, len(x)), color=COL[g], label=g, lw=1.4 if g == "PI-DRL" else 1.0)
        ax.set_xlabel({"J1": r"$J_1$ (peak drift ratio)", "J2": r"$J_2$ (peak acceleration ratio)", "JT": r"$J_T$ (composite)"}[j])
        if j == "J2": ax.axvline(1.0, color="k", lw=0.5, ls=":")
        ax.set_xlim(0, {"J1": 1.2, "J2": 2.0, "JT": 7.0}[j])
    axs[r, 0].set_ylabel(("in-distribution" if kind == "mc_in" else "out-of-distribution") + "\nempirical CDF")
axs[0, 0].legend(fontsize=6.5, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_cdf.png")); plt.close(fig)

# ------------------------------------------------------------------ Fig: intensity adaptivity
m = meta("mc_in"); pga = m["pga"]
edges = np.exp(np.linspace(np.log(0.015), np.log(0.20), 7)); ctr = np.sqrt(edges[1:] * edges[:-1])
fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.4))
T["by_pga"] = {"pga": ctr.tolist()}
for g, names in GROUPS.items():
    _, avg = group_stats("mc_in", names)
    if avg is None: continue
    T["by_pga"][g] = {}
    for ax, j in zip(axs, ("J1", "J2", "JT", "vmean")):
        y = [avg[j][(pga >= lo) & (pga < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
        T["by_pga"][g][j] = [float(v) for v in y]
        ax.semilogx(ctr, y, "-o", ms=2.5, color=COL[g], label=g, lw=1.4 if g == "PI-DRL" else 1.0)
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
for ax, lab in zip(axs, (r"mean $J_1$", r"mean $J_2$", r"mean $J_T$", r"mean voltage / $V_{max}$")):
    ax.set_xlabel("PGA (g)"); ax.set_title(lab, fontsize=8)
    ax.xaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2])); ax.xaxis.set_major_formatter(FixedFormatter(["0.02", "0.05", "0.1", "0.2"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=9, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_intensity.png")); plt.close(fig)

# ------------------------------------------------------------------ Fig: robustness sweeps
SW = [("sw_stiff", "storey-stiffness scale"), ("sw_cap", "damper yield-force scale"), ("sw_noise", "sensor-noise level"),
      ("sw_delay", "command delay (ms)")]
fig, axs = plt.subplots(2, 4, figsize=(7.6, 4.0), sharex="col")
T["sweeps"] = {}
for c, (kind, lab) in enumerate(SW):
    if not os.path.exists(os.path.join(RES, f"meta__{kind}.npz")): continue
    mm = meta(kind); vals = np.unique(mm["val"]); T["sweeps"][kind] = {"val": vals.tolist()}
    xs = vals * (10.0 if kind == "sw_delay" else 1.0)
    for g, names in GROUPS.items():
        D = [load(kind, n) for n in names]; D = [d for d in D if d is not None]
        if not D: continue
        T["sweeps"][kind][g] = {}
        for r, j in enumerate(("J1", "J2")):
            Y = np.array([[d[j][mm["val"] == v].mean() for v in vals] for d in D])
            axs[r, c].plot(xs, Y.mean(0), "-o", ms=2.5, color=COL[g], label=g, lw=1.4 if g == "PI-DRL" else 1.0)
            if len(D) > 1: axs[r, c].fill_between(xs, Y.min(0), Y.max(0), color=COL[g], alpha=0.15)
            T["sweeps"][kind][g][j] = Y.mean(0).tolist()
        T["sweeps"][kind][g]["JT"] = np.array([[d["JT"][mm["val"] == v].mean() for v in vals] for d in D]).mean(0).tolist()
    axs[1, c].set_xlabel(lab)
    if kind in ("sw_stiff", "sw_cap", "sw_noise"):
        lo, hi = {"sw_stiff": (0.75, 1.25), "sw_cap": (0.8, 1.2), "sw_noise": (0, 1)}[kind]
        for r in range(2): axs[r, c].axvspan(lo, hi, color="gold", alpha=0.12, lw=0)
    else:
        for r in range(2): axs[r, c].axvspan(0, 20, color="gold", alpha=0.12, lw=0)
axs[0, 0].set_ylabel(r"mean $J_1$"); axs[1, 0].set_ylabel(r"mean $J_2$")
h_, l_ = axs[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=9, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_sweeps.png")); plt.close(fig)

# ------------------------------------------------------------------ Fragility
def fit_frag(x, k, n):
    def nll(p):
        pr = np.clip(sst.norm.cdf(np.log(x / np.exp(p[0])) / np.exp(p[1])), 1e-9, 1 - 1e-9)
        return -(k * np.log(pr) + (n - k) * np.log(1 - pr)).sum()
    best = min((optimize.minimize(nll, [np.log(t0), np.log(b0)], method="Nelder-Mead") for t0 in (0.03, 0.08, 0.2, 0.4) for b0 in (0.3, 0.6)),
               key=lambda r: r.fun)
    return float(np.exp(best.x[0])), float(np.exp(best.x[1]))

if os.path.exists(os.path.join(RES, "meta__frag.npz")):
    mf = meta("frag"); lv = np.unique(mf["pga"])
    LIM = {"drift": 3.0e-3, "acc": 2.0}
    T["fragility"] = {"levels": lv.tolist(), "limits": LIM}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    xx = np.linspace(0.005, 0.32, 300)
    ref = load("frag", "P-BEST")
    series = {"UNC": [dict(pk_d=ref["unc_pk_d"], pk_a=ref["unc_pk_a"])]}
    for g, names in GROUPS.items():
        D = [load("frag", n) for n in names]; D = [d for d in D if d is not None]
        if D: series[g] = D
    for g, D in series.items():
        T["fragility"][g] = {}
        for ax, (q, key) in zip(axs, (("drift", "pk_d"), ("acc", "pk_a"))):
            k = np.array([sum((d[key][mf["pga"] == l] > LIM[q]).sum() for d in D) for l in lv], dtype=float)
            n = np.array([sum((mf["pga"] == l).sum() for d in D) for l in lv], dtype=float)
            th, be = fit_frag(lv, k, n)
            T["fragility"][g][q] = dict(theta=th, beta=be, frac=(k / n).tolist())
            ax.plot(lv, k / n, "o", ms=3, color=COL[g]); ax.plot(xx, sst.norm.cdf(np.log(xx / th) / be), color=COL[g], label=g, lw=1.4 if g == "PI-DRL" else 1.0)
    axs[0].set_title(f"P(peak inter-storey drift > {LIM['drift']*1e3:.0f} mm | PGA)", fontsize=8)
    axs[1].set_title(f"P(peak floor acceleration > {LIM['acc']:.1f} m/s$^2$ | PGA)", fontsize=8)
    for ax in axs: ax.set_xlabel("PGA (g)"); ax.set_xlim(0, 0.30); ax.set_ylim(-0.02, 1.02)
    axs[0].set_ylabel("probability of exceedance"); axs[1].legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_fragility.png")); plt.close(fig)

# ------------------------------------------------------------------ time histories (El Centro, PGA 0.07 g, nominal plant)
if not A.skip_traj:
    from simcore import Building, sample_params, G
    from env import MREnv, Passive, ClippedOptimal, ClippedOptimalDC, rollout
    from ppo import load_actor, make_policy_fn
    import evaluate as ev
    bld = ev.bld
    PG = 0.07
    p = sample_params(bld, 1, np.random.default_rng(0), nominal=True)
    cl = {c[0]: c for c in ev.controllers()}
    tr = {}; tt = {}
    from env import AmpSched
    for g in ("P-OFF", "P-BEST", "P-BEST2"):
        env = MREnv(bld, 1, **ev.RATES["slow"]); ag = ev.unit_records(env.dt)[0:1] * PG * G
        tr[g] = rollout(env, Passive(cl[g][2]), ag, p, record=0); tt[g] = env.ctrl_dt
    env = MREnv(bld, 1, **ev.RATES["slow"]); ag = ev.unit_records(env.dt)[0:1] * PG * G
    tr["AS"] = rollout(env, AmpSched(cl["AS"][2]), ag, p, record=0); tt["AS"] = env.ctrl_dt
    for g in ("CO-NOM", "CO-ROB"):
        wa, r, q = cl[g][2]
        env = MREnv(bld, 1, kf_q=q, **ev.RATES["fast"]); ag = ev.unit_records(env.dt)[0:1] * PG * G
        tr[g] = rollout(env, ClippedOptimal(env.nom.lqr_energy(bld, 1.0, wa, r)), ag, p, record=0); tt[g] = env.ctrl_dt
    if "CO-DC" in cl:
        r_, d_, q = cl["CO-DC"][2]
        env = MREnv(bld, 1, kf_q=q, **ev.RATES["fast"]); ag = ev.unit_records(env.dt)[0:1] * PG * G
        tr["CO-DC"] = rollout(env, ClippedOptimalDC(env.nom.lqr_energy(bld, 1.0, 0.0, r_), d_), ag, p, record=0); tt["CO-DC"] = env.ctrl_dt
    actor, args = load_actor(os.path.join(RUNS, MAIN + "_s0.pt"))
    env = MREnv(bld, 1, act_mode=args["act"], **ev.RATES["slow"]); ag = ev.unit_records(env.dt)[0:1] * PG * G
    tr["PI-DRL"] = rollout(env, None, ag, p, policy_fn=make_policy_fn(actor), obs_mode=args["obs"], K=args["K"], record=0); tt["PI-DRL"] = env.ctrl_dt
    b0 = Building("six"); b0.dcount = np.zeros(2)
    env = MREnv(b0, 1, **ev.RATES["slow"])
    tr["UNC"] = rollout(env, Passive(0.0), ag, p, record=0); tt["UNC"] = env.ctrl_dt
    tx = lambda g: np.arange(len(tr[g]["traj"]["x"])) * tt[g]
    fig, axs = plt.subplots(4, 1, figsize=(7.2, 6.8), sharex=True)
    for g in ("UNC", "CO-NOM", "AS", "PI-DRL"):
        axs[0].plot(tx(g), np.asarray(tr[g]["traj"]["x"])[:, -1] * 100, color=COL[g], lw=0.7, label=g)
        axs[1].plot(tx(g), np.asarray(tr[g]["traj"]["acc"])[:, -1], color=COL[g], lw=0.5, label=g)
    axs[2].plot(tx("CO-NOM"), np.asarray(tr["CO-NOM"]["traj"]["v"])[:, 0], color=COL["CO-NOM"], lw=0.3, alpha=0.35, label="CO-NOM (storey 1)")
    axs[2].plot(tx("PI-DRL"), np.asarray(tr["PI-DRL"]["traj"]["v"])[:, 0], color=COL["PI-DRL"], lw=1.0, label="PI-DRL (storey 1)")
    axs[2].plot(tx("PI-DRL"), np.asarray(tr["PI-DRL"]["traj"]["v"])[:, 1], color="#ff9896", lw=1.0, label="PI-DRL (storey 2)")
    axs[2].plot(tx("AS"), np.asarray(tr["AS"]["traj"]["v"])[:, 0], color=COL["AS"], lw=0.8, ls="--", label="AS (storey 1)")
    axs[2].plot(tx("AS"), np.asarray(tr["AS"]["traj"]["v"])[:, 1], color="#ffbb78", lw=0.8, ls="--", label="AS (storey 2)")
    axs[3].plot(tx("CO-NOM"), np.asarray(tr["CO-NOM"]["traj"]["f"])[:, 0], color=COL["CO-NOM"], lw=0.4, label="CO-NOM")
    axs[3].plot(tx("PI-DRL"), np.asarray(tr["PI-DRL"]["traj"]["f"])[:, 0], color=COL["PI-DRL"], lw=0.6, label="PI-DRL")
    axs[0].set_ylabel("roof displ. (cm)"); axs[1].set_ylabel("roof abs. acc. (m/s$^2$)"); axs[2].set_ylabel("command voltage (V)")
    axs[3].set_ylabel("storey-1 damper force (N)"); axs[3].set_xlabel("time (s)")
    for a_ in axs: a_.legend(fontsize=6, frameon=True, framealpha=0.9, edgecolor="none", ncol=5, loc="upper right")
    axs[0].set_xlim(0, ev.TMAX)   # show the whole 40 s record used in the metrics
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_timehist.png")); plt.close(fig)
    T["elcentro"] = {g: {j: float(tr[g][j][0]) for j in ("J1", "J2", "J3", "J4", "J5", "JT", "vmean")} for g in ("P-OFF", "P-BEST", "P-BEST2", "AS", "CO-NOM", "CO-ROB", "CO-DC", "PI-DRL") if g in tr}
    T["elcentro"]["unc"] = dict(pk_d_mm=float(tr["P-BEST"]["unc_pk_d"][0] * 1e3), pk_a=float(tr["P-BEST"]["unc_pk_a"][0]),
                                roof_cm=float(np.abs(np.asarray(tr["UNC"]["traj"]["x"])[:, -1]).max() * 100), pga=PG)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True, sharex=True)
    for ax, g in zip(axs, ("AS", "CO-NOM", "PI-DRL")):
        ax.plot(np.asarray(tr[g]["traj"]["x"])[:, 0] * 1e3, np.asarray(tr[g]["traj"]["f"])[:, 0], color=COL[g], lw=0.4); ax.set_title(g, fontsize=8)
        ax.set_xlabel("storey-1 drift (mm)")
    axs[0].set_ylabel("damper force (N)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_hysteresis.png")); plt.close(fig)

# ------------------------------------------------------------------ training-budget comparison (100 vs current iterations)
T["budget"] = {}
for g, names in ABL.items():
    T["budget"][g] = {}
    for kind in ("mc_in", "mc_ood"):
        row = {}
        for tag, folder in (("b100", RES100), ("now", RES)):
            vals = []
            for n in names:
                if not os.path.exists(os.path.join(RES100, f"{kind}__{n}.npz")): continue   # seeds trained at both budgets only
                f = os.path.join(folder, f"{kind}__{n}.npz")
                if os.path.exists(f):
                    vals.append(dict(np.load(f)))
            if not vals: continue
            per = [float(d["JT"].mean()) for d in vals]
            row[tag] = dict(seeds=per, mean=float(np.mean(per)), sd=float(np.std(per, ddof=1)) if len(per) > 1 else 0.0,
                            J1=float(np.mean([d["J1"].mean() for d in vals])), J2=float(np.mean([d["J2"].mean() for d in vals])),
                            cvar90=float(np.mean([st(d["JT"])["cvar90"] for d in vals])))
            row[tag + "_avg"] = np.mean([d["JT"] for d in vals], axis=0)
        if "b100" in row and "now" in row:
            d = row["now_avg"] - row["b100_avg"]
            row["paired"] = dict(diff=float(d.mean()), ci=boot_ci(d), win=float((d < 0).mean()))
        row.pop("b100_avg", None); row.pop("now_avg", None)
        T["budget"][g][kind] = row
for g, names in ABL.items():
    for tag, folder in (("b100", RUNS100), ("now", RUNS)):
        L = [json.load(open(os.path.join(folder, n + ".json"))) for n in names
             if os.path.exists(os.path.join(folder, n + ".json")) and os.path.exists(os.path.join(RUNS100, n + ".json"))]
        if L:
            T["budget"][g]["train_" + tag] = dict(best_val=float(np.mean([min(l["val_cost"]) for l in L])),
                                                  decisions=float(L[0]["steps"][-1]), iters=int(L[0]["args"]["iters"]),
                                                  minutes=float(np.mean([l["time"][-1] for l in L]) / 60))
# pairwise ablation comparisons; the training seed is the unit of replication
def seed_boot(a1, a0, B=20000):
    r_ = np.random.default_rng(1)
    d = a1[r_.integers(0, len(a1), (B, len(a1)))].mean(1) - a0[r_.integers(0, len(a0), (B, len(a0)))].mean(1)
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


T["abl_pair"] = {}
for kind in ("mc_in", "mc_ood"):
    T["abl_pair"][kind] = {}
    P0 = [load(kind, n) for n in ABL["PI-DRL (proposed)"]]
    if not P0 or any(p is None for p in P0): continue
    a0 = np.array([p["JT"].mean() for p in P0])
    T["abl_pair"][kind]["_proposed"] = dict(seeds=a0.tolist(), n=int(len(a0)), mean=float(a0.mean()), sd=float(a0.std(ddof=1)),
                                             min=float(a0.min()), max=float(a0.max()))
    for g, names in list(ABL.items())[1:]:
        D = [load(kind, n) for n in names]
        if not D or any(d is None for d in D): continue
        a1 = np.array([d["JT"].mean() for d in D])
        dd = np.mean([d["JT"] for d in D], axis=0) - np.mean([p["JT"] for p in P0], axis=0)
        w = sst.ttest_ind(a1, a0, equal_var=False)
        v1, v0 = a1.var(ddof=1) / len(a1), a0.var(ddof=1) / len(a0)
        dof = (v1 + v0) ** 2 / (v1 ** 2 / (len(a1) - 1) + v0 ** 2 / (len(a0) - 1))
        half = sst.t.ppf(0.975, dof) * np.sqrt(v1 + v0)
        sp = np.sqrt(((len(a1) - 1) * a1.var(ddof=1) + (len(a0) - 1) * a0.var(ddof=1)) / (len(a1) + len(a0) - 2))
        k3 = min(3, len(a1), len(a0))
        T["abl_pair"][kind][g] = dict(
            diff=float(a1.mean() - a0.mean()), welch_p=float(w.pvalue), welch_ci=[float(a1.mean() - a0.mean() - half), float(a1.mean() - a0.mean() + half)],
            mwu_p=float(sst.mannwhitneyu(a1, a0, alternative="two-sided").pvalue), seed_ci=seed_boot(a1, a0),
            cohen_d=float((a1.mean() - a0.mean()) / sp), ci=boot_ci(dd), overlap=bool(a1.min() <= a0.max() and a0.min() <= a1.max()),
            n1=int(len(a1)), n0=int(len(a0)), seeds=a1.tolist(), mean=float(a1.mean()), sd=float(a1.std(ddof=1)),
            min=float(a1.min()), max=float(a1.max()),
            n_worse_than_all=int((a1 > a0.max()).sum()), n_better_than_median=int((a1 < np.median(a0)).sum()),
            first3=dict(diff=float(a1[:k3].mean() - a0[:k3].mean()), welch_p=float(sst.ttest_ind(a1[:k3], a0[:k3], equal_var=False).pvalue)),
            # spread: Brown-Forsythe test of equal variances; sensitivity: the comparison without the poorest ablated run
            bf_p=float(sst.levene(a1, a0, center="median").pvalue), median=float(np.median(a1)), median0=float(np.median(a0)),
            drop_worst=dict(seed=int(np.argmax(a1)), value=float(a1.max()), diff=float(np.delete(a1, np.argmax(a1)).mean() - a0.mean()),
                            welch_p=float(sst.ttest_ind(np.delete(a1, np.argmax(a1)), a0, equal_var=False).pvalue),
                            sd=float(np.delete(a1, np.argmax(a1)).std(ddof=1))) if len(a1) > 3 else None,
            n_within_range=int((a1 <= a0.max()).sum()))
# delay-compensated clipped-optimal control: summary incl. the oracle-delay reference
T["codc"] = {}
if os.path.exists(os.path.join(RES, "meta__sw_delay.npz")):
    mm = meta("sw_delay"); vals = np.unique(mm["val"])
    for g in ("CO-NOM", "CO-ROB", "CO-DC", "CO-DCo", "AS"):
        d = load("sw_delay", g)
        if d is not None:
            T["codc"][g] = {"sw_delay": {j: [float(d[j][mm["val"] == v].mean()) for v in vals] for j in ("J1", "J2", "JT")}}
            for kind in ("nominal", "mc_in", "mc_ood"):
                dk = load(kind, g)
                if dk is not None:
                    T["codc"][g][kind] = {j: float(dk[j].mean()) for j in JS} | {"cvar90": st(dk["JT"])["cvar90"]}
    mi = meta("mc_in")
    T["codc"]["by_delay"] = {}
    for g in ("CO-NOM", "CO-ROB", "CO-DC", "CO-DCo", "AS", "P-BEST2"):
        d = load("mc_in", g)
        if d is not None:
            T["codc"]["by_delay"][g] = {str(int(v)): {j: float(d[j][mi["delay"] == v].mean()) for j in ("J1", "J2", "JT")} for v in np.unique(mi["delay"])}
    _, avg = group_stats("mc_in", GROUPS["PI-DRL"])
    if avg is not None:
        T["codc"]["by_delay"]["PI-DRL"] = {str(int(v)): {j: float(avg[j][mi["delay"] == v].mean()) for j in ("J1", "J2", "JT")} for v in np.unique(mi["delay"])}
    T["codc"]["by_delay"]["n"] = {str(int(v)): int((mi["delay"] == v).sum()) for v in np.unique(mi["delay"])}
json.dump(T, open(os.path.join(RES, "tables.json"), "w"), indent=1)
print("tables keys:", list(T.keys()))
for kind in ("mc_in", "mc_ood"):
    print("==", kind)
    for g, s in T[kind].items():
        print(f"{g:8s} " + " ".join(f"{j}={s[j]['mean']:.3f}/{s[j]['cvar90']:.2f}" for j in JS))
    for g, s in T[kind + "_abl"].items():
        print(f"  abl {g[:38]:38s} " + " ".join(f"{j}={s[j]['mean']:.3f}±{s[j]['seed_std']:.3f}" for j in JS))
    for b, s in T[kind + "_paired"].items():
        print(f"  vs {b:7s} " + " ".join(f"{j}:{100*s[j]['rel']:+.1f}% win {100*s[j]['win']:.0f}%" for j in JS))
