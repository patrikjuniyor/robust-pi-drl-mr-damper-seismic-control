"""build_paper.py -- assembles the manuscript (.docx) from the result files. Every number in tables and most numbers
in the text are read from ../results/*.json / *.npz so that the manuscript cannot drift from the computations."""
import json, os
import numpy as np
from paperlib import Paper
from simcore import Building, G
from env import MREnv

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); FIG = os.path.join(HERE, "..", "figures")
T = json.load(open(os.path.join(RES, "tables.json")))
B = json.load(open(os.path.join(RES, "baselines.json")))
VD = json.load(open(os.path.join(RES, "validation_dyke.json")))
IC = json.load(open(os.path.join(RES, "integration_check.json")))
KF = json.load(open(os.path.join(RES, "kf_tuning.json")))
bld = Building("six"); env = MREnv(bld, 1)
freq = bld.modal()[0]
W = bld.mass.sum() * G

# ----------------------------------------------------------------------------- submission switches
# BLIND=1        : anonymised reviewer copy (no name, affiliation, e-mail; author contributions withheld)
# SUBMISSION=1   : journal-ready copy (no draft banner, no placeholder line)
BLIND = os.environ.get("BLIND", "0") == "1"
SUBMISSION = os.environ.get("SUBMISSION", "0") == "1"
AUTHOR = "Hassan Alizadeh"
AFFIL = "Semnan University, Semnan, Iran"
EMAIL = "h.alizadeh@semnan.ac.ir"
WARN = ("This is a non-peer-reviewed manuscript; all results are numerical and no experimental validation is "
        "reported.")

f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"
pc = lambda x: f"{100 * x:.0f}%"
pc1 = lambda x: f"{100 * x:+.1f}%"
def pv(p):
    return "<0.001" if p < 1e-3 else f"{p:.3f}"
def M(kind, g, j, s="mean"):
    return T[kind][g][j][s]
def load(kind, name):
    return dict(np.load(os.path.join(RES, f"{kind}__{name}.npz")))

ORDER = ["P-OFF", "P-ON", "P-BEST", "P-BEST2", "AS", "CO-NOM", "CO-ROB", "CO-DC", "PI-DRL"]
N_IN, N_OOD = T["mc_in_N"], T["mc_ood_N"]
d_in, d_ood = M("mc_in", "PI-DRL", "JT"), M("mc_ood", "PI-DRL", "JT")
sd_in, sd_ood = T["mc_in"]["PI-DRL"]["JT"]["seed_std"], T["mc_ood"]["PI-DRL"]["JT"]["seed_std"]
rel = lambda kind, b, j="JT": T[kind + "_paired"][b][j]["rel"]
nomJT = {g: float(np.mean([v["JT"] for v in T["nominal"][g].values()])) for g in T["nominal"]}
wa_n, r_n = [float(x) for x in B["best_co_nom"].split("_")]
wa_r, r_r = [float(x) for x in B["best_co_rob"].split("_")]
pb = float(B["best_passive"]); pb2 = B["best_passive2"]; gas = B["best_as"]
r_dc, d_dc = float(B["best_co_dc"].split("_")[0]), int(B["best_co_dc"].split("_")[1])     # CO-DC: control weight, design delay (steps of 2 ms)
CD = T["codc"]
DG = json.load(open(os.path.join(RES, "codc_diag.json")))

# ---- training-budget and ablation statistics -----------------------------------------------------------------
from scipy import stats as _sst
T0 = json.load(open(os.path.join(RES, "tables_b100.json")))            # results of the earlier 100-iteration budget
RUNS = os.path.join(HERE, "..", "runs")
BU = T["budget"]; AP = T["abl_pair"]
PROP, NODR, NOOBS, SYM = ("PI-DRL (proposed)", "No domain randomisation", "No physics-based observer (raw history)",
                          "Symmetric critic (no privileged information)")
SEEDS = T["seeds"]                                   # seeds per configuration that were trained and evaluated
NS, NS_NODR, NS_OBS, NS_SYM = (len(SEEDS[g]) for g in (PROP, NODR, NOOBS, SYM))
WORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
NSW, NSW_NODR, NSW_OBS, NSW_SYM = WORD[NS], WORD[NS_NODR], WORD[NS_OBS], WORD[NS_SYM]
NRUNS = NS + NS_NODR + NS_OBS + NS_SYM
ST_in, ST_ood = T["mc_in_seedtest"], T["mc_ood_seedtest"]
bu = BU[PROP]
ITERS = bu["train_now"]["iters"]; ITERS0 = bu["train_b100"]["iters"]
DEC = bu["train_now"]["decisions"]; DEC0 = bu["train_b100"]["decisions"]
MINUTES = float(np.mean([BU[g]["train_now"]["minutes"] for g in BU]))
VAL = [T["train"][PROP]["val_at"][str(it)] for it in (100, 200, 300, 400)]
sAS = ST_in["AS"]["diffs"]; p_seedAS = ST_in["AS"]["p"]
pAS, pASo = T["mc_in_paired"]["AS"], T["mc_ood_paired"]["AS"]
relAS, relAS0 = pAS["JT"]["rel"], T0["mc_in_paired"]["AS"]["JT"]["rel"]
# ---- outcome descriptors: the prose is assembled from these, so that it follows the statistics of the current result files
def sig(p):
    return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"
A_in, A_ood = AP["mc_in"], AP["mc_ood"]
P0_in, P0_ood = A_in["_proposed"], A_ood["_proposed"]
dr_in, dr_ood = A_in[NODR], A_ood[NODR]
DR_SIG_IN, DR_SIG_OOD = dr_in["welch_p"] < 0.05, dr_ood["welch_p"] < 0.05
DR_REL_IN, DR_REL_OOD = dr_in["diff"] / P0_in["mean"], dr_ood["diff"] / P0_ood["mean"]
assert dr_in["diff"] > 0 and dr_ood["diff"] > 0                      # randomisation helps in the mean (sign only)
DR_STATUS = "established" if (DR_SIG_IN and DR_SIG_OOD) else "partial" if (DR_SIG_IN or DR_SIG_OOD) else "unresolved"
as_in, as_ood = ST_in["AS"], ST_ood["AS"]
AS_SIG_IN, AS_SIG_OOD = as_in["p"] < 0.05, as_ood["p"] < 0.05
AS_STATUS = "ahead" if (AS_SIG_IN and as_in["mean"] < 0) else "behind" if (AS_SIG_IN and as_in["mean"] > 0) else "tie"
assert pAS["J2"]["rel"] < 0
sy_d = A_in[SYM]; SY_SIG = sy_d["welch_p"] < 0.05 and A_ood[SYM]["welch_p"] < 0.05
assert sy_d["diff"] > 0
SY_REL = sy_d["diff"] / P0_in["mean"]
ABS_SY = (f"only the privileged critic shows a benefit that is nominally significant ({100 * SY_REL:.0f}%, Welch {sig(sy_d['welch_p'])}, "
          f"not corrected for the three comparisons)" if SY_SIG else
          "the privileged critic mainly makes training faster and more reliable")
_s3 = [T["mc_in_seeds"][f"seed {i}"]["dJT_vs_AS"] for i in range(3)]
relAS3 = float(np.mean(_s3)) / M("mc_in", "AS", "JT")                 # seeds 0-2 only (the seeds of the budget comparison)
assert relAS3 < 0 < relAS0
ABS_AS = {"ahead": f"across {NSW} training seeds the learned policies are ahead by {abs(100 * as_in['mean'] / M('mc_in', 'AS', 'JT')):.1f}% on "
                   f"average, a margin that is statistically detectable (seed-level {sig(as_in['p'])}) but has no practical weight",
          "tie": f"across {NSW} training seeds the mean difference of {abs(100 * as_in['mean'] / M('mc_in', 'AS', 'JT')):.1f}% is not "
                 f"statistically distinguishable from zero (seed-level {sig(as_in['p'])}), and individual seeds fall on either side",
          "behind": f"across {NSW} training seeds the learned policies are behind by {abs(100 * as_in['mean'] / M('mc_in', 'AS', 'JT')):.1f}% on "
                    f"average (seed-level {sig(as_in['p'])})"}[AS_STATUS]
ABS_DR = {"established": f"domain randomisation gives a small but statistically established gain ({abs(100 * DR_REL_IN):.1f}% in "
                         f"distribution, {abs(100 * DR_REL_OOD):.1f}% out of distribution; Welch {sig(dr_in['welch_p'])} and "
                         f"{sig(dr_ood['welch_p'])} with {NSW} seeds per group)",
          "partial": f"domain randomisation gives a small gain ({abs(100 * DR_REL_IN):.1f}% in distribution, "
                     f"{abs(100 * DR_REL_OOD):.1f}% out of distribution) whose statistical support remains borderline even with "
                     f"{NSW} seeds per group (Welch {sig(dr_in['welch_p'])} and {sig(dr_ood['welch_p'])})",
          "unresolved": f"domain randomisation gives a gain of {abs(100 * DR_REL_IN):.1f}% in distribution and "
                        f"{abs(100 * DR_REL_OOD):.1f}% out of distribution that is not statistically established even with {NSW} "
                        f"seeds per group (Welch {sig(dr_in['welch_p'])} and {sig(dr_ood['welch_p'])})"}[DR_STATUS]
print(f"outcomes: seeds {NS}/{NS_NODR}/{NS_OBS}/{NS_SYM} | DR {DR_STATUS} (in {dr_in['diff']:+.3f} p={dr_in['welch_p']:.4f}, "
      f"ood {dr_ood['diff']:+.3f} p={dr_ood['welch_p']:.4f}) | vs AS {AS_STATUS} (in {as_in['mean']:+.4f} p={as_in['p']:.3f}, "
      f"ood {as_ood['mean']:+.4f} p={as_ood['p']:.3f})")

P = Paper()
c = P.cite

# ================================================================================================ title
P.P("**Robust Physics-Informed Deep Reinforcement Learning for Semi-Active Seismic Control of Multi-Storey Buildings "
    "With MR Dampers**", align="center", size=15, space_after=8)
if not BLIND:
    P.P(f"**{AUTHOR}**", align="center", size=11.5, space_after=2)
    P.P(AFFIL, align="center", size=10, space_after=1)
    P.P(f"E-mail address: {EMAIL} (corresponding author)", align="center", size=9.5, space_after=1)
    if not SUBMISSION:
        P.P("*Faculty and full postal address to be added to match the journal template.*", align="center",
            size=8.5, space_after=8)
    else:
        P.P("", align="center", size=6, space_after=8)
P.P("*All numerical results were produced by the simulation code that accompanies this document; no experimental "
    "data are reported.*", align="center", size=9.5, space_after=12)

P.H("Abstract", 1)
P.P(f"Deep reinforcement learning (DRL) is increasingly applied to structural control, mostly on nominal models with full-state "
    f"feedback. This paper studies voltage control of magnetorheological (MR) dampers in a six-storey shear frame with four "
    f"shear-mode devices under explicitly modelled uncertainty; the policy sees only floor accelerations, damper strokes and damper "
    f"forces. It is trained with proximal policy optimisation (PPO) on synthetic ground motions with randomised structural, "
    f"device, sensor-noise and delay parameters, a nominal-model Kalman observer, a twin-normalised reward and a privileged "
    f"critic. Evaluation uses eleven recorded accelerograms unseen in training, {N_IN} in- and {N_OOD} out-of-distribution Monte "
    f"Carlo scenarios, tail statistics, robustness sweeps, fragility curves and seed-level tests. Over {NSW} seeds the "
    f"controller lowers the composite objective J_{{T}} by {pc(-rel('mc_in', 'P-BEST'))} relative to the best constant voltage, "
    f"{pc(-rel('mc_in', 'CO-ROB'))} relative to a robustly tuned clipped-optimal LQG controller and {pc(-rel('mc_in', 'CO-DC'))} "
    f"relative to a delay-compensated one, in and out of distribution. Three results temper this. First, a two-parameter "
    f"amplitude-scheduled law is practically as good (J_{{T}} = {f2(M('mc_in', 'AS', 'JT'))} against {f2(d_in)} ± {f2(sd_in)}: a "
    f"{abs(100 * as_in['mean'] / M('mc_in', 'AS', 'JT')):.1f}% margin that is statistically detectable, seed-level "
    f"{sig(as_in['p'])}, but of no practical weight). Second, quadrupling the training budget cut the mean J_{{T}} by only "
    f"{abs(100 * bu['mc_in']['paired']['diff'] / bu['mc_in']['b100']['mean']):.0f}%. Third, domain randomisation gives a gain of "
    f"{abs(100 * DR_REL_IN):.1f}% with borderline support (Welch {sig(dr_in['welch_p'])} and {sig(dr_ood['welch_p'])}), the "
    f"Kalman observer gives no resolvable gain, and only the privileged critic gives a nominally significant "
    f"{100 * SY_REL:.0f}% ({sig(sy_d['welch_p'])}, uncorrected). The learned behaviour is essentially intensity-adaptive "
    f"damping, its robustness stemming mainly from the slow, dissipative command structure.")
P.P("**Keywords:** semi-active control; magnetorheological damper; deep reinforcement learning (PPO); domain randomisation; "
    "partial observability; seismic fragility.", align="left")

# ================================================================================================ 1 intro
P.H("1. Introduction", 1)
P.P(f"Structural control aims to limit the response of buildings to earthquakes and wind by passive, active, hybrid or semi-active "
    f"devices {c('housner1997', 'spencer2003', 'soong2002', 'saaed2015')}. Semi-active devices occupy a practical middle ground: "
    f"their properties can be changed in real time with very little power, yet they can only dissipate energy and therefore cannot "
    f"destabilise the structure in the bounded-input sense {c('symans1999', 'spencer2003')}. The MR damper is the most studied "
    f"device of this class. Its force depends on a command voltage through strongly hysteretic dynamics, commonly described by "
    f"Bouc–Wen-type phenomenological models {c('wen1976', 'spencer1997', 'yang2002')}.")
P.P(f"Because the damper force cannot be commanded directly, the standard model-based solution is clipped-optimal control: a "
    f"linear optimal (LQG) controller computes a desired force from acceleration feedback and a bang-bang rule switches the voltage "
    f"so that the measured force tracks it {c('dyke1996', 'dyke1998', 'jansen2000', 'yi2001')}. The approach, and related Lyapunov, "
    f"fuzzy and neural schemes {c('jansen2000', 'schurter2001', 'chang2002', 'banihani2006', 'bitaraf2010')}, has been "
    f"validated on laboratory structures and benchmark problems {c('ohtori2004', 'yoshida2004')}. It nevertheless depends on a "
    f"nominal linear model and on fast switching, both of which are sensitive to modelling error and actuation delay.")
P.P(f"Deep reinforcement learning {c('sutton2018', 'mnih2015', 'lillicrap2015', 'schulman2017', 'fujimoto2018', 'haarnoja2018')} "
    f"offers a model-free alternative that can in principle handle the damper nonlinearity directly. A growing body of work applies "
    f"it to structural control. For active systems, studies report online learning with time delay and shake-table tests "
    f"{c('khalat2019')}, learning-based frameworks for smart structures {c('rahmani2019')}, a five-storey benchmark controller "
    f"that outperforms LQG {c('eshkevari2021')}, model-free active vibration control {c('zhang2023')}, DRL-modified LQR "
    f"{c('gheni2024')}, LQR-guided safe exploration {c('thorat2025')}, PPO training on Kanai–Tajimi excitation {c('asai2026')}, "
    f"a knowledge-guided framework with randomised stiffness degradation {c('wang2026')} and an RL-controlled hybrid mass damper on "
    f"a real tower {c('kouts2026')}. For semi-active MR devices, deep Q-learning has been used for a mid-storey isolation system "
    f"{c('kim2023')} and combined with damage identification {c('hormozabad2024')}, and TD3 has been applied to MR vehicle "
    f"suspensions {c('babaahmadi2023')}.")
P.P(f"Three gaps motivate this paper. (i) Most studies assume that the full state is available to the agent, whereas a building "
    f"controller measures a few accelerations and device quantities; the problem is a partially observable Markov decision process "
    f"{c('kaelbling1998')}. (ii) Policies are usually trained and tested on the same nominal model, although robotics has shown "
    f"that policies trained in one simulator can fail under small model changes and that randomising the simulator is a practical "
    f"remedy {c('tobin2017', 'peng2018', 'rajeswaran2016', 'openai2018', 'muratore2022', 'zhao2020')}. (iii) Performance is "
    f"typically reported for a handful of deterministic records, without seed variability {c('henderson2018', 'agarwal2021')}, "
    f"tail statistics or strong, carefully tuned baselines.")
P.P("The contributions of this paper are the following.")
P.bullets([
    "A partially observed, uncertainty-aware formulation of MR-damper voltage control in which stiffness, mass, damping, device "
    "parameters, sensor noise and command delay are randomised, and a fast batched simulator of the hysteretic system.",
    "A robust physics-informed DRL controller that combines a nominal-model Kalman observer, a twin-normalised reward and an "
    "asymmetric actor–critic with a privileged critic, trained with domain randomisation on synthetic ground motions only.",
    "An evaluation protocol on recorded ground motions that reports distributions, tail risk, paired confidence intervals, "
    "out-of-distribution performance, robustness sweeps and fragility curves, against six tuned non-learning baselines, "
    "including a delay-compensated clipped-optimal controller, and that tests claims about the training method with the "
    "training seed as the unit of replication.",
    "A candid assessment: the learned controller is clearly better than constant-voltage and clipped-optimal control under "
    "uncertainty, with or without delay compensation, but a two-parameter amplitude-scheduled law comes within about one per "
    "cent of it, a fourfold training budget adds little, and the ablations show only a small benefit of domain randomisation "
    "and none of the observer. We report these findings as they are.",
])

# ================================================================================================ 2 related work
P.H("2. Background and related work", 1)
P.H("2.1 Semi-active control with MR dampers", 2)
P.P(f"Dyke et al. {c('dyke1996')} introduced clipped-optimal control with acceleration feedback on a three-storey model with one "
    f"MR damper and reported that it outperformed both passive modes of the device. Jansen and Dyke {c('jansen2000')} compared "
    f"several semi-active algorithms on a six-storey structure with MR dampers on the two lowest storeys; that configuration is "
    f"the test bed of this paper. Later work scaled the device and the model to full size {c('yang2002', 'yoshida2004')}, defined "
    f"nonlinear benchmark buildings {c('ohtori2004')} and examined alternative strategies {c('yi2001', 'bitaraf2010')}. An effect "
    f"that is central to our results is that the best passive setting of an MR damper depends on the excitation level: a device "
    f"that is well sized for a strong motion locks the storey and transmits acceleration during a weak one.")
P.H("2.2 Deep reinforcement learning for structural control", 2)
P.P(f"The studies cited in the introduction differ in algorithm (DQN, DDPG, TD3, SAC, PPO), device and structure, but share "
    f"several features. Rewards are usually hand-weighted sums of responses. Training excitations are a few records or filtered "
    f"white noise {c('asai2026', 'kim2023')}. Robustness, where addressed, concerns one factor such as time delay {c('khalat2019')} "
    f"or stiffness degradation {c('wang2026')}. Comparisons are mostly against LQR/LQG or simple passive and skyhook-type laws. We "
    f"are not aware of a study of MR-damper voltage control that combines partial observation, multi-factor randomisation and a "
    f"distributional evaluation; we note, however, that this literature is growing quickly and that our survey was not systematic.")
P.H("2.3 Robust and physics-informed reinforcement learning", 2)
P.P(f"Domain randomisation trains a policy on a distribution of simulators so that the real system appears as one more sample "
    f"{c('tobin2017', 'peng2018', 'muratore2022')}. Asymmetric actor–critic methods give the critic privileged simulator "
    f"information that the deployed actor does not need {c('pinto2018', 'openai2018')}. Physics-informed machine learning "
    f"embeds prior physical knowledge in the learning problem {c('raissi2019', 'karniadakis2021')}; in reinforcement learning the "
    f"knowledge can enter through the state representation, the reward, the action space or the model {c('banerjee2023')}. In this "
    f"paper it enters through the observation (a model-based state estimate {c('kalman1960')}), the reward (structural energies "
    f"and an uncontrolled reference) and the critic input (true physical state and parameters). No physics residual is added to "
    f"the loss; the term ‘physics-informed’ is used in this broader sense.")

# ================================================================================================ 3 model
P.H("3. System model and uncertainty description", 1)
P.H("3.1 Structure", 2)
P.P(f"The test bed is a six-storey, single-bay laboratory-scale shear frame with two shear-mode MR dampers between the ground and "
    f"the first floor and two between the first and second floors, patterned after the numerical example of Jansen and Dyke "
    f"{c('jansen2000')}. Each floor has mass 22.7 kg (0.227 N·s²/cm), each storey has stiffness 297 N/cm and every mode has 0.5% "
    f"damping. The computed natural frequencies are {', '.join(f'{x:.2f}' for x in freq)} Hz and the weight is {W:.0f} N. The "
    f"equation of motion is")
P.EQ("M ẍ + C ẋ + K x = Γ f − M 1 ẍ_{g}")
P.P("where x collects the floor displacements relative to the ground, f the two storey-level damper forces (each the sum of two "
    "devices), Γ their influence matrix and ẍ_{g} the ground acceleration. The frame is linear elastic throughout.")
P.P(f"*Provenance of the parameters.* We could not access the full text of {c('jansen2000')}. The damper parameters below were "
    f"taken from a secondary source that lists them with attribution to that paper {c('patrascu2012')}. For the structure, a second "
    f"secondary source {c('chojung')} gives the same stiffness and damping but a floor mass of 0.277 N·s²/cm, whereas we use "
    f"0.227 N·s²/cm. The model should therefore be read as *patterned after* the original example, not as an exact replica, and "
    f"our numbers are not directly comparable with those of {c('jansen2000')}.")
P.H("3.2 MR damper", 2)
P.P("Each shear-mode device follows the simple Bouc–Wen model with first-order voltage dynamics:")
P.EQ("f_{d} = c_{0}(u) ẋ_{d} + α(u) z ,     ż = −γ |ẋ_{d}| z |z|^{n−1} − β ẋ_{d} |z|^{n} + A ẋ_{d}")
P.EQ("c_{0}(u) = c_{0a} + c_{0b} u ,     α(u) = α_{a} + α_{b} u ,     u̇ = −η (u − v)")
P.P("where ẋ_{d} is the storey drift velocity, z the hysteretic variable, v the command voltage and u its filtered value. "
    "Table 1 lists the parameters. With n = 2 the hysteretic variable saturates at (A/(γ+β))^{1/2} = 0.447 cm, so one device "
    "delivers about 3.9 N at 0 V and about 23.7 N at 5 V, plus a small viscous term; the two devices of a storey deliver about "
    "47 N, or 3.5% of the building weight. The two devices of a storey share one command, so the controller has two outputs.")
P.TAB("Model parameters (nominal values).",
      ["Item", "Value", "Item", "Value"],
      [["Floor mass (each)", "22.7 kg", "c_{0a}", "0.0064 N·s/cm"],
       ["Storey stiffness (each)", "297 N/cm", "c_{0b}", "0.0052 N·s/(cm·V)"],
       ["Modal damping (all modes)", "0.5%", "α_{a}", "8.66 N/cm"],
       ["Natural frequencies", ", ".join(f"{x:.2f}" for x in freq) + " Hz", "α_{b}", "8.86 N/(cm·V)"],
       ["Dampers", "2 at storey 1, 2 at storey 2", "γ, β", "300 cm^{−2}, 300 cm^{−2}"],
       ["Maximum voltage V_{max}", "5 V", "A, n", "120, 2"],
       ["Building weight", f"{W:.0f} N", "η", "80 s^{−1}"]], widths=[4.2, 4.6, 2.0, 4.0])
P.H("3.3 Sensors, actuation and uncertainty", 2)
P.P("The controller measures the six absolute floor accelerations, the two damper strokes (drifts of storeys 1 and 2) and the two "
    "storey-level damper forces, sampled every 10 ms. Displacements and velocities of the upper floors and the hysteretic "
    "variables are not measured. White measurement noise is added to every sample and the command voltage reaches the damper "
    "after a delay. Table 2 lists the quantities that are randomised independently for every simulated earthquake. The "
    "‘training’ column is the distribution used for domain randomisation and for the in-distribution test set; the "
    "‘out-of-distribution’ column widens every range by a factor 1.6. On the *nominal* plant all scale factors are one and "
    "noise and delay are zero.")
P.TAB("Randomised parameters. U denotes a uniform distribution; noise level 1 corresponds to standard deviations of 0.02 m/s² "
      "(acceleration), 0.02 mm (stroke) and 0.3 N (force) per sample.",
      ["Quantity", "Training / in-distribution", "Out-of-distribution"],
      [["Storey stiffness scale (per storey)", "U(0.75, 1.25)", "U(0.60, 1.40)"],
       ["Floor mass scale (per floor)", "U(0.90, 1.10)", "U(0.84, 1.16)"],
       ["Damping scale (whole matrix)", "log-uniform (0.5, 2.0)", "log-uniform (0.33, 3.03)"],
       ["Damper force scale α_{a}, α_{b} (per storey)", "U(0.80, 1.20)", "U(0.68, 1.32)"],
       ["Damper viscous scale c_{0a}, c_{0b} (per storey)", "U(0.80, 1.20)", "U(0.68, 1.32)"],
       ["Damper response rate η scale (per storey)", "U(0.70, 1.30)", "U(0.52, 1.48)"],
       ["Command delay", "0, 10 or 20 ms", "0, 10, 20 or 30 ms"],
       ["Sensor-noise level", "U(0, 1)", "U(0, 1.6)"],
       ["Peak ground acceleration", "log-uniform (0.015, 0.20) g", "same"]], widths=[6.2, 4.6, 4.2])
P.H("3.4 Ground motions", 2)
P.P(f"*Training motions* are synthetic. Gaussian white noise is passed through a Kanai–Tajimi filter {c('kanai1957', 'tajimi1958')} "
    f"with site frequency log-uniform in 6–35 rad/s and damping uniform in 0.3–0.9, then through a Clough–Penzien high-pass filter "
    f"{c('clough1975')}, and multiplied by a random envelope with quadratic rise, plateau and exponential decay, in the spirit of "
    f"separable non-stationary models {c('rezaeian2008')}. With probability 0.3 a Gabor-wavelet acceleration pulse (0.4–2.5 Hz, "
    f"1–2.5 cycles) is added. Each motion lasts 16 s and is scaled to a peak ground acceleration (PGA) drawn log-uniformly from "
    f"0.015–0.20 g. A fresh batch is generated at every training iteration, so no motion is seen twice.")
P.P(f"*Test motions* are eleven recorded accelerograms (Table 3): the north–south El Centro 1940 record from {c('vibrationdata')} "
    f"and ten records distributed with the gmspy package {c('gmspy')}, which attributes them to the PEER strong-motion database "
    f"{c('peer')}. Records are truncated to 40 s and the few that are shorter (Trinidad 1983 lasts 21 s) are zero-padded to that "
    f"length; each record is normalised to unit PGA and scaled to the PGA required by each scenario {c('vamvatsikos2002')}. The "
    f"El Centro version used here has an as-recorded PGA of 0.349 g (Table 3). Records are not compressed in time, and none of "
    f"them was used for training, validation or model selection.")
P.TAB("Recorded accelerograms used for testing (as-recorded PGA; station labels as given in the source files).",
      ["Event", "Station / component", "PGA (g)", "Event", "Station / component", "PGA (g)"],
      [["El Centro 1940", "NS component", "0.349", "Loma Prieta 1989", "CDMG 47381 (090)", "0.367"],
       ["Kobe 1995", "Kakogawa (CUE90)", "0.345", "Landers 1992", "SCE station 24 (000)", "0.780"],
       ["Northridge 1994", "CDMG 24278 (090)", "0.568", "Imperial Valley 1979", "USGS 5115", "0.315"],
       ["Chi-Chi 1999", "TCU045", "0.361", "Friuli 1976", "Tolmezzo (000)", "0.351"],
       ["Kocaeli 1999", "Yarimca (KOERI330)", "0.349", "Hollister 1961", "USGS 1028", "0.195"],
       ["Trinidad 1983", "CDMG 1498 (090)", "0.194", "–", "–", "–"]], widths=[2.8, 3.6, 1.4, 2.9, 3.4, 1.4])
P.P("*Intensity range.* The PGA range was chosen so that the best constant voltage varies from zero at the weakest motions to "
    "V_{max} at the strongest (Section 6.3). This makes amplitude adaptation matter by design. At the upper end the uncontrolled "
    "linear model reaches storey drifts of several centimetres, which a physical frame of this size would not sustain elastically; "
    "the uncontrolled response serves as a normalising reference, not as a prediction for a real frame.")
P.H("3.5 Numerical integration", 2)
P.P(f"The coupled equations are integrated with a batched fourth-order Runge–Kutta scheme compiled with Numba {c('lam2015')}, "
    f"with linear interpolation of the ground acceleration. Because the hysteretic variable becomes stiff at high velocity, each "
    f"step is subdivided adaptively so that the product of the local stiffness estimate n(γ+β)|ẋ_{{d}}| z_{{max}}^{{n−1}} and the "
    f"sub-step stays below 1.4. Training uses a base step of 2 ms. All reported results use 1 ms (constant-voltage, scheduled and "
    f"learned controllers, 100 Hz control rate) or 0.5 ms (clipped-optimal controllers, 500 Hz control rate). Section 6.1 verifies "
    f"the code and quantifies the step-size error.")

# ================================================================================================ 4 method
P.H("4. Robust physics-informed DRL controller", 1)
P.FIG(os.path.join(FIG, "fig_schematic.png"), "Training set-up. The actor uses only quantities that are measurable on the "
      "structure; the uncontrolled twin and the privileged critic exist only in simulation.", 15.0)
P.H("4.1 Partially observed decision process", 2)
P.P("Every 0.1 s (ten control samples) the agent chooses an action a ∈ [−1, 1]² that is mapped to the two storey voltages "
    "v = V_{max}(a + 1)/2 and held until the next decision. The hidden state comprises the structural state, the hysteretic "
    "variables, the filtered voltages, the delay buffer and the plant parameters of the episode. An episode is one earthquake. "
    "The slow decision rate was a deliberate choice made during pilot runs: it shortens the credit-assignment horizon to 160 "
    "decisions per episode and restricts the agent to slowly varying voltages.")
P.H("4.2 Physics-informed observation", 2)
P.P(f"The actor input has {env.obs_dims['pi']} components, all computable from the measurements:")
P.bullets([
    "*Model-based state estimate (12).* A steady-state discrete Kalman filter built on the nominal linear model estimates floor "
    "displacements and velocities from the measured accelerations and strokes. The measured damper force enters as a known input, "
    "so the observer needs no damper model; the ground acceleration is treated as white process noise. The filter always uses the "
    "nominal matrices, so its estimate is biased whenever the plant is perturbed.",
    "*Measurement history (100).* The ten measured channels at lags 0, 1, 2, 3, 4, 6, 8, 12, 16 and 24 control samples (up to "
    "0.24 s), which lets a feed-forward network infer velocities, phase and delay.",
    "*Intensity features (16).* The exponentially weighted (1 s) root-mean-square of each channel and the running peak of each "
    "floor acceleration.",
    "*Previous action (2).*",
])
P.P(f"The process-noise intensity and a measurement-noise inflation factor of 30 were selected by minimising the state-estimation "
    f"error on randomised plants; with the chosen values the relative RMS estimation error is "
    f"{KF['0.2_30.0'][0]:.2f} at 0.1 V_{{max}} and {KF['0.2_30.0'][1]:.2f} at V_{{max}}, most of it in the velocities.")
P.H("4.3 Twin-normalised reward", 2)
P.P("For every training episode the same plant is simulated once without dampers under the same motion (the *uncontrolled "
    "twin*). Its peak storey drift D^{unc}, peak floor acceleration A^{unc}, mean strain energy E_{s}^{unc} and mean inertial "
    "energy E_{a}^{unc} = mean(½ Σ m_{i} a_{i}²) normalise the reward, which makes episodes of very different intensity "
    "commensurable. With m_{d}(k) and m_{a}(k) the running maxima of normalised drift and acceleration, the reward at control "
    "sample k is")
P.EQ("r_{k} = −[m_{d}(k) − m_{d}(k−1)] − [m_{a}(k) − m_{a}(k−1)] − (w_{e}/2N) [E_{s}(k)/E_{s}^{unc} + E_{a}(k)/E_{a}^{unc}]")
P.P("with N the number of control samples and w_{e} = 4. The first two terms telescope, so the undiscounted return of an episode "
    "equals minus the composite objective")
P.EQ("J_{T} = J_{1} + J_{2} + w_{e} J_{E}")
P.P("where J_{1} and J_{2} are the peak drift and peak acceleration normalised by the uncontrolled twin and J_{E} is the mean of "
    "the two normalised energies. J_{T} = 6 corresponds to no reduction at all. The peak terms are sparse but exact; the energy "
    "term is dense and rewards damping of the whole response. The weights were fixed before the final runs and were not tuned on "
    "test data.")
P.H("4.4 Asymmetric actor–critic and training", 2)
P.P(f"The policy is trained with PPO {c('schulman2017')} and generalised advantage estimation {c('schulman2015')} using the Adam "
    f"optimiser {c('kingma2014')} in PyTorch {c('paszke2019')}. The actor is a Gaussian policy with state-independent standard "
    f"deviation. The critic is used only during training and receives a {env.priv_dim}-dimensional privileged input "
    f"{c('pinto2018')}: the true structural state, damper force, hysteretic variables and filtered voltages, the intensity "
    f"features, the randomised parameters of the episode, the ground acceleration at four upcoming instants and its RMS over the "
    f"next 0.4 s, the elapsed time fraction, the twin normalisers and the running peaks. Table 4 lists the hyperparameters. "
    f"Every five iterations the deterministic policy is evaluated on a fixed validation set of 64 synthetic motions with "
    f"randomised plants, and the checkpoint with the lowest validation J_{{T}} is kept.")
P.TAB("Network and training settings.",
      ["Setting", "Value", "Setting", "Value"],
      [["Actor network", f"{env.obs_dims['pi']}–128–128–2, tanh", "Discount γ, GAE λ", "0.97, 0.92"],
       ["Critic network", f"{env.priv_dim}–256–256–1, tanh", "Clip range", "0.2"],
       ["Parallel environments", "64", "Epochs / minibatch", "10 / 2048"],
       ["Episode length", "16 s (160 decisions)", "Learning rate", "10^{−3} → 10^{−4}, linear"],
       ["Iterations", f"{ITERS}", "KL early stop", "0.03"],
       ["Decisions per run", f"{DEC / 1e6:.2f} × 10^{{6}}", "Gradient-norm clip", "0.5"],
       ["Initial policy std", "exp(−0.7)", "Value loss", "Huber, weight 0.5"],
       ["Seeds per configuration", f"{NS} (proposed), {NS_NODR} / {NS_OBS} / {NS_SYM} (ablations)", "Wall-clock per run", f"≈{MINUTES:.0f} min, one CPU core"]], widths=[3.8, 4.4, 3.4, 3.8])
P.H("4.5 Domain randomisation", 2)
P.P("At every iteration each of the 64 environments receives a new synthetic motion and a new draw of all quantities in Table 2. "
    "The policy is therefore never trained on the nominal plant alone, except in the ablation of Section 6.7.")

# ================================================================================================ 5 baselines + protocol
P.H("5. Baselines and evaluation protocol", 1)
P.H("5.1 Baseline controllers", 2)
P.P("Eight non-learning controllers are compared with the learned one. All tuning used synthetic motions from the training "
    "distribution; the recorded test motions were not used.")
P.bullets([
    "**P-OFF, P-ON.** Constant 0 V and constant V_{max}.",
    f"**P-BEST.** The best common constant voltage, {pb:g} V_{{max}}, selected among seven values on 128 randomised synthetic "
    f"scenarios.",
    f"**P-BEST2.** The best pair of constant voltages for storeys 1 and 2, ({pb2[0]:g}, {pb2[1]:g}) V_{{max}}, selected among 37 "
    f"pairs on the same scenarios.",
    f"**AS (amplitude-scheduled).** v_{{j}} = V_{{max}} min(1, g_{{j}} ρ_{{j}}/d_{{ref}}), where ρ_{{j}} is the exponentially weighted "
    f"(1 s) RMS of the measured stroke of storey j and d_{{ref}} = 5 mm. The gains (g_{{1}}, g_{{2}}) = ({gas[0]:g}, {gas[1]:g}) were "
    f"selected among 16 pairs on the same scenarios. The law uses a single measured quantity per storey and no model. It was "
    f"added *after* the first DRL results were available, as a check on whether the learned gain could be explained by "
    f"intensity scheduling alone.",
    f"**CO-NOM.** Clipped-optimal LQG control {c('dyke1996', 'jansen2000')}: an LQR on the nominal model with weight on the strain "
    f"energy (and optionally on the inertial energy) and control weight r, fed by the same type of Kalman filter, with the "
    f"bang-bang rule v = V_{{max}} H[(f_{{c}} − f) f]. It runs at 500 Hz. Weights were selected on the *nominal* plant "
    f"(w_{{a}} = {wa_n:g}, r = {r_n:g}).",
    f"**CO-ROB.** The same controller with weights selected on *randomised* plants (w_{{a}} = {wa_r:g}, r = {r_r:g}).",
    f"**CO-DC (delay-compensated).** Time delay is known to degrade semi-active MR-damper control {c('cha2013')}, and "
    f"compensation by prediction over the delay is a standard remedy in structural control {c('agrawal2000')}. CO-DC is the "
    f"clipped-optimal controller with a model-based predictor. The command issued at step k reaches the damper after the delay d, and the commands issued within the last d "
    f"are known to the controller. At every 2 ms step the nominal model, i.e. the linear structure together with the "
    f"Bouc–Wen dampers and the first-order voltage filter, is integrated over the design delay d_{{c}}, starting from the Kalman "
    f"estimate of the structural state, the controller's own record of the queued commands, and a hysteretic variable recovered "
    f"from the measured force, z = (f − c_{{0}}(u) ẋ)/α(u). The ground acceleration, which is not measured, is set to zero over the "
    f"horizon. The switching rule is then evaluated on the *predicted* state and force, v = V_{{max}} H[(f_{{c}}(x̂_{{k+d}}) − "
    f"f̂_{{k+d}}) f̂_{{k+d}}]. One fixed design delay is used for all plants, since the true delay (0, 10 or 20 ms) is unknown "
    f"to the controller. Control weight and design delay were selected on the randomised plants used for CO-ROB "
    f"(r = {r_dc:g}, d_{{c}} = {2 * d_dc} ms). With d_{{c}} = 0 the controller is identical to CO-NOM/CO-ROB.",
])
P.P(f"The clipped-optimal grid covered 18 weight combinations and was extended until the optimum was interior; it was evaluated on "
    f"64 synthetic motions. On that set CO-NOM attains J_{{T}} = {B['co_nom'][B['best_co_nom']][0]:.2f} on the nominal plant but "
    f"{B['co_rob'][B['best_co_nom']][0]:.2f} on randomised plants, whereas CO-ROB attains {B['co_rob'][B['best_co_rob']][0]:.2f} "
    f"on randomised plants; the robust tuning chooses a large control weight and therefore switches the dampers on rarely. "
    f"CO-NOM and CO-ROB have no delay compensation. Running at 500 Hz, the clipped-optimal controllers receive five times as "
    f"many samples as the other controllers, with the same noise level per sample; the command delay is the same in milliseconds.")
cdg = B["co_dc"]; rs_dc = sorted({float(k.split("_")[0]) for k in cdg})
rows = [[f"{r:g}"] + [f"{cdg[f'{r}_{d}'][0]:.2f} ({cdg[f'{r}_{d}'][1]:.2f} / {cdg[f'{r}_{d}'][2]:.2f})" for d in (5, 10)] for r in rs_dc]
P.TAB("Tuning of the delay-compensated clipped-optimal controller on 64 synthetic motions with randomised plants (the set used "
      "for CO-ROB): mean J_{T} (J_{1} / J_{2}) for each control weight r and design delay d_{c}.",
      ["r", "d_{c} = 10 ms", "d_{c} = 20 ms"], rows, widths=[2.5, 5.0, 5.0], bold_rows=(rs_dc.index(r_dc),))
t_dctune = P.ntab
assert min(rs_dc) < r_dc < max(rs_dc)
_orc = cdg[f"{r_dc}_None"]
P.P(f"Table {P.ntab} gives the tuning grid of CO-DC. The optimum is interior in r, and the two design delays differ by about "
    f"{abs(100 * (cdg[f'{r_dc}_5'][0] / cdg[f'{r_dc}_10'][0] - 1)):.0f}%. The selected setting attains J_{{T}} = "
    f"{cdg[B['best_co_dc']][0]:.2f} on this set, against {B['co_rob'][B['best_co_rob']][0]:.2f} for CO-ROB and "
    f"{B['co_rob'][B['best_co_nom']][0]:.2f} for CO-NOM. The predictor thus permits a control weight ten times smaller than the "
    f"robust tuning. As a reference, the same controller given the *true* delay of every plant (an oracle that is not "
    f"realisable) attains {_orc[0]:.2f}; not knowing the delay within 0–20 ms therefore costs the predictor about "
    f"{100 * (cdg[B['best_co_dc']][0] / _orc[0] - 1):.0f}%.")
P.H("5.2 Test sets", 2)
P.bullets([
    "**Nominal.** 11 records × 3 PGA levels (0.035, 0.07, 0.14 g) on the nominal plant, without noise or delay (33 scenarios).",
    f"**In-distribution Monte Carlo.** 11 records × 30 draws of Table 2, training ranges ({N_IN} scenarios).",
    f"**Out-of-distribution Monte Carlo.** The same with the widened ranges ({N_OOD} scenarios).",
    "**Sweeps.** One factor varied on the otherwise nominal plant at PGA 0.07 g: stiffness scale 0.5–1.3, damper force scale "
    "0.4–1.2, noise level 0–4, command delay 0–50 ms (11 records per value).",
    "**Fragility.** 8 PGA levels from 0.015 to 0.25 g × 11 records × 4 draws of Table 2 (352 scenarios).",
])
P.P("Every controller is run on exactly the same scenarios (same record, PGA, plant parameters and noise seed), which permits "
    "paired comparisons.")
P.H("5.3 Metrics and statistics", 2)
P.P(f"For each scenario we compute J_{{1}} (peak storey drift), J_{{2}} (peak absolute floor acceleration), J_{{3}} and J_{{4}} (their "
    f"RMS counterparts) and J_{{T}}, each normalised by the uncontrolled twin of that scenario, in the manner of the benchmark "
    f"evaluation criteria {c('ohtori2004')}. Over a scenario set we report the mean, the median, the 95th percentile and the "
    f"conditional value-at-risk CVaR_{{90}}, i.e. the mean of the worst 10% of scenarios {c('rockafellar2000')}. Confidence "
    f"intervals are 95% percentile bootstrap intervals over scenarios with 2000 resamples {c('efron1979')}. Paired differences "
    f"between the learned controller and each baseline are tested with the Wilcoxon signed-rank test {c('wilcoxon1945')}. For the "
    f"learned controller, scenario-level values are averaged over the {NSW} training seeds before these statistics are computed, "
    f"and the sample standard deviation across seeds is reported separately. Because scenario-level intervals treat the trained "
    f"policies as fixed, claims about the training *method* are tested with the seed as the unit of replication "
    f"{c('henderson2018', 'agarwal2021')}: a one-sample t-test on the {NSW} per-seed mean differences for comparisons with a "
    f"baseline, and Welch's unequal-variance t-test {c('welch1947')}, complemented by the Mann–Whitney U test {c('mann1947')}, on the "
    f"per-seed means for comparisons between training configurations. The proposed configuration and the one without domain randomisation were "
    f"trained with {NSW} seeds each, the other two ablations with {NSW_OBS}. Fragility curves are lognormal functions of PGA fitted by maximum likelihood "
    f"{c('baker2015')} to exceedance counts of a peak storey drift of 3 mm and a peak floor acceleration of 2 m/s²; these "
    f"thresholds are illustrative and carry no code meaning. The p-values of the ablation study are given unadjusted: three "
    f"configurations are compared with the proposed one on two test sets, so six tests are made in total, and any single "
    f"ablation difference should be judged after an appropriate correction (a Bonferroni correction, for instance, raises the "
    f"smallest of the six values from 0.027 to about 0.16).")

# ================================================================================================ 6 results
P.H("6. Results", 1)
P.P(f"*Provenance of the reported numbers.* All results for learned controllers in this section, i.e. every PI-DRL entry in the "
    f"tables and figures of Sections 6.2–6.7 and 6.9, come from policies trained for {ITERS} iterations "
    f"({DEC / 1e6:.2f} million decisions). ‘PI-DRL’ denotes the average over {NSW} independently trained policies (seeds 0–{NS - 1}) "
    f"unless a single seed is named. Results of the earlier {ITERS0}-iteration budget ({DEC0 / 1e6:.2f} million decisions) appear "
    f"in Section 6.8 and as the dotted curve of the training figure, and nowhere else; that comparison uses the three seeds "
    f"(0–2) that were trained at both budgets. The non-learning baselines do not depend on the training budget.")
P.H("6.1 Verification of the simulator", 2)
P.P(f"*Reproduction of a published case.* Since we could not compare with {c('jansen2000')} directly, the code was verified on "
    f"the three-storey, single-damper experiment-based model of Dyke et al. {c('dyke1996')}, which uses the more elaborate "
    f"modified Bouc–Wen damper model {c('spencer1997')} and the El Centro record compressed five times in time. Structural "
    f"matrices, damper parameters and reference responses were read from the authors' full text {c('dyke1996pdf')}. Table 6 "
    f"compares peak responses. The passive cases agree within about 2% in all quantities, except the small first-storey "
    f"displacement in passive-on (0.079 against 0.076 cm, 4%). The uncontrolled case agrees within 1–4%; the difference is "
    f"presumably due to the version of the El Centro record. Our clipped-optimal implementation reproduces displacements within "
    f"1%, drifts and force within about 3% and accelerations within about 5%; it is not identical to the original (discrete "
    f"Kalman filter with measured force input, 500 Hz sampling, control weight chosen by us). To match the published damper forces, the constant accumulator offset of the modified model had to be "
    f"omitted. This test verifies the integrator, the damper routines and the clipped-optimal logic; it does not validate the "
    f"six-storey configuration.")
ref = VD["reference"]; rows = []
cases = [("Uncontrolled", "Uncontrolled"), ("Passive-off", "Passive-off"), ("Passive-on", "Passive-on"), ("Clipped-optimal r=1e-15", "Clipped-optimal")]
for q, lab, fm in (("x", "x_{%d} (cm)", "{:.3f}"), ("d", "d_{%d} (cm)", "{:.3f}"), ("a", "a_{%d} (cm/s²)", "{:.0f}")):
    for i in range(3):
        rows.append([lab % (i + 1)] + [f"{fm.format(VD[s][q][i])} / {fm.format(ref[r][q][i])}" for s, r in cases])
rows.append(["f (N)"] + ["–"] + [f"{VD[s]['f']:.0f} / {ref[r]['f']:.0f}" for s, r in cases[1:]])
P.TAB("Verification on the three-storey model of Dyke et al. (1996), El Centro record: peak relative displacements x, storey "
      "drifts d, absolute accelerations a and damper force f. Each cell gives *this study / published value*.",
      ["Quantity", "Uncontrolled", "Passive-off", "Passive-on", "Clipped-optimal"], rows, widths=[2.6, 3.2, 3.2, 3.2, 3.2])
P.P("*Step-size error.* Table 7 compares peak responses of the six-storey model on 32 randomised synthetic scenarios with a "
    "reference computed with a 0.1 ms step. At the steps used for the reported results, the error of smooth-voltage "
    "controllers is negligible. The bang-bang clipped-optimal controller is more sensitive, because small differences can change "
    "individual switching decisions; its median error is still below 0.1%, with a worst case of a few per cent in peak "
    "acceleration.")
rows = []
for case, d in IC.items():
    for step, v in d.items():
        rows.append([case, step, f"{100 * v['drift_med']:.3f}", f"{100 * v['drift_max']:.2f}", f"{100 * v['acc_med']:.3f}", f"{100 * v['acc_max']:.2f}"])
P.TAB("Relative difference (%) of peak storey drift and peak floor acceleration with respect to a 0.1 ms reference "
      "(32 randomised scenarios; sensor noise switched off so that runs are comparable).",
      ["Controller", "Step", "Drift, median", "Drift, max", "Acceleration, median", "Acceleration, max"], rows,
      widths=[4.0, 3.0, 2.1, 2.1, 2.4, 2.4])

P.H("6.2 Training", 2)
tr = T["train"]["PI-DRL (proposed)"]
P.FIG(os.path.join(FIG, "fig_learning.png"), "Validation performance during training (64 synthetic motions, randomised plants). "
      f"Lines: mean over seeds; bands: range over seeds (n: number of seeds). Dotted: proposed configuration in the earlier 100-iteration runs (three seeds).", 15.5)
f_learn = P.nfig
P.P(f"Figure {P.nfig} shows the validation curves. The proposed configuration reaches a validation J_{{T}} of "
    f"{tr['best_val']:.2f} ± {tr['best_val_sd']:.2f} (mean and standard deviation of the {NSW} seeds' best checkpoints) after {DEC / 1e6:.1f} million decisions, which corresponds "
    f"to {DEC * 0.1 / 3600:.0f} hours of simulated shaking and about {MINUTES:.0f} minutes of computation per run. Most of the "
    f"improvement occurs in the first million decisions: the mean validation J_{{T}} is {VAL[0]:.2f}, {VAL[1]:.2f}, {VAL[2]:.2f} "
    f"and {VAL[3]:.2f} after 100, 200, 300 and 400 iterations, so the curves are nearly flat over the last quarter of the budget. "
    f"The configuration without domain randomisation ends slightly higher on this (randomised) validation set, and the "
    f"symmetric-critic configuration learns more slowly and has by far the widest spread between seeds. Section 6.8 quantifies "
    f"what the longer training buys on the test scenarios.")

P.H("6.3 Nominal plant", 2)
rows = []
for g in ORDER:
    d = T["nominal"][g]
    rows.append([g] + [f"{d[l]['J1']:.2f} / {d[l]['J2']:.2f} / {d[l]['JT']:.2f}" for l in ("0.035", "0.07", "0.14")] + [f2(nomJT[g])])
P.TAB("Nominal plant, eleven recorded motions per PGA level: mean J_{1} / J_{2} / J_{T}, and mean J_{T} over the three levels. "
      f"PI-DRL: mean of {NSW} seeds.",
      ["Controller", "PGA 0.035 g", "PGA 0.07 g", "PGA 0.14 g", "J_{T}, all"], rows, widths=[2.4, 3.6, 3.6, 3.6, 2.0],
      bold_rows=(ORDER.index("PI-DRL"),))
t_nom = P.ntab
nm = T["nominal"]
assert nm["CO-NOM"]["0.14"]["JT"] < min(nm["PI-DRL"]["0.14"]["JT"], nm["AS"]["0.14"]["JT"])
assert abs(nm["CO-NOM"]["0.07"]["JT"] - nm["PI-DRL"]["0.07"]["JT"]) < 0.02 and nm["PI-DRL"]["0.035"]["JT"] < nm["AS"]["0.035"]["JT"] < nm["CO-NOM"]["0.035"]["JT"]
assert nomJT["PI-DRL"] < nomJT["AS"] < nomJT["CO-NOM"] < nomJT["CO-DC"] < nomJT["CO-ROB"]
P.P(f"On the nominal plant (Table {P.ntab}) the picture is the classical one. Among constant voltages, passive-off is best at "
    f"0.035 g and passive-on at 0.14 g; any fixed setting is a compromise. Three adaptive controllers, PI-DRL, AS and CO-NOM, are close together. "
    f"The nominally tuned clipped-optimal controller has the lowest J_{{T}} at 0.14 g ({nm['CO-NOM']['0.14']['JT']:.2f}) and ties "
    f"with the learned controller at 0.07 g ({nm['CO-NOM']['0.07']['JT']:.2f} against {nm['PI-DRL']['0.07']['JT']:.2f}), but at "
    f"0.035 g its switching raises the peak acceleration (J_{{2}} = {nm['CO-NOM']['0.035']['J2']:.2f}). The learned controller "
    f"has the lowest overall mean ({f2(nomJT['PI-DRL'])}, against {f2(nomJT['AS'])} for the amplitude-scheduled law and "
    f"{f2(nomJT['CO-NOM'])} for CO-NOM), owing to its advantage at the lowest intensity. With eleven records per level, "
    f"differences of a few hundredths are within the noise; the learned controller is on a par with, not clearly better than, a "
    f"well-tuned model-based controller when the model is exact and there is no delay. The delay-compensated controller is "
    f"designed for a delay of {2 * d_dc} ms that is absent in this set and pays for it: its mean J_{{T}} of {f2(nomJT['CO-DC'])} lies "
    f"between CO-NOM and CO-ROB ({f2(nomJT['CO-ROB'])}).")

P.H("6.4 Monte Carlo evaluation under uncertainty", 2)
def mc_table(kind, cap):
    rows = []
    for g in ORDER:
        s = T[kind][g]
        rows.append([g, f3(s["J1"]["mean"]), f3(s["J2"]["mean"]), f3(s["J3"]["mean"]), f3(s["J4"]["mean"]),
                     f"{s['JT']['mean']:.2f} [{s['JT']['ci'][0]:.2f}, {s['JT']['ci'][1]:.2f}]", f2(s["JT"]["median"]), f2(s["JT"]["p95"]),
                     f2(s["JT"]["cvar90"]), f2(s["J1"]["cvar90"]), f2(s["J2"]["cvar90"])])
    P.TAB(cap, ["Controller", "J_{1}", "J_{2}", "J_{3}", "J_{4}", "J_{T} mean [95% CI]", "J_{T} median", "J_{T} P95", "J_{T} CVaR_{90}",
                "J_{1} CVaR_{90}", "J_{2} CVaR_{90}"], rows, widths=[1.8, 1.05, 1.05, 1.05, 1.05, 2.9, 1.2, 1.1, 1.3, 1.35, 1.35], size=7.6,
          bold_rows=(ORDER.index("PI-DRL"),))
mc_table("mc_in", f"In-distribution Monte Carlo ({N_IN} scenarios on recorded motions). Means unless stated. PI-DRL: scenario "
         f"values averaged over {NSW} seeds; standard deviation of the mean J_{{T}} across seeds: {sd_in:.3f}.")
t_in = P.ntab
mc_table("mc_ood", f"Out-of-distribution Monte Carlo ({N_OOD} scenarios, ranges widened by 1.6). PI-DRL: standard deviation of the "
         f"mean J_{{T}} across seeds: {sd_ood:.3f}.")
t_ood = P.ntab
P.FIG(os.path.join(FIG, "fig_cdf.png"), "Empirical distributions of J_{1}, J_{2} and J_{T} over the Monte Carlo scenarios "
      "(top: in-distribution; bottom: out-of-distribution). PI-DRL pools the " + NSW + " seeds. J_{2} > 1 means that the controller "
      "increases the peak floor acceleration relative to the uncontrolled structure.", 15.5)
f_cdf = P.nfig
pin = T["mc_in_paired"]; pood = T["mc_ood_paired"]
rows = []
for b in ["P-OFF", "P-ON", "P-BEST", "P-BEST2", "AS", "CO-NOM", "CO-ROB", "CO-DC"]:
    a, o = pin[b]["JT"], pood[b]["JT"]
    rows.append([b, f"{a['diff']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}]", pc1(a["rel"]), pc(a["win"]), pv(a["p"]), pv(ST_in[b]["p"]),
                 f"{o['diff']:+.3f} [{o['ci'][0]:+.3f}, {o['ci'][1]:+.3f}]", pc1(o["rel"]), pc(o["win"]), pv(o["p"]), pv(ST_ood[b]["p"])])
P.TAB(f"Paired comparison of PI-DRL (average of {NSW} seeds) with each baseline: mean difference in J_{{T}} with bootstrap 95% CI, "
      "relative difference of the means, share of scenarios in which PI-DRL is better, Wilcoxon signed-rank p-value over "
      "scenarios (p), and p-value of the one-sample t-test over the per-seed mean differences (p_{seed}). "
      "Negative differences favour PI-DRL.",
      ["Baseline", "In-dist. ΔJ_{T} [CI]", "Rel.", "Better in", "p", "p_{seed}", "OOD ΔJ_{T} [CI]", "Rel.", "Better in", "p", "p_{seed}"], rows,
      widths=[1.6, 3.2, 1.1, 1.1, 1.05, 1.05, 3.2, 1.1, 1.1, 1.05, 1.05], size=7.6)
t_pair = P.ntab
P.P(f"Tables {t_in}–{t_pair} and Figure {f_cdf} summarise the main experiment. Under in-distribution uncertainty the learned "
    f"controller reduces the mean peak drift to {pc(M('mc_in', 'PI-DRL', 'J1'))} and the mean peak acceleration to "
    f"{pc(M('mc_in', 'PI-DRL', 'J2'))} of the uncontrolled values. Its mean J_{{T}} of {f2(d_in)} is {pc(-rel('mc_in', 'P-BEST'))} "
    f"lower than that of the best constant voltage, {pc(-rel('mc_in', 'P-BEST2'))} lower than that of the best pair of constant "
    f"voltages, {pc(-rel('mc_in', 'CO-ROB'))} lower than that of the robustly tuned clipped-optimal controller and "
    f"{pc(-rel('mc_in', 'CO-DC'))} lower than that of the delay-compensated one; the paired "
    f"intervals exclude zero by a wide margin, and the learned controller is better in "
    f"{100 * min(pin[b]['JT']['win'] for b in ('P-BEST', 'P-BEST2', 'CO-ROB', 'CO-DC')):.0f}–"
    f"{100 * max(pin[b]['JT']['win'] for b in ('P-BEST', 'P-BEST2', 'CO-ROB', 'CO-DC')):.0f}% of the scenarios. These four "
    f"differences are also significant when the training seed is taken as the unit (p_{{seed}} < 0.001 in Table {t_pair}). The "
    f"advantage extends to the tails: CVaR_{{90}} of J_{{T}} is {f2(M('mc_in', 'PI-DRL', 'JT', 'cvar90'))} against "
    f"{f2(M('mc_in', 'P-BEST2', 'JT', 'cvar90'))} for P-BEST2, {f2(M('mc_in', 'CO-ROB', 'JT', 'cvar90'))} for CO-ROB and "
    f"{f2(M('mc_in', 'CO-DC', 'JT', 'cvar90'))} for CO-DC.")
assert all(ST_in[b]["p"] < 1e-3 and ST_ood[b]["p"] < 1e-3 for b in ("P-BEST", "P-BEST2", "CO-ROB", "CO-DC"))
ps = T["mc_in_psweep"]
P.P(f"As a guard against an unlucky choice of the constant voltage, all seven constant voltages were also run on the test sets. "
    f"The best of them *in hindsight* attains J_{{T}} = {min(v['JT'] for v in ps.values()):.2f} in distribution and "
    f"{min(v['JT'] for v in T['mc_ood_psweep'].values()):.2f} out of distribution, essentially the P-BEST values, so the conclusion "
    f"does not depend on that choice.")
P.P(f"The nominally tuned clipped-optimal controller, among the best on the nominal plant, is the worst under uncertainty: its mean "
    f"peak acceleration exceeds the uncontrolled one (J_{{2}} = {f2(M('mc_in', 'CO-NOM', 'J2'))}), and in the worst tenth of the "
    f"scenarios it roughly triples it (CVaR_{{90}} of J_{{2}} = {f2(M('mc_in', 'CO-NOM', 'J2', 'cvar90'))}), although its drift "
    f"remains among the lowest. Section 6.5 traces this to the command delay. The robust tuning removes the failure but, by nearly switching "
    f"the dampers off, ends up no better than passive-off.")
bd = CD["by_delay"]
assert M("mc_in", "CO-DC", "JT") < M("mc_in", "CO-ROB", "JT") and M("mc_ood", "CO-DC", "JT") < M("mc_ood", "CO-ROB", "JT")
assert M("mc_in", "CO-DC", "JT", "cvar90") > M("mc_in", "CO-ROB", "JT", "cvar90") and M("mc_ood", "CO-DC", "JT", "cvar90") > M("mc_ood", "CO-ROB", "JT", "cvar90")
assert bd["CO-DC"]["0"]["JT"] > bd["CO-NOM"]["0"]["JT"] and bd["CO-DC"]["1"]["JT"] < bd["CO-ROB"]["1"]["JT"] and bd["CO-DC"]["2"]["JT"] < bd["CO-ROB"]["2"]["JT"]
P.P(f"**Delay compensation recovers only part of the loss.** The delay-compensated controller CO-DC attains a mean J_{{T}} of "
    f"{f2(M('mc_in', 'CO-DC', 'JT'))} in distribution and {f2(M('mc_ood', 'CO-DC', 'JT'))} out of distribution. This is "
    f"{100 * (1 - M('mc_in', 'CO-DC', 'JT') / M('mc_in', 'CO-ROB', 'JT')):.0f}% and "
    f"{100 * (1 - M('mc_ood', 'CO-DC', 'JT') / M('mc_ood', 'CO-ROB', 'JT')):.0f}% below CO-ROB and about half of the CO-NOM "
    f"values. The improvement is in drift (J_{{1}} = {f2(M('mc_in', 'CO-DC', 'J1'))} against {f2(M('mc_in', 'CO-ROB', 'J1'))} for "
    f"CO-ROB), which the smaller control weight makes possible. The accelerations remain high (J_{{2}} = "
    f"{f2(M('mc_in', 'CO-DC', 'J2'))}, CVaR_{{90}} of J_{{2}} = {f2(M('mc_in', 'CO-DC', 'J2', 'cvar90'))}), so that the tail of "
    f"J_{{T}} is somewhat worse than that of CO-ROB (CVaR_{{90}} = {f2(M('mc_in', 'CO-DC', 'JT', 'cvar90'))} against "
    f"{f2(M('mc_in', 'CO-ROB', 'JT', 'cvar90'))} in distribution, {f2(M('mc_ood', 'CO-DC', 'JT', 'cvar90'))} against "
    f"{f2(M('mc_ood', 'CO-ROB', 'JT', 'cvar90'))} out of distribution). Grouping the in-distribution scenarios by the true delay of "
    f"the plant shows the trade-off of a fixed design delay: for plants without delay CO-DC attains "
    f"{f2(bd['CO-DC']['0']['JT'])} against {f2(bd['CO-NOM']['0']['JT'])} for CO-NOM; with 10 ms, {f2(bd['CO-DC']['1']['JT'])} "
    f"against {f2(bd['CO-NOM']['1']['JT'])}; with 20 ms, {f2(bd['CO-DC']['2']['JT'])} against {f2(bd['CO-NOM']['2']['JT'])}. "
    f"The learned controller attains {f2(bd['PI-DRL']['0']['JT'])}, {f2(bd['PI-DRL']['1']['JT'])} and "
    f"{f2(bd['PI-DRL']['2']['JT'])} in the three groups. Even an oracle version of CO-DC that is given the true delay of every "
    f"plant only reaches {f2(CD['CO-DCo']['mc_in']['JT'])}. The weakness of clipped-optimal control under uncertainty in this "
    f"study is therefore only partly a matter of missing delay compensation: the predictor inherits the parameter errors of the "
    f"nominal model and the error of the state estimate, and fast switching on an imperfect prediction still excites the "
    f"accelerations.")
P.P(f"Out of distribution (Table {t_ood}) all controllers lose a little and the ranking is unchanged. The learned controller's "
    f"mean J_{{T}} rises from {f2(d_in)} to {f2(d_ood)} and its margins over P-BEST, P-BEST2, CO-ROB and CO-DC remain "
    f"{pc(-rel('mc_ood', 'P-BEST'))}, {pc(-rel('mc_ood', 'P-BEST2'))}, {pc(-rel('mc_ood', 'CO-ROB'))} and "
    f"{pc(-rel('mc_ood', 'CO-DC'))}.")
si = T["mc_in_seeds"]
_as_verdict = {
    "ahead": f"The training method therefore yields policies that are on average slightly better than AS, by "
             f"{abs(100 * as_in['mean'] / M('mc_in', 'AS', 'JT')):.1f}%, and the difference is statistically detectable; it has no "
             f"practical weight.",
    "tie": f"At the seed level the difference from AS is therefore not distinguishable from zero: the training method produces "
           f"policies on a par with the two-parameter law, some marginally better and some marginally worse.",
    "behind": f"The training method therefore yields policies that are on average slightly worse than AS."}[AS_STATUS]
_drift = (f", while AS retains a {100 * pin['AS']['J1']['rel']:.1f}% lower peak drift" if pin['AS']['J1']['rel'] > 0 else "")
P.P(f"**The amplitude-scheduled law is practically as good as the learned controller.** AS attains J_{{T}} = "
    f"{f2(M('mc_in', 'AS', 'JT'))} in distribution and {f2(M('mc_ood', 'AS', 'JT'))} out of distribution, against {f2(d_in)} and "
    f"{f2(d_ood)} for the {NSW}-seed average of PI-DRL. The scenario-paired mean difference is {pin['AS']['JT']['diff']:+.3f} "
    f"[{pin['AS']['JT']['ci'][0]:+.3f}, {pin['AS']['JT']['ci'][1]:+.3f}] in distribution and {pood['AS']['JT']['diff']:+.3f} "
    f"[{pood['AS']['JT']['ci'][0]:+.3f}, {pood['AS']['JT']['ci'][1]:+.3f}] out of distribution, i.e. "
    f"{pc1(relAS)} and {pc1(pood['AS']['JT']['rel'])}; the learned controller is better in {pc(pin['AS']['JT']['win'])} of the "
    f"in-distribution scenarios (signed-rank p = {pv(pin['AS']['JT']['p'])}). These scenario-level statistics treat the {NSW} "
    f"trained policies as fixed. With the seed as the unit, the per-seed mean differences from AS range from "
    f"{min(sAS):+.3f} to {max(sAS):+.3f}: {as_in['better']} of the {NS} policies have a lower mean J_{{T}} than AS, "
    f"{as_in['sig_better']} of them significantly so by their own paired interval, and {as_in['sig_worse']} "
    f"{'is' if as_in['sig_worse'] == 1 else 'are'} significantly worse. The mean seed-level difference is {as_in['mean']:+.3f} "
    f"with a 95% interval of [{as_in['ci'][0]:+.3f}, {as_in['ci'][1]:+.3f}] ({sig(as_in['p'])}) in distribution and "
    f"{as_ood['mean']:+.3f} [{as_ood['ci'][0]:+.3f}, {as_ood['ci'][1]:+.3f}] ({sig(as_ood['p'])}) out of distribution. "
    f"{_as_verdict} Where the learned controller gains, it is in the tail (CVaR_{{90}} of "
    f"J_{{T}}: {f2(M('mc_in', 'PI-DRL', 'JT', 'cvar90'))} against {f2(M('mc_in', 'AS', 'JT', 'cvar90'))}) and in acceleration "
    f"(J_{{2}} {abs(100 * pin['AS']['J2']['rel']):.1f}% lower){_drift}. We conclude that, in this problem, DRL does not "
    f"improve in any practically relevant way on a two-parameter heuristic that schedules the voltage on the measured stroke "
    f"amplitude.")
assert M('mc_in', 'PI-DRL', 'JT', 'cvar90') < M('mc_in', 'AS', 'JT', 'cvar90')

P.H("6.5 Sensitivity to individual uncertainties", 2)
P.FIG(os.path.join(FIG, "fig_sweeps.png"), "One-at-a-time sweeps on the otherwise nominal plant (PGA 0.07 g, eleven records per "
      "point). Shaded: range covered by training randomisation. PI-DRL: mean and range over " + NSW + " seeds.", 16.0)
sw = T["sweeps"]
P.P(f"Figure {P.nfig} varies one factor at a time. *Command delay* is the decisive factor for clipped-optimal control: with "
    f"10 ms of delay the mean J_{{2}} of CO-NOM jumps from {sw['sw_delay']['CO-NOM']['J2'][0]:.2f} to "
    f"{sw['sw_delay']['CO-NOM']['J2'][1]:.2f} and J_{{T}} from {sw['sw_delay']['CO-NOM']['JT'][0]:.2f} to "
    f"{sw['sw_delay']['CO-NOM']['JT'][1]:.2f}. The bang-bang rule then switches out of phase with the fast stick–slip dynamics "
    f"of the dampers and injects high-frequency acceleration. The learned and scheduled controllers, which change the voltage "
    f"slowly, are far less sensitive. Over delays up to 50 ms, well beyond the 20 ms seen in training, J_{{T}} of PI-DRL rises "
    f"gradually from {sw['sw_delay']['PI-DRL']['JT'][0]:.2f} to {sw['sw_delay']['PI-DRL']['JT'][-1]:.2f}, whereas that of AS stays "
    f"at {min(sw['sw_delay']['AS']['JT']):.2f}–{max(sw['sw_delay']['AS']['JT']):.2f}, so that AS is slightly better than the "
    f"learned controller for delays of 20 ms and more. *Sensor noise* up to four "
    f"times the reference level has little effect: J_{{T}} of CO-NOM rises from {sw['sw_noise']['CO-NOM']['JT'][0]:.2f} to "
    f"{sw['sw_noise']['CO-NOM']['JT'][-1]:.2f}, CO-ROB improves slightly, and the other controllers change by 0.02 or less. "
    f"*Stiffness* errors affect all controllers in a similar "
    f"way through the change of the structure itself; CO-NOM additionally loses acceleration performance when the structure "
    f"differs from its model, most strongly when it is softer. *Damper capacity:* with a weaker damper all controllers lose drift "
    f"performance; here CO-NOM, evaluated without delay, is the best at reduced capacity, and the learned controller is "
    f"better than AS at 40% capacity "
    f"({sw['sw_cap']['PI-DRL']['JT'][0]:.2f} against {sw['sw_cap']['AS']['JT'][0]:.2f}).")
dl = [int(round(10 * v)) for v in sw["sw_delay"]["val"]]
dsw = {"CO-NOM": CD["CO-NOM"]["sw_delay"], "CO-ROB": CD["CO-ROB"]["sw_delay"], f"CO-DC (d_{{c}} = {2 * d_dc} ms)": CD["CO-DC"]["sw_delay"],
       "CO-DC, oracle delay": CD["CO-DCo"]["sw_delay"], "AS": CD["AS"]["sw_delay"], "PI-DRL": sw["sw_delay"]["PI-DRL"]}
rows = [[g] + [f"{v['JT'][i]:.2f} ({v['J2'][i]:.2f})" for i in range(len(dl))] for g, v in dsw.items()]
P.TAB("Command-delay sweep on the otherwise nominal plant (PGA 0.07 g, eleven records per delay): mean J_{T} (mean J_{2}). "
      "Training and tuning covered delays of 0–20 ms. The oracle variant of CO-DC is given the true delay and is not realisable.",
      ["Controller"] + [f"{d} ms" for d in dl], rows, widths=[3.8] + [2.0] * len(dl), size=8.5, bold_rows=(len(rows) - 1,))
t_delay = P.ntab
dc_, dco_ = CD["CO-DC"]["sw_delay"]["JT"], CD["CO-DCo"]["sw_delay"]["JT"]
assert max(dc_[:5]) < 1.70 and dc_[0] > CD["CO-NOM"]["sw_delay"]["JT"][0] + 0.1 and all(dc_[i] < CD["CO-ROB"]["sw_delay"]["JT"][i] for i in range(len(dl)))
assert all(dc_[i] > max(CD["AS"]["sw_delay"]["JT"][i], sw["sw_delay"]["PI-DRL"]["JT"][i]) + 0.05 for i in range(len(dl)))
g10, g20 = DG["10ms"], DG["20ms"]
P.P(f"*Delay compensation.* Table {P.ntab} isolates the effect of the predictor. CO-DC keeps J_{{T}} within "
    f"{min(dc_[:5]):.2f}–{max(dc_[:5]):.2f} for delays from 0 to 40 ms and reaches {dc_[5]:.2f} at 50 ms, whereas CO-NOM "
    f"exceeds 2.5 for every non-zero delay. The predictor thus removes most of the delay sensitivity of clipped-optimal control "
    f"on the nominal plant, and it does so with a control weight ten times smaller than that of CO-ROB, which it outperforms at "
    f"every delay. The price is paid when there is no delay ({dc_[0]:.2f} against {CD['CO-NOM']['sw_delay']['JT'][0]:.2f} for "
    f"CO-NOM). Two limits are visible. First, compensation does not restore the delay-free performance: even with the true "
    f"delay the controller attains {min(dco_[1:]):.2f}–{max(dco_[1:]):.2f} for delays of 10–50 ms, against "
    f"{dco_[0]:.2f} without delay. The prediction starts from an imperfect state estimate, since the ground acceleration is not "
    f"measured: in these runs the relative RMS error of the estimated floor velocities is about "
    f"{100 * g10['kf_vel_relerr']:.0f}%, and the predicted damper force has the correct sign in "
    f"{100 * g10['force_sign_pred']:.0f}% of the strong-motion samples for a 10 ms horizon and in "
    f"{100 * g20['force_sign_pred']:.0f}% for 20 ms. Second, at every delay CO-DC remains above the slowly varying AS and "
    f"PI-DRL commands, which need no compensation at all.")

P.H("6.6 Fragility", 2)
P.FIG(os.path.join(FIG, "fig_fragility.png"), "Fragility curves under in-distribution plant uncertainty (markers: observed "
      "exceedance fractions, 44 scenarios per PGA level and controller, PI-DRL pooled over " + NSW + " seeds; lines: lognormal fits). "
      "UNC: uncontrolled.", 15.5)
fr = T["fragility"]
rows = [[g, f"{fr[g]['drift']['theta']:.3f}", f2(fr[g]["drift"]["beta"]), f"{fr[g]['acc']['theta']:.3f}", f2(fr[g]["acc"]["beta"])]
        for g in ["UNC"] + ORDER]
P.TAB("Fitted lognormal fragility parameters: median PGA θ (g) at which the limit is exceeded with 50% probability, and "
      "dispersion β.", ["Controller", "Drift > 3 mm: θ (g)", "β", "Acceleration > 2 m/s²: θ (g)", "β"], rows,
      widths=[2.6, 3.4, 1.4, 4.2, 1.4], bold_rows=(len(rows) - 1,))
P.P(f"Figure {P.nfig} and Table {P.ntab} express the results as exceedance probabilities. For the drift limit, the learned "
    f"controller, AS and passive-on raise the median capacity from {fr['UNC']['drift']['theta']:.3f} g (uncontrolled) to about "
    f"{fr['PI-DRL']['drift']['theta']:.3f} g; passive-on achieves this only at the price of poor performance at low intensity, "
    f"which a single limit state does not show. For the acceleration limit, the learned controller and AS give the highest "
    f"median capacities ({fr['PI-DRL']['acc']['theta']:.3f} and {fr['AS']['acc']['theta']:.3f} g), while CO-NOM is *worse than "
    f"no control* ({fr['CO-NOM']['acc']['theta']:.3f} g against {fr['UNC']['acc']['theta']:.3f} g). Delay compensation removes "
    f"this defect: CO-DC reaches {fr['CO-DC']['acc']['theta']:.3f} g for the acceleration limit and "
    f"{fr['CO-DC']['drift']['theta']:.3f} g for the drift limit, between CO-ROB and the leading controllers. With 44 scenarios per level "
    f"the curves are indicative; differences of a few thousandths of g between the leading controllers are not significant.")

P.H("6.7 Ablation study", 2)
rows = []
for lab in (PROP, NODR, NOOBS, SYM):
    a, o = T["mc_in_abl"][lab]["JT"], T["mc_ood_abl"][lab]["JT"]
    qi = P0_in if lab == PROP else A_in[lab]; qo = P0_ood if lab == PROP else A_ood[lab]
    rows.append([lab, str(a["nseeds"]), f"{qi['mean']:.3f} ± {qi['sd']:.3f}", f"{qi['min']:.2f}–{qi['max']:.2f}", f2(a["cvar90"]),
                 f"{qo['mean']:.3f} ± {qo['sd']:.3f}", f"{qo['min']:.2f}–{qo['max']:.2f}", f2(o["cvar90"])])
P.TAB("Ablations: mean J_{T} on the Monte Carlo test sets per training seed, summarised as mean ± sample standard deviation and "
      "range across the n seeds, and CVaR_{90} (mean over seeds).",
      ["Configuration", "n", "In-dist. mean ± s.d.", "Range", "CVaR_{90}", "OOD mean ± s.d.", "Range", "CVaR_{90}"],
      rows, widths=[4.2, 0.8, 2.5, 1.9, 1.3, 2.5, 1.9, 1.3], size=7.6, bold_rows=(0,))
t_abl = P.ntab
rows = []
for lab in (NODR, NOOBS, SYM):
    r = [lab, f"{A_in[lab]['n1']} vs {A_in[lab]['n0']}"]
    for A_ in (A_in, A_ood):
        q = A_[lab]
        r += [f"{q['diff']:+.3f} [{q['welch_ci'][0]:+.3f}, {q['welch_ci'][1]:+.3f}]", pv(q["welch_p"]), pv(q["mwu_p"])]
    rows.append(r)
P.TAB("Seed-level tests of the ablations against the proposed configuration: difference of the seed means of J_{T} "
      "(ablation − proposed; positive values mean that the removed element helps) with the 95% confidence interval and p-value of "
      "Welch's t-test, and the p-value of the Mann–Whitney U test.",
      ["Configuration", "Seeds", "In-dist. Δ [95% CI]", "Welch p", "U-test p", "OOD Δ [95% CI]", "Welch p", "U-test p"],
      rows, widths=[3.8, 1.3, 3.15, 1.35, 1.2, 3.15, 1.35, 1.2], size=7.6)
t_abl2 = P.ntab
ab = T["mc_in_abl"]; abo = T["mc_ood_abl"]
sy = A_in[SYM]; sy_in = sorted(sy["seeds"]); ob = A_in[NOOBS]
_dr_verdict = {
    "established": "With the seed as the unit, the effect is thus statistically established on both test sets",
    "partial": f"With the seed as the unit, the evidence is thus borderline: the difference reaches the conventional 5% level "
               f"{'in distribution but not out of distribution' if DR_SIG_IN else 'out of distribution but not in distribution'}, "
               f"and both p-values lie close to 0.05",
    "unresolved": "With the seed as the unit, the effect is thus not statistically established on either test set"}[DR_STATUS]
dw = dr_in["drop_worst"]
assert dr_in["overlap"] and dr_in["sd"] > 1.5 * P0_in["sd"] and dw["diff"] > 0
_overlap = (f"The seed ranges overlap: {dr_in['n_worse_than_all']} of the {NS_NODR} policies trained without "
            f"randomisation are worse than every randomised policy, and {dr_in['n_better_than_median']} "
            f"{'is' if dr_in['n_better_than_median'] == 1 else 'are'} better than the median randomised one, so an individual "
            f"training run without randomisation can turn out as good as one with it. The spread across seeds is larger without "
            f"randomisation, {dr_in['sd']:.3f} against {P0_in['sd']:.3f}, but this is largely due to one poor run (seed "
            f"{dw['seed']}, J_{{T}} = {dw['value']:.3f}) and the difference in variance is "
            f"{'significant' if dr_in['bf_p'] < 0.05 else 'not significant'} (Brown–Forsythe {sig(dr_in['bf_p'])}). "
            f"The comparison does not hinge on that run alone, but it weakens without it: excluding it, the difference is "
            f"{dw['diff']:+.3f} (Welch {sig(dw['welch_p'])}).")
P.P(f"Tables {t_abl} and {t_abl2} remove one element at a time. *Domain randomisation.* Policies trained on the nominal plant "
    f"only (no parameter variation, noise or delay) are compared with the proposed ones using {NSW + ' seeds per configuration' if NS == NS_NODR else NSW + ' seeds for the proposed and ' + NSW_NODR + ' for the ablated configuration'}. "
    f"They attain J_{{T}} = "
    f"{dr_in['mean']:.3f} ± {dr_in['sd']:.3f} against {P0_in['mean']:.3f} ± {P0_in['sd']:.3f} in distribution "
    f"({pc1(DR_REL_IN)}) and {dr_ood['mean']:.3f} ± {dr_ood['sd']:.3f} against {P0_ood['mean']:.3f} ± {P0_ood['sd']:.3f} out of "
    f"distribution ({pc1(DR_REL_OOD)}). The difference of the seed means is {dr_in['diff']:+.3f} with a 95% interval of "
    f"[{dr_in['welch_ci'][0]:+.3f}, {dr_in['welch_ci'][1]:+.3f}] in distribution (Welch {sig(dr_in['welch_p'])}; Mann–Whitney "
    f"{sig(dr_in['mwu_p'])}) and {dr_ood['diff']:+.3f} [{dr_ood['welch_ci'][0]:+.3f}, {dr_ood['welch_ci'][1]:+.3f}] out of "
    f"distribution (Welch {sig(dr_ood['welch_p'])}; Mann–Whitney {sig(dr_ood['mwu_p'])}). {_dr_verdict}. Its size is "
    f"{abs(100 * DR_REL_IN):.1f}% in distribution, with an interval from "
    f"{100 * dr_in['welch_ci'][0] / P0_in['mean']:.1f}% to {100 * dr_in['welch_ci'][1] / P0_in['mean']:.1f}%, and "
    f"{abs(100 * DR_REL_OOD):.1f}% out of distribution: the relative gain is "
    f"{'no larger' if DR_REL_OOD <= DR_REL_IN * 1.25 else 'larger'} outside the training ranges than inside. {_overlap} "
    f"With the first three seeds alone, as in the earlier version of this study, the same comparison gave "
    f"{dr_in['first3']['diff']:+.3f} with Welch {sig(dr_in['first3']['welch_p'])}. Two caveats apply: the ablated policies were "
    f"still trained on randomised ground motions and intensities, and their checkpoints were selected on the randomised "
    f"validation set, which favours the ablation.")
_ob_null = ob["welch_ci"][0] < 0 < ob["welch_ci"][1] and A_ood[NOOBS]["welch_ci"][0] < 0 < A_ood[NOOBS]["welch_ci"][1]
assert _ob_null, "observer ablation is no longer a null result: rewrite the paragraph"
_sy_sig = sy["welch_p"] < 0.05
P.P(f"*Physics-based observer.* Replacing the Kalman estimate by the raw measurement history alone ({NSW_OBS} seeds) changes "
    f"J_{{T}} by {ob['diff']:+.3f} in distribution and {A_ood[NOOBS]['diff']:+.3f} out of distribution; both intervals include "
    f"zero (Welch {sig(ob['welch_p'])} and {sig(A_ood[NOOBS]['welch_p'])}). No benefit of the model-based observer can be "
    f"resolved; given the interval, a benefit larger than about {100 * max(ob['welch_ci'][1], 0) / P0_in['mean']:.0f}% is "
    f"unlikely. *Privileged critic.* A critic that sees only the actor's observation ({NSW_SYM} seeds) gives the worst mean on "
    f"both sets ({pc1(sy['diff'] / P0_in['mean'])} in distribution), the slowest learning in Figure {f_learn} and by far the "
    f"largest spread between seeds, {sy['sd']:.3f} against {P0_in['sd']:.3f}: the seed means range from {sy_in[0]:.2f} to "
    f"{sy_in[-1]:.2f}, and {sy['n_worse_than_all']} of the {NS_SYM} runs are worse than every run of the proposed configuration. "
    + (f"The difference in means is significant (Welch {sig(sy['welch_p'])}; Mann–Whitney {sig(sy['mwu_p'])})."
       if _sy_sig else
       f"Because of this spread the difference in means is not significant in Welch's test ({sig(sy['welch_p'])}; "
       f"Mann–Whitney {sig(sy['mwu_p'])}); what the data show is a loss of reliability, not a shift of the best achievable result.")
    + (f" The privileged critic thus improves the mean by about {100 * SY_REL:.0f}% and, above all, the reliability of training: "
       f"without it a run can still reach the level of the proposed configuration ({sy_in[0]:.2f}), but most runs do not."
       if _sy_sig else " The privileged critic thus mainly makes training faster and more reliable."))
P.P({"established": "In short, the ablations establish a small benefit of domain randomisation, ",
     "partial": "In short, the ablations indicate a small benefit of domain randomisation with borderline statistical support, ",
     "unresolved": "In short, the ablations do not establish a benefit of domain randomisation, they show "}[DR_STATUS]
    + ("a benefit of the privileged critic that is nominally significant in the mean but uncorrected for the three comparisons, and a "
     "clear benefit in reliability, " if SY_SIG else "a reliability benefit of the privileged critic ")
    + "and no benefit of the model-based observer. A plausible explanation is "
    "given in Section 7.")

# ---------------------------------------------------------------- 6.8 training budget
P.H("6.8 Effect of the training budget", 2)
rows = []
for lab in (PROP, NODR, NOOBS, SYM):
    r = [lab]
    for kind in ("mc_in", "mc_ood"):
        b = BU[lab][kind]
        r += [f"{b['b100']['mean']:.3f} ± {b['b100']['sd']:.3f}", f"{b['now']['mean']:.3f} ± {b['now']['sd']:.3f}",
              f"{b['paired']['diff']:+.3f} [{b['paired']['ci'][0]:+.3f}, {b['paired']['ci'][1]:+.3f}]"]
    rows.append(r)
P.TAB(f"Effect of the training budget on test performance for the three seeds (0–2) trained at both budgets: J_{{T}} (mean ± "
      f"sample standard deviation across these seeds) after "
      f"{ITERS0} iterations ({DEC0 / 1e6:.2f} million decisions) and after {ITERS} iterations ({DEC / 1e6:.2f} million decisions), "
      f"and scenario-paired difference of the seed averages with bootstrap 95% CI. Negative differences favour the larger budget.",
      ["Configuration", f"In-dist., {ITERS0} it.", f"In-dist., {ITERS} it.", "Δ [CI]", f"OOD, {ITERS0} it.", f"OOD, {ITERS} it.", "Δ [CI]"],
      rows, widths=[3.2, 1.9, 1.9, 2.95, 1.9, 1.9, 2.95], size=7.6, bold_rows=(0,))
b_in, b_ood = bu["mc_in"], bu["mc_ood"]
s0, s1 = b_in["b100"]["seeds"], b_in["now"]["seeds"]
lag = int(np.argmax(s0)); oth = [i for i in range(3) if i != lag]
assert all(abs(s1[i] - s0[i]) < 0.01 for i in oth) and s0[lag] - s1[lag] > 0.05
gains = {g: -BU[g]["mc_in"]["paired"]["diff"] / BU[g]["mc_in"]["b100"]["mean"] for g in (NODR, NOOBS, SYM)}
P.P(f"A first version of this study trained every policy for {ITERS0} iterations ({DEC0 / 1e6:.2f} million decisions). The "
    f"validation curves were then still decreasing, and it was unclear whether the comparison with AS and the ablation results "
    f"were artefacts of a short budget. All twelve policies were therefore retrained from scratch for {ITERS} iterations with "
    f"otherwise identical settings (the linear learning-rate decay is stretched over the longer run), and all results in the "
    f"other sections refer to the longer budget. The {NRUNS - 12} further policies with new seeds were trained at the longer "
    f"budget only and do not enter this comparison, so the {ITERS}-iteration columns of Table {P.ntab} are three-seed values and "
    f"differ slightly from the {NSW}-seed values of Table {t_abl}. Table {P.ntab} compares the two budgets on the same test "
    f"scenarios.")
P.P(f"For the proposed configuration the fourfold budget lowers the mean J_{{T}} from {b_in['b100']['mean']:.3f} to "
    f"{b_in['now']['mean']:.3f} in distribution ({pc1(b_in['paired']['diff'] / b_in['b100']['mean'])}) and from "
    f"{b_ood['b100']['mean']:.3f} to {b_ood['now']['mean']:.3f} out of distribution "
    f"({pc1(b_ood['paired']['diff'] / b_ood['b100']['mean'])}). The gain comes almost entirely from the seed that lagged at the "
    f"smaller budget ({s0[lag]:.2f} → {s1[lag]:.2f}); the other two seeds are unchanged ({s0[oth[0]]:.2f} → {s1[oth[0]]:.2f} and "
    f"{s0[oth[1]]:.2f} → {s1[oth[1]]:.2f}). Accordingly the standard deviation across seeds falls from {b_in['b100']['sd']:.3f} to "
    f"{b_in['now']['sd']:.3f}. The ablated configurations gain {100 * min(gains.values()):.0f}–{100 * max(gains.values()):.0f}% in "
    f"distribution, the symmetric-critic configuration the most.")
P.P(f"For these three seeds, three conclusions changed with the budget. (i) Relative to AS, the learned controller moved from "
    f"{pc1(relAS0)} (slightly worse) to {pc1(relAS3)} (slightly better). (ii) The benefit of domain randomisation, which was hidden in the seed spread at "
    f"{ITERS0} iterations ({BU[NODR]['mc_in']['b100']['mean']:.3f} ± {BU[NODR]['mc_in']['b100']['sd']:.3f} against "
    f"{b_in['b100']['mean']:.3f} ± {b_in['b100']['sd']:.3f}), appeared to become visible, with seed ranges that no longer overlapped; Section 6.7 re-examines it with more seeds. (iii) The symmetric critic, which was worse in the mean with "
    f"little spread at {ITERS0} iterations ({BU[SYM]['mc_in']['b100']['mean']:.3f} ± {BU[SYM]['mc_in']['b100']['sd']:.3f}), reached "
    f"the level of the proposed configuration in one of three seeds. One conclusion did not change: the model-based observer shows no "
    f"benefit. Since the validation curves are flat and the two best seeds did not improve, we do not expect that still longer "
    f"training of this architecture at this decision rate would alter the picture, although this is an extrapolation.")

P.H("6.9 What the policy learned", 2)
P.FIG(os.path.join(FIG, "fig_intensity.png"), "In-distribution Monte Carlo results binned by PGA: mean J_{1}, J_{2}, J_{T} and "
      "mean command voltage.", 16.0)
f_int = P.nfig
bp = T["by_pga"]
P.FIG(os.path.join(FIG, "fig_timehist.png"), "El Centro record scaled to 0.07 g, nominal plant: roof displacement, roof "
      "acceleration, command voltages and storey-1 damper force (PI-DRL: seed 0).", 15.0)
f_th = P.nfig
ec = T["elcentro"]
_ah = [bp['PI-DRL']['JT'][i] < bp['AS']['JT'][i] for i in range(6)]; KA = sum(_ah)
assert _ah == [True] * KA + [False] * (6 - KA) and 2 <= KA <= 5, _ah
P.P(f"Figure {f_int} shows why constant voltages lose. The mean J_{{T}} of P-BEST is {bp['P-BEST']['JT'][0]:.2f} in the lowest PGA "
    f"bin and {bp['P-BEST']['JT'][-1]:.2f} in the highest, with a minimum of {min(bp['P-BEST']['JT']):.2f} in between, whereas the "
    f"learned controller and AS stay at or below {max(bp['PI-DRL']['JT'] + bp['AS']['JT']):.2f} in every bin. The learned policy raises its mean voltage from "
    f"{bp['PI-DRL']['vmean'][0]:.2f} V_{{max}} in the lowest bin to {bp['PI-DRL']['vmean'][-1]:.2f} V_{{max}} in the highest, "
    f"and AS from {bp['AS']['vmean'][0]:.2f} to {bp['AS']['vmean'][-1]:.2f} V_{{max}}. Both implement intensity-adaptive "
    f"damping; the policy inferred it from data, the heuristic has it built in. The learned controller is ahead of AS in the "
    f"{WORD[KA]} lower bins ({bp['PI-DRL']['JT'][0]:.2f} against {bp['AS']['JT'][0]:.2f} in the lowest) and behind it in the "
    f"{WORD[6 - KA] if 6 - KA > 1 else 'highest one'}{' highest' if 6 - KA > 1 else ''} "
    f"({bp['PI-DRL']['JT'][-1]:.2f} against {bp['AS']['JT'][-1]:.2f} in the highest).")
P.P(f"The time histories in Figure {f_th} show how the two laws differ within a record. The AS voltages follow the envelope of "
    f"the response on a time scale of seconds. The learned policy raises the voltage when the strong motion arrives and then "
    f"holds it, with fluctuations from decision to decision, at roughly 3–4.5 V on the first storey and 1–3 V on the "
    f"second for the rest of the record; its mean command is {ec['PI-DRL']['vmean']:.2f} V_{{max}} against "
    f"{ec['AS']['vmean']:.2f} V_{{max}} for AS. In this example the learned adaptation is thus mainly across intensity levels and "
    f"much less within a record. The clipped-optimal controller instead switches between 0 and V_{{max}} hundreds of times. On this "
    f"particular noise-free, delay-free case CO-NOM is clearly the best (J_{{T}} = {ec['CO-NOM']['JT']:.2f} against "
    f"{ec['AS']['JT']:.2f} for AS and {ec['PI-DRL']['JT']:.2f} for the learned policy), which illustrates what fast dissipative "
    f"switching can gain when its assumptions hold, and what the slow policy leaves unexploited. CO-DC, whose predictor "
    f"assumes a delay of {2 * d_dc} ms that is absent here, attains {ec['CO-DC']['JT']:.2f}.")
assert ec['CO-NOM']['JT'] < ec['CO-DC']['JT'] < min(ec['AS']['JT'], ec['PI-DRL']['JT'])

# ================================================================================================ 7 discussion
P.H("7. Discussion", 1)
rels = sorted(-100 * rel('mc_in', b) for b in ('P-BEST', 'P-BEST2', 'CO-ROB', 'CO-DC'))
HW = 100 * 0.5 * (dr_in["welch_ci"][1] - dr_in["welch_ci"][0]) / P0_in["mean"]         # resolution of the seed-level comparison, %
P.P("**What DRL achieved.** Trained only on synthetic motions and randomised models, with realistic measurements, the policy "
    "transferred to recorded motions and to plants outside the training ranges without failure. It was "
    f"{rels[0]:.0f}–{rels[-1]:.0f}% better in the "
    "composite objective than the best constant voltages and the robustly tuned and delay-compensated clipped-optimal "
    "controllers, in the mean and in the "
    "tails, and it avoided the acceleration amplification that the nominal clipped-optimal controller shows once a small delay "
    f"is present. The {NSW} independently trained policies agree closely: the standard deviation of their mean J_{{T}} is "
    f"{P0_in['sd']:.3f}, about {100 * P0_in['sd'] / P0_in['mean']:.0f}% of the mean.")
P.P("**What it did not achieve.** It did not beat the nominal clipped-optimal controller on the nominal plant, and "
    + {"ahead": f"its margin over a two-parameter amplitude-scheduled law is about {abs(100 * as_in['mean'] / M('mc_in', 'AS', 'JT')):.0f} per cent, "
                "statistically detectable with ten seeds but of no practical weight, even after a fourfold increase of the "
                "training budget. ",
       "behind": "it did not quite match a two-parameter amplitude-scheduled law. "}[AS_STATUS]
    + "The policy discovered that the voltage should grow with the response amplitude, the dominant effect in this problem, "
      "but found little further gain. This is partly our own design: one decision every 0.1 s cannot exploit switching within a "
      "vibration cycle. Whether a faster or hierarchically structured policy could combine the nominal performance of "
      "clipped-optimal control with the robustness of slow scheduling is the obvious open question.")
P.P("**Delay compensation helps clipped-optimal control, but not enough.** The failure of CO-NOM under uncertainty is "
    "caused by the command delay, and a model-based predictor is the textbook remedy. It works on the nominal plant, where "
    "CO-DC is almost insensitive to delays up to 40 ms, and it improves on the robust tuning by "
    f"{100 * (1 - M('mc_in', 'CO-DC', 'JT') / M('mc_in', 'CO-ROB', 'JT')):.0f}% in distribution and "
    f"{100 * (1 - M('mc_ood', 'CO-DC', 'JT') / M('mc_ood', 'CO-ROB', 'JT')):.0f}% out of distribution. It does not bring "
    f"clipped-optimal control close to the slowly varying controllers, however: the learned policy and AS remain about "
    f"{pc(-rel('mc_in', 'CO-DC'))} better, and even exact knowledge of the delay leaves a gap of "
    f"{100 * (CD['CO-DCo']['mc_in']['JT'] / d_in - 1):.0f}%. A predictor is only as good as its model and its state estimate, "
    "and bang-bang switching amplifies their errors into acceleration. For this device and these uncertainties, the choice "
    "between fast switching and slow scheduling matters more than the choice between model-based and learned control.")
P.P("**Why randomisation mattered so little.** The robustness seen here comes mainly from structure and only to a small "
    "extent from training: a semi-active device can only dissipate energy, so no voltage sequence can destabilise the plant, "
    "and a voltage that varies over seconds is by construction insensitive to delays of tens of milliseconds and to moderate "
    "model errors. Randomised training added "
    f"{abs(100 * DR_REL_IN):.1f}% in distribution and {abs(100 * DR_REL_OOD):.1f}% out of distribution"
    + {"established": f", an effect that {NSW} seeds per group establish statistically but that remains small. ",
       "partial": f", an effect whose statistical support remains borderline with {NSW} seeds per group. ",
       "unresolved": f", an effect that {NSW} seeds per group do not establish statistically. "}[DR_STATUS] +
    "Domain randomisation would be expected to matter more for policies that act within the vibration cycle. Our results "
    "therefore show that it was a minor ingredient in this configuration, not that it is unimportant for semi-active control.")
_n3 = dr_in["first3"]; p3AS = float(_sst.ttest_1samp(_s3, 0.0).pvalue)
P.P("**What the additional seeds changed.** The earlier version used three seeds per configuration. Domain randomisation: "
    f"{_n3['diff']:+.3f} with non-overlapping ranges and Welch {sig(_n3['welch_p'])}, against {dr_in['diff']:+.3f} "
    f"({sig(dr_in['welch_p'])}) with {NSW} seeds, whose ranges "
    + ("overlap" if dr_in["overlap"] else "still do not overlap") + ". "
    f"The comparison with AS: {pc1(relAS3)} ({sig(p3AS)}) against {pc1(as_in['mean'] / M('mc_in', 'AS', 'JT'))} "
    f"({sig(as_in['p'])}). The privileged critic: {sy_d['first3']['diff']:+.3f} (Welch {sig(sy_d['first3']['welch_p'])}) "
    f"against {sy_d['diff']:+.3f} ({sig(sy_d['welch_p'])}) for {NSW_SYM} and {NSW} seeds. The estimates changed little and "
    "their precision improved, but the impression from three seeds that randomised and non-randomised policies are cleanly "
    f"separated did not survive: {dr_in['n_within_range']} of the {NS_NODR} ablated runs fall within the randomised range. "
    f"With {NSW} runs per group the seed-level test resolves about {HW:.1f}% in J_{{T}}; effects of one to two per cent cannot "
    "be judged from three runs, and scenario-level intervals do not substitute for replication of training.")
P.P("**More training is not a substitute for a better formulation.** Four times as many decisions bought two per cent, mostly "
    "by rescuing one slow seed. The limit reached here appears to be set by what a slowly varying voltage can do, not by "
    "optimisation; "
    "further progress would have to come from the action structure, the decision rate or the observation, not from the budget.")
P.P(f"**Implications for evaluation practice.** Four practices changed the conclusions of this study and seem worth adopting. "
    f"(i) Evaluating under uncertainty: CO-NOM goes from near the top of Table {t_nom} to the bottom of Table {t_in}. (ii) "
    f"Including simple, tuned baselines: without the AS law the gain of about 20% over the best passive setting could easily "
    f"have been credited to deep learning as such. (iii) Reporting seeds and testing at the seed level: without this, each "
    f"ablation difference could have been presented as an effect. (iv) Reporting the sensitivity to the training budget: two ablation "
    f"conclusions and the sign of the comparison with AS changed between one and four million decisions (Section 6.8).")
P.H("7.1 Limitations", 2)
P.P("*Simulation only.* No experiment was performed, and ‘out-of-distribution’ means wider parameter ranges within the same "
    "model class, not unmodelled physics; the robustness claimed in the title should be read in that sense. *Model fidelity.* "
    "The frame is linear elastic, the upper intensity range implies unrealistically large uncontrolled drifts, and the "
    "structural parameters were taken from secondary sources with one unresolved discrepancy in the floor mass (Section 3.1); "
    "the simple Bouc–Wen model ignores temperature and frequency dependence. *Scale.* One small laboratory-scale structure with "
    "two controlled storeys; the findings need not carry over to full-scale buildings, large dampers with slower dynamics or "
    "nonlinear frames. *Baselines.* The clipped-optimal controllers were tuned on coarse grids with 64 motions and the "
    "constant-voltage and AS baselines on 128 motions from a separately generated set. CO-DC uses one fixed design delay, a "
    "predictor built on the nominal model and zero ground acceleration over the prediction horizon; an on-line delay estimate, "
    "a predictor robust to parameter errors, a measured ground acceleration or an H∞-type design might do better, and the "
    "oracle-delay result of Section 6.4 bounds only what exact knowledge of the delay would add. The AS law was introduced after "
    "the first DRL results were known, and CO-DC later still, both tuned on synthetic motions only.")
P.P("*Design history and statistics.* The intensity range was chosen so that amplitude adaptation matters; an earlier attempt "
    "on a three-storey single-damper model, where constant maximum voltage was close to the best constant setting, showed no "
    "improvement in short pilot runs and was not pursued. Reward weights, decision interval and hyperparameters were set in "
    "pilot runs without a systematic search, and all seeds share one set of hyperparameters. Every policy was trained for about "
    f"four million decisions (about twenty minutes on one CPU core); the proposed configuration and the no-randomisation "
    f"ablation used {NSW} seeds each, the observer and critic ablations "
    f"{NSW_OBS} and {NSW_SYM}, and the budget comparison of Section 6.8 rests on three. The seed-level tests resolve about "
    f"{HW:.1f}% for the domain-randomisation comparison and assume approximately normal seed means, so smaller effects such as a "
    "possible benefit of the observer remain unresolved; recurrent policies, other algorithms or a faster decision rate may "
    "change these comparisons. *Metrics.* J_{T} is one particular scalarisation, the fragility thresholds are illustrative, and "
    "records were amplitude-scaled without regard to spectral shape.")

# ================================================================================================ 8 conclusions
P.H("8. Conclusions", 1)
P.P(f"We formulated MR-damper voltage control as a partially observed problem with randomised structural, device, sensing and "
    f"actuation parameters, trained a PPO policy with a model-based observer, a twin-normalised reward and a privileged critic, "
    f"and evaluated it on {N_IN + N_OOD} Monte Carlo scenarios built from recorded motions. The main findings are:")
P.bullets([
    f"Under uncertainty the learned controller lowers the composite objective by {pc(-rel('mc_in', 'P-BEST'))} relative to the "
    f"best constant voltage, {pc(-rel('mc_in', 'P-BEST2'))} relative to the best pair of constant voltages, "
    f"{pc(-rel('mc_in', 'CO-ROB'))} relative to a robustly tuned clipped-optimal controller and {pc(-rel('mc_in', 'CO-DC'))} "
    f"relative to a delay-compensated one, with similar margins out of distribution; these differences are significant at the "
    f"seed level ({NSW} seeds).",
    "A nominally tuned clipped-optimal controller is competitive on the nominal plant but amplifies floor accelerations beyond "
    "the uncontrolled level when the command is delayed by 10 ms or more.",
    f"A model-based delay predictor makes clipped-optimal control nearly insensitive to delay on the nominal plant and improves "
    f"on the robust tuning by {100 * (1 - M('mc_in', 'CO-DC', 'JT') / M('mc_in', 'CO-ROB', 'JT')):.0f}% under uncertainty, but its "
    f"composite objective stays about {100 * (M('mc_in', 'CO-DC', 'JT') / d_in - 1):.0f}% above that of the learned controller "
    f"and its tail is no better than that of the robust tuning.",
    {"ahead": f"A two-parameter amplitude-scheduled voltage law comes within about "
              f"{max(1, round(abs(100 * as_in['mean'] / M('mc_in', 'AS', 'JT')))):.0f} per cent of the learned policies; the difference is "
              f"statistically detectable across {NSW} seeds and practically irrelevant. ",
     "tie": f"A two-parameter amplitude-scheduled voltage law is statistically indistinguishable from the learned policies "
            f"across {NSW} seeds (mean difference {pc1(as_in['mean'] / M('mc_in', 'AS', 'JT'))}, {sig(as_in['p'])}). ",
     "behind": f"A two-parameter amplitude-scheduled voltage law is slightly better than the learned policies across {NSW} seeds. "}[AS_STATUS]
    + "What the policy learned is essentially intensity-adaptive damping.",
    f"For the three seeds trained at both budgets, quadrupling the training budget to {DEC / 1e6:.1f} million decisions improved the mean objective by "
    f"{abs(100 * bu['mc_in']['paired']['diff'] / bu['mc_in']['b100']['mean']):.0f}%, almost entirely by rescuing one lagging seed, "
    f"and reduced the standard deviation across seeds from {bu['mc_in']['b100']['sd']:.3f} to {bu['mc_in']['now']['sd']:.3f}.",
    {"established": f"With {NSW} seeds per group, domain randomisation shows a statistically established but small benefit of "
                    f"{abs(100 * DR_REL_IN):.1f}% in distribution and {abs(100 * DR_REL_OOD):.1f}% out of distribution "
                    f"(Welch {sig(dr_in['welch_p'])} and {sig(dr_ood['welch_p'])})",
     "partial": f"With {NSW} seeds per group, domain randomisation shows a small benefit of {abs(100 * DR_REL_IN):.1f}% in "
                f"distribution and {abs(100 * DR_REL_OOD):.1f}% out of distribution with borderline statistical support "
                f"(Welch {sig(dr_in['welch_p'])} and {sig(dr_ood['welch_p'])})",
     "unresolved": f"Even with {NSW} seeds per group, the benefit of domain randomisation ({abs(100 * DR_REL_IN):.1f}% in "
                   f"distribution, {abs(100 * DR_REL_OOD):.1f}% out of distribution) is not statistically established "
                   f"(Welch {sig(dr_in['welch_p'])} and {sig(dr_ood['welch_p'])})"}[DR_STATUS]
    + "; no benefit of the model-based observer could be resolved; "
    + (f"the privileged critic gave a nominally significant benefit of {100 * SY_REL:.0f}% (Welch {sig(sy_d['welch_p'])}, "
       f"uncorrected), mostly by making training more reliable."
       if SY_SIG else "the privileged critic made training faster and more reliable."),
    "Evaluation under uncertainty, strong simple baselines, seed-level tests with an adequate number of seeds and a budget check each changed the conclusions, and we "
    "recommend them for future DRL studies in structural control.",
])
P.P("Future work should examine policies that act within the vibration cycle, where both the potential gain and the need for "
    "robust training are larger, as well as nonlinear full-scale benchmark buildings and experimental validation.")

P.H("Data and code availability", 1)
P.P("The simulation, training, evaluation and analysis code (Python: NumPy, SciPy, Numba, PyTorch), the trained policies, the "
    "per-scenario result files and the script that generates this document (build_paper.py) accompany the manuscript. All "
    "numerical results reported here were produced by that code. The recorded accelerograms come from the sources cited in "
    "Section 3.4: the El Centro record from [57] and ten records bundled with the gmspy package [58], which attributes them to "
    "the PEER strong-motion database [59]. They are included for reproducibility only and remain subject to the terms of their "
    "original sources; gmspy itself is distributed under the GNU General Public License. The parameters of the six-storey model "
    "were taken from secondary sources (Section 3.1), so the model is patterned after the original example rather than being an "
    "exact replica of it." if BLIND else
    "exact replica of it. Correspondence and requests for the code and data should be addressed to Hassan Alizadeh, "
    "Semnan University, Semnan, Iran (h.alizadeh@semnan.ac.ir).")

P.H("Declaration", 1)
P.P("**Funding.** This research received no specific grant from any funding agency in the public, commercial or not-for-profit "
    "sectors.")
P.P("**Competing interests.** The author declares no competing interests.")
if not BLIND:
    P.P(f"**Author contributions.** {AUTHOR} designed the study, implemented the simulator, the controller and the evaluation "
        "pipeline, performed all simulations and statistical analyses, and wrote the manuscript.")
P.P("**Use of AI-assisted tools.** Generative AI tools were used to help implement the simulation, analysis and typesetting "
    "code and to draft and edit the text. The author has checked the code, the numerical results and the reference list against "
    "their stated sources and takes full responsibility for the content of the manuscript.")
if not (SUBMISSION or BLIND):
    P.P(f"**Disclaimer.** {WARN}")

P.references()
_sb = "manuscript_anonymous.docx" if BLIND else ("manuscript_submission.docx" if SUBMISSION else "manuscript.docx")
out = os.path.join(HERE, "..", _sb)
P.save(out)
# ---- document properties (title page / metadata) ------------------------------------------
import datetime
from docx import Document as _Document
_doc = _Document(out)
_cp = _doc.core_properties
_cp.title = ("Robust Physics-Informed Deep Reinforcement Learning for Semi-Active Seismic Control of Multi-Storey "
             "Buildings With MR Dampers")
_cp.author = "Anonymous" if BLIND else AUTHOR
_cp.last_modified_by = "Anonymous" if BLIND else AUTHOR
_cp.subject = "Semi-active seismic control with MR dampers; physics-informed deep reinforcement learning"
_cp.keywords = ("semi-active control; magnetorheological damper; deep reinforcement learning (PPO); domain randomisation; "
                "partial observability; seismic fragility")
_cp.comments = ""
_cp.category = "Manuscript"
_cp.modified = datetime.datetime.now()
_cp.revision = 1
_doc.save(out)
open(os.path.join(HERE, "..", "references.md"), "w").write("# References (in order of citation)\n\n" + P.references_md() + "\n")
print("saved", out, "| refs cited:", len(P.order), "| figures", P.nfig, "tables", P.ntab)
from paperlib import REFS
print("uncited refs:", [k for k in REFS if k not in P.order])
