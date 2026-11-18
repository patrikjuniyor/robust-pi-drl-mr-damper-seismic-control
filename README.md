# Robust physics-informed deep reinforcement learning for semi-active seismic control of MR-damped buildings

Code, data, trained policies and result files for the paper

> **Robust Physics-Informed Deep Reinforcement Learning for Semi-Active Seismic Control of Multi-Storey Buildings With MR Dampers**
> Hassan Alizadeh — Semnan University, Semnan, Iran — h.alizadeh@semnan.ac.ir

This repository contains everything needed to reproduce the paper: the structural and damper simulator, the
reinforcement-learning environment, the training code, all baseline controllers, the evaluation and statistics pipeline,
the trained policies, the per-scenario result files, the figures, and the script that assembles the manuscript.

Every number in the tables of the paper — and most numbers in the running text — is read from the files in `results/`
when the manuscript is built, so the document cannot drift away from the computations.

---

## Layout

| Path | Contents |
| --- | --- |
| `src/code/` | All Python code (simulator, environment, PPO, baselines, evaluation, statistics, figures, manuscript builder) |
| `src/data/` | Recorded accelerograms: El Centro (vibrationdata.com) and the ten records shipped with gmspy 0.1.3 |
| `src/figures/` | The seven figures of the paper at 450 dpi, plus one figure that is produced but not used |
| `src/results/` | `tables.json` (all statistics used in the paper), `baselines.json`, tuning and diagnostic files, and one `.npz` per (scenario set, controller) with per-scenario indices |
| `src/runs/` | The 30 trained policies (400 iterations = 4.1 million decisions), with training logs |
| `src/runs_b100/`, `src/results/b100/` | Policies and results of the earlier 100-iteration budget, used only for the budget comparison of Section 6.8 |
| `src/refs/` | Crossref / arXiv metadata used to build the reference list |
| `src/paper.docx` | The manuscript (journal-ready version) |
| `src/references.md` | The 73 cited references, in order of citation |

`src/README.md` holds the detailed description of the contents written for the original reproduction package.

## Which numbers come from which runs

- Every PI-DRL number in the tables and figures of Sections 6.2–6.7 and 6.9 comes from the 400-iteration
  (4.1 M-decision) policies; **PI-DRL** is the average over the ten seeds of `V_pi_dr`.
- The 100-iteration (1.0 M-decision) results appear only in Section 6.8 (budget table), as the dotted curve of the
  training figure, and in two sentences that refer to that section. That comparison uses seeds 0–2 at both budgets.
- The non-learning baselines (P-OFF, P-ON, P-BEST, P-BEST2, AS, CO-NOM, CO-ROB, CO-DC) do not depend on the budget.

## Trained policies

| Directory prefix | Configuration | Seeds |
| --- | --- | --- |
| `V_pi_dr` | proposed: partial observation, privileged critic, domain randomisation | 0–9 |
| `V_pi_nom` | no domain randomisation (ablation) | 0–9 |
| `V_raw_dr` | no physics-based observer (ablation) | 0–4 |
| `V_pi_sym` | symmetric critic, no privileged information (ablation) | 0–4 |

`*_s{seed}.pt` is the best-validation checkpoint; `*_s{seed}_last.pt` is the final checkpoint; `*.json` are the training
logs and `*.done` mark completed runs.

## Reproduce

Requirements: Python 3, `numpy`, `scipy`, `numba`, `torch` (CPU is enough), `matplotlib`, `python-docx`.
All commands are run inside `src/code/`.

1. `python validate_dyke.py` — verification against Dyke et al. (1996), Table 2.
2. `python tune_kf.py` — observer tuning.
3. `python tune_baselines.py`, then `python tune_ext.py` (clipped-optimal grid extension), then `python tune_extra.py`
   (P-BEST, P-BEST2, AS), then `python tune_codc.py` (delay-compensated controller, control weight × design delay).
4. `bash run_queue.sh 400` — the 30 PPO runs listed in `queue.txt`: 400 iterations, about 21 min each on one CPU core,
   two at a time (about 5.5 h in total). Finished runs are skipped and interrupted runs resume from the last validation
   point.
5. `python evaluate.py` — all controllers on all scenario sets (about 2 h on one core).
6. `python check_integration.py`, `python diag_codc.py`, `python analyze.py`, `python fig_schematic.py`,
   `python build_paper.py`.

`analyze.py` uses every seed that has finished training and has been evaluated; the prose of the manuscript is assembled
from the resulting statistics (see the `outcomes:` line printed by `build_paper.py`). To rebuild only the journal-ready
document, run `SUBMISSION=1 python build_paper.py`; `BLIND=1 python build_paper.py` produces the anonymised copy used for
double-blind review.

Steps 1–5 are already done here: the trained policies and all result files are included, so the statistics, figures and
manuscript can be regenerated from step 6 onward without retraining. `analyze.py` reproduces `results/tables.json`
byte-for-byte.

## Code map

| File | Role |
| --- | --- |
| `simcore.py` | Building model, Bouc–Wen MR damper, integrator, ground motions |
| `env.py` | Environment, Kalman observer, passive/AS/clipped-optimal controllers (incl. `ClippedOptimalDC`) |
| `ppo.py` | PPO training loop, actor–critic networks, checkpointing |
| `evaluate.py` | Evaluation of all controllers on all scenario sets |
| `analyze.py` | Statistics, hypothesis tests, tables and figures |
| `fig_schematic.py` | Figure 1 (training set-up) |
| `paperlib.py`, `build_paper.py` | Assembling the manuscript (.docx) from the result files |
| `test_codc.py`, `diag_codc.py` | Development test and prediction-quality diagnostic of the delay-compensated controller |
| `eval_loop.sh` | Evaluates runs as they finish |

## Data sources and licences

- El Centro 1940 (north–south component): T. Irvine, *El Centro earthquake page*, vibrationdata.com.
- Ten further accelerograms: bundled with the `gmspy` package (v0.1.3, GNU General Public License), attributed there to the
  PEER NGA-West2 strong-motion database. They are included for reproducibility only and remain subject to the terms of
  their original sources.
- The six-storey structural parameters were taken from secondary sources, documented in Section 3.1 of the paper.

## Scope of the results

This is a **simulation study**. No experimental validation is reported. "Out-of-distribution" refers to wider parameter
ranges within the same model class, not to unmodelled physics.

## Citation

If you use this code or these results, please cite the paper above. A machine-readable citation will be added here once a
DOI is available.
