"""
ppo.py -- Proximal Policy Optimisation with an asymmetric (privileged) critic for the batched
          MR-damper environment.  Usage example:
    python ppo.py --name pi_dr_s0 --obs pi --dr 1 --seed 0 --iters 120
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from simcore import Building, sample_params, synth_motions
from env import MREnv, rollout

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")


def mlp(i, hs, o, act=nn.Tanh):
    layers, d = [], i
    for h in hs:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128, 128), log_std0=-0.7):
        super().__init__()
        self.net = mlp(obs_dim, hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std0))
        nn.init.uniform_(self.net[-1].weight, -1e-2, 1e-2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, o):
        return self.net(o)

    def dist(self, o):
        return torch.distributions.Normal(self.net(o), self.log_std.clamp(-3.0, 0.5).exp())


def make_policy_fn(actor):
    def fn(obs):
        with torch.no_grad():
            return actor(torch.as_tensor(obs, dtype=torch.float32)).numpy().astype(np.float64)
    return fn


def load_actor(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = Actor(ck["obs_dim"], ck["act_dim"])
    a.load_state_dict(ck["actor"])
    a.eval()
    return a, ck["args"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--building", default="six")
    ap.add_argument("--obs", default="pi", choices=["pi", "raw", "full"])
    ap.add_argument("--act", default="sched", choices=["volt", "sched"])
    ap.add_argument("--K", type=int, default=10, help="control steps per agent decision")
    ap.add_argument("--dr", type=float, default=1.0, help="domain-randomisation level (0 = nominal plant)")
    ap.add_argument("--sym_critic", type=int, default=0, help="1 = critic sees only the actor observation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--nenv", type=int, default=64)
    ap.add_argument("--T", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--lam", type=float, default=0.92)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--mb", type=int, default=2048)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--w_e", type=float, default=4.0)
    ap.add_argument("--w_p", type=float, default=1.0)
    ap.add_argument("--rscale", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(RUNS, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(1000 + args.seed)
    bld = Building(args.building)
    if args.T <= 0:
        args.T = bld.T_train
    env = MREnv(bld, args.nenv, act_mode=args.act, seed=args.seed, w_e=args.w_e, w_p=args.w_p)
    obs_dim, act_dim = env.obs_dims[args.obs], env.adim
    cdim = obs_dim if args.sym_critic else env.priv_dim
    actor = Actor(obs_dim, act_dim)
    critic = mlp(cdim, (256, 256), 1)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr, eps=1e-5)

    # fixed validation set (in-distribution, randomised plant) used for model selection / curves
    vrng = np.random.default_rng(777)
    NV = 64
    venv = MREnv(bld, NV, act_mode=args.act, w_e=args.w_e, w_p=args.w_p)
    v_ag, _ = synth_motions(NV, args.T, venv.dt, vrng, bld.time_scale, bld.pga_range)
    v_par = sample_params(bld, NV, vrng, level=1.0)

    log = dict(args=vars(args), iter=[], steps=[], ret=[], cost_e=[], cost_p=[], J1=[], J2=[], J3=[], J4=[],
               val_iter=[], val_J1=[], val_J2=[], val_J3=[], val_J4=[], val_cost=[], time=[])
    best = np.inf
    t0 = time.time()
    steps = 0
    it0 = 0
    # ---- resume from a full training state (written at every validation point) ---------------------------
    state_f = os.path.join(RUNS, args.name + "_state.pt")
    if os.path.exists(state_f):
        st = torch.load(state_f, map_location="cpu", weights_only=False)
        if st["args"] == vars(args):
            actor.load_state_dict(st["actor"]); critic.load_state_dict(st["critic"]); opt.load_state_dict(st["opt"])
            rng.bit_generator.state = st["rng"]; env.rng.bit_generator.state = st["env_rng"]
            torch.set_rng_state(st["torch_rng"])
            log, best, steps, it0 = st["log"], st["best"], st["steps"], st["it"] + 1
            t0 = time.time() - (log["time"][-1] if log["time"] else 0.0)
            print(f"[{args.name}] resumed at iteration {it0}", flush=True)
    for it in range(it0, args.iters):
        frac = 1.0 - it / args.iters
        for g in opt.param_groups:
            g["lr"] = args.lr * (0.1 + 0.9 * frac)
        ag, _ = synth_motions(args.nenv, args.T, env.dt, rng, bld.time_scale, bld.pga_range)
        par = sample_params(bld, args.nenv, rng, level=args.dr, nominal=(args.dr == 0))
        env.reset(ag, par)
        Tn = int(np.ceil(env.nsteps / args.K))
        O = np.zeros((Tn, args.nenv, obs_dim), dtype=np.float32)
        Pv = np.zeros((Tn + 1, args.nenv, cdim), dtype=np.float32)
        A = np.zeros((Tn, args.nenv, act_dim), dtype=np.float32)
        LP = np.zeros((Tn, args.nenv), dtype=np.float32)
        R = np.zeros((Tn, args.nenv), dtype=np.float32)
        for t in range(Tn):
            o = env.obs(args.obs).astype(np.float32)
            O[t] = o
            Pv[t] = o if args.sym_critic else env.priv_obs()
            with torch.no_grad():
                d = actor.dist(torch.from_numpy(o))
                a = d.sample()
                LP[t] = d.log_prob(a).sum(-1).numpy()
            A[t] = a.numpy()
            r, done = env.step(A[t].astype(np.float64), args.K)
            R[t] = r * args.rscale
        Pv[Tn] = env.obs(args.obs) if args.sym_critic else env.priv_obs()
        steps += Tn * args.nenv
        for nm_, arr_ in (("obs", O), ("critic input", Pv), ("reward", R)):
            if not np.isfinite(arr_).all():
                tb, eb = np.argwhere(~np.isfinite(arr_.reshape(arr_.shape[0], arr_.shape[1], -1)).all(-1))[0]
                raise FloatingPointError(f"non-finite {nm_} at decision {tb}, env {eb}, cols "
                                         f"{np.where(~np.isfinite(arr_[tb, eb]))[0][:8] if arr_.ndim == 3 else ''}")
        with torch.no_grad():
            V = critic(torch.from_numpy(Pv.reshape(-1, cdim))).numpy().reshape(Tn + 1, args.nenv)
        adv = np.zeros_like(R)
        last = 0.0
        for t in reversed(range(Tn)):
            delta = R[t] + args.gamma * V[t + 1] - V[t]      # time-limit truncation -> always bootstrap
            last = delta + args.gamma * args.lam * last
            adv[t] = last
        ret = adv + V[:-1]
        # ---- PPO update --------------------------------------------------------------------
        o_t = torch.from_numpy(O.reshape(-1, obs_dim)); p_t = torch.from_numpy(Pv[:-1].reshape(-1, cdim))
        a_t = torch.from_numpy(A.reshape(-1, act_dim)); lp_t = torch.from_numpy(LP.reshape(-1))
        adv_t = torch.from_numpy(adv.reshape(-1)); ret_t = torch.from_numpy(ret.reshape(-1))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        nb = o_t.shape[0]
        stop = False
        for ep in range(args.epochs):
            perm = torch.randperm(nb)
            for i in range(0, nb, args.mb):
                idx = perm[i:i + args.mb]
                d = actor.dist(o_t[idx])
                lp = d.log_prob(a_t[idx]).sum(-1)
                ratio = (lp - lp_t[idx]).exp()
                s1 = ratio * adv_t[idx]; s2 = ratio.clamp(1 - args.clip, 1 + args.clip) * adv_t[idx]
                pl = -torch.min(s1, s2).mean()
                vl = nn.functional.smooth_l1_loss(critic(p_t[idx]).squeeze(-1), ret_t[idx])
                loss = pl + 0.5 * vl - args.ent * d.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                opt.step()
                kl = (lp_t[idx] - lp).mean().item()
                if kl > 0.03:
                    stop = True; break
            if stop:
                break
        J = env.indices()
        log["iter"].append(it); log["steps"].append(steps); log["ret"].append(float(R.sum(0).mean() / args.rscale))
        log["cost_e"].append(float(J["JE"].mean())); log["cost_p"].append(float(J["JT"].mean()))
        for k in ("J1", "J2", "J3", "J4"):
            log[k].append(float(J[k].mean()))
        log["time"].append(time.time() - t0)
        if it % 5 == 4 or it == args.iters - 1:
            Jv = rollout(venv, None, v_ag, v_par, noise_seed=1, policy_fn=make_policy_fn(actor), obs_mode=args.obs, K=args.K)
            vc = float(Jv["JT"].mean())
            log["val_iter"].append(it); log["val_cost"].append(vc)
            for k in ("J1", "J2", "J3", "J4"):
                log["val_" + k].append(float(Jv[k].mean()))
            ck = dict(actor=actor.state_dict(), obs_dim=obs_dim, act_dim=act_dim, args=vars(args))
            torch.save(ck, os.path.join(RUNS, args.name + "_last.pt"))
            if vc < best:
                best = vc
                torch.save(ck, os.path.join(RUNS, args.name + ".pt"))
            print(f"[{args.name}] it {it+1:4d} steps {steps/1e6:5.2f}M ret {log['ret'][-1]:8.4f} "
                  f"val cost {vc:.4f} J1 {log['val_J1'][-1]:.3f} J2 {log['val_J2'][-1]:.3f} J3 {log['val_J3'][-1]:.3f} "
                  f"J4 {log['val_J4'][-1]:.3f} std {actor.log_std.exp().mean().item():.3f} t {time.time()-t0:6.0f}s", flush=True)
            json.dump(log, open(os.path.join(RUNS, args.name + ".json"), "w"))
            tmp = state_f + ".tmp"
            torch.save(dict(args=vars(args), actor=actor.state_dict(), critic=critic.state_dict(), opt=opt.state_dict(),
                            rng=rng.bit_generator.state, env_rng=env.rng.bit_generator.state,
                            torch_rng=torch.get_rng_state(), log=log, best=best, steps=steps, it=it), tmp)
            os.replace(tmp, state_f)
    if os.path.exists(state_f):
        os.remove(state_f)
    open(os.path.join(RUNS, args.name + ".done"), "w").write(str(args.iters))


if __name__ == "__main__":
    main()
