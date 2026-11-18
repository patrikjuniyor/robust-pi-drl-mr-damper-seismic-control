"""
env.py -- batched semi-active control environment (partial observation, domain randomisation),
          nominal-model Kalman filter, clipped-optimal LQG baseline, performance indices.
"""
import numpy as np
from scipy import linalg
from simcore import (Building, sample_params, build_matrices, sim_block, synth_motions, G)

LAGS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24)  # measurement-history lags (control steps)
# normalisation constants and reference sensor-noise std (noise level = 1), per building
SCALES = dict(dyke3=dict(acc=10.0, drift=2.0e-3, vel=0.1, n_acc=0.15, n_drift=2.0e-5, n_f=10.0, kf_q=100.0),
              six=dict(acc=3.0, drift=5.0e-3, vel=0.08, n_acc=0.02, n_drift=2.0e-5, n_f=0.3, kf_q=0.2))


class NominalModel:
    """Linear nominal model, Kalman filter and LQR gains used by the observer and CO baseline."""

    def __init__(self, bld, ctrl_dt, kf_q=1.0, kf_rinfl=1.0):
        n, nd = bld.n, bld.nd
        sc = SCALES[bld.name]; NOISE_ACC, NOISE_DRIFT = sc['n_acc'], sc['n_drift']
        self.n, self.nd = n, nd
        Minv = np.diag(1.0 / bld.mass)
        Gam = np.zeros((n, nd))
        for j, st in enumerate(bld.dstory):
            Gam[st, j] -= bld.dcount[j]
            if st > 0:
                Gam[st - 1, j] += bld.dcount[j]
        A = np.block([[np.zeros((n, n)), np.eye(n)], [-Minv @ bld.K, -Minv @ bld.C]])
        B = np.vstack([np.zeros((n, nd)), Minv @ Gam])
        E = np.vstack([np.zeros((n, 1)), -np.ones((n, 1))])
        Ca = np.hstack([-Minv @ bld.K, -Minv @ bld.C]); Da = Minv @ Gam           # absolute accelerations
        Cd = np.zeros((nd, 2 * n))
        for j, st in enumerate(bld.dstory):
            Cd[j, st] = 1.0
            if st > 0:
                Cd[j, st - 1] = -1.0
        Cy = np.vstack([Ca, Cd]); Dy = np.vstack([Da, np.zeros((nd, nd))])
        self.A, self.B, self.E, self.Ca, self.Da, self.Cy, self.Dy = A, B, E, Ca, Da, Cy, Dy
        # ---- discrete-time steady-state Kalman filter (ground acceleration = white sequence) ----
        # x_{k+1} = Ad x_k + Bd f_k + Ed w_k ,   y_k = Cy x_k + Dy f_k + v_k   (f = measured damper force)
        nx = 2 * n
        Mx = linalg.expm(np.block([[A, B, E], [np.zeros((nd + 1, nx + nd + 1))]]) * ctrl_dt)
        Ad, Bd, Ed = Mx[:nx, :nx], Mx[:nx, nx:nx + nd], Mx[:nx, nx + nd:]
        Rv = np.diag(np.r_[np.full(n, NOISE_ACC ** 2), np.full(nd, NOISE_DRIFT ** 2)]) * kf_rinfl
        Pp = linalg.solve_discrete_are(Ad.T, Cy.T, kf_q * Ed @ Ed.T, Rv)
        Lf = Pp @ Cy.T @ np.linalg.inv(Cy @ Pp @ Cy.T + Rv)          # filter gain
        self.Ad, self.Bd, self.Lf = Ad, Bd, Lf

    def kf_update(self, xpred, y, f):
        """xpred = x_{k|k-1};  returns filtered x_{k|k} and the next prediction x_{k+1|k}."""
        innov = y - xpred @ self.Cy.T - f @ self.Dy.T
        xf = xpred + innov @ self.Lf.T
        return xf, xf @ self.Ad.T + f @ self.Bd.T

    def lqr(self, Q, N, R):
        P = linalg.solve_continuous_are(self.A, self.B, Q, R, s=N)
        return np.linalg.solve(R, self.B.T @ P + N.T)

    def lqr_energy(self, bld, w_s, w_a, r):
        """LQR weighting the nominal strain energy and the inertial acceleration energy."""
        n = self.n
        Qs = np.zeros((2 * n, 2 * n)); Qs[:n, :n] = 0.5 * bld.K
        Mh = 0.5 * np.diag(bld.mass)
        Q = w_s * Qs + w_a * self.Ca.T @ Mh @ self.Ca
        N = w_a * self.Ca.T @ Mh @ self.Da
        R = w_a * self.Da.T @ Mh @ self.Da + r * np.eye(self.nd)
        return self.lqr(Q, N, R)

    def lqr_dyke(self, r=1e-17, floor=-1):
        """Original design of Dyke et al. (1996): weight on the top-floor absolute acceleration."""
        c = self.Ca[floor:floor + 1] if floor != -1 else self.Ca[-1:]
        d = self.Da[floor:floor + 1] if floor != -1 else self.Da[-1:]
        return self.lqr(c.T @ c, c.T @ d, d.T @ d + r * np.eye(self.nd))


class MREnv:
    """Batched environment.  One call to step() = one control interval."""

    def __init__(self, bld, N, ctrl_dt=None, nsub=None, act_mode="volt", x0flag=0.0, seed=0,
                 w_e=4.0, w_p=1.0, w_u=0.0, kf_q=None, kf_rinfl=30.0):
        self.bld, self.N = bld, N
        if ctrl_dt is None:
            ctrl_dt, nsub = (0.005, 10) if bld.name == 'dyke3' else (0.01, 5)
        self.n, self.nd = bld.n, bld.nd
        self.ctrl_dt, self.nsub, self.dt = ctrl_dt, nsub, ctrl_dt / nsub
        self.act_mode, self.x0flag = act_mode, x0flag
        self.rng = np.random.default_rng(seed)
        # command delay unit: one nominal control period (10 ms for 'six', 5 ms for 'dyke3'), independent of ctrl_dt
        self.dmult = max(1, int(round((0.01 if bld.name == 'six' else 0.005) / ctrl_dt)))
        self.nom = NominalModel(bld, ctrl_dt, SCALES[bld.name]['kf_q'] if kf_q is None else kf_q, kf_rinfl)
        self.w_e, self.w_p, self.w_u = w_e, w_p, w_u
        self.ny = self.n + 2 * self.nd
        self.adim = 2 * self.nd if act_mode == "sched" else self.nd
        nf = self.ny + self.n + self.adim            # intensity features + previous action
        self.obs_dims = dict(raw=len(LAGS) * self.ny + nf,
                             pi=2 * self.n + len(LAGS) * self.ny + nf,
                             full=2 * self.n + 3 * self.nd + nf)
        self.priv_dim = self.obs_dims["full"] + (2 * self.n + 1 + 3 * self.nd + 2) + 10
        # primary (nominal-model LQG) controller used by the clipped-optimal inner loop
        wa, r = (0.0, 1e-3) if bld.name == "six" else (0.0, 1e-8)
        self.Kco = self.nom.lqr_energy(bld, 1.0, wa, r)
        self.ew_c = np.exp(-self.ctrl_dt / (1.0 / bld.time_scale))     # 1 s (prototype time) forgetting
        sc = SCALES[bld.name]; self.sc = sc
        self.yscale = np.r_[np.full(self.n, sc['acc']), np.full(self.nd, sc['drift']), np.full(self.nd, bld.fscale)]
        self.xscale = np.r_[np.full(self.n, sc['drift'] * 3), np.full(self.n, sc['vel'])]
        self.nstd = np.r_[np.full(self.n, sc['n_acc']), np.full(self.nd, sc['n_drift']), np.full(self.nd, sc['n_f'])][None]

    # ------------------------------------------------------------------------------------------
    def _alloc_metrics(self, N):
        n, nd = self.n, self.nd
        return dict(pk_d=np.zeros((N, n)), pk_a=np.zeros((N, n)), pk_x=np.zeros((N, n)), pk_f=np.zeros((N, nd)),
                    ss_d=np.zeros((N, n)), ss_a=np.zeros((N, n)))

    def _run_uncontrolled(self):
        """Uncontrolled twin (same structure, same excitation, no damper)."""
        N, n, nd = self.N, self.n, self.nd
        S = np.zeros((N, 2 * n + 3 * nd)); v = np.zeros((N, nd))
        m = self._alloc_metrics(N)
        m_acc = np.zeros((N, n)); m_f = np.zeros((N, nd)); e_s = np.zeros(N); e_a = np.zeros(N)
        act = np.ones(N, dtype=np.bool_)
        Es = np.zeros(N); Ea = np.zeros(N)
        spd = np.zeros((N, n)); spa = np.zeros((N, n))
        zero = np.zeros(nd)
        # integrate the whole record with large blocks
        blk = 50 * self.nsub
        k = 0
        nblk = 0
        while k + blk <= self.nsteps * self.nsub:
            sim_block(S, v, self.ag, k, blk, self.dt, self.M["MK"], self.M["MC"], self.M["mass"], self.M["K"],
                      self.bld.dstory, zero, self.M["DP"], self.x0flag, act, m_acc, m_f, e_s, e_a,
                      m["pk_d"], m["pk_a"], m["pk_x"], m["pk_f"], m["ss_d"], m["ss_a"], spd, spa)
            Es += e_s; Ea += e_a; k += blk; nblk += 1
        rem = self.nsteps * self.nsub - k
        if rem > 0:
            sim_block(S, v, self.ag, k, rem, self.dt, self.M["MK"], self.M["MC"], self.M["mass"], self.M["K"],
                      self.bld.dstory, zero, self.M["DP"], self.x0flag, act, m_acc, m_f, e_s, e_a,
                      m["pk_d"], m["pk_a"], m["pk_x"], m["pk_f"], m["ss_d"], m["ss_a"], spd, spa)
            Es = (Es * blk + e_s * rem) / (nblk * blk + rem); Ea = (Ea * blk + e_a * rem) / (nblk * blk + rem)
        else:
            Es /= nblk; Ea /= nblk
        if not (np.isfinite(Es).all() and np.isfinite(Ea).all() and np.isfinite(S).all()):
            raise FloatingPointError('uncontrolled twin diverged')
        self.unc = m
        self.Es_unc = Es + 1e-12; self.Ea_unc = Ea + 1e-12
        self.D_unc = m["pk_d"].max(1) + 1e-12; self.A_unc = m["pk_a"].max(1) + 1e-12

    def reset(self, ag, params, noise_seed=None, unc=None):
        """ag [N, L] ground acceleration (m/s^2) at the integration step; params from sample_params."""
        N, n, nd = self.N, self.n, self.nd
        assert ag.shape[0] == N
        self.ag = np.ascontiguousarray(ag)
        self.nsteps = (ag.shape[1] - 2) // self.nsub
        self.p = params
        self.M = build_matrices(self.bld, params)
        if unc is None:
            self._run_uncontrolled()
        else:                                   # re-use a cached uncontrolled twin
            self.unc, self.Es_unc, self.Ea_unc, self.D_unc, self.A_unc = unc
        self.S = np.zeros((N, 2 * n + 3 * nd))
        self.met = self._alloc_metrics(N)
        self.m_acc = np.zeros((N, n)); self.m_f = np.zeros((N, nd)); self.e_s = np.zeros(N); self.e_a = np.zeros(N)
        self.sp_d = np.zeros((N, n)); self.sp_a = np.zeros((N, n))
        self.active = np.ones(N, dtype=np.bool_)
        self.k = 0
        self.xhat = np.zeros((N, 2 * n)); self.xpred = np.zeros((N, 2 * n))
        self.hist = np.zeros((N, LAGS[-1] + 1, self.ny))      # ring buffer of scaled measurements
        self.vbuf = np.zeros((N, 8 * self.dmult, nd))         # command buffer for actuation delay
        self.v_prev = np.zeros((N, nd))
        self.a_prev = np.zeros((N, self.adim))
        self.ew = np.zeros((N, self.ny)); self.rpk = np.zeros((N, n))
        self.nrng = np.random.default_rng(noise_seed) if noise_seed is not None else self.rng
        self.sensor_gain = np.ones((N, self.ny))             # for sensor-fault studies
        self.Es_sum = np.zeros(N); self.Ea_sum = np.zeros(N); self.v_sum = np.zeros(N)
        self.cost_e = np.zeros(N); self.Md = np.zeros(N); self.Ma = np.zeros(N)
        self._measure()
        return self

    def _measure(self):
        N, n, nd = self.N, self.n, self.nd
        drift = np.empty((N, nd))
        for j, st in enumerate(self.bld.dstory):
            drift[:, j] = self.S[:, st] - (self.S[:, st - 1] if st > 0 else 0.0)
        y = np.concatenate([self.m_acc, drift, self.m_f], axis=1)
        lvl = self.p["noise"][:, None]
        y = (y + lvl * self.nstd * self.nrng.standard_normal(y.shape)) * self.sensor_gain
        self.y = y
        # physics-based observer: Kalman filter built on the *nominal* linear model, driven by the
        # measured accelerations / damper stroke and by the measured damper force (known input)
        ny1 = n + nd
        self.xhat, self.xpred = self.nom.kf_update(self.xpred, y[:, :ny1], y[:, ny1:])
        self.hist = np.roll(self.hist, 1, axis=1)
        ys = y / self.yscale
        self.hist[:, 0] = ys
        # causal intensity features: exponentially-weighted RMS of every channel, running acceleration peak
        self.ew = self.ew_c * self.ew + (1.0 - self.ew_c) * ys ** 2
        self.rpk = np.maximum(self.rpk, np.abs(ys[:, :n]))

    # ------------------------------------------------------------------------------------------
    def obs(self, mode):
        N = self.N
        feat = np.concatenate([np.sqrt(self.ew), self.rpk, self.a_prev], axis=1)
        if mode == "raw":
            return np.concatenate([self.hist[:, LAGS].reshape(N, -1), feat], axis=1)
        if mode == "pi":
            return np.concatenate([self.xhat / self.xscale, self.hist[:, LAGS].reshape(N, -1), feat], axis=1)
        if mode == "full":
            n, nd = self.n, self.nd
            return np.concatenate([self.S[:, :2 * n] / self.xscale, self.m_f / self.bld.fscale,
                                   self.S[:, 2 * n + nd:2 * n + 2 * nd], self.S[:, 2 * n + 2 * nd:] / self.bld.vmax,
                                   feat], axis=1)
        raise ValueError(mode)

    def priv_obs(self):
        """Privileged critic input: true state, device state, randomised parameters, excitation."""
        n, nd = self.n, self.nd
        p = self.p
        k0 = self.k * self.nsub
        L = self.ag.shape[1]
        idx = lambda s: min(k0 + s * self.nsub, L - 1)
        agn = np.stack([self.ag[:, idx(0)], self.ag[:, idx(2)], self.ag[:, idx(6)], self.ag[:, idx(16)]], axis=1) / self.sc['acc']
        w = self.ag[:, k0:min(k0 + 40 * self.nsub, L)]
        fut = np.sqrt((w ** 2).mean(axis=1, keepdims=True)) / self.sc['acc']
        par = np.concatenate([p["sk"] - 1, p["sm"] - 1, np.log(p["sc"])[:, None], p["sf"] - 1, p["sv"] - 1,
                              p["seta"] - 1, p["delay"][:, None] / 2.0, p["noise"][:, None]], axis=1)
        nrm = np.stack([np.log10(self.Es_unc), np.log10(self.Ea_unc), self.Md, self.Ma], axis=1) / np.array([3.0, 3.0, 1.0, 1.0])
        return np.concatenate([self.obs("full"), par, agn, fut, np.full((self.N, 1), self.k / self.nsteps), nrm], axis=1)

    # ------------------------------------------------------------------------------------------
    def voltage_from_action(self, a):
        """Map the (clipped) action to command voltages.
        volt : direct voltage command
        sched: physics-informed structured action.  Per device the agent sets a base voltage v_lo and a
               top voltage v_hi >= v_lo; the fast inner loop applies the clipped-optimal (dissipativity)
               switching rule between them using the nominal-model LQG force demand:
                   v = v_lo + (v_hi - v_lo) H[(f_c - f) f]"""
        nd, vmax = self.nd, self.bld.vmax
        if self.act_mode == "volt":
            return vmax * 0.5 * (a + 1.0)
        v_lo = vmax * 0.5 * (a[:, :nd] + 1.0)
        v_hi = v_lo + (vmax - v_lo) * 0.5 * (a[:, nd:] + 1.0)
        fc = -self.xhat @ self.Kco.T
        fm = self.y[:, self.n + nd:]
        return v_lo + (v_hi - v_lo) * (((fc - fm) * fm) > 0.0)

    def step(self, action, K=1):
        """Apply one agent decision held for K control steps; returns the summed reward."""
        a = np.clip(action, -1.0, 1.0)
        R = np.zeros(self.N)
        done = False
        for _ in range(K):
            r, done = self.step_voltage(self.voltage_from_action(a), a)
            R += r
            if done:
                break
        return R, done

    def step_voltage(self, v, a=None):
        N, nd = self.N, self.nd
        if a is None:
            a = np.zeros((N, self.adim))
        self.vbuf = np.roll(self.vbuf, 1, axis=1)
        self.vbuf[:, 0] = v
        v_app = self.vbuf[np.arange(N), self.p["delay"] * self.dmult]   # delay is expressed in units of DELAY_UNIT
        v_app = np.ascontiguousarray(v_app)
        m = self.met
        sim_block(self.S, v_app, self.ag, self.k * self.nsub, self.nsub, self.dt, self.M["MK"], self.M["MC"],
                  self.M["mass"], self.M["K"], self.bld.dstory, self.bld.dcount, self.M["DP"], self.x0flag,
                  self.active, self.m_acc, self.m_f, self.e_s, self.e_a,
                  m["pk_d"], m["pk_a"], m["pk_x"], m["pk_f"], m["ss_d"], m["ss_a"], self.sp_d, self.sp_a)
        self.k += 1
        cs = self.e_s / self.Es_unc; ca = self.e_a / self.Ea_unc            # twin-normalised energies
        # twin-normalised peak-increment terms: their undiscounted sums equal the peak indices J1, J2
        sd = self.sp_d.max(axis=1) / self.D_unc; sa = self.sp_a.max(axis=1) / self.A_unc
        inc_d = np.maximum(sd - self.Md, 0.0); inc_a = np.maximum(sa - self.Ma, 0.0)
        self.Md = np.maximum(self.Md, sd); self.Ma = np.maximum(self.Ma, sa)
        rew = -self.w_p * (inc_d + inc_a) - self.w_e * 0.5 * (cs + ca) / self.nsteps
        self.cost_e += 0.5 * (cs + ca)
        self.Es_sum += self.e_s; self.Ea_sum += self.e_a; self.v_sum += v.mean(axis=1)
        self.a_prev = a.copy(); self.v_prev = v.copy()
        self._measure()
        return rew, self.k >= self.nsteps

    # ------------------------------------------------------------------------------------------
    def indices(self):
        """Performance indices normalised by the uncontrolled twin."""
        m, u = self.met, self.unc
        W = (self.M["mass"].sum(axis=1) * G)
        J = dict(
            JT=self.w_p * (self.Md + self.Ma) + self.w_e * self.cost_e / self.k,   # training objective
            J1=m["pk_d"].max(1) / u["pk_d"].max(1),                       # peak inter-storey drift
            J2=m["pk_a"].max(1) / u["pk_a"].max(1),                       # peak absolute acceleration
            J3=np.sqrt(m["ss_d"]).max(1) / np.sqrt(u["ss_d"]).max(1),     # RMS drift
            J4=np.sqrt(m["ss_a"]).max(1) / np.sqrt(u["ss_a"]).max(1),     # RMS acceleration
            J5=m["pk_f"].sum(1) / W,                                      # peak control force / weight
            JE=0.5 * (self.Es_sum / self.k / self.Es_unc + self.Ea_sum / self.k / self.Ea_unc),  # energy cost
            vmean=self.v_sum / self.k / self.bld.vmax,
            pk_d=m["pk_d"].max(1), pk_a=m["pk_a"].max(1), unc_pk_d=u["pk_d"].max(1), unc_pk_a=u["pk_a"].max(1),
        )
        return J


# ----------------------------------------------------------------------------------------------
# Controllers operating on an environment
# ----------------------------------------------------------------------------------------------
class Passive:
    def __init__(self, frac):
        self.frac = frac
    def reset(self, env): pass
    def act_voltage(self, env):
        return np.zeros((env.N, env.nd)) + np.asarray(self.frac) * env.bld.vmax      # scalar or one value per storey


class AmpSched:
    """Amplitude-scheduled passive heuristic: the voltage of each device group is proportional to the running
    (exponentially weighted, 1 s) RMS of its own measured stroke:  v_j = Vmax * min(1, g_j * rms_j / d_ref)."""
    def __init__(self, gains):
        self.g = np.asarray(gains, dtype=float)
    def reset(self, env): pass
    def act_voltage(self, env):
        rms = np.sqrt(env.ew[:, env.n:env.n + env.nd])          # stroke RMS / d_ref (d_ref = drift scale)
        return env.bld.vmax * np.clip(self.g * rms, 0.0, 1.0)


class ClippedOptimal:
    """Clipped-optimal LQG controller (Dyke et al. 1996) with acceleration feedback."""
    def __init__(self, Kgain):
        self.K = Kgain
    def reset(self, env): pass
    def act_voltage(self, env):
        fc = -env.xhat @ self.K.T
        fm = env.y[:, env.n + env.nd:]
        return env.bld.vmax * (((fc - fm) * fm) > 0.0)


class ClippedOptimalDC:
    """Delay-compensated clipped-optimal LQG controller (model-based predictor).

    The command issued at step k reaches the damper d steps later.  The commands issued during the last d steps are
    known to the controller, so the *nominal* model (linear structure + Bouc-Wen dampers) is integrated d steps ahead
    from the current estimate, with those queued commands and zero ground acceleration, and the clipped-optimal
    switching rule is evaluated on the predicted state and the predicted damper force:
        v_k = Vmax H[(f_c(x_pred) - f_pred) f_pred].
    Initial condition of the predictor: Kalman estimate of the structural state; filtered voltage u from the
    controller's own first-order model of the command path; hysteretic variable z from the measured force,
    z = (f/n_dev - c0(u) xdot)/alpha(u).
    d_steps: design delay in control steps; None -> oracle (true delay of every environment; reference only).
    """
    def __init__(self, Kgain, d_steps=None, ag_mode=0, f_mode=0):
        # ag_mode: 0 = zero ground acceleration in the predictor; 1 = hold the current ground acceleration (requires a
        # ground accelerometer); 2 = true future ground acceleration (diagnostic only, not realisable)
        self.K, self.d_steps, self.ag_mode = Kgain, d_steps, ag_mode
        self.f_mode = f_mode            # 0: predicted damper force in the switching rule; 1: measured force (held)

    def reset(self, env):
        from simcore import ALA, ALB, C0A, C0B, GAM, BET, AA, NEXP, ETA
        N, n, nd, bld = env.N, env.n, env.nd, env.bld
        self.M = build_matrices(bld, sample_params(bld, N, np.random.default_rng(0), nominal=True))
        self.D = ((env.p["delay"] * env.dmult).astype(np.int64) if self.d_steps is None
                  else np.full(N, int(self.d_steps), dtype=np.int64))
        self.Dmax = int(self.D.max())
        self.q = np.zeros((N, self.Dmax + 1, nd))            # q[:, i] = command issued i+1 steps ago
        self.u = np.zeros((N, nd))
        mr = bld.mr
        self.decay = np.exp(-mr[ETA] * env.ctrl_dt)
        self.ala, self.alb, self.c0a, self.c0b = mr[ALA], mr[ALB], mr[C0A], mr[C0B]
        self.zmax = (mr[AA] / (mr[GAM] + mr[BET])) ** (1.0 / mr[NEXP])
        self.Sp = np.zeros((N, 2 * n + 3 * nd)); self.agz = np.zeros((N, 4))
        z = lambda *s: np.zeros(s)
        self.scr = [z(N, n), z(N, nd), z(N), z(N), z(N, n), z(N, n), z(N, n), z(N, nd), z(N, n), z(N, n), z(N, n), z(N, n)]
        self.idx = np.arange(N)

    def act_voltage(self, env):
        N, n, nd, bld = env.N, env.n, env.nd, env.bld
        fm = env.y[:, n + nd:]
        if self.Dmax == 0:
            fc = -env.xhat @ self.K.T
            return bld.vmax * (((fc - fm) * fm) > 0.0)
        D = self.D
        if env.k > 0:                                       # voltage that acted during the last interval
            v_last = self.q[self.idx, D]
            self.u = v_last + (self.u - v_last) * self.decay
        Sp = self.Sp
        Sp[:] = 0.0
        Sp[:, :2 * n] = env.xhat
        for j, st in enumerate(bld.dstory):
            dv = env.xhat[:, n + st] - (env.xhat[:, n + st - 1] if st > 0 else 0.0)
            al = self.ala + self.alb * self.u[:, j]; c0 = self.c0a + self.c0b * self.u[:, j]
            Sp[:, 2 * n + nd + j] = np.clip((fm[:, j] / bld.dcount[j] - c0 * 100.0 * dv) / al, -self.zmax, self.zmax)
            Sp[:, 2 * n + 2 * nd + j] = self.u[:, j]
        m_acc, m_f = self.scr[0], self.scr[1]
        m_f[:] = fm
        M = self.M
        for i in range(self.Dmax):
            act = D > i
            vq = np.ascontiguousarray(self.q[self.idx, np.maximum(D - 1 - i, 0)])
            if self.ag_mode == 1:
                self.agz[:, 0] = self.agz[:, 1] = env.ag[:, env.k * env.nsub]
            elif self.ag_mode == 2:
                L = env.ag.shape[1] - 1
                self.agz[:, 0] = env.ag[:, min((env.k + i) * env.nsub, L)]; self.agz[:, 1] = env.ag[:, min((env.k + i + 1) * env.nsub, L)]
            sim_block(Sp, vq, self.agz, 0, 1, env.ctrl_dt, M["MK"], M["MC"], M["mass"], M["K"], bld.dstory, bld.dcount,
                      M["DP"], env.x0flag, act, m_acc, m_f, *self.scr[2:])
        fc = -Sp[:, :2 * n] @ self.K.T
        fs = fm if self.f_mode == 1 else m_f
        v = bld.vmax * (((fc - fs) * fs) > 0.0)
        self.q = np.roll(self.q, 1, axis=1)
        self.q[:, 0] = v
        return v


def rollout(env, controller, ag, params, noise_seed=0, policy_fn=None, obs_mode=None, record=None, fault=None, K=1, unc=None):
    """Run one batch of episodes.  controller: object with act_voltage(env), or policy_fn(obs)->action."""
    env.reset(ag, params, noise_seed=noise_seed, unc=unc)
    if controller is not None:
        controller.reset(env)
    if fault is not None:
        env.sensor_gain[:] = fault
    traj = None
    if record is not None:
        traj = {k: [] for k in ("x", "acc", "f", "v", "u")}
    done = False
    while not done:
        if policy_fn is not None:
            if env.k % K == 0:
                a = np.clip(policy_fn(env.obs(obs_mode)), -1.0, 1.0)
            _, done = env.step_voltage(env.voltage_from_action(a), a)
        else:
            v = controller.act_voltage(env)
            _, done = env.step_voltage(v)
        if traj is not None:
            e = record
            traj["x"].append(env.S[e, :env.n].copy()); traj["acc"].append(env.m_acc[e].copy())
            traj["f"].append(env.m_f[e].copy()); traj["v"].append(env.v_prev[e].copy())
            traj["u"].append(env.S[e, 2 * env.n + 2 * env.nd:].copy())
    J = env.indices()
    if not np.isfinite(env.S).all():
        raise FloatingPointError('simulation diverged')
    J["unc_cache"] = (env.unc, env.Es_unc, env.Ea_unc, env.D_unc, env.A_unc)
    if traj is not None:
        J["traj"] = {k: np.array(v) for k, v in traj.items()}
    return J
