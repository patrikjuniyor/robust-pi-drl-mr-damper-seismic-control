"""
simcore.py  --  Simulation core for semi-active control of shear buildings with MR dampers.

* n-storey shear-building model (SI units: m, kg, N, s)
* MR damper: modified Bouc-Wen model (Spencer et al. 1997) with first-order voltage dynamics
  (damper-internal units: cm, N, V as in Dyke et al. 1996)
* batched fixed-step RK4 integrator compiled with numba
* synthetic non-stationary Kanai-Tajimi / Clough-Penzien ground motions (+ optional pulse)
* recorded ground motions (time-compressed for the scaled model)
"""
import os
import numpy as np
from numba import njit
from scipy import signal

G = 9.81
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# ----------------------------------------------------------------------------------------------
# Building / damper definitions
# ----------------------------------------------------------------------------------------------
# damper parameter vector layout
(C0A, C0B, K0, C1A, C1B, K1, X0, ALA, ALB, GAM, BET, AA, NEXP, ETA) = range(14)

# Prototype MR damper identified by Dyke et al. (1996) / Spencer et al. (1997)  [N, cm, s, V]
MR_DYKE = np.array([21.0, 3.50, 46.9, 283.0, 2.95, 5.00, 14.3, 140.0, 695.0, 363.0, 363.0, 301.0, 2.0, 190.0])
VMAX_DYKE = 2.25
# Shear-mode MR damper, simple Bouc-Wen model f = c0 xdot + alpha z   [N, cm, s, V]
MR_SHEAR = np.array([0.0064, 0.0052, 0.0, 0.0, 0.0, 0.0, 0.0, 8.66, 8.86, 300.0, 300.0, 120.0, 2.0, 80.0])


class Building:
    """Nominal structural model."""

    def __init__(self, name="dyke3"):
        self.name = name
        if name == "dyke3":
            # three-storey scaled test structure, Dyke et al. (1996)
            self.n = 3
            self.mass = np.array([98.3, 98.3, 98.3])
            self.K = 1e5 * np.array([[12.0, -6.84, 0.0], [-6.84, 13.7, -6.84], [0.0, -6.84, 6.84]])
            self.C = np.array([[175.0, -50.0, 0.0], [-50.0, 100.0, -50.0], [0.0, -50.0, 50.0]])
            self.kstory = 1e5 * np.array([5.16, 6.84, 6.84])
            self.dstory = np.array([0], dtype=np.int64)      # damper between ground and 1st floor
            self.dcount = np.array([1.0])
            self.mr = MR_DYKE.copy()
            self.vmax = VMAX_DYKE
            self.time_scale = 5.0                              # earthquakes reproduced 5x faster
            self.fscale = 1000.0                               # force normalisation [N]
            self.pga_range = (0.05, 1.0); self.T_train = 4.0
        elif name == "six":
            # six-storey laboratory-scale shear frame with four shear-mode MR dampers
            # (two on each of the two lowest storeys), patterned after Jansen & Dyke (2000)
            self.n = 6
            self.mass = np.full(6, 22.7)                       # 0.227 N s^2/cm
            self.kstory = np.full(6, 29700.0)                  # 297 N/cm
            mats = self.story_matrices()
            self.K = sum(self.kstory[i] * mats[i] for i in range(6))
            w2, phi = np.linalg.eigh(self.K / 22.7)
            phi = phi / np.sqrt(22.7)                          # mass-normalised modes
            M = np.diag(self.mass)
            self.C = M @ phi @ np.diag(2 * 0.005 * np.sqrt(w2)) @ phi.T @ M     # 0.5 % in every mode
            self.dstory = np.array([0, 1], dtype=np.int64)
            self.dcount = np.array([2.0, 2.0])
            self.mr = MR_SHEAR.copy()
            self.vmax = 5.0
            self.time_scale = 1.0
            self.fscale = 40.0
            self.pga_range = (0.015, 0.20); self.T_train = 16.0  # amplitude-scaled excitation (lab model)
        else:
            raise ValueError(name)
        self.nd = len(self.dstory)

    def story_matrices(self):
        """Unit-stiffness contribution matrices of each storey (for perturbing K)."""
        n = self.n
        mats = []
        for i in range(n):
            Ki = np.zeros((n, n))
            Ki[i, i] += 1.0
            if i > 0:
                Ki[i - 1, i - 1] += 1.0
                Ki[i, i - 1] -= 1.0
                Ki[i - 1, i] -= 1.0
            mats.append(Ki)
        return mats

    def modal(self, K=None, mass=None):
        K = self.K if K is None else K
        mass = self.mass if mass is None else mass
        w2, phi = np.linalg.eig(np.linalg.solve(np.diag(mass), K))
        idx = np.argsort(w2.real)
        return np.sqrt(w2.real[idx]) / 2 / np.pi, phi.real[:, idx]


def sample_params(bld, N, rng, level=1.0, nominal=False):
    """Domain randomisation of structural / device / sensing / actuation parameters.

    level scales the half-width of every range (level=1 -> training distribution).
    Returns a dict of per-environment arrays.
    """
    n, nd = bld.n, bld.nd
    def U(lo, hi, size):
        if nominal:
            return np.ones(size)
        return 1.0 + level * rng.uniform(lo - 1.0, hi - 1.0, size)
    p = {}
    p["sk"] = U(0.75, 1.25, (N, n))            # storey stiffness scale
    p["sm"] = U(0.90, 1.10, (N, n))            # floor mass scale
    p["sc"] = np.exp(level * rng.uniform(np.log(0.5), np.log(2.0), N)) if not nominal else np.ones(N)  # damping scale
    p["sf"] = U(0.80, 1.20, (N, nd))           # MR yield-force scale (alpha_a, alpha_b)
    p["sv"] = U(0.80, 1.20, (N, nd))           # MR viscous scale (c0, c1)
    p["seta"] = U(0.70, 1.30, (N, nd))         # MR response-rate scale (eta)
    if nominal:
        p["delay"] = np.zeros(N, dtype=np.int64)
        p["noise"] = np.zeros(N)
    else:
        p["delay"] = rng.integers(0, int(round(2 * level)) + 1, N)   # command delay in control steps
        p["noise"] = level * rng.uniform(0.0, 1.0, N)                # sensor-noise level (x reference std)
    return p


def build_matrices(bld, p):
    """Assemble per-environment matrices from randomised parameters."""
    N = p["sk"].shape[0]
    n, nd = bld.n, bld.nd
    mats = bld.story_matrices()
    K = np.repeat(bld.K[None], N, axis=0)
    for i in range(n):
        K = K + ((p["sk"][:, i] - 1.0) * bld.kstory[i])[:, None, None] * mats[i][None]
    mass = bld.mass[None] * p["sm"]
    C = bld.C[None] * p["sc"][:, None, None]
    MK = K / mass[:, :, None]
    MC = C / mass[:, :, None]
    DP = np.repeat(np.repeat(bld.mr[None, None], N, axis=0), nd, axis=1).copy()
    DP[:, :, ALA] *= p["sf"]; DP[:, :, ALB] *= p["sf"]
    for j in (C0A, C0B, C1A, C1B):
        DP[:, :, j] *= p["sv"]
    DP[:, :, ETA] *= p["seta"]
    return dict(K=np.ascontiguousarray(K), mass=np.ascontiguousarray(mass), MK=np.ascontiguousarray(MK),
                MC=np.ascontiguousarray(MC), DP=np.ascontiguousarray(DP))


# ----------------------------------------------------------------------------------------------
# numba kernels
# ----------------------------------------------------------------------------------------------
@njit(cache=True, fastmath=True, inline='always')
def _deriv(n, nd, e, s, ds, agv, MK, MC, minv, dstory, dcount, DP, vcmd, x0flag, acc_abs, fout):
    """State derivative for environment e.   s = [x(n), v(n), y(nd), z(nd), u(nd)]"""
    for j in range(nd):
        if dcount[j] <= 0.0:          # no device installed (uncontrolled twin): freeze the device states
            fout[j] = 0.0
            ds[2 * n + j] = 0.0; ds[2 * n + nd + j] = 0.0; ds[2 * n + 2 * nd + j] = 0.0
            continue
        st = dstory[j]
        if st == 0:
            dr = s[0]; dv = s[n]
        else:
            dr = s[st] - s[st - 1]; dv = s[n + st] - s[n + st - 1]
        xc = 100.0 * dr; xdc = 100.0 * dv
        y = s[2 * n + j]; z = s[2 * n + nd + j]; u = s[2 * n + 2 * nd + j]
        al = DP[e, j, ALA] + DP[e, j, ALB] * u
        c0 = DP[e, j, C0A] + DP[e, j, C0B] * u
        c1 = DP[e, j, C1A] + DP[e, j, C1B] * u
        if c1 > 0.0:      # modified Bouc-Wen model (Spencer et al. 1997)
            yd = (al * z + c0 * xdc + DP[e, j, K0] * (xc - y)) / (c0 + c1)
        else:             # simple Bouc-Wen model (shear-mode device): no internal dof
            yd = 0.0
        rel = xdc - yd
        az = abs(z)
        ne = DP[e, j, NEXP]
        if ne == 2.0:
            zd = -DP[e, j, GAM] * abs(rel) * z * az - DP[e, j, BET] * rel * z * z + DP[e, j, AA] * rel
        elif ne == 1.0:
            zd = -DP[e, j, GAM] * abs(rel) * z - DP[e, j, BET] * rel * az + DP[e, j, AA] * rel
        else:
            zd = -DP[e, j, GAM] * abs(rel) * z * az ** (ne - 1.0) - DP[e, j, BET] * rel * az ** ne + DP[e, j, AA] * rel
        if c1 > 0.0:
            f = (c1 * yd + DP[e, j, K1] * (xc - x0flag * DP[e, j, X0])) * dcount[j]
        else:
            f = (c0 * xdc + al * z) * dcount[j]
        fout[j] = f
        ds[2 * n + j] = yd
        ds[2 * n + nd + j] = zd
        ds[2 * n + 2 * nd + j] = -DP[e, j, ETA] * (u - vcmd[e, j])
    for i in range(n):
        a = 0.0
        for k in range(n):
            a -= MK[e, i, k] * s[k] + MC[e, i, k] * s[n + k]
        acc_abs[i] = a
    for j in range(nd):
        st = dstory[j]
        acc_abs[st] -= fout[j] * minv[st]
        if st > 0:
            acc_abs[st - 1] += fout[j] * minv[st - 1]
    for i in range(n):
        ds[i] = s[n + i]
        ds[n + i] = acc_abs[i] - agv


@njit(cache=True, fastmath=True)
def sim_block(S, vcmd, ag, k0, nsub, dt, MK, MC, mass, K, dstory, dcount, DP, x0flag, active,
              m_acc, m_f, e_s, e_a, pk_d, pk_a, pk_x, pk_f, ss_d, ss_a, sp_d, sp_a):
    """Advance all environments by one control step (nsub RK4 sub-steps).

    S      [N, 2n+3nd]  state (in/out)
    vcmd   [N, nd]      command voltage held over the control step
    ag     [N, L]       ground acceleration sampled at the integration step
    Outputs: m_acc [N,n] absolute accelerations and m_f [N,nd] damper forces at end of step,
             e_s, e_a [N]  step-averaged strain energy and inertial 'acceleration energy',
             sp_d, sp_a [N,n] largest |drift| and |absolute acceleration| within the control step,
             running peaks (pk_*) and sums of squares (ss_*).
    """
    N = S.shape[0]
    n = MK.shape[1]
    nd = dstory.shape[0]
    ns = 2 * n + 3 * nd
    k1 = np.zeros(ns); k2 = np.zeros(ns); k3 = np.zeros(ns); k4 = np.zeros(ns); tmp = np.zeros(ns)
    s = np.zeros(ns)
    acc = np.zeros(n); fo = np.zeros(nd); minv = np.zeros(n)
    for e in range(N):
        if not active[e]:
            continue
        for i in range(n):
            minv[i] = 1.0 / mass[e, i]
            sp_d[e, i] = 0.0; sp_a[e, i] = 0.0
        for i in range(ns):
            s[i] = S[e, i]
        es = 0.0; ea = 0.0
        _deriv(n, nd, e, s, k1, ag[e, k0], MK, MC, minv, dstory, dcount, DP, vcmd, x0flag, acc, fo)
        for q in range(nsub):
            a0 = ag[e, k0 + q]; a1 = ag[e, k0 + q + 1]
            # --- adaptive micro-stepping: keep (stiffness of the hysteretic variable) x (step) <= 1.4
            lam = 0.0
            for j in range(nd):
                if dcount[j] > 0.0:
                    st = dstory[j]
                    dv = s[n] if st == 0 else s[n + st] - s[n + st - 1]
                    rel = abs(100.0 * dv - k1[2 * n + j])
                    u = s[2 * n + 2 * nd + j]
                    al = DP[e, j, ALA] + DP[e, j, ALB] * u
                    cc = DP[e, j, C0A] + DP[e, j, C0B] * u + DP[e, j, C1A] + DP[e, j, C1B] * u
                    if DP[e, j, C1A] > 0.0:
                        lj = 2.0 * (DP[e, j, GAM] + DP[e, j, BET]) * rel + DP[e, j, AA] * al / cc
                    else:
                        gb = DP[e, j, GAM] + DP[e, j, BET]; ne_ = DP[e, j, NEXP]
                        lj = ne_ * gb * rel * (DP[e, j, AA] / gb) ** ((ne_ - 1.0) / ne_)
                    if lj > lam: lam = lj
            mm = int(np.ceil(lam * dt / 1.4))
            if mm < 1: mm = 1
            if mm > 400: mm = 400
            h = dt / mm
            for r in range(mm):
                b0 = a0 + (a1 - a0) * r / mm; b1 = a0 + (a1 - a0) * (r + 1) / mm; bm = 0.5 * (b0 + b1)
                for i in range(ns): tmp[i] = s[i] + 0.5 * h * k1[i]
                _deriv(n, nd, e, tmp, k2, bm, MK, MC, minv, dstory, dcount, DP, vcmd, x0flag, acc, fo)
                for i in range(ns): tmp[i] = s[i] + 0.5 * h * k2[i]
                _deriv(n, nd, e, tmp, k3, bm, MK, MC, minv, dstory, dcount, DP, vcmd, x0flag, acc, fo)
                for i in range(ns): tmp[i] = s[i] + h * k3[i]
                _deriv(n, nd, e, tmp, k4, b1, MK, MC, minv, dstory, dcount, DP, vcmd, x0flag, acc, fo)
                for i in range(ns): s[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
                # outputs at the new state (also provides k1 of the next micro-step)
                _deriv(n, nd, e, s, k1, b1, MK, MC, minv, dstory, dcount, DP, vcmd, x0flag, acc, fo)
            e1 = 0.0
            for i in range(n):
                kx = 0.0
                for k in range(n):
                    kx += K[e, i, k] * s[k]
                e1 += 0.5 * s[i] * kx
                d = s[i] if i == 0 else s[i] - s[i - 1]
                ad = abs(d); aa = abs(acc[i]); ax = abs(s[i])
                if ad > pk_d[e, i]: pk_d[e, i] = ad
                if aa > pk_a[e, i]: pk_a[e, i] = aa
                if ax > pk_x[e, i]: pk_x[e, i] = ax
                if ad > sp_d[e, i]: sp_d[e, i] = ad
                if aa > sp_a[e, i]: sp_a[e, i] = aa
                ss_d[e, i] += d * d
                ss_a[e, i] += acc[i] * acc[i]
                ea += 0.5 * mass[e, i] * acc[i] * acc[i]
            es += e1
            for j in range(nd):
                af = abs(fo[j])
                if af > pk_f[e, j]: pk_f[e, j] = af
        for i in range(ns):
            S[e, i] = s[i]
        e_s[e] = es / nsub
        e_a[e] = ea / nsub
        for i in range(n): m_acc[e, i] = acc[i]
        for j in range(nd): m_f[e, j] = fo[j]


# ----------------------------------------------------------------------------------------------
# Ground motions
# ----------------------------------------------------------------------------------------------
def load_records(time_scale=1.0):
    """Recorded accelerograms (in g).  Returns dict name -> (dt, acc[g]) with time compressed."""
    recs = {}
    d = np.loadtxt(os.path.join(DATA_DIR, "elcentro_NS.dat"))
    recs["El Centro 1940"] = (0.02 / time_scale, d[:, 1].copy())
    names = {"Kobe": "Kobe 1995", "Northridge": "Northridge 1994", "ChiChi": "Chi-Chi 1999",
             "Kocaeli": "Kocaeli 1999", "Loma_Prieta": "Loma Prieta 1989", "Landers": "Landers 1992",
             "Imperial_Valley": "Imperial Valley 1979", "Friuli": "Friuli 1976",
             "Hollister": "Hollister 1961", "Trinidad": "Trinidad 1983"}
    for f, nm in names.items():
        d = np.loadtxt(os.path.join(DATA_DIR, f + ".dat"), skiprows=5)
        recs[nm] = (0.01 / time_scale, d[:, 1].copy())
    return recs


def resample(dt_rec, acc, dt, T=None):
    t_rec = np.arange(len(acc)) * dt_rec
    T = t_rec[-1] if T is None else T
    t = np.arange(int(round(T / dt)) + 2) * dt
    return np.interp(t, t_rec, acc, right=0.0)


def synth_motions(N, T, dt, rng, time_scale=1.0, pga_range=(0.05, 1.0)):
    """Non-stationary filtered-white-noise accelerograms (Kanai-Tajimi + Clough-Penzien),
    with random site frequency/damping, envelope, PGA and an optional velocity-pulse-like
    component.  All frequencies are multiplied by time_scale (scaled-model similitude).
    Returns ag [N, L] in m/s^2 and the PGA vector."""
    L = int(round(T / dt)) + 2
    t = np.arange(L) * dt
    out = np.zeros((N, L))
    pgas = np.zeros(N)
    for e in range(N):
        wg = time_scale * np.exp(rng.uniform(np.log(6.0), np.log(35.0)))   # rad/s
        zg = rng.uniform(0.3, 0.9)
        wf = 0.1 * wg; zf = 0.6
        w = rng.standard_normal(L)
        # Kanai-Tajimi:  H1(s) = (2 zg wg s + wg^2)/(s^2 + 2 zg wg s + wg^2)
        b, a = signal.bilinear([2 * zg * wg, wg ** 2], [1.0, 2 * zg * wg, wg ** 2], fs=1.0 / dt)
        y = signal.lfilter(b, a, w)
        # Clough-Penzien high-pass: H2(s) = s^2/(s^2 + 2 zf wf s + wf^2)
        b, a = signal.bilinear([1.0, 0.0, 0.0], [1.0, 2 * zf * wf, wf ** 2], fs=1.0 / dt)
        y = signal.lfilter(b, a, y)
        t1 = rng.uniform(0.05, 0.25) * T
        t2 = t1 + rng.uniform(0.1, 0.4) * T
        cdec = rng.uniform(2.0, 6.0) / (T - t2)
        env = np.where(t < t1, (t / t1) ** 2, np.where(t < t2, 1.0, np.exp(-cdec * (t - t2))))
        y = y * env
        y /= np.abs(y).max() + 1e-12
        if rng.uniform() < 0.3:          # pulse-like component (Gabor wavelet in acceleration)
            fp = time_scale * rng.uniform(0.4, 2.5)
            tp = rng.uniform(0.15, 0.5) * T
            ncyc = rng.uniform(1.0, 2.5)
            sig = ncyc / fp / 2.0
            pulse = np.exp(-0.5 * ((t - tp) / sig) ** 2) * np.cos(2 * np.pi * fp * (t - tp) + rng.uniform(0, 2 * np.pi))
            y = y + rng.uniform(0.3, 1.0) * pulse
            y /= np.abs(y).max()
        pga = np.exp(rng.uniform(np.log(pga_range[0]), np.log(pga_range[1])))
        out[e] = y * pga * G
        pgas[e] = pga
    return out, pgas
