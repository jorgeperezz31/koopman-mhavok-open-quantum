"""
jhavok_setup.py
===============

Simulation and mHAVOK routines behind

    J. E. Perez-Garcia, C. Colchero, and J. C. Gutierrez-Vega,
    "Spectral analysis of the Koopman operator as a framework for recovering
    Hamiltonian parameters in open quantum systems",
    Phys. Rev. A 113, 042222 (2026).

Consolidated from the working notebooks of the project (Sep 2025 - Feb 2026).
The companion module ``jhavok_plots`` draws the figures.

Contents
--------
1. Default parameters (Table I)
2. Lindblad simulations with QuTiP
   (2D QHO, Kerr terms, Jaynes-Cummings, parametric frequency modulation)
3. mHAVOK: block Hankel matrix, SVD, linear/nonlinear split, forced linear model
4. Optimal cutoff rank
5. Spectral analysis of A: recovering Hamiltonian parameters
6. Baseline estimators (matrix pencil, FFT) and the parameter sweeps
   that produce the data of Figs. 3 and 5

Note on pysindy
---------------
The original code used pysindy (FiniteDifference(order=4), PolynomialLibrary(degree=1),
STLSQ(threshold=0, normalize_columns=True)). With threshold 0, pysindy ends with an
"unbias" step that refits by ordinary least squares on the whole library, so the
regression is exactly ``lstsq([1, V], dV/dt)``. The NumPy implementation below
reproduces pysindy 1.7.5 and 2.1.0 to round-off, so pysindy is no longer needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
import pandas as pd
import qutip as qt
from scipy.linalg import block_diag, schur
from scipy.signal import StateSpace, detrend, get_window, lsim


# =============================================================================
# 1. Default parameters (Table I of the paper)
# =============================================================================

OMEGA_X = 2 * np.pi     # oscillation frequency of the x mode
OMEGA_Y = np.pi         # oscillation frequency of the y mode
T_BATH = 2.0            # environment temperature (hbar = k_B = 1)
KAPPA = 0.1             # damping rate
N_FOCK = 10             # Fock-space truncation of each mode
T_FINAL = 50.0          # simulation time
DT = 0.01               # time step
ALPHA = 1.0             # coherent-state amplitude of the initial state
TAU = 0.95              # R^2 threshold separating linear / nonlinear modes
HANKEL_ROWS = 100       # rows of the block Hankel matrix, m * q


def embedding_dimension(num_channels, hankel_rows=HANKEL_ROWS):
    """Number of delays m such that the block Hankel matrix has ~hankel_rows rows.

    All runs in the paper used m = ceil(100 / q) for q observables
    (m = 25 for the four quadratures, m = 17 for the six Jaynes-Cummings channels).
    """
    return int(np.ceil(hankel_rows / num_channels))


def time_grid(t_final=T_FINAL, dt=DT):
    return np.arange(0.0, t_final, dt)


# =============================================================================
# 2. Lindblad simulations (QuTiP)
# =============================================================================

def bose_einstein(omega, T):
    """Mean thermal occupation, Eq. (33)."""
    return 1.0 / (np.exp(omega / T) - 1.0)


def mode_operators(Nx=N_FOCK, Ny=N_FOCK, qubit=False):
    """Ladder, quadrature and number operators of the 2D QHO.

    With ``qubit=True`` a two-level system is appended as a third tensor
    factor and sigma_-, sigma_+, sigma_z are included.
    """
    factors = [qt.qeye(Nx), qt.qeye(Ny)] + ([qt.qeye(2)] if qubit else [])

    def embed(op, k):
        ops = list(factors)
        ops[k] = op
        return qt.tensor(*ops)

    a_x = embed(qt.destroy(Nx), 0)
    a_y = embed(qt.destroy(Ny), 1)
    ops = {
        "a_x": a_x,
        "a_y": a_y,
        "qx": (a_x + a_x.dag()) / np.sqrt(2),
        "px": (a_x - a_x.dag()) / (1j * np.sqrt(2)),
        "qy": (a_y + a_y.dag()) / np.sqrt(2),
        "py": (a_y - a_y.dag()) / (1j * np.sqrt(2)),
        "nx": a_x.dag() * a_x,
        "ny": a_y.dag() * a_y,
    }
    if qubit:
        sm = embed(qt.sigmam(), 2)
        ops.update(sm=sm, sp=sm.dag(), sz=embed(qt.sigmaz(), 2))
    return ops


def qho_hamiltonian(ops, omega_x=OMEGA_X, omega_y=OMEGA_Y):
    """H0 of Eq. (38) (zero of energy shifted, hbar = 1)."""
    return omega_x * ops["nx"] + omega_y * ops["ny"]


def kerr_hamiltonian(ops, chi_x=0.0, chi_y=0.0, chi_xy=0.0):
    """Self- and cross-Kerr terms, Eq. (41)."""
    a_x, a_y = ops["a_x"], ops["a_y"]
    return (0.5 * chi_x * a_x.dag() ** 2 * a_x ** 2
            + 0.5 * chi_y * a_y.dag() ** 2 * a_y ** 2
            + chi_xy * ops["nx"] * ops["ny"])


def common_bath_c_ops(ops, kappa=KAPPA, nbar_x=0.0, nbar_y=0.0, phi=0.0,
                      averaged=False):
    """Jump operators of the common bath, Eqs. (32), (34) and (35).

    L_c = a_x + exp(i phi) a_y. By default one decay/excitation pair is added per
    mode (four operators, as in most of the paper). ``averaged=True`` uses a single
    pair with nbar = (nbar_x + nbar_y) / 2, which is what the time-dependent
    simulation of Fig. 8 used.
    """
    L = ops["a_x"] + np.exp(1j * phi) * ops["a_y"]
    if averaged:
        nbar = 0.5 * (nbar_x + nbar_y)
        return [np.sqrt(kappa * (nbar + 1.0)) * L, np.sqrt(kappa * nbar) * L.dag()]
    return [
        np.sqrt(kappa * (nbar_x + 1.0)) * L,
        np.sqrt(kappa * nbar_x) * L.dag(),
        np.sqrt(kappa * (nbar_y + 1.0)) * L,
        np.sqrt(kappa * nbar_y) * L.dag(),
    ]


def _thermal_occupations(omega_x, omega_y, T, nbar):
    if nbar is not None:
        return nbar
    return bose_einstein(omega_x, T), bose_einstein(omega_y, T)


def coherent_product_state(alpha=ALPHA, Nx=N_FOCK, Ny=N_FOCK, qubit_ket=None):
    kets = [qt.coherent(Nx, alpha), qt.coherent(Ny, alpha)]
    if qubit_ket is not None:
        kets.append(qubit_ket)
    psi0 = qt.tensor(*kets)
    return psi0 * psi0.dag()


def simulate_2d_qho(chi_x=0.0, chi_y=0.0, chi_xy=0.0, kappa=KAPPA, T=T_BATH,
                    open_system=True, nbar=None, phi=0.0,
                    omega_x=OMEGA_X, omega_y=OMEGA_Y, alpha=ALPHA,
                    Nx=N_FOCK, Ny=N_FOCK, t_final=T_FINAL, dt=DT,
                    return_numbers=False):
    """2D QHO (+ optional Kerr terms) coupled to a common thermal bath.

    Returns ``tlist`` and ``traj`` with columns [<x>, <y>, <p_x>, <p_y>]
    (the column order used to build the Hankel matrix in the paper).
    ``nbar = (nbar_x, nbar_y)`` overrides the Bose-Einstein occupations;
    ``open_system=False`` removes all jump operators (Fig. 6).
    With ``return_numbers=True`` a third output {'nx': ..., 'ny': ...} is returned.
    """
    ops = mode_operators(Nx, Ny)
    H = qho_hamiltonian(ops, omega_x, omega_y) + kerr_hamiltonian(ops, chi_x, chi_y, chi_xy)

    if open_system:
        nbar_x, nbar_y = _thermal_occupations(omega_x, omega_y, T, nbar)
        c_ops = common_bath_c_ops(ops, kappa, nbar_x, nbar_y, phi)
    else:
        c_ops = []

    rho0 = coherent_product_state(alpha, Nx, Ny)
    tlist = time_grid(t_final, dt)
    e_ops = [ops["qx"], ops["qy"], ops["px"], ops["py"], ops["nx"], ops["ny"]]
    expect = qt.mesolve(H, rho0, tlist, c_ops, e_ops=e_ops).expect

    traj = np.column_stack([np.real(e) for e in expect[:4]])
    if return_numbers:
        return tlist, traj, {"nx": np.real(expect[4]), "ny": np.real(expect[5])}
    return tlist, traj


def simulate_jaynes_cummings(g=0.15, omega_q=2 * np.pi, gamma_relax=0.1,
                             gamma_phi=0.01, kappa=KAPPA, T=T_BATH,
                             omega_x=OMEGA_X, omega_y=OMEGA_Y, alpha=ALPHA,
                             Nx=N_FOCK, Ny=N_FOCK, t_final=T_FINAL, dt=DT):
    """2D QHO whose x mode is coupled to a qubit (Sec. IV C, Eqs. (45)-(47)).

    Each mode has its own thermal bath; the qubit has relaxation sqrt(gamma) sigma_-
    and dephasing sqrt(gamma_phi) sigma_z. The qubit starts in ``qt.basis(2, 0)``,
    as in the original code (in QuTiP's convention this is the sigma_z = +1 state).

    Returns ``tlist`` and ``traj`` with the six channels of Eq. (50):
    [<x>, <p_x>, <y>, <p_y>, Re<sigma_+>, Im<sigma_+>].
    """
    ops = mode_operators(Nx, Ny, qubit=True)
    a_x, sm, sp, sz = ops["a_x"], ops["sm"], ops["sp"], ops["sz"]

    H = (qho_hamiltonian(ops, omega_x, omega_y)
         + 0.5 * omega_q * sz
         + g * (a_x * sp + a_x.dag() * sm))

    nbar_x, nbar_y = bose_einstein(omega_x, T), bose_einstein(omega_y, T)
    c_ops = [
        np.sqrt(kappa * (nbar_x + 1.0)) * a_x,
        np.sqrt(kappa * nbar_x) * a_x.dag(),
        np.sqrt(kappa * (nbar_y + 1.0)) * ops["a_y"],
        np.sqrt(kappa * nbar_y) * ops["a_y"].dag(),
        np.sqrt(gamma_relax) * sm,
        np.sqrt(gamma_phi) * sz,
    ]

    rho0 = coherent_product_state(alpha, Nx, Ny, qubit_ket=qt.basis(2, 0))
    tlist = time_grid(t_final, dt)
    e_ops = [ops["qx"], ops["px"], ops["qy"], ops["py"], sp]
    qx_t, px_t, qy_t, py_t, sp_t = qt.mesolve(H, rho0, tlist, c_ops, e_ops=e_ops).expect

    traj = np.column_stack([np.real(qx_t), np.real(px_t), np.real(qy_t), np.real(py_t),
                            np.real(sp_t), np.imag(sp_t)])
    return tlist, traj


def _modulation(t, args):
    return args["delta"] * np.cos(args["omega_f"] * t)


def simulate_modulated_qho(delta=0.25, omega_f=np.pi / np.e, chi_x=0.0, chi_y=0.0,
                           chi_xy=0.0, kappa=KAPPA, T=T_BATH, phi=0.0,
                           omega_x=OMEGA_X, omega_y=OMEGA_Y, alpha=ALPHA,
                           Nx=N_FOCK, Ny=N_FOCK, t_final=T_FINAL, dt=DT):
    """2D QHO with H_f(t) = delta cos(omega_f t) a_x^dag a_x, Eqs. (51)-(52).

    Uses the averaged common bath (one decay/excitation pair), as in the original
    simulation of Fig. 8. Returns ``tlist`` and [<x>, <y>, <p_x>, <p_y>].
    """
    ops = mode_operators(Nx, Ny)
    H_static = qho_hamiltonian(ops, omega_x, omega_y) + kerr_hamiltonian(ops, chi_x, chi_y, chi_xy)
    H = [H_static, [ops["nx"], _modulation]]

    nbar_x, nbar_y = bose_einstein(omega_x, T), bose_einstein(omega_y, T)
    c_ops = common_bath_c_ops(ops, kappa, nbar_x, nbar_y, phi, averaged=True)

    rho0 = coherent_product_state(alpha, Nx, Ny)
    tlist = time_grid(t_final, dt)
    e_ops = [ops["qx"], ops["qy"], ops["px"], ops["py"]]
    expect = qt.mesolve(H, rho0, tlist, c_ops, e_ops=e_ops,
                        args={"delta": float(delta), "omega_f": float(omega_f)}).expect
    return tlist, np.column_stack([np.real(e) for e in expect])


# =============================================================================
# 3. mHAVOK (Appendix A 1)
# =============================================================================

def hankel_matrix(xdata, m):
    """Block Hankel matrix, Eqs. (A2)-(A3). Row i*q + j holds channel j delayed by i."""
    xdata = np.real(np.asarray(xdata))
    q = xdata.shape[1]
    n_snap = xdata.shape[0] - m
    H = np.zeros((q * m, n_snap))
    for i in range(m):
        H[i * q:(i + 1) * q, :] = xdata[i:i + n_snap, :].T
    return H


def hankel_svd(H):
    """Economy SVD H = U diag(S) V^T, Eq. (A4). Returns U, S, V (not V^T)."""
    U, S, Vh = np.linalg.svd(H, full_matrices=False)
    return U, S, Vh.conj().T


def _fd_weights(offsets, dt, deriv=1):
    """Finite-difference weights for the given stencil offsets (Vandermonde solve)."""
    h = np.asarray(offsets, dtype=float) * dt
    n = len(h)
    M = h[np.newaxis, :] ** np.arange(n)[:, np.newaxis]
    b = np.zeros(n)
    b[deriv] = factorial(deriv)
    return np.linalg.solve(M, b)


def finite_difference(x, dt, order=4):
    """First time derivative along axis 0, as pysindy's FiniteDifference(order=order).

    Centered (order + 1)-point stencil in the interior and one-sided
    (order + 1)-point stencils for the order/2 points at each end.
    """
    x = np.asarray(x, dtype=float)
    n, h = x.shape[0], order // 2
    dx = np.empty_like(x)

    w = _fd_weights(np.arange(-h, h + 1), dt)
    interior = np.zeros_like(x[h:n - h])
    for j, wj in enumerate(w):
        if wj != 0:
            interior = interior + wj * x[j:n - 2 * h + j]
    dx[h:n - h] = interior

    left = np.arange(order + 1)
    right = np.arange(n - order - 1, n)
    for i in range(h):
        dx[i] = np.tensordot(_fd_weights(left - i, dt), x[left], axes=1)
        k = n - 1 - i
        dx[k] = np.tensordot(_fd_weights(right - k, dt), x[right], axes=1)
    return dx


def r2_score(y, y_hat):
    """Coefficient of determination, Eq. (A6) (same convention as sklearn)."""
    y, y_hat = np.asarray(y), np.asarray(y_hat)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _affine_least_squares(X, Y):
    """Fit Y ~ [1, X] Xi by least squares. Returns the library Theta and Xi."""
    Theta = np.column_stack([np.ones(X.shape[0]), X])
    Xi = np.linalg.lstsq(Theta, Y, rcond=None)[0]
    return Theta, Xi


@dataclass
class MHAVOKResult:
    """Output of :func:`run_mhavok`.

    A, B          dynamical matrices of the forced linear model, Eq. (A10)
    good_dims     indices I_c of the linear components, Eq. (A7)
    forcing_dims  indices I_f of the nonlinear (forcing) components, Eq. (A8)
    r2_modes      first-pass R^2 of each of the r modes, Eq. (A6)
    x_recovered   reconstructed Hankel rows; the first q columns are the observables
    r2_mean       mean R^2 between the input observables and their reconstruction
    """
    A: np.ndarray
    B: np.ndarray
    good_dims: list
    forcing_dims: list
    r2_modes: list
    Vr: np.ndarray
    U: np.ndarray
    S: np.ndarray
    t_out: np.ndarray
    y_out: np.ndarray
    system: StateSpace
    x_recovered: np.ndarray
    r2_each: list
    r: int
    m: int

    @property
    def r2_mean(self):
        return float(np.mean(self.r2_each))

    @property
    def eigenvalues(self):
        """Spectrum of A sorted by magnitude (approximates the Koopman spectrum)."""
        return sorted_eigenvalues(self.A)


def mhavok_regression(V, r, dt, tau=TAU):
    """Linear/nonlinear split and forced linear model on the first r right-singular vectors.

    Returns a dict with A, B, good_dims, forcing_dims, r2_modes, Vr, t_out, y_out, system.
    """
    Vr = V[:, :r]
    Vr_dot = finite_difference(Vr, dt, order=4)

    # (A5)-(A8): one linear fit of all r modes, keep those with R^2 >= tau as linear
    Theta, Xi = _affine_least_squares(Vr, Vr_dot)
    r2_modes = [r2_score(Vr_dot[:, i], Theta @ Xi[:, i]) for i in range(r)]
    good_dims = [i for i, s in enumerate(r2_modes) if s >= tau]
    forcing_dims = [i for i in range(r) if i not in good_dims]
    if not good_dims:
        raise ValueError(f"r = {r}: no mode passed the R^2 >= {tau} test, so A is empty.")

    # (A10): refit the linear modes using linear + forcing modes as regressors
    V_core, V_force = Vr[:, good_dims], Vr[:, forcing_dims]
    _, Xi2 = _affine_least_squares(np.hstack([V_core, V_force]), Vr_dot[:, good_dims])
    n_core, n_force = len(good_dims), len(forcing_dims)
    A = Xi2[1:1 + n_core, :].T
    B = Xi2[1 + n_core:1 + n_core + n_force, :].T

    # (A11): simulate x' = A x + B u with u the measured forcing modes
    system = StateSpace(A, B, np.eye(n_core), np.zeros((n_core, n_force)))
    steps = np.arange(V.shape[0] - 1)
    u_in = Vr[np.ix_(steps, forcing_dims)]
    t_vec = np.arange(len(u_in)) * dt
    t_out, y_out, _ = lsim(system, U=u_in, T=t_vec, X0=Vr[0, good_dims])
    y_out = np.asarray(y_out).reshape(len(t_out), -1)

    return dict(A=A, B=B, good_dims=good_dims, forcing_dims=forcing_dims,
                r2_modes=r2_modes, Vr=Vr, t_out=t_out, y_out=y_out, system=system)


def originalcoords(U, S, good_dims, y_out):
    """Map the simulated linear coordinates back to Hankel rows, Eq. (A12)."""
    U_core = U[:, good_dims]
    S_core = np.diag(S[good_dims])
    return (U_core @ S_core @ y_out.T).T


def run_mhavok(traj, dt, r, m=None, tau=TAU):
    """Full mHAVOK pipeline on a (time x q) array of observables.

    ``m`` defaults to :func:`embedding_dimension` (about 100 Hankel rows).
    """
    traj = np.real(np.asarray(traj))
    if m is None:
        m = embedding_dimension(traj.shape[1])
    U, S, V = hankel_svd(hankel_matrix(traj, m))
    fit = mhavok_regression(V, r, dt, tau)
    x_rec = originalcoords(U, S, fit["good_dims"], fit["y_out"])
    r2_each = [r2_score(traj[:len(x_rec), i], x_rec[:, i]) for i in range(traj.shape[1])]
    return MHAVOKResult(A=fit["A"], B=fit["B"], good_dims=fit["good_dims"],
                        forcing_dims=fit["forcing_dims"], r2_modes=fit["r2_modes"],
                        Vr=fit["Vr"], U=U, S=S, t_out=fit["t_out"], y_out=fit["y_out"],
                        system=fit["system"], x_recovered=x_rec, r2_each=r2_each,
                        r=r, m=m)


def HAVOK(xdata, dt, r=19, m=35, lam=0, tau=TAU):
    """Backward-compatible wrapper returning the tuple of the original code:

    good_dims, forcing_dims, Vr, A, B, t_out, y_out, sys, r2_modes, V_core, V_force, S, U
    """
    if lam != 0:
        raise NotImplementedError("Only lam = 0 (plain least squares) is implemented; "
                                  "all runs in the paper used lam = 0.")
    res = run_mhavok(xdata, dt, r=r, m=m, tau=tau)
    return (res.good_dims, res.forcing_dims, res.Vr, res.A, res.B, res.t_out, res.y_out,
            res.system, res.r2_modes, res.Vr[:, res.good_dims], res.Vr[:, res.forcing_dims],
            res.S, res.U)


# Names used in the dated notebooks
Hankel_v1 = hankel_matrix


def SVD(H):
    return hankel_svd(H)


# =============================================================================
# 4. Optimal cutoff rank (Appendix A 2)
# =============================================================================

def evaluate_ranks(traj, dt, r_min, r_max, m=None, tau=TAU):
    """Condition number of B and mean reconstruction R^2 for r = r_min..r_max.

    Ranks with no linear component are skipped.
    """
    metrics = {}
    for r in range(r_min, r_max + 1):
        try:
            res = run_mhavok(traj, dt, r=r, m=m, tau=tau)
        except ValueError:
            continue
        if res.B.size:
            svals = np.linalg.svd(res.B, compute_uv=False)
            cond_B = svals[0] / (svals[-1] + 1e-12)
        else:
            cond_B = 0.0
        metrics[r] = {"cond_number_B": cond_B, "mean_R2": res.r2_mean,
                      "n_linear": len(res.good_dims), "n_forcing": len(res.forcing_dims)}
    return metrics


def select_optimal_rank(metrics, q_percent=100):
    """Rank selection as implemented in the original code (``analyze_q_scores``).

    Keeps the top ``q_percent`` % of ranks by condition number of B and, among
    them, returns the one with the smallest Q = 1 - mean R^2.
    With q_percent = 100 (used throughout) this is the rank with the best reconstruction.
    """
    ranked = sorted(metrics, key=lambda r: metrics[r]["cond_number_B"], reverse=True)
    kept = sorted(ranked[:int(np.ceil(len(ranked) * q_percent / 100.0))])
    scores = [(r, 1.0 - metrics[r]["mean_R2"]) for r in kept if metrics[r]["mean_R2"] > 0]
    scores.sort(key=lambda rq: rq[1])
    return scores[0][0]


def metrics_table(metrics):
    return pd.DataFrame.from_dict(metrics, orient="index").rename_axis("r")


def find_optimal_rank(traj, dt, r_min=4, r_max=30, m=None, tau=TAU, q_percent=100):
    """Sweep r and return (r_opt, metrics DataFrame)."""
    metrics = evaluate_ranks(traj, dt, r_min, r_max, m=m, tau=tau)
    return select_optimal_rank(metrics, q_percent), metrics_table(metrics)


# =============================================================================
# 5. Spectral analysis of A
# =============================================================================

def percent_error(est, true):
    """Eq. (37)."""
    return 100.0 * np.abs(est - true) / (np.abs(true) if true != 0 else 1.0)


def sorted_eigenvalues(A):
    """Eigenvalues of A sorted by magnitude; within a conjugate pair, Im > 0 first."""
    mu = np.linalg.eigvals(A)
    order = np.lexsort((-mu.imag, np.round(np.abs(mu), 10)))
    return mu[order]


def analyze_linear_oscillators(A, tol=1e-10, sort_by="omega"):
    """Real Schur form of A as a sum of damped oscillators (Sec. IV A).

    Returns a dict with the Schur factors 'Z', 'T', the oscillatory 2x2 'blocks'
    (each with 'kappa' = -2 Re(lambda) and 'omega' = |Im(lambda)|) and the canonical
    block-diagonal 'A_canon' = diag([[-k/2, w], [-w, -k/2]], ...).
    """
    T, Z = schur(A, output="real")
    n = A.shape[0]
    i = 0
    blocks = []
    while i < n:
        if i + 1 < n and abs(T[i + 1, i]) > tol:
            a = 0.5 * (T[i, i] + T[i + 1, i + 1])
            b, c = T[i, i + 1], T[i + 1, i]
            blocks.append({"idx": (i, i + 1), "a": a, "b": b, "c": c,
                           "kappa": -2.0 * a, "omega": np.sqrt(max(0.0, -b * c))})
            i += 2
        else:
            a = T[i, i]
            blocks.append({"idx": (i,), "a": a, "b": 0.0, "c": 0.0,
                           "kappa": -2.0 * a, "omega": 0.0})
            i += 1

    osc = [b for b in blocks if len(b["idx"]) == 2 and b["omega"] > tol]
    if sort_by == "omega":
        order = np.argsort([b["omega"] for b in osc])[::-1]
    elif sort_by == "kappa":
        order = np.argsort([b["kappa"] for b in osc])
    else:
        order = np.arange(len(osc))

    canon = [np.array([[-osc[k]["kappa"] / 2, osc[k]["omega"]],
                       [-osc[k]["omega"], -osc[k]["kappa"] / 2]]) for k in order]
    A_canon = block_diag(*canon) if canon else np.zeros_like(A)
    return {"Z": Z, "T": T, "blocks": osc, "A_canon": A_canon, "order": list(order)}


def extract_params_from_A(A, omega_x=OMEGA_X, omega_y=OMEGA_Y):
    """omega_x, omega_y and kappa from the eigenvalues of A closest to the bare frequencies.

    kappa_est is the mean of -Re(lambda) over the two selected eigenvalues
    (the convention used to build the CSVs of Figs. 3 and 5).
    """
    lam = np.linalg.eigvals(A)

    def closest(target):
        return lam[np.argmin(np.abs(np.abs(lam.imag) - target))]

    lam_y, lam_x = closest(omega_y), closest(omega_x)
    return {
        "omega_x_est": float(abs(lam_x.imag)),
        "omega_y_est": float(abs(lam_y.imag)),
        "kappa_est": float(np.mean([-lam_y.real, -lam_x.real])),
        "eigvals": lam,
    }


def extract_kerr_from_A(A, idx_y=(2, 0), idx_x=(6, 4)):
    """Kerr strengths from the frequency ladder of A (Sec. IV B).

    Eigenvalues are sorted by magnitude; chi_y = |Im l[idx_y[0]]| - |Im l[idx_y[1]]|
    and likewise for chi_x. The defaults follow the text of the paper (third minus
    first and seventh minus fifth eigenvalue). Which indices are right depends on
    the spectrum, so always inspect ``sorted_eigenvalues(A)``.
    """
    lam = sorted_eigenvalues(A)
    w = np.abs(lam.imag)
    return float(w[idx_x[0]] - w[idx_x[1]]), float(w[idx_y[0]] - w[idx_y[1]]), lam


def dressed_frequencies(omega_x, omega_q, g):
    """omega_+- of the Jaynes-Cummings model, Eq. (49)."""
    delta = omega_q - omega_x
    root = np.sqrt(4 * g ** 2 + delta ** 2)
    return 0.5 * (omega_q + omega_x + root), 0.5 * (omega_q + omega_x - root)


def coupling_from_lower_dressed(omega_minus, omega_x, omega_q):
    """Invert Eq. (49) for g given omega_-."""
    delta = omega_q - omega_x
    return float(np.sqrt(np.abs(0.25 * (omega_x + omega_q - 2 * omega_minus) ** 2
                                - (delta / 2) ** 2)))


def estimate_jc_coupling(A, omega_x, omega_q, index=None):
    """Estimate g from the spectrum of A.

    omega_- lies below both bare frequencies, so by default the eigenfrequency
    closest to min(omega_x, omega_q) from below is taken as omega_-. Pass
    ``index`` (into ``sorted_eigenvalues(A)``) to pick it by hand instead.
    Returns (g_est, omega_minus_est).
    """
    lam = sorted_eigenvalues(A)
    w = np.abs(lam.imag)
    if index is None:
        below = w[w < min(omega_x, omega_q)]
        omega_minus = below.max()
    else:
        omega_minus = w[index]
    return coupling_from_lower_dressed(omega_minus, omega_x, omega_q), float(omega_minus)


# =============================================================================
# 6. Baseline estimators and parameter sweeps (data of Figs. 3 and 5)
# =============================================================================

def _preprocess(x, apply_detrend=True, window="hann"):
    x = np.asarray(x, float)
    if apply_detrend:
        x = detrend(x, type="constant")
    if window is not None:
        x = x * get_window(window, x.size, fftbins=True)
    return x


def fft_peak_freq(x, dt):
    """Angular frequency of the largest non-DC FFT bin."""
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=dt) * 2 * np.pi
    mag = np.abs(X)
    if freqs.size > 1:
        mag[0] = 0.0
    return freqs[np.argmax(mag)] if freqs.size else 0.0


def estimate_omega_fft_parabolic(x, dt, zero_pad=8, window="hann", detr=True):
    """FFT frequency estimate with zero padding and parabolic peak interpolation."""
    x = _preprocess(x, apply_detrend=detr, window=window)
    Z = int(2 ** np.ceil(np.log2(x.size))) * max(1, int(zero_pad))
    mag = np.abs(np.fft.rfft(x, n=Z))
    freqs = np.fft.rfftfreq(Z, d=dt) * 2 * np.pi
    if mag.size <= 1:
        return 0.0
    mag[0] = 0.0
    k = int(np.argmax(mag))
    if k == 0 or k == mag.size - 1:
        return float(freqs[k])
    m1, m0, p1 = mag[k - 1], mag[k], mag[k + 1]
    delta = 0.5 * (m1 - p1) / ((m1 - 2 * m0 + p1) + 1e-18)
    return float(((k + delta) / Z) * (2 * np.pi / dt))


def matrix_pencil_poles(x, dt, L=None, r=None, svd_frac=0.01):
    """Continuous-time poles of a real series by the matrix pencil method [7, 8]."""
    x = np.asarray(x, float)
    N = x.size
    L = N // 3 if L is None else L
    M = N - L
    if M <= 2:
        raise ValueError("Time series too short for chosen L.")
    X0 = np.column_stack([x[i:i + L] for i in range(M)])
    X1 = np.column_stack([x[i + 1:i + L + 1] for i in range(M)])
    U, s, Vh = np.linalg.svd(X0, full_matrices=False)
    if r is None:
        r = max(2, min(int(np.sum(s > svd_frac * s[0])), 40))
    U1, S1, V1 = U[:, :r], np.diag(s[:r]), Vh.conj().T[:, :r]
    z = np.linalg.eigvals(U1.T @ X1 @ V1 @ np.linalg.inv(S1))
    return np.log(z) / dt


def pick_dominant_pole(lam, x, dt, side="positive"):
    """Pole closest to the FFT peak (weighted) and least damped: (omega, kappa, pole)."""
    lam = np.asarray(lam, complex)
    lam = lam[np.abs(lam.imag) > 1e-9]
    if side == "positive":
        lam = lam[lam.imag > 0]
    if lam.size == 0:
        return np.nan, np.nan, np.nan
    w_fft = fft_peak_freq(x, dt)
    score = [5.0 * np.abs(l.imag - w_fft) + np.maximum(0.0, -l.real) for l in lam]
    best = lam[int(np.argmin(score))]
    return np.abs(best.imag), -best.real, best


def estimate_omega_kappa_mp(x, dt, L=None, r=None, svd_frac=0.01, side="positive",
                            detr=True, window="hann"):
    """Matrix-pencil estimate of (omega, kappa) for one channel."""
    x_p = _preprocess(x, apply_detrend=detr, window=window)
    lam = matrix_pencil_poles(x_p, dt, L=L, r=r, svd_frac=svd_frac)
    omega, kappa, _ = pick_dominant_pole(lam, x_p, dt, side=side)
    return omega, kappa


def _resolve_rank(r, traj, dt, m, tau, r_search):
    if r == "auto":
        return find_optimal_rank(traj, dt, *r_search, m=m, tau=tau)[0]
    return r


def sweep_kappa(kappa_values, chi_x=0.0, chi_y=0.0, chi_xy=0.0, r=4, m=None, tau=TAU,
                nbar=(0.0, 0.0), r_search=(10, 30), save_csv_path=None, verbose=True,
                **sim_kwargs):
    """mHAVOK vs matrix pencil vs FFT frequency/damping errors over kappa.

    Fig. 3:  chi = (0, 0), r = 4   (data/fig3_kappa_sweep_no_kerr.csv)
    Fig. 5b: chi = (2, 3), r = 20  (data/fig5b_kappa_sweep_kerr.csv)
    Both CSVs were generated with a vacuum bath, nbar = (0, 0).
    ``r='auto'`` picks the optimal rank for every kappa (slow).
    """
    rows = []
    for kappa in kappa_values:
        _, traj = simulate_2d_qho(chi_x=chi_x, chi_y=chi_y, chi_xy=chi_xy, kappa=kappa,
                                  nbar=nbar, **sim_kwargs)
        dt = sim_kwargs.get("dt", DT)
        omega_x = sim_kwargs.get("omega_x", OMEGA_X)
        omega_y = sim_kwargs.get("omega_y", OMEGA_Y)

        r_used = _resolve_rank(r, traj, dt, m, tau, r_search)
        res = run_mhavok(traj, dt, r=r_used, m=m, tau=tau)
        est = extract_params_from_A(res.A, omega_x, omega_y)

        wy_mp, ky_mp = estimate_omega_kappa_mp(traj[:, 1], dt)
        wx_mp, kx_mp = estimate_omega_kappa_mp(traj[:, 0], dt)
        k_mp = float(np.nanmean([ky_mp, kx_mp]))
        wy_fft = estimate_omega_fft_parabolic(traj[:, 1], dt)
        wx_fft = estimate_omega_fft_parabolic(traj[:, 0], dt)

        row = {
            "kappa_true": kappa, "r": r_used,
            "kappa_est_havok": est["kappa_est"],
            "omega_x_est_havok": est["omega_x_est"],
            "omega_y_est_havok": est["omega_y_est"],
            "err_kappa_pct_havok": percent_error(est["kappa_est"], kappa),
            "err_omega_x_pct_havok": percent_error(est["omega_x_est"], omega_x),
            "err_omega_y_pct_havok": percent_error(est["omega_y_est"], omega_y),
            "kappa_est_mp": k_mp, "omega_x_est_mp": wx_mp, "omega_y_est_mp": wy_mp,
            "err_kappa_pct_mp": percent_error(k_mp, kappa),
            "err_omega_x_pct_mp": percent_error(wx_mp, omega_x),
            "err_omega_y_pct_mp": percent_error(wy_mp, omega_y),
            "omega_x_est_fft": wx_fft, "omega_y_est_fft": wy_fft,
            "err_omega_x_pct_fft": percent_error(wx_fft, omega_x),
            "err_omega_y_pct_fft": percent_error(wy_fft, omega_y),
            "r2_mean_havok": res.r2_mean,
        }
        for method in ("havok", "mp", "fft"):
            row[f"avg_freq_err_pct_{method}"] = 0.5 * (row[f"err_omega_x_pct_{method}"]
                                                       + row[f"err_omega_y_pct_{method}"])
        rows.append(row)
        if verbose:
            print(f"kappa = {kappa:.2f} (r = {r_used}): freq. error mHAVOK "
                  f"{row['avg_freq_err_pct_havok']:.3g} %, MP {row['avg_freq_err_pct_mp']:.3g} %, "
                  f"FFT {row['avg_freq_err_pct_fft']:.3g} %")

    df = pd.DataFrame(rows).sort_values("kappa_true").reset_index(drop=True)
    if save_csv_path:
        df.to_csv(save_csv_path, index=False)
    return df


def sweep_kerr_grid(chi_x_values, chi_y_values, kappa=KAPPA, chi_xy=0.0, r=20, m=None,
                    tau=TAU, nbar=(0.0, 0.0), idx_y=(2, 0), idx_x=(6, 4),
                    r_search=(10, 30), save_csv_path=None, verbose=True, **sim_kwargs):
    """Kerr-strength recovery error over a (chi_x, chi_y) grid (Fig. 5a).

    The paper's heatmap (data/fig5a_kerr_sweep.csv) is a 10 x 10 grid with
    chi_x, chi_y in linspace(3, 5, 10). ``r='auto'`` picks the optimal rank per point.
    """
    rows = []
    dt = sim_kwargs.get("dt", DT)
    for chi_x in chi_x_values:
        for chi_y in chi_y_values:
            _, traj = simulate_2d_qho(chi_x=chi_x, chi_y=chi_y, chi_xy=chi_xy, kappa=kappa,
                                      nbar=nbar, **sim_kwargs)
            r_used = _resolve_rank(r, traj, dt, m, tau, r_search)
            res = run_mhavok(traj, dt, r=r_used, m=m, tau=tau)
            cx, cy, _ = extract_kerr_from_A(res.A, idx_y=idx_y, idx_x=idx_x)
            ex, ey = percent_error(cx, chi_x), percent_error(cy, chi_y)
            rows.append({"chi_x_true": chi_x, "chi_y_true": chi_y, "chi_x_est": cx,
                         "chi_y_est": cy, "err_chi_x_pct": ex, "err_chi_y_pct": ey,
                         "avg_err_pct": 0.5 * (ex + ey), "kappa_fixed": kappa, "r": r_used,
                         "r2_mean_havok": res.r2_mean})
            if verbose:
                print(f"chi = ({chi_x:.3g}, {chi_y:.3g}), r = {r_used}: "
                      f"mean error {0.5 * (ex + ey):.3g} %")
    df = pd.DataFrame(rows)
    if save_csv_path:
        df.to_csv(save_csv_path, index=False)
    return df
