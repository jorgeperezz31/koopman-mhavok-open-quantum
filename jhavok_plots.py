"""
jhavok_plots.py
===============

Figures of Phys. Rev. A 113, 042222 (2026). Each ``plot_figN`` function draws one
figure from data produced by ``jhavok_setup`` (or from the CSVs in ``data/``) and
returns the matplotlib Figure. Pass ``save_path`` to also write it to disk.

Styles (colors, font sizes, panel-label positions) are copied from the working
notebooks that produced the published figures.
"""

from __future__ import annotations

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jhavok_setup import sorted_eigenvalues


# =============================================================================
# Style
# =============================================================================

PAPER_RC = {
    "font.family": "serif",
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.dpi": 100,
    "axes.linewidth": 1.2,
}

NAVY = "#0A2362"        # Fig. 2
TEAL = "#03b389"        # Fig. 4, chi = (2, 3)
GREEN = "#058d0c"       # Fig. 4, chi = (5, 5)
DARK_BLUE = "#12277a"   # Fig. 6, r below the optimum
LIGHT_BLUE = "#309bc1"  # Fig. 6, high r
CRIMSON = "#A30E2C"     # Figs. 1 and 7
METHOD_STYLES = {       # Figs. 3 and 5b
    "mHAVOK": dict(color="#1f77b4", marker="o", linestyle="-", lw=2),
    "Matrix pencil": dict(color="#d62728", marker="s", linestyle="--", lw=2),
    "FFT": dict(color="#2ca02c", marker="^", linestyle="-.", lw=2),
}


def use_paper_style(usetex=None):
    """Apply the rcParams of the paper. LaTeX text is used when a LaTeX install is found."""
    if usetex is None:
        usetex = shutil.which("latex") is not None
    plt.rcParams.update({**PAPER_RC, "text.usetex": bool(usetex)})


use_paper_style()


def _pct():
    return r"\%" if plt.rcParams["text.usetex"] else "%"


def _finish(fig, save_path):
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def _as_df(df_or_csv):
    if isinstance(df_or_csv, (str, os.PathLike)):
        return pd.read_csv(df_or_csv)
    return df_or_csv.copy()


def _phase_panel(ax, x, y, color, xlabel, ylabel, fontsize, lw=1.2, labelpad=20,
                 equal=False):
    ax.plot(x, y, color=color, linewidth=lw)
    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=labelpad, rotation=0)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.grid(True, alpha=0.15)
    if equal:
        ax.set_aspect("equal", adjustable="box")


# =============================================================================
# Fig. 1 - insets of the pipeline schematic
# =============================================================================

def plot_fig1_insets(x_ring, t_jc, x_jc, color=CRIMSON, save_path=None):
    """The two small plots inside Fig. 1, panel (4).

    x_ring : reconstruction whose columns 0 and 2 are <x> and <p_x>
             (the closed Kerr oscillator of Fig. 6 gives the ring).
    t_jc, x_jc : time and Jaynes-Cummings reconstruction (column 4 is Re<sigma_+>).
    The schematic itself was drawn outside Python.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot(x_ring[:, 0], x_ring[:, 2], color=color, linewidth=0.8)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel(r"$\langle x \rangle$", fontsize=25)
    ax1.set_ylabel(r"$\langle p_x \rangle$", fontsize=25)

    n = min(len(t_jc), len(x_jc))
    ax2.plot(t_jc[:n], x_jc[:n, 4], color=color, linewidth=0.8)
    ax2.set_xlabel(r"$t$", fontsize=25)
    ax2.set_ylabel(r"$\mathrm{Re}(\sigma_+)$", fontsize=25)

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return _finish(fig, save_path)


# =============================================================================
# Fig. 2 - open 2D QHO, two damping rates
# =============================================================================

def plot_fig2_qho_phase_space(x_rec_a, x_rec_b, color=NAVY, fontsize=35, save_path=None):
    """<y> vs <x> reconstructed by mHAVOK for (a) kappa = 0.1 and (b) kappa = 1.0.

    Inputs are ``MHAVOKResult.x_recovered`` (columns <x>, <y>, <p_x>, <p_y>).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.set_constrained_layout_pads(wspace=0.08, hspace=0.05)
    for ax, x_rec in zip(axes, (x_rec_a, x_rec_b)):
        _phase_panel(ax, x_rec[:, 0], x_rec[:, 1], color,
                     r"$\langle x \rangle$", r"$\langle y \rangle$", fontsize)
        ax.locator_params(nbins=3)
    fig.text(0.012, 0.97, r"(a)", fontsize=fontsize, va="top", ha="left")
    fig.text(0.525, 0.97, r"(b)", fontsize=fontsize, va="top", ha="left")
    return _finish(fig, save_path)


# =============================================================================
# Figs. 3 and 5b - mHAVOK vs matrix pencil vs FFT
# =============================================================================

def plot_compare_errors(df_or_csv, kind="freq", ax=None, xcol="kappa_true",
                        fontsize=25, save_path=None):
    """Percent error vs kappa for mHAVOK, matrix pencil and FFT (log scale).

    kind="freq": average oscillation-frequency error (Figs. 3 and 5b).
    kind="kappa": damping-rate error (mHAVOK and matrix pencil only).
    """
    df = _as_df(df_or_csv)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if kind == "freq":
        cols = {"mHAVOK": "avg_freq_err_pct_havok", "Matrix pencil": "avg_freq_err_pct_mp",
                "FFT": "avg_freq_err_pct_fft"}
        ylabel = r"$e_{\omega_i \hat{\omega}_i}$"
    elif kind == "kappa":
        cols = {"mHAVOK": "err_kappa_pct_havok", "Matrix pencil": "err_kappa_pct_mp"}
        ylabel = r"$e_{\kappa \hat{\kappa}}$"
    else:
        raise ValueError("kind must be 'freq' or 'kappa'")

    for label, col in cols.items():
        ax.plot(df[xcol].values, df[col].values, label=label, **METHOD_STYLES[label])

    ax.set_xlabel(r"$\kappa$", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize, rotation=0, labelpad=15)
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.legend()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return ax


def plot_fig3_method_comparison(df_or_csv, save_path=None):
    """Fig. 3: frequency error of the 2D QHO (no Kerr terms) vs kappa."""
    fig, ax = plt.subplots()
    plot_compare_errors(df_or_csv, kind="freq", ax=ax)
    fig.tight_layout()
    return _finish(fig, save_path)


# =============================================================================
# Fig. 4 - Kerr nonlinearities, open system
# =============================================================================

def plot_fig4_kerr_phase_space(x_rec_left, x_rec_right, colors=(TEAL, GREEN), fontsize=35,
                               save_path=None):
    """Fig. 4: Kerr phase spaces, chi = (2, 3) on the left and chi = (7, 8) on the right.

    Top row: <p_x> vs <x>. Bottom row: <p_y> vs <y>. The right column is the run where the
    chi = (5, 5) Kerr term was added on top of chi = (2, 3).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [(x_rec_left, 0, 2, colors[0], r"$\langle x \rangle$", r"$\langle p_x \rangle$"),
              (x_rec_right, 0, 2, colors[1], r"$\langle x \rangle$", r"$\langle p_x \rangle$"),
              (x_rec_left, 1, 3, colors[0], r"$\langle y \rangle$", r"$\langle p_y \rangle$"),
              (x_rec_right, 1, 3, colors[1], r"$\langle y \rangle$", r"$\langle p_y \rangle$")]
    for ax, (xr, i, j, c, xl, yl) in zip(axes.flat, panels):
        _phase_panel(ax, xr[:, i], xr[:, j], c, xl, yl, fontsize)

    fig.text(0.055, 0.96, r"(a)", fontsize=fontsize, va="top", ha="left")
    fig.text(0.542, 0.96, r"(b)", fontsize=fontsize, va="top", ha="left")
    fig.text(0.032, 0.46, r"(c)", fontsize=fontsize, va="top", ha="left")
    fig.text(0.517, 0.46, r"(d)", fontsize=fontsize, va="top", ha="left")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    return _finish(fig, save_path)


# =============================================================================
# Fig. 5 - Kerr-strength heatmap and method comparison with Kerr terms
# =============================================================================

def _grid_from_df(df, z_col="avg_err_pct"):
    xs = np.sort(df["chi_x_true"].unique())
    ys = np.sort(df["chi_y_true"].unique())
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    lut = {(row["chi_x_true"], row["chi_y_true"]): row[z_col] for _, row in df.iterrows()}
    Z = np.array([[lut.get((cx, cy), np.nan) for cx in xs] for cy in ys], dtype=float)
    return X, Y, Z


def plot_chi_heatmap(df_or_csv, which="avg", cmap="summer", ax=None, add_colorbar=True,
                     cbar_ticks=(1, 2, 3, 4, 5), save_path=None):
    """Percent error of the recovered Kerr strengths over (chi_x, chi_y).

    which: "avg" (mean of both), "chi_x" or "chi_y". Returns (ax, image).
    """
    df = _as_df(df_or_csv)
    zcol = {"avg": "avg_err_pct", "chi_x": "err_chi_x_pct", "chi_y": "err_chi_y_pct"}[which]
    X, Y, Z = _grid_from_df(df, z_col=zcol)
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(Z, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()],
                   aspect="auto", cmap=cmap)
    ax.set_xlabel(r"$\chi_x$", fontsize=25)
    ax.set_ylabel(r"$\chi_y$", rotation=0, labelpad=20, fontsize=25)
    if add_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label(rf"Percent error ({_pct()})")
        if cbar_ticks is not None:
            cbar.set_ticks(list(cbar_ticks))
    if save_path:
        ax.figure.savefig(save_path, bbox_inches="tight")
    return ax, im


def _panel_label(ax, s, x=-0.21, y=0.99, fontsize=25):
    ax.text(x, y, s, transform=ax.transAxes, ha="left", va="top", fontsize=fontsize,
            fontweight="bold")


def plot_fig5_kerr_errors(chi_csv, kappa_csv, save_path=None):
    """Fig. 5: (a) Kerr-strength error heatmap, (b) frequency error vs kappa with Kerr terms."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)
    _, im = plot_chi_heatmap(chi_csv, which="avg", ax=ax1, add_colorbar=False)
    _panel_label(ax1, r"(a)")
    plot_compare_errors(kappa_csv, kind="freq", ax=ax2)
    _panel_label(ax2, r"(b)")
    cbar = fig.colorbar(im, ax=[ax1])
    cbar.set_label(rf"Percent error ({_pct()})")
    cbar.set_ticks([1, 2, 3, 4, 5])
    return _finish(fig, save_path)


# =============================================================================
# Fig. 6 - closed Kerr oscillator, sensitivity to the cutoff rank
# =============================================================================

def plot_fig6_closed_kerr_rank(x_rec_dark, x_rec_light,
                               labels=(r"$r=13$", r"$r_\mathrm{opt}=16$"),
                               colors=(DARK_BLUE, LIGHT_BLUE), fontsize=35, save_path=None):
    """Fig. 6: closed Kerr oscillator, dark = chi = (10, 10) at r = 25, light = chi = (5, 5) at r = 30.

    ``labels`` are the column titles; the default reproduces the published ones.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    panels = [(x_rec_dark, 0, 2, colors[0], r"$\langle x \rangle$", r"$\langle p_x \rangle$"),
              (x_rec_light, 0, 2, colors[1], r"$\langle x \rangle$", r"$\langle p_x \rangle$"),
              (x_rec_dark, 1, 3, colors[0], r"$\langle y \rangle$", r"$\langle p_y \rangle$"),
              (x_rec_light, 1, 3, colors[1], r"$\langle y \rangle$", r"$\langle p_y \rangle$")]
    for ax, (xr, i, j, c, xl, yl) in zip(axes.flat, panels):
        _phase_panel(ax, xr[:, i], xr[:, j], c, xl, yl, fontsize, lw=1.0, labelpad=28,
                     equal=True)

    for (x, y), s in zip([(0.033, 0.95), (0.52, 0.95), (0.033, 0.46), (0.52, 0.46)],
                         [r"(a)", r"(b)", r"(c)", r"(d)"]):
        fig.text(x, y, s, fontsize=fontsize, va="top", ha="left")
    for y in (1.012, 1.012 / 2):
        fig.text(0.27, y, labels[0], fontsize=fontsize, va="top", ha="left")
        fig.text(0.75, y, labels[1], fontsize=fontsize, va="top", ha="left")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    return _finish(fig, save_path)


# =============================================================================
# Fig. 7 - Jaynes-Cummings interaction
# =============================================================================

def plot_fig7_jaynes_cummings(t, x_rec, color=CRIMSON, fontsize=37, save_path=None):
    """(a) <y> vs <x> and (b) Re<sigma_+>(t) from the forced linear model.

    ``x_rec`` columns follow Eq. (50): <x>, <p_x>, <y>, <p_y>, Re<sigma_+>, Im<sigma_+>.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(x_rec[:, 0], x_rec[:, 2], color=color, linewidth=1.0)
    ax1.set_xlabel(r"$\langle x \rangle$", fontsize=fontsize, labelpad=10)
    ax1.set_ylabel(r"$\langle y \rangle$", fontsize=fontsize, labelpad=35, rotation=0)

    n = min(len(t), len(x_rec))
    ax2.plot(t[:n], x_rec[:n, 4], color=color, linewidth=1.0)
    ax2.set_xlabel(r"$t$", fontsize=fontsize, labelpad=10)
    ax2.set_ylabel(r"$\mathrm{Re}(\sigma_+)$", fontsize=fontsize, labelpad=0)
    ax2.set_xticks(np.linspace(t.min(), t.max(), 3))
    ax2.set_yticks([-0.4, 0, 0.4])

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.15)
        ax.tick_params(axis="both", labelsize=fontsize)
    fig.text(0.028, 0.92, r"(a)", fontsize=fontsize + 3, va="top", ha="left")
    fig.text(0.506, 0.92, r"(b)", fontsize=fontsize + 3, va="top", ha="left")
    fig.tight_layout()
    return _finish(fig, save_path)


# =============================================================================
# Fig. 8 - spectrum of A under parametric frequency modulation
# =============================================================================

FIG8_LABELS = [r"$\hat{\omega}_y$", r"$-\hat{\omega}_y$",
               r"$\hat{\omega}_x - \hat{\omega}_f$", r"$\hat{\omega}_f - \hat{\omega}_x$",
               r"$\hat{\omega}_x$", r"$-\hat{\omega}_x$",
               r"$\hat{\omega}_x + \hat{\omega}_f$", r"$-\hat{\omega}_x - \hat{\omega}_f$"]
# label position relative to each point (taken from the published figure)
FIG8_OFFSETS = [(0.1, -1.34), (-0.45, 1.14), (0.0, -1.6), (-0.6, 1.2),
                (0.1, -1.5), (-0.7, 1.1), (0.0, -1.85), (-1.25, 1.45)]


def plot_fig8_modulated_spectrum(A_or_eigs, labels=FIG8_LABELS, offsets=FIG8_OFFSETS,
                                 fontsize=25, save_path=None):
    """Imaginary parts of the eigenvalues of A, labelled as sidebands omega_x +- omega_f.

    Accepts A itself or its eigenvalues. Eigenvalues are sorted by magnitude with
    Im > 0 first in each conjugate pair; labels/offsets assume the 8-eigenvalue
    spectrum of the paper (r = 10).
    """
    A_or_eigs = np.asarray(A_or_eigs)
    lam = sorted_eigenvalues(A_or_eigs) if A_or_eigs.ndim == 2 else A_or_eigs
    x = np.arange(len(lam))
    y = lam.imag

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.scatter(x, y, color=plt.cm.Greens(np.linspace(0.3, 1, len(x))), s=80, zorder=2)
    for sign in (1, -1):
        sel = sign * y > 0
        ax.plot(x[sel], y[sel], color="green", alpha=0.3, linewidth=2, zorder=1)
    if labels is not None and len(labels) == len(lam):
        for xi, yi, s, (dx, dy) in zip(x, y, labels, offsets):
            ax.text(xi + dx, yi + dy, s, fontsize=fontsize, color="black")

    ax.set_xticks(x)
    ax.set_yticks([-3, 3, 0, -6, 6, -8, 8])
    ax.set_xlabel(r"$i$-th eigenvalue of $\mathbf{A}$", fontsize=fontsize)
    ax.set_ylabel(r"$|\mathrm{Im}(\lambda_i)|$", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return _finish(fig, save_path)


# =============================================================================
# Extra diagnostics (not in the paper)
# =============================================================================

def plot_rank_metrics(metrics_df, r_opt=None):
    """Condition number of B and 1 - mean R^2 against the cutoff rank r."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.semilogy(metrics_df.index, metrics_df["cond_number_B"].replace(0, np.nan), "o-")
    ax1.set_xlabel(r"$r$")
    ax1.set_ylabel(r"$\kappa(\mathbf{B})$")
    ax2.semilogy(metrics_df.index, 1 - metrics_df["mean_R2"], "o-", color="k")
    ax2.set_xlabel(r"$r$")
    ax2.set_ylabel(r"$1 - \bar{R}^2$")
    if r_opt is not None:
        for ax in (ax1, ax2):
            ax.axvline(r_opt, ls="--", color="gray")
    fig.tight_layout()
    return fig
