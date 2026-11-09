# P2: code and figures for PRA 113, 042222 (2026)

J. E. Pérez-García, C. Colchero, and J. C. Gutiérrez-Vega, *Spectral analysis of the Koopman operator as a framework for recovering Hamiltonian parameters in open quantum systems*, Phys. Rev. A **113**, 042222 (2026). DOI: [10.1103/v5zg-gybr](https://doi.org/10.1103/v5zg-gybr)

This folder holds the code that regenerates every plot in the paper. 

Please cite both the PRA and the original mHAVOK papers in your research; it'll help us a lot. The original mHAVOK paper can be found here: https://doi.org/10.1063/5.0303718.

## Contents

| file | what it is |
|---|---|
| `paper_figures.ipynb` | Regenerates Figs. 2 to 8. Each figure is a simulate → mHAVOK → plot call, and the recovered parameters are printed next to it. |
| `jhavok_setup.py` | QuTiP simulations, mHAVOK, cutoff-rank selection, spectral analysis of **A**, matrix-pencil/FFT baselines and the κ / χ sweeps. |
| `jhavok_plots.py` | One `plot_figN` function per figure, styled like the published version. |
| `data/` | CSVs behind Figs. 3 and 5. |
| `figures/` | PDFs written by the notebook (`SAVE_FIGS = True`). |
| `LICENSE` | MIT license. |

## Running it

Requirements: Python 3.11 with `numpy`, `scipy`, `pandas`, `matplotlib` and `qutip` 5 (tested with 5.2.0). A LaTeX install (MiKTeX here) is used for text if present; without it the figures fall back to mathtext. 

Open `paper_figures.ipynb` and run all cells. It takes about 3 to 5 minutes; the Jaynes-Cummings simulation is the slowest. Two flags at the top:

- `SAVE_FIGS`: write `figures/FigN.pdf`.
- `RECOMPUTE_SWEEPS`: re-run the sweeps behind Figs. 3 and 5 (takes about an hour). The new CSVs go to `data/recomputed/`; the paper's CSVs are never overwritten.

Minimal use of the modules:

```python
import jhavok_setup as js
import jhavok_plots as jp

t, traj = js.simulate_2d_qho(chi_x=2, chi_y=3, kappa=0.1)   # columns <x>, <y>, <p_x>, <p_y>
res = js.run_mhavok(traj, js.DT, r=20)                      # m = ceil(100 / 4) = 25 delays
print(res.eigenvalues)                                       # spectrum of A, sorted by |lambda|
r_opt, metrics = js.find_optimal_rank(traj, js.DT, 4, 30)   # Appendix A 2
```

`HAVOK(...)` and `originalcoords(...)` keep the signatures of the old notebooks, so old code can import these modules too.

## Figure map

All simulations use Table I (ω_x = 2π, ω_y = π, T = 2, Fock cutoff 10, t_f = 50, Δt = 0.01, coherent α = 1) unless stated, with τ = 0.95 and about 100 Hankel rows (m = 25 delays for 4 observables, 17 for 6).

| fig. | system | cutoff rank |
|---|---|---|
| 1 | pipeline schematic, drawn outside Python (`jp.plot_fig1_insets` draws its two small plots) | n/a |
| 2 | 2D QHO, κ = 0.1 and κ = 1.0, no Kerr | r = 4 |
| 3 | κ sweep 0.1 to 3.0, no Kerr, vacuum bath (n̄ = 0); `data/fig3_kappa_sweep_no_kerr.csv` | r = 4 |
| 4 | open, κ = 0.1: (a), (c) χ = (2, 3); (b), (d) χ = (7, 8), i.e. χ = (5, 5) added on top of (2, 3) | r = 30 and r = 25 |
| 5a | 10 × 10 grid χ_x, χ_y ∈ [3, 5], κ = 0.1, vacuum bath; `data/fig5a_kerr_sweep.csv` | per point |
| 5b | κ sweep 0.1 to 2.9 with χ = (2, 3), vacuum bath; `data/fig5b_kappa_sweep_kerr.csv` | r = 20 |
| 6 | closed (no jump operators): light χ = (5, 5), dark χ = (10, 10), i.e. χ = (5, 5) added twice | light r = 30, dark r = 25 |
| 7 | Jaynes-Cummings on x: g = 0.15, ω_q = 2π, γ = 0.1, γ_φ = 0.01, independent baths per mode | r = 20 |
| 8 | H_f(t) = δ cos(ω_f t) a_x†a_x with δ = 0.25, ω_f = π/e, averaged common bath | r = 10 |

## Implementation notes

- **Simulations** reproduce the original QuTiP cells exactly (max difference 2e-13 or zero).
- **Cutoff rank.** `select_optimal_rank` reproduces the old `analyze_q_scores`: keep the top `q_percent` of ranks by condition number of **B**, then take the best mean R². With `q_percent = 100` (used throughout) this is simply the best reconstruction.

## License

MIT, see [`LICENSE`](LICENSE).



