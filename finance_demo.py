"""
finance_demo.py — End-to-end MCMC pricing of an autocallable on the S&P 500.

Uses the CS 4850 Metropolis-Hastings engine (Eq 6.1) in two passes:

Pass 1 — Empirical return distribution (DataDrivenProblem)
    State space X : R  (one daily log-return)
    w(x)          : Gaussian KDE fitted to 15 years of S&P 500 daily log-returns
    K             : Gaussian random walk, step = 2.38 * sigma  (auto-selected)
    kappa(x)      : 1 for all x  (symmetric K)
    x0            : mean(log_returns)

Pass 2 — Autocallable pricing (FormulaProblem, 4-D path space)
    State space X : R^4  (one quarterly log-return per observation date)
    w(x)          : GBM risk-neutral density — product of 4 Gaussian densities
                    log w(x) = -0.5 * ||x - mu_rn_q||^2 / sigma_q^2
    K             : Multivariate Gaussian random walk  (auto-selected)
    kappa(x)      : 1 for all x  (symmetric K)
    x0            : zero vector  (i.e. flat path — always in support of w)

Autocallable structure
    Underlying    : S&P 500
    Observation   : quarterly (4 dates, 1 year total)
    Autocall      : triggered if S_t >= 100 % of S0 at any observation date
    Coupon        : 8 % per annum, paid pro-rata on call (2 % per quarter)
    Downside      : if not called and S_T < 70 % of S0 at maturity, capital at risk
    Otherwise     : full notional returned at maturity
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")        # non-interactive backend — saves to file
import matplotlib.pyplot as plt
from unittest.mock import patch

# Make sure we run from the project root so imports work
sys.path.insert(0, os.path.dirname(__file__))
from main import MCMCSampler
from problems.continuous import FormulaProblem
from core.mh import MetropolisHastings

import yfinance as yf


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MARKET DATA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — DOWNLOADING MARKET DATA")
print("=" * 60)

sp500   = yf.download("^GSPC", start="2010-01-01", end="2024-12-31",
                       auto_adjust=True, progress=False)
prices  = sp500["Close"].dropna().to_numpy().flatten()
log_ret = np.diff(np.log(prices))           # 3 772 daily log-returns

S0          = float(prices[-1])             # current S&P 500 level
mu_hist     = float(log_ret.mean())         # historical daily mean
sigma_daily = float(log_ret.std())          # historical daily vol
sigma_ann   = sigma_daily * np.sqrt(252)

tbill = yf.download("^IRX", start="2024-10-01", end="2024-12-31",
                     auto_adjust=True, progress=False)
r_ann   = float(tbill["Close"].dropna().to_numpy().flatten()[-1]) / 100
r_daily = r_ann / 252

print(f"  S&P 500 current level  : {S0:>10.2f}")
print(f"  Historical vol (annual): {sigma_ann:>10.4f}  ({sigma_ann*100:.1f} %)")
print(f"  Historical mean (daily): {mu_hist:>10.6f}")
print(f"  Risk-free rate (annual): {r_ann:>10.4f}  ({r_ann*100:.2f} %)")
print(f"  Daily log-returns      : {len(log_ret):>10,}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PASS 1 — MH ON EMPIRICAL RETURN DISTRIBUTION  (DataDrivenProblem)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 2 — MH PASS 1: EMPIRICAL RETURN DISTRIBUTION")
print("  w(x) = KDE of 3 772 daily log-returns")
print("  K    = Gaussian random walk")
print("=" * 60)

with patch("builtins.input", return_value="y"):
    result_empirical = MCMCSampler().run({
        "problem_type": "continuous_data",
        "data":          log_ret,
        "notes":         "S&P 500 daily log-returns 2010-2024",
    })

emp_samples = np.array(result_empirical.samples)

print(f"\n  Acceptance rate : {result_empirical.acceptance_rate:.3f}  "
      f"(target ~0.44 per Roberts et al.)")
print(f"  ESS             : {result_empirical.diagnostics['effective_sample_size']:.0f}")
print(f"  Samples drawn   : {result_empirical.diagnostics['n_samples']}")
print(f"  Proposal kernel : {result_empirical.proposal_config['type']}")
print(f"  Step size h     : {result_empirical.proposal_config['step_size']:.6f}  "
      f"(= 2.38 × {sigma_daily:.6f})")

# Quick sanity check: MH sample mean vs data mean
print(f"\n  Data mean (daily return): {log_ret.mean():.6f}")
print(f"  MH sample mean          : {emp_samples.mean():.6f}")
print(f"  Data std                : {log_ret.std():.6f}")
print(f"  MH sample std           : {emp_samples.std():.6f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PASS 2 — MH ON 4-D PATH SPACE FOR AUTOCALLABLE  (FormulaProblem)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 3 — MH PASS 2: AUTOCALLABLE PATH SAMPLING")
print("  X    = R^4  (one quarterly log-return per obs. date)")
print("  w(x) = GBM risk-neutral density over 4 quarters")
print("  K    = Multivariate Gaussian random walk")
print("=" * 60)

T_PERIODS      = 4          # quarterly observation dates
DAYS_PER_Q     = 63         # trading days per quarter

mu_rn_daily    = r_daily - 0.5 * sigma_daily**2    # risk-neutral daily drift
mu_rn_q        = mu_rn_daily  * DAYS_PER_Q         # quarterly mean log-return
sigma_q        = sigma_daily  * np.sqrt(DAYS_PER_Q) # quarterly vol

print(f"\n  Quarterly GBM parameters:")
print(f"    mu_rn_q  (quarterly drift) : {mu_rn_q:.6f}")
print(f"    sigma_q  (quarterly vol)   : {sigma_q:.6f}  ({sigma_q*100:.2f} %)")

def log_w_path(x: np.ndarray) -> float:
    """
    Eq 6.1 target weight for a 4-quarterly-return path x.

    Under GBM, each quarterly log-return is i.i.d. N(mu_rn_q, sigma_q^2).
    The joint density of the 4-vector x is the product of 4 Gaussians, so
    log w(x) = -0.5 * sum_t (x_t - mu_rn_q)^2 / sigma_q^2

    The normalization constant cancels in every MH acceptance ratio,
    so we drop it (Lemma 6.12 — only the ratio w_y / w_x matters).
    """
    residuals = np.asarray(x, dtype=float) - mu_rn_q
    return float(-0.5 * np.dot(residuals, residuals) / sigma_q**2)

# The MCMCSampler classifier would assign step_size=1.0 for a multi-D formula
# problem (no data to infer scale from).  That is 11× too large for our
# σ_q = 0.086 target and would give ~0% acceptance.  The theoretically correct
# step for a d-dimensional Gaussian target is  2.38/√d × σ  (Roberts et al.),
# which targets the 23.4% optimal acceptance rate for d > 1.
# We therefore instantiate FormulaProblem + MetropolisHastings directly —
# this is the same Eq 6.1 engine, just with the step size we need.

BURN_IN_PATH    = 4_000
N_SAMPLES_PATH  = 20_000

step_size_path  = (2.38 / np.sqrt(T_PERIODS)) * sigma_q   # 0.103 for d=4
proposal_cfg    = {
    "type":      "multivariate_gaussian",
    "step_size": step_size_path,
    "cov":       (step_size_path ** 2) * np.eye(T_PERIODS),
}

prob_path = FormulaProblem(
    log_w_path, proposal_cfg,
    dimension=T_PERIODS, seed=42,
    notes=f"GBM risk-neutral path, {T_PERIODS} quarters",
)
mh_path = MetropolisHastings(prob_path, seed=42)

print(f"  Step size (2.38/√{T_PERIODS} × σ_q)  : {step_size_path:.6f}")
print(f"  Running chain: {BURN_IN_PATH} burn-in + {N_SAMPLES_PATH} samples …")

raw = mh_path.sample(
    np.full(T_PERIODS, mu_rn_q),       # x0: start at GBM drift
    n_steps=BURN_IN_PATH + N_SAMPLES_PATH,
    burn_in=BURN_IN_PATH,
)
path_samples = np.array(raw)           # shape (N_SAMPLES_PATH, 4)

# Compute ESS manually for the first dimension (representative)
def _ess_1d(x: np.ndarray) -> float:
    n = len(x)
    x_c = x - x.mean()
    var = float(np.var(x_c))
    if var == 0:
        return 1.0
    acf_sum = 0.0
    for lag in range(1, n):
        rho = float(np.dot(x_c[:-lag], x_c[lag:]) / (n * var))
        if rho <= 0:
            break
        acf_sum += rho
    return n / (1 + 2 * acf_sum)

acc_path = mh_path.acceptance_rate
ess_path = float(np.min([_ess_1d(path_samples[:, d]) for d in range(T_PERIODS)]))

print(f"Done.  Acceptance rate: {acc_path:.3f}")
print(f"\n  Acceptance rate : {acc_path:.3f}  (target ~0.234 for d=4 Gaussian)")
print(f"  ESS             : {ess_path:.0f}")
print(f"  Samples drawn   : {N_SAMPLES_PATH}")
print(f"  Proposal kernel : multivariate_gaussian")
print(f"  Path shape      : {path_samples.shape}")

# Sanity: each quarter's sample mean and std should match GBM parameters
for q in range(T_PERIODS):
    col = path_samples[:, q]
    print(f"  Q{q+1}: sample mean={col.mean():.5f}  (GBM: {mu_rn_q:.5f})  "
          f"std={col.std():.5f}  (GBM: {sigma_q:.5f})")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PRICE THE AUTOCALLABLE
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 4 — PRICING THE AUTOCALLABLE")
print("  Structure:")
print("    Underlying        : S&P 500")
print("    Observation dates : quarterly (Q1 – Q4)")
print("    Autocall trigger  : S_t >= 100 % of S0")
print("    Coupon            : 8 % p.a. (2 % per quarter, paid on call)")
print("    Downside barrier  : 70 % of S0 at maturity")
print("    Capital at risk   : if S_T < 70 % → receive S_T / S0")
print("=" * 60)

NOTIONAL         = 1.0
COUPON_ANNUAL    = 0.08
COUPON_Q         = COUPON_ANNUAL / 4      # 2 % per quarter
AUTOCALL_TRIGGER = 1.00                   # 100 % of S0
BARRIER          = 0.70                   # 70 % of S0


def autocall_payoff(quarterly_returns: np.ndarray) -> float:
    """
    Compute the discounted payoff of the autocallable for one simulated path.

    Parameters
    ----------
    quarterly_returns : ndarray, shape (4,)
        Log-returns for each of the 4 quarterly observation periods.

    Returns
    -------
    float
        Present value of the payoff (discounted at the risk-free rate r_ann).
    """
    cumulative = np.cumsum(quarterly_returns)
    S_t        = S0 * np.exp(cumulative)     # price at each observation date

    for t in range(T_PERIODS):
        quarters_elapsed = t + 1
        discount         = np.exp(-r_ann * quarters_elapsed / 4)

        if S_t[t] >= S0 * AUTOCALL_TRIGGER:
            # Autocalled: return notional + accumulated coupon
            return NOTIONAL * (1.0 + COUPON_Q * quarters_elapsed) * discount

    # Maturity: autocall never triggered
    discount_T = np.exp(-r_ann * 1.0)

    if S_t[-1] < S0 * BARRIER:
        # Capital at risk: proportional loss
        return NOTIONAL * (S_t[-1] / S0) * discount_T
    else:
        # Full principal returned
        return NOTIONAL * discount_T


payoffs      = np.array([autocall_payoff(row) for row in path_samples])
price        = float(payoffs.mean())
price_se     = float(payoffs.std() / np.sqrt(len(payoffs)))
ci_low       = price - 1.96 * price_se
ci_high      = price + 1.96 * price_se

# Breakdown by scenario
called_mask  = np.array([
    any(S0 * np.exp(np.cumsum(row)[t]) >= S0 * AUTOCALL_TRIGGER
        for t in range(T_PERIODS))
    for row in path_samples
])
barrier_hit  = np.array([
    (not called_mask[i]) and
    (S0 * np.exp(np.cumsum(path_samples[i])[-1]) < S0 * BARRIER)
    for i in range(len(path_samples))
])
principal_ret = ~called_mask & ~barrier_hit

pct_called   = called_mask.mean()
pct_barrier  = barrier_hit.mean()
pct_principal = principal_ret.mean()

# Call timing
call_quarter = []
for row in path_samples[called_mask]:
    cumulative = np.cumsum(row)
    S_t = S0 * np.exp(cumulative)
    for t in range(T_PERIODS):
        if S_t[t] >= S0 * AUTOCALL_TRIGGER:
            call_quarter.append(t + 1)
            break
call_quarter = np.array(call_quarter)

print(f"\n  ─── FAIR VALUE ───────────────────────────────────")
print(f"  Price (% of notional)  : {price*100:.4f} %")
print(f"  Standard error         : {price_se*100:.4f} %")
print(f"  95 % confidence interval: [{ci_low*100:.4f} %, {ci_high*100:.4f} %]")
print(f"  N paths                : {len(payoffs):,}")
print(f"  ESS                    : {ess_path:.0f}")
print()
print(f"  ─── SCENARIO BREAKDOWN (% of all paths) ──────────")
print(f"  Autocalled early       : {pct_called*100:.1f} %")
for q in range(1, T_PERIODS + 1):
    pct_q = (np.sum(call_quarter == q) / len(path_samples)) * 100
    print(f"    -> called at Q{q}       : {pct_q:.1f} % of all paths")
print(f"  Capital at risk (< 70%): {pct_barrier*100:.1f} %")
print(f"  Full principal returned: {pct_principal*100:.1f} %")
avg_loss = payoffs[barrier_hit].mean() if barrier_hit.any() else float("nan")
print(f"  Avg payoff | barrier   : {avg_loss*100:.2f} % of notional" if not np.isnan(avg_loss)
      else "  Avg payoff | barrier   : N/A (0 paths hit barrier in this sample)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("output", exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"MH MCMC — S&P 500 Autocallable  |  S0 = {S0:,.0f}  "
    f"σ = {sigma_ann*100:.1f}%  r = {r_ann*100:.2f}%",
    fontsize=13, fontweight="bold"
)

# ── Panel A: Historical returns + MH empirical samples ────────────────────
ax = axes[0, 0]
bins = np.linspace(-0.06, 0.06, 80)
ax.hist(log_ret,     bins=bins, density=True, alpha=0.45,
        color="steelblue", label="Historical returns (2010–2024)")
ax.hist(emp_samples, bins=bins, density=True, alpha=0.55,
        color="tomato",    label="MH samples (empirical w)")
ax.set_xlabel("Daily log-return")
ax.set_ylabel("Density")
ax.set_title(
    f"Pass 1 — Empirical distribution\n"
    f"acc={result_empirical.acceptance_rate:.2f}  "
    f"ESS={result_empirical.diagnostics['effective_sample_size']:.0f}"
)
ax.legend(fontsize=8)

# ── Panel B: Sample paths from 4-D MH chain ───────────────────────────────
ax = axes[0, 1]
n_show = 200
for i in range(n_show):
    row  = path_samples[i]
    S_t  = S0 * np.exp(np.cumsum(row))
    full = np.concatenate([[S0], S_t])
    color = ("gold" if any(S_t[t] >= S0 * AUTOCALL_TRIGGER for t in range(T_PERIODS))
             else ("firebrick" if S_t[-1] < S0 * BARRIER else "steelblue"))
    ax.plot(range(T_PERIODS + 1), full, alpha=0.15, lw=0.7, color=color)

# Overlay threshold lines
ax.axhline(S0 * AUTOCALL_TRIGGER, color="gold",     lw=1.5, ls="--",
           label=f"Autocall {AUTOCALL_TRIGGER*100:.0f}%")
ax.axhline(S0 * BARRIER,          color="firebrick", lw=1.5, ls="--",
           label=f"Barrier {BARRIER*100:.0f}%")
ax.axhline(S0,                     color="black",     lw=1.0, ls=":",
           label=f"S0 = {S0:,.0f}")
from matplotlib.ticker import FixedLocator
ax.xaxis.set_major_locator(FixedLocator(range(T_PERIODS + 1)))
ax.set_xticklabels(["Now", "Q1", "Q2", "Q3", "Q4"])
ax.set_ylabel("S&P 500 level")
ax.set_title(
    f"Pass 2 — {n_show} sampled paths\n"
    f"acc={acc_path:.2f}  ESS={ess_path:.0f}"
)
ax.legend(fontsize=8)

# ── Panel C: Payoff distribution ──────────────────────────────────────────
ax = axes[1, 0]
ax.hist(payoffs * 100, bins=60, density=True,
        color="mediumpurple", alpha=0.75, edgecolor="white", lw=0.3)
ax.axvline(price * 100, color="black",  lw=2.0, label=f"Fair value = {price*100:.3f}%")
ax.axvline(ci_low  * 100, color="gray", lw=1.0, ls="--")
ax.axvline(ci_high * 100, color="gray", lw=1.0, ls="--", label="95% CI")
ax.set_xlabel("Payoff (% of notional)")
ax.set_ylabel("Density")
ax.set_title(
    f"Autocallable payoff distribution\n"
    f"Fair value = {price*100:.3f}% ± {price_se*100:.3f}%"
)
ax.legend(fontsize=8)

# ── Panel D: Scenario breakdown ────────────────────────────────────────────
ax = axes[1, 1]

# Call timing bars
q_labels = [f"Called Q{q}" for q in range(1, T_PERIODS + 1)]
q_counts  = [(np.sum(call_quarter == q) / len(path_samples)) * 100 for q in range(1, T_PERIODS + 1)]
bar_labels  = q_labels + ["Barrier hit", "Principal returned"]
bar_heights = q_counts + [pct_barrier * 100, pct_principal * 100]
bar_colors  = ["gold", "goldenrod", "darkgoldenrod", "olive",
               "firebrick", "steelblue"]

bars = ax.bar(bar_labels, bar_heights, color=bar_colors, edgecolor="white", lw=0.5)
ax.set_ylabel("% of all paths")
ax.set_title("Scenario breakdown (% of all paths)")
ax.set_xticks(range(len(bar_labels)))
ax.set_xticklabels(bar_labels, rotation=25, ha="right", fontsize=8)
for bar, h in zip(bars, bar_heights):
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
            f"{h:.1f}%", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig("output/autocallable_results.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved → output/autocallable_results.png")
