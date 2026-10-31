"""Bayesian Online Change Point Detection (Adams & MacKay, 2007).

Gaussian observations with unknown mean and variance (Normal-Gamma conjugate prior),
constant hazard 1/hazard_days, run length truncated at max_run_length.

Note: with a constant hazard, P(r_t = 0 | x_1:t) equals the hazard exactly, so it carries no
information. The alarm signal used here is P(r_t <= k | x_1:t) - the posterior probability
that a new regime started within the last k steps. It is causal: the value at step t uses
only x_1..x_t, so it can be precomputed once and read "as of" any time in a backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp


def _student_logpdf(x, mu, kappa, alpha, beta):
    nu = 2 * alpha
    scale2 = beta * (kappa + 1) / (alpha * kappa)
    z = (x - mu) ** 2 / (nu * scale2)
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * nu * scale2)
            - (nu + 1) / 2 * np.log1p(z))


def bocpd(x: np.ndarray, hazard: float, max_run: int, recent: int,
          mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 1.0) -> pd.DataFrame:
    """Run BOCPD over a 1-D series. Returns per-step cp_prob, map_run_length, expected_run_length."""
    n = len(x)
    log_h, log_1mh = np.log(hazard), np.log1p(-hazard)
    log_r = np.array([0.0])  # log P(r_0 = 0) = 0
    mu = np.array([mu0]); kappa = np.array([kappa0]); alpha = np.array([alpha0]); beta = np.array([beta0])
    cp_prob = np.full(n, np.nan); map_rl = np.zeros(n, dtype=int); exp_rl = np.full(n, np.nan)

    for t in range(n):
        xt = x[t]
        if np.isnan(xt):  # missing observation: carry the posterior, no update
            cp_prob[t] = cp_prob[t - 1] if t else 0.0
            map_rl[t] = map_rl[t - 1] + 1 if t else 0
            exp_rl[t] = exp_rl[t - 1] + 1 if t else 0.0
            continue
        pred = _student_logpdf(xt, mu, kappa, alpha, beta)
        growth = log_r + pred + log_1mh
        cp = logsumexp(log_r + pred + log_h)
        new = np.concatenate([[cp], growth])
        # parameter update (prepend the prior for r = 0)
        mu_n = (kappa * mu + xt) / (kappa + 1)
        beta_n = beta + kappa * (xt - mu) ** 2 / (2 * (kappa + 1))
        mu = np.concatenate([[mu0], mu_n]); kappa = np.concatenate([[kappa0], kappa + 1])
        alpha = np.concatenate([[alpha0], alpha + 0.5]); beta = np.concatenate([[beta0], beta_n])
        if len(new) > max_run:  # truncate: fold the tail mass into the last kept run length
            new[max_run - 1] = logsumexp(new[max_run - 1:])
            new = new[:max_run]
            mu, kappa, alpha, beta = mu[:max_run], kappa[:max_run], alpha[:max_run], beta[:max_run]
        log_r = new - logsumexp(new)
        p = np.exp(log_r)
        # burn-in: during the first `recent` steps the run length is trivially short
        cp_prob[t] = p[: recent + 1].sum() if t > recent else 0.0
        map_rl[t] = int(np.argmax(p))
        exp_rl[t] = float((np.arange(len(p)) * p).sum())
    return pd.DataFrame({"cp_prob": cp_prob, "map_run_length": map_rl, "expected_run_length": exp_rl})


def alarms(signal: pd.DataFrame, threshold: float, cooldown: int) -> pd.DataFrame:
    """Threshold crossings with a cooldown. break_estimate = alarm step - MAP run length."""
    out, last = [], None
    for i, (ts, row) in enumerate(signal.iterrows()):
        if row["cp_prob"] > threshold and (last is None or i - last >= cooldown):
            est = signal.index[max(0, i - int(row["map_run_length"]))]
            out.append({"alarm": ts, "break_estimate": est, "cp_prob": row["cp_prob"]})
            last = i
    df = pd.DataFrame(out, columns=["alarm", "break_estimate", "cp_prob"])
    for c in ("alarm", "break_estimate"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df
