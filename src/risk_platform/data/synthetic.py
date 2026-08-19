from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "transaction_amount",
    "account_age_days",
    "transactions_24h",
    "avg_amount_30d",
    "international",
    "high_risk_country",
    "device_new",
    "failed_logins_24h",
    "transaction_hour",
    "customer_tenure_years",
]


def generate_synthetic_transactions(n_rows: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate an industry-neutral synthetic risk dataset; no real customer data are used."""
    rng = np.random.default_rng(seed)
    avg_amount = np.clip(rng.lognormal(mean=4.4, sigma=0.8, size=n_rows), 5, 20_000)
    amount_multiplier = rng.lognormal(mean=0.0, sigma=0.7, size=n_rows)
    amount = np.clip(avg_amount * amount_multiplier, 1, 100_000)
    account_age = rng.integers(1, 5000, n_rows)
    tx_24h = rng.poisson(4.0, n_rows)
    international = rng.binomial(1, 0.16, n_rows)
    high_risk_country = rng.binomial(1, 0.05, n_rows)
    device_new = rng.binomial(1, 0.12, n_rows)
    failed_logins = rng.poisson(0.18, n_rows)
    hour = rng.integers(0, 24, n_rows)
    tenure = np.clip(account_age / 365.25 + rng.normal(0, 0.35, n_rows), 0, 40)

    amount_ratio = amount / np.maximum(avg_amount, 1)
    night = ((hour <= 5) | (hour >= 23)).astype(int)
    velocity = np.log1p(tx_24h)
    # A deliberately nonlinear latent process produces realistic class imbalance (~5-10%).
    signal = (
        1.15 * np.log1p(amount_ratio)
        + 1.35 * international
        + 2.15 * high_risk_country
        + 1.25 * device_new
        + 0.75 * np.log1p(failed_logins)
        + 0.55 * night
        + 0.42 * velocity
        - 0.18 * np.log1p(account_age)
        + 0.80 * (high_risk_country * international)
        + 0.55 * (device_new * (failed_logins > 0))
    )
    logit = -6.0 + 1.8 * signal
    prob = 1.0 / (1.0 + np.exp(-logit))
    label = rng.binomial(1, prob)

    frame = pd.DataFrame(
        {
            "transaction_amount": amount.round(2),
            "account_age_days": account_age,
            "transactions_24h": tx_24h,
            "avg_amount_30d": avg_amount.round(2),
            "international": international,
            "high_risk_country": high_risk_country,
            "device_new": device_new,
            "failed_logins_24h": failed_logins,
            "transaction_hour": hour,
            "customer_tenure_years": tenure.round(2),
            "is_high_risk": label,
        }
    )
    return frame
