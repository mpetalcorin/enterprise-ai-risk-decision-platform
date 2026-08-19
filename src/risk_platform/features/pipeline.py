from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_transaction_amount",
    "log_account_age_days",
    "log_transactions_24h",
    "amount_to_avg_ratio",
    "international",
    "high_risk_country",
    "device_new",
    "log_failed_logins_24h",
    "night_transaction",
    "customer_tenure_years",
    "country_international_interaction",
    "new_device_failed_login_interaction",
]


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["log_transaction_amount"] = np.log1p(frame["transaction_amount"].astype(float))
    out["log_account_age_days"] = np.log1p(frame["account_age_days"].astype(float))
    out["log_transactions_24h"] = np.log1p(frame["transactions_24h"].astype(float))
    out["amount_to_avg_ratio"] = frame["transaction_amount"].astype(float) / np.maximum(
        frame["avg_amount_30d"].astype(float), 1e-6
    )
    out["international"] = frame["international"].astype(float)
    out["high_risk_country"] = frame["high_risk_country"].astype(float)
    out["device_new"] = frame["device_new"].astype(float)
    out["log_failed_logins_24h"] = np.log1p(frame["failed_logins_24h"].astype(float))
    out["night_transaction"] = (
        (frame["transaction_hour"].astype(int) <= 5) | (frame["transaction_hour"].astype(int) >= 23)
    ).astype(float)
    out["customer_tenure_years"] = frame["customer_tenure_years"].astype(float)
    out["country_international_interaction"] = (
        out["high_risk_country"] * out["international"]
    )
    out["new_device_failed_login_interaction"] = out["device_new"] * (
        frame["failed_logins_24h"].astype(float) > 0
    ).astype(float)
    return out[FEATURE_COLUMNS]
