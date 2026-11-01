"""Generate a synthetic telecom customer dataset with realistic churn patterns.

All data is invented. The churn probability is driven by a known formula so the
model's discovered drivers can be checked against the ground truth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENTS = ["Electronic check", "Credit card", "Bank transfer", "Mailed check"]
INTERNET = ["Fiber optic", "DSL", "None"]


def generate(n: int = 7000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.24, 0.21])
    tenure = np.where(
        contract == "Month-to-month",
        rng.gamma(1.6, 10, n),
        np.where(contract == "One year", rng.gamma(3.0, 12, n), rng.gamma(5.0, 11, n)),
    ).clip(1, 72).round().astype(int)
    internet = rng.choice(INTERNET, n, p=[0.44, 0.34, 0.22])
    payment = rng.choice(PAYMENTS, n, p=[0.34, 0.22, 0.22, 0.22])
    tech_support = np.where(internet == "None", "No internet", rng.choice(["Yes", "No"], n, p=[0.3, 0.7]))
    paperless = rng.choice(["Yes", "No"], n, p=[0.6, 0.4])
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    support_calls = rng.poisson(1.2, n)

    base = {"Fiber optic": 75, "DSL": 50, "None": 22}
    monthly = np.array([base[i] for i in internet]) + rng.normal(0, 12, n)
    monthly = monthly.clip(18, 120).round(2)
    total = (monthly * tenure * rng.uniform(0.95, 1.05, n)).round(2)

    # Ground-truth churn logic (log-odds)
    logit = (
        -1.6
        + 1.3 * (contract == "Month-to-month")
        - 0.9 * (contract == "Two year")
        - 0.045 * tenure
        + 0.7 * (internet == "Fiber optic")
        + 0.6 * (payment == "Electronic check")
        - 0.5 * (tech_support == "Yes")
        + 0.35 * support_calls
        + 0.3 * senior
        + 0.2 * (paperless == "Yes")
        + 0.012 * (monthly - 60)
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet,
            "tech_support": tech_support,
            "payment_method": payment,
            "paperless_billing": paperless,
            "senior_citizen": senior,
            "support_calls_90d": support_calls,
            "monthly_charges": monthly,
            "total_charges": total,
            "churned": churn.astype(int),
        }
    )
    # Realistic mess: a few missing totals for brand-new customers
    missing = rng.choice(n, 11, replace=False)
    df.loc[missing, "total_charges"] = np.nan
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=7000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/customers.csv")
    a = p.parse_args()
    df = generate(a.rows, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df):,} rows to {a.out} (churn rate {df.churned.mean():.1%})")


if __name__ == "__main__":
    main()
