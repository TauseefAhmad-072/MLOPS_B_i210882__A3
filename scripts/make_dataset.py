"""Generate a synthetic, fully offline customer-churn dataset (no personal data, no downloads).
v1 = 2000 customers, v2 = refreshed snapshot (different customers, slightly higher churn rate)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--version", choices=["v1", "v2"], default="v1")
p.add_argument("--out", default="data/customer_churn.csv")
a = p.parse_args()

rng = np.random.default_rng(42 if a.version == "v1" else 7)
n = 2000 if a.version == "v1" else 2400
tenure = rng.integers(1, 72, n)
monthly = rng.normal(65, 25, n).clip(15, 150).round(2)
tickets = rng.poisson(1.5, n)
usage = rng.gamma(3.0, 40.0, n).round(1)
late = rng.poisson(0.6, n)
contract = rng.choice([1, 12, 24], n, p=[0.55, 0.3, 0.15])
age = rng.integers(18, 80, n)
products = rng.integers(1, 6, n)
shift = 0.0 if a.version == "v1" else 0.25
logit = 3.5 * (-0.04 * tenure + 0.025 * (monthly - 65) + 0.35 * tickets + 0.5 * late - 0.06 * contract
         - 0.003 * (usage - 120) - 0.2 * products + 0.9 + shift) + rng.normal(0, 0.3, n)
churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

df = pd.DataFrame({"tenure_months": tenure, "monthly_charges": monthly, "support_tickets": tickets, "usage_gb": usage,
                   "late_payments": late, "contract_months": contract, "age": age, "num_products": products, "target": churn})
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(df)} rows ({a.version}), churn rate {churn.mean():.2%}")
