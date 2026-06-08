"""
generate_lms_repayment_schedule.py
====================================
NPA Early Warning System — Model 2 Data Pipeline
Generates `lms_repayment_schedule.csv`: a time-series relational table
containing 24–36 months of rolling monthly payment history for every
loan account in `lms_advanced_master.csv`.

Architecture
------------
  Master CSV  →  Vectorized EMI Computation  →  Phase-Aware Behavioral
  Simulation  →  Row Explosion via np.repeat  →  Output CSV

The behavioural engine encodes three distinct financial trajectories:
  • Class 0 (Repaid)  : Healthy payments with rare, self-correcting noise
  • Class 1 (Default) : Progressive deterioration across three phases,
                        culminating in terminal zero-payment months
"""

import os
import time
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────
INPUT_PATH   = "credit_scoring_engine/data/lms_advanced_master.csv"
OUTPUT_PATH  = "npa_early_warning_sys/data/lms_repayment_schedule.csv"
RANDOM_SEED  = 42
SCHEDULE_CAP = 36          # Maximum months to generate per loan (cap long tenures)
SCHEDULE_MIN = 24          # Minimum months to generate per loan

# Nepal base interest rate band (NRB regulated, annualised, %)
INTEREST_RATE_MIN = 0.08   # 8%  — subsidised / priority sector
INTEREST_RATE_MAX = 0.18   # 18% — unsecured / high-risk personal

# ─────────────────────────────────────────────────
# 1.  LOAD MASTER DATA
# ─────────────────────────────────────────────────
def load_master(path: str) -> pd.DataFrame:
    print(f"[1/7] Loading master dataset from '{path}' ...")
    df = pd.read_csv(path)
    df.insert(0, "loan_account_id", range(1, len(df) + 1))
    print(f"      → {len(df):,} loan accounts loaded  |  "
          f"Class-0 (Repaid): {(df.defaulted==0).sum():,}  |  "
          f"Class-1 (Default): {(df.defaulted==1).sum():,}")
    return df


# ─────────────────────────────────────────────────
# 2.  DERIVE INTEREST RATE
# ─────────────────────────────────────────────────
def derive_interest_rates(df: pd.DataFrame) -> pd.Series:
    """
    Back-calculate a plausible annual interest rate for each loan.
    Strategy: blend borrower-risk proxies (LTV, DSCR, collateral,
    EMI ratio) into a normalised risk score, then map linearly onto
    the NRB rate band [8%, 18%].
    """
    # Higher risk → higher rate
    risk = (
        0.30 * df["ltv_ratio"].clip(0, 1)                       # LTV: 0=safe, 1=risky
      + 0.25 * (1 / df["dscr"].clip(0.5, 5)).pipe(             # DSCR: low coverage = risky
                    lambda s: (s - s.min()) / (s.max() - s.min()))
      + 0.20 * (1 - df["collateral_score"].clip(0, 1))          # Low collateral = risky
      + 0.15 * df["emi_income_ratio"].clip(0, 1)                # High burden = risky
      + 0.10 * df["hard_enquiry_6m"].clip(0, 10).div(10)        # Many enquiries = risky
    )
    # Normalise to [0, 1]
    risk_norm = (risk - risk.min()) / (risk.max() - risk.min() + 1e-9)
    annual_rate = INTEREST_RATE_MIN + risk_norm * (INTEREST_RATE_MAX - INTEREST_RATE_MIN)
    return annual_rate


# ─────────────────────────────────────────────────
# 3.  COMPUTE FIXED MONTHLY EMI (PMT formula)
# ─────────────────────────────────────────────────
def compute_emi(principal: np.ndarray,
                annual_rate: np.ndarray,
                tenure_months: np.ndarray) -> np.ndarray:
    """
    Standard amortising EMI:
        EMI = P * r * (1+r)^n / ((1+r)^n - 1)
    where r = monthly_rate, n = tenure_months.
    For zero-interest edge case falls back to simple division.
    """
    r = annual_rate / 12.0
    n = tenure_months.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(
            r > 0,
            r * (1 + r) ** n / ((1 + r) ** n - 1),
            1.0 / n
        )
    return np.round(principal * factor, 2)


# ─────────────────────────────────────────────────
# 4.  COMPUTE SCHEDULE LENGTH PER LOAN
# ─────────────────────────────────────────────────
def compute_schedule_lengths(tenure_months: np.ndarray) -> np.ndarray:
    """
    Clamp each loan's observable window to [SCHEDULE_MIN, SCHEDULE_CAP].
    Short-tenure loans (< SCHEDULE_MIN) observe the full tenure.
    """
    return np.clip(tenure_months, SCHEDULE_MIN, SCHEDULE_CAP).astype(int)


# ─────────────────────────────────────────────────
# 5.  SIMULATE PAYMENT RATIOS — VECTORISED CORE
# ─────────────────────────────────────────────────

def _ratios_repaid(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Class 0: Healthy borrower.
    • 95–99% of months → full payment (ratio = 1.0)
    • ~1–5% chance of a single random "forgotten" month (ratio 0.0)
      followed immediately by a catch-up month (ratio 2.0), keeping
      cumulative payments on track.
    • Tiny amount-noise (~±3%) on regular months simulates round-ups
      / partial over-payments common in Nepali banking practice.
    """
    ratios = rng.normal(loc=1.0, scale=0.015, size=n).clip(0.97, 1.05)

    # Rare single-month skip with catch-up
    if n >= 3 and rng.random() < 0.06:          # 6% of healthy borrowers
        skip_idx = rng.integers(1, n - 1)        # never skip first or last
        ratios[skip_idx]     = 0.0               # missed
        ratios[skip_idx + 1] = min(2.0, ratios[skip_idx + 1] + 1.0)  # catch-up

    return ratios.round(4)


def _ratios_defaulted(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Class 1: Defaulted borrower — three-phase financial deterioration.

    Phase 1 — STABLE   : First ~40% of schedule, full payments with minor noise.
    Phase 2 — STRESS   : Middle ~35%, progressive partial payments and missed months.
                         Modelled as a downward sigmoid plus Bernoulli skip events.
    Phase 3 — TERMINAL : Final 3–6 months locked to 0.0 (hard default run-up).
    """
    ratios = np.zeros(n, dtype=np.float64)

    terminal_len = rng.integers(3, min(7, n))       # 3–6 terminal months
    terminal_start = n - terminal_len

    phase1_end = max(1, int(n * rng.uniform(0.30, 0.45)))
    phase2_end = terminal_start

    # ── Phase 1: stable ────────────────────────────────────────────────
    p1_len = phase1_end
    if p1_len > 0:
        ratios[:p1_len] = rng.normal(1.0, 0.02, size=p1_len).clip(0.96, 1.04)

    # ── Phase 2: stress (downward slope + random misses) ────────────────
    p2_len = phase2_end - phase1_end
    if p2_len > 0:
        # Sigmoid descent from ~0.95 down to ~0.20 across phase 2
        t = np.linspace(-3, 3, p2_len)
        sigmoid = 1.0 / (1.0 + np.exp(t))                   # 0.95 → 0.12
        base = 0.20 + 0.75 * sigmoid                         # scale to [0.20, 0.95]

        noise = rng.normal(0.0, 0.07, size=p2_len)
        stress_ratios = (base + noise).clip(0.0, 1.05)

        # Inject Bernoulli missed-payment events (probability rises with time)
        miss_prob = np.linspace(0.05, 0.45, p2_len)
        missed_mask = rng.random(size=p2_len) < miss_prob
        stress_ratios[missed_mask] = rng.uniform(0.0, 0.15, size=missed_mask.sum())

        ratios[phase1_end:phase2_end] = stress_ratios

    # ── Phase 3: terminal — hard zeros ─────────────────────────────────
    ratios[terminal_start:] = 0.0

    return ratios.round(4)


def simulate_all_payment_ratios(ids:       np.ndarray,
                                 n_months:  np.ndarray,
                                 defaulted: np.ndarray,
                                 seed:      int) -> list[np.ndarray]:
    """
    Loop over each loan account and dispatch to the appropriate
    simulation function.  Returns a list of per-loan ratio arrays.
    Pure Python loop is acceptable here: 8 000 calls × tiny arrays
    is ~0.5 s; Cython/numba overhead would dwarf the gain.
    """
    rng = np.random.default_rng(seed)
    all_ratios = []

    for i in range(len(ids)):
        n   = int(n_months[i])
        cls = int(defaulted[i])
        if cls == 0:
            all_ratios.append(_ratios_repaid(n, rng))
        else:
            all_ratios.append(_ratios_defaulted(n, rng))

    return all_ratios


# ─────────────────────────────────────────────────
# 6.  EXPLODE INTO LONG-FORMAT DATAFRAME
# ─────────────────────────────────────────────────

def build_schedule_dataframe(df:        pd.DataFrame,
                              emi:       np.ndarray,
                              n_months:  np.ndarray,
                              all_ratios: list[np.ndarray],
                              close_year: np.ndarray) -> pd.DataFrame:
    """
    Vectorised row-explosion using np.repeat + concatenated ratio arrays.
    Building the DataFrame in one shot avoids iterative appends, keeping
    memory allocation O(total_rows) rather than O(n_loans * total_rows).
    """
    print("[6/7] Exploding loan schedules into long-format rows ...")

    total_rows = int(n_months.sum())
    print(f"      → Target row count: {total_rows:,}  "
          f"({len(df):,} accounts × avg {n_months.mean():.1f} months)")

    # ── Repeated columns (one value per loan, tiled across months) ──────
    loan_ids_rep  = np.repeat(df["loan_account_id"].values, n_months)
    emi_rep       = np.repeat(emi, n_months)

    # ── month_index: [1,2,...,n] per loan, concatenated ──────────────────
    month_idx = np.concatenate([np.arange(1, n + 1) for n in n_months])

    # ── due_date: approximate loan disbursement then roll forward ────────
    # Derive disbursement month: close_year minus tenure gives approx start
    start_years_float = close_year.astype(float) - (n_months / 12.0)
    start_year_int    = start_years_float.astype(int)
    start_month_frac  = (start_years_float - start_year_int) * 12
    start_month_int   = np.round(start_month_frac).astype(int).clip(1, 12)

    # Convert each loan's start to a (year, month) absolute-month integer
    # abs_month = year*12 + (month-1)  →  unique monotone integer per calendar month
    abs_month_per_loan = start_year_int * 12 + (start_month_int - 1)   # shape (n_loans,)

    # Repeat per loan, then offset by (month_index - 1)
    abs_month_rep = np.repeat(abs_month_per_loan, n_months) + (month_idx - 1)

    # Convert back to calendar (year, month) — pure integer arithmetic
    due_year  = abs_month_rep // 12
    due_month = abs_month_rep %  12 + 1   # 1-based month

    # Build date strings as YYYY-MM-DD (last day of due month)
    # Use pandas DatetimeIndex for vectorised last-day calculation
    due_starts   = pd.to_datetime(
        {"year": due_year, "month": due_month, "day": 1}
    )
    due_dates    = due_starts + pd.offsets.MonthEnd(0)   # snap to month-end
    due_date_str = due_dates.dt.strftime("%Y-%m-%d")

    # ── Payment ratios concatenated ─────────────────────────────────────
    payment_ratios = np.concatenate(all_ratios)

    # ── Derived columns ─────────────────────────────────────────────────
    amount_paid = np.round(emi_rep * payment_ratios, 2)
    missed      = (payment_ratios < 0.95).astype(np.int8)

    schedule = pd.DataFrame({
        "loan_account_id": loan_ids_rep.astype(np.int32),
        "month_index":     month_idx.astype(np.int16),
        "due_date":        due_date_str,
        "amount_due":      emi_rep.astype(np.float32).round(2),
        "amount_paid":     amount_paid.astype(np.float32),
        "payment_ratio":   payment_ratios.astype(np.float32),
        "missed":          missed,
    })

    return schedule


# ─────────────────────────────────────────────────
# 7.  MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()
    print("=" * 65)
    print("  NPA EARLY WARNING SYSTEM — REPAYMENT SCHEDULE GENERATOR")
    print("  Model 2 | mbb_loan_schedule Table Builder")
    print("=" * 65)

    # ── Step 1: Load ──────────────────────────────────────────────────
    df = load_master(INPUT_PATH)

    # ── Step 2: Derive interest rates ────────────────────────────────
    print("[2/7] Deriving per-loan annual interest rates ...")
    annual_rates = derive_interest_rates(df).values
    print(f"      → Rate band  : {annual_rates.min()*100:.2f}% – "
          f"{annual_rates.max()*100:.2f}%  |  "
          f"Mean: {annual_rates.mean()*100:.2f}%")

    # ── Step 3: Compute EMI ──────────────────────────────────────────
    print("[3/7] Computing fixed monthly EMI for all accounts ...")
    emi = compute_emi(
        principal    = df["loan_amount_npr"].values.astype(float),
        annual_rate  = annual_rates,
        tenure_months= df["loan_tenure_months"].values.astype(float),
    )
    print(f"      → EMI range  : NPR {emi.min():,.0f} – {emi.max():,.0f}  |  "
          f"Mean: NPR {emi.mean():,.0f}")

    # ── Step 4: Compute schedule window per loan ──────────────────────
    print("[4/7] Capping observable schedule window "
          f"[{SCHEDULE_MIN}–{SCHEDULE_CAP} months] ...")
    n_months = compute_schedule_lengths(df["loan_tenure_months"].values)
    total    = n_months.sum()
    print(f"      → Total rows to generate: {total:,}")

    # ── Step 5: Simulate payment ratios ──────────────────────────────
    print("[5/7] Simulating phase-aware payment behaviour ...")
    print(f"      (Repaid accounts: {(df.defaulted==0).sum():,}  |  "
          f"Defaulted accounts: {(df.defaulted==1).sum():,})")
    all_ratios = simulate_all_payment_ratios(
        ids       = df["loan_account_id"].values,
        n_months  = n_months,
        defaulted = df["defaulted"].values,
        seed      = RANDOM_SEED,
    )

    # ── Step 6: Build long-format DataFrame ──────────────────────────
    schedule = build_schedule_dataframe(
        df         = df,
        emi        = emi,
        n_months   = n_months,
        all_ratios = all_ratios,
        close_year = df["close_year"].values,
    )

    # ── Step 7: Save output ───────────────────────────────────────────
    print("[7/7] Saving output CSV ...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    schedule.to_csv(OUTPUT_PATH, index=False)

    elapsed = time.perf_counter() - t0

    # ── Validation summary ────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  GENERATION COMPLETE — OUTPUT VALIDATION REPORT")
    print("=" * 65)
    print(f"  Output file     : {OUTPUT_PATH}")
    print(f"  Total rows      : {len(schedule):,}")
    print(f"  Unique accounts : {schedule['loan_account_id'].nunique():,}")
    print(f"  Elapsed time    : {elapsed:.2f}s")
    print()
    print("  Payment Ratio Distribution:")
    print(f"    Mean ratio     : {schedule['payment_ratio'].mean():.4f}")
    print(f"    Missed rows    : {schedule['missed'].sum():,}  "
          f"({schedule['missed'].mean()*100:.1f}%)")
    print()

    # Per-class sanity check
    id_class = df[["loan_account_id", "defaulted"]].copy()
    merged   = schedule.merge(id_class, on="loan_account_id")
    for cls, label in [(0, "Repaid (Class 0)"), (1, "Defaulted (Class 1)")]:
        sub = merged[merged["defaulted"] == cls]
        print(f"  {label}:")
        print(f"    Avg payment ratio : {sub['payment_ratio'].mean():.4f}")
        print(f"    Miss rate         : {sub['missed'].mean()*100:.1f}%")
        print(f"    Zero-payment rows : {(sub['payment_ratio']==0).sum():,}")
    print()
    print("  Schema:")
    print(schedule.dtypes.to_string())
    print()
    print("  Sample (first 5 rows):")
    print(schedule.head(5).to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()
