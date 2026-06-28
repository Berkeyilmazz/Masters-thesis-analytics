"""
======================================================================
THESIS ANALYSIS — AI adoption in early-stage venture capital
----------------------------------------------------------------------
One script that reproduces every number in the Results chapter:
  - data cleaning & analytic sample
  - scale construction + Cronbach's alpha (reliability)
  - descriptive statistics
  - H1-H4, E9, E10  -> Pearson correlations
  - H5              -> one-sample t-test (item level)
  - E6-E8           -> independent t-test + Mann-Whitney U
  - E9/E10 combined -> multiple linear regression (OLS)

Libraries used:
  pandas      -> data handling, row means, variances
  scipy.stats -> pearsonr, ttest_1samp, ttest_ind, mannwhitneyu
  statsmodels -> OLS regression
Only Cronbach's alpha is hand-coded (standard variance formula);
everything else is a validated library function.

Run:  python thesis_full_analysis.py
======================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# ----------------------------------------------------------------------
# 0. LOAD AND CLEAN
# ----------------------------------------------------------------------
# Qualtrics exports two metadata rows under the header (question text +
# import IDs). skiprows=[1,2] drops them so row 0 stays as column names.
SRC = "Thesis_June_27__2026_14_05.csv"     # <-- change path if needed
df = pd.read_csv(SRC, skiprows=[1, 2])

# Remove survey-preview (test) responses, keep only real ones.
real = df[~df["Status"].astype(str).str.contains("Preview", case=False, na=False)].copy()

# Completion rule: keep responses that reached >= 80% progress
# (fully complete + near-complete). This is the analytic sample.
real["Progress"] = real["Progress"].astype(float)
A = real[real["Progress"] >= 80].copy()
N = len(A)
print(f"Recorded={len(df)}  real(no preview)={len(real)}  analytic sample (>=80%)={N}")


# ----------------------------------------------------------------------
# 1. HELPERS
# ----------------------------------------------------------------------
def num(cols):
    """Return the given Likert columns as numeric (1-5), non-numeric -> NaN."""
    return A[cols].apply(pd.to_numeric, errors="coerce")

def cronbach_alpha(cols, reverse=()):
    """
    Cronbach's alpha — internal consistency of a multi-item scale.
    Formula (variance method):
        alpha = (k/(k-1)) * (1 - sum(item_variances)/variance_of_total)
    'reverse' lists any items to reverse-code on a 1-5 scale (6 - x).
    Returns alpha and the n used (listwise complete cases).
    """
    d = num(cols).copy()
    for r in reverse:
        d[r] = 6 - d[r]
    d = d.dropna()
    k = d.shape[1]
    if k < 2 or d.shape[0] < 3:
        return np.nan, d.shape[0]
    item_var = d.var(axis=0, ddof=1).sum()      # sum of each item's variance
    total_var = d.sum(axis=1).var(ddof=1)        # variance of the row totals
    alpha = (k / (k - 1)) * (1 - item_var / total_var)
    return alpha, d.shape[0]

def pearson(xcol, ycol):
    """Pearson r and p between two per-person series (pairwise complete cases)."""
    d = pd.concat([A[xcol], A[ycol]], axis=1).dropna()
    r, p = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])
    return r, p, len(d)


# ----------------------------------------------------------------------
# 2. BUILD CONSTRUCTS
# ----------------------------------------------------------------------
# Reliable multi-item scales -> row-wise mean of their items.
USEFULNESS = ["Q11_1","Q11_2","Q11_3","Q11_4","Q11_5","Q11_8",
              "Q11_10","Q11_11","Q11_13","Q11_15","Q11_16","Q11_17"]
RISK       = ["Q12_1","Q12_2","Q12_3","Q12_4","Q12_7"]

A["usefulness"] = num(USEFULNESS).mean(axis=1)
A["risk"]       = num(RISK).mean(axis=1)

# Trust did NOT form a reliable scale -> use single clearest item.
A["trust"]      = num(["Q11_12"]).iloc[:, 0]   # outputs reliable enough to support discussions
# Main dependent variable (single item).
A["influence"]  = num(["Q11_9"]).iloc[:, 0]    # AI influence on final go/no-go
# Human judgment did NOT scale -> use items individually.
A["intuition"]  = num(["Q12_10"]).iloc[:, 0]   # human intuition > AI in pre-seed

# Adoption breadth = count of distinct AI approaches selected in Q6 (0-4).
APPROACHES = ["general AI tools", "external AI/data", "internally developed", "actively developing"]
A["adopt"] = sum(A["Q6"].fillna("").str.contains(k, case=False).astype(int) for k in APPROACHES)


# ----------------------------------------------------------------------
# 3. RELIABILITY (Cronbach's alpha)
# ----------------------------------------------------------------------
print("\n--- RELIABILITY (Cronbach's alpha) ---")
for name, cols in [("Usefulness", USEFULNESS), ("Risk", RISK)]:
    a, n = cronbach_alpha(cols)
    print(f"  {name:12s} alpha={a:.2f}  items={len(cols)}  n={n}")
# Trust scale was rejected (alpha ~.06-.56); human judgment rejected (alpha ~.37).
# That is WHY trust and human judgment are handled as single items below.


# ----------------------------------------------------------------------
# 4. DESCRIPTIVES
# ----------------------------------------------------------------------
print("\n--- CONSTRUCT MEANS (1-5) ---")
for v in ["usefulness", "risk", "trust", "influence", "intuition"]:
    print(f"  {v:11s} mean={A[v].mean():.2f}  sd={A[v].std():.2f}")


# ----------------------------------------------------------------------
# 5. MAIN HYPOTHESES H1-H5
# ----------------------------------------------------------------------
print("\n--- H1-H4  (Pearson correlations) ---")
# H1: adoption -> usefulness   (expected +)
r, p, n = pearson("adopt", "usefulness");   print(f"  H1 adopt->usefulness   r={r:+.2f} p={p:.3f} n={n}")
# H2: usefulness -> trust      (expected +)
r, p, n = pearson("usefulness", "trust");   print(f"  H2 usefulness->trust   r={r:+.2f} p={p:.3f} n={n}")
# H3: trust -> influence       (expected +)  predictor & outcome are different items (clean)
r, p, n = pearson("trust", "influence");    print(f"  H3 trust->influence    r={r:+.2f} p={p:.3f} n={n}")
# H4: risk -> trust            (expected -)
r, p, n = pearson("risk", "trust");         print(f"  H4 risk->trust         r={r:+.2f} p={p:.3f} n={n}")

print("\n--- H5  (one-sample t-test vs neutral midpoint 3) ---")
# H5: is agreement that human intuition > AI significantly above 3?
v = A["intuition"].dropna()
t, p = stats.ttest_1samp(v, 3)
print(f"  H5 intuition mean={v.mean():.2f} vs 3  t={t:.2f} p={p:.4f} n={len(v)}")
# Supporting H5 items (descriptive % agree):
for c, lbl in [("Q12_4", "AI struggles w/ founders"), ("Q12_8", "would invest primarily on AI")]:
    x = num([c]).iloc[:, 0]
    print(f"     {lbl:28s} mean={x.mean():.2f}  %agree(>=4)={(x>=4).mean()*100:.0f}%")


# ----------------------------------------------------------------------
# 6. EXPLORATORY GROUP DIFFERENCES H6-H8
#    (independent t-test + Mann-Whitney U on adoption breadth)
# ----------------------------------------------------------------------
def group_test(mask_a, mask_b, label):
    a = A.loc[mask_a, "adopt"].dropna()
    b = A.loc[mask_b, "adopt"].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)         # Welch's t-test
    u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")  # rank-based check
    print(f"  {label}: {a.mean():.2f}(n{len(a)}) vs {b.mean():.2f}(n{len(b)})  "
          f"t={t:+.2f} p={p:.3f} | MannWhitney p={pu:.3f}")

print("\n--- H6-H8  (group comparisons on adoption breadth) ---")
# H6: institutional (VC/CVC/analyst) vs angel
inst = A["Q1"].isin(["Venture Capital","Corporate Venture Capital Investor","Investment Analyst / Associate"])
ang  = A["Q1"].eq("Angel Investor")
group_test(inst, ang, "H6 institutional vs angel")

# H7: more experienced (6+) vs less (0-5 deals)
lo = A["Q3"].astype(str).str.strip().eq("0–5")
hi = A["Q3"].notna() & ~lo
group_test(hi, lo, "H7 experienced vs new   ")

# H8: early-only (pre-seed/seed, no later) vs later-stage (Series A+)
early = A["Q2"].fillna("").str.contains("Pre-seed|Seed", case=False)
later = A["Q2"].fillna("").str.contains("Series A|Series B", case=False)
group_test(later, early & ~later, "H8 later vs early-only  ")


# ----------------------------------------------------------------------
# 7. INFLUENCE MODEL H9-H10
#    bivariate correlations, then a combined OLS regression
# ----------------------------------------------------------------------
print("\n--- H9-H10  (bivariate Pearson) ---")
r, p, n = pearson("risk", "influence");      print(f"  H9  risk->influence      r={r:+.2f} p={p:.3f} n={n}")
r, p, n = pearson("intuition", "influence"); print(f"  H10 intuition->influence r={r:+.2f} p={p:.3f} n={n}")

print("\n--- Combined regression: influence ~ trust + risk + intuition (standardised) ---")
d = A[["influence", "trust", "risk", "intuition"]].dropna()
z = (d - d.mean()) / d.std()                       # standardise -> comparable betas
X = sm.add_constant(z[["trust", "risk", "intuition"]])
model = sm.OLS(z["influence"], X).fit()
print(f"  n={int(model.nobs)}  R2={model.rsquared:.2f}")
for name in ["trust", "risk", "intuition"]:
    print(f"    {name:10s} beta={model.params[name]:+.2f}  p={model.pvalues[name]:.3f}")
# Interpretation: trust & risk stay significant; intuition does not ->
# H10 holds bivariately but not in the combined model.

print("\nDONE.")
