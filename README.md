# Masters-thesis-analytics

# Thesis analysis — AI adoption in early-stage venture capital

This folder contains the analysis script that reproduces every statistic
reported in the Results chapter, plus the figures and tables it supports.

## Files
- `thesis_full_analysis.py` — the complete, commented analysis script.
- `Thesis_June_27__2026_14_05.csv` — the Qualtrics survey export (recorded responses).

## How to run
1. Install the libraries:
   ```
   pip install pandas numpy scipy statsmodels
   ```
2. Put the CSV next to the script (or edit the `SRC` path at the top of the script).
3. Run:
   ```
   python thesis_full_analysis.py
   ```
   All results print to the screen.

## What the script does (in order)
1. **Cleaning & sample.** Drops Qualtrics' two metadata rows, removes survey-preview
   (test) responses, and keeps responses with >= 80% completion. Final analytic
   sample = 66.
2. **Construct building.** Averages reliable multi-item scales (usefulness, risk).
   Trust and human judgment did not form reliable scales, so they are used as
   single items (see below). Adoption breadth = count of AI approaches in Q6 (0-4).
3. **Reliability.** Cronbach's alpha for each candidate scale.
4. **Descriptives.** Means and SDs of the constructs.
5. **Hypotheses H1-H10** with the appropriate test for each.

## Measurement decisions (important)
- **Usefulness** (12 items, alpha = .85) and **Risk** (5 items, alpha = .73) are
  reliable scales -> used as item averages.
- **Trust** items did not cohere (alpha .06-.56). Inter-item analysis showed two
  separate ideas, so trust is measured with the single clearest item
  (Q11_12, "AI outputs reliable enough to support discussions").
- **Human judgment** items did not cohere (alpha = .37) -> interpreted as
  individual items (Q12_10 intuition, Q12_4 founder qualities, Q12_8 invest-on-AI).
- **AI influence** (Q11_9) is the main dependent variable (single item).

## Statistical methods and where each is used
| Test | Library function | Hypotheses |
|------|------------------|------------|
| Pearson correlation (r, p) | `scipy.stats.pearsonr` | H1, H2, H3, H4, H9, H10 |
| One-sample t-test | `scipy.stats.ttest_1samp` | H5 (vs neutral 3) |
| Welch's independent t-test | `scipy.stats.ttest_ind(equal_var=False)` | H6, H7, H8 |
| Mann-Whitney U (rank check) | `scipy.stats.mannwhitneyu` | H6, H7, H8 |
| OLS regression (betas, R2) | `statsmodels.api.OLS` | H9/H10 combined model |
| Cronbach's alpha | hand-coded (variance formula) | scale reliability |

Only Cronbach's alpha is hand-coded (standard variance-based formula); it matches
the value SPSS or the `pingouin` library would return. All inferential tests use
validated library functions.

## Hypotheses summary
| H | Relationship | Test | Result |
|---|--------------|------|--------|
| H1 | adoption -> usefulness | Pearson | not supported (r=.09, p=.47) |
| H2 | usefulness -> trust | Pearson | supported (r=.50, p<.001) |
| H3 | trust -> AI influence | Pearson | supported (r=.54, p<.001) |
| H4 | risk -> trust | Pearson | not supported (r=.01, p=.93) |
| H5 | human judgment > neutral | one-sample t | supported (t=8.1, p<.001) |
| H6 | institutional > angel adoption | t + MWU | supported (p=.002) |
| H7 | experienced > new adoption | t + MWU | supported (p=.021) |
| H8 | early-only < later-stage adoption | t + MWU | supported (p=.003) |
| H9 | risk -> AI influence (neg) | Pearson | supported (r=-.28, p=.040) |
| H10 | intuition -> AI influence (neg) | Pearson / OLS | supported bivariate only |

## Notes / caveats
- Correlations use pairwise complete cases, so n varies slightly per test (55-60).
- Significance threshold: p < .05.
- H6-H10 are exploratory (post-hoc subgroups / additional relationships) and use
  small subgroups; report as "this sample suggests", not causal claims.
- Results are associational; correlation does not establish causation.
