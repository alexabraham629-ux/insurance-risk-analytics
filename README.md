# Insurance Risk Analytics

Financial statement analysis, market-risk and portfolio analysis, claim-risk prediction and anomaly screening for the Indian insurance sector.

**Author:** Alex Abraham · MBA (Business Analytics), CHRIST (Deemed to be University), Pune Lavasa Campus
**Course:** Financial and Risk Analytics (MBA441B), CIA-4 Sector Analysis

---

## What's inside

| Component | Data | Method | Headline result |
|---|---|---|---|
| Financial statement analysis | ICICI Lombard annual reports, FY2024–FY2026 | Horizontal/vertical analysis, IRDAI ratios, adapted Piotroski score | Combined ratio 102.8–103.4%: profit comes from investment income, not underwriting. Solvency 2.62–2.69× vs 1.50× minimum |
| Market risk and portfolios | 104 monthly returns: ICICI Lombard, GIC Re, SBI Life, NIFTY 50 | Volatility, beta, CAPM, Sharpe/Treynor, VaR, drawdown, mean-variance optimisation | Max-Sharpe portfolio (ICICI Lombard + SBI Life) Sharpe 0.372 vs NIFTY 0.291 |
| Claim-occurrence model | freMTPL2freq, 678,013 French motor policies (5.02% claim rate) | Logistic regression, random forest, HistGradientBoosting; validation-only selection, one-time test | Test ROC-AUC 0.665 (95% CI 0.658–0.671), AP 0.126; top 5% riskiest policies have a 17.7% claim rate (3.5× lift) |
| Claim-frequency model | Same, exposure-aware | Poisson GLM vs Poisson gradient boosting | 8.2% deviance explained; 7,221 predicted vs 7,225 observed test claims |
| Anomaly screening | Same | Two Isolation Forest screens (prospective / retrospective) with reason codes | ~1% flagged per screen; seed-stability Jaccard 0.69 / 0.77 |
| Decision support | — | ipywidgets Policy Claim Probability Predictor, batch scoring, model card | Probability, risk band, annual frequency, partial-dependence explanations |

## Repository structure

```
insurance-risk-analytics/
├── src/
│   └── claim_risk_analytics.py              # End-to-end pipeline (single Jupyter cell)
├── reports/
│   ├── report.pdf                           # Full written report (48 pages)
│   ├── presentation.pptx                    # Presentation deck
│   └── financial_and_portfolio_analysis.xlsx  # Financials, ratios, Piotroski, stock & portfolio sheets
├── requirements.txt
└── README.md
```

## Running the claim-risk pipeline

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Get the **freMTPL2freq** dataset from the [CASdatasets](https://github.com/dutangc/CASdatasets) R package (also on OpenML / Kaggle) and save it as `data/freMTPL2freq.csv`. The code expects the renamed columns used in the report (`Policy ID`, `Number of claims`, `Exposure period`, `Vehicle power`, `Vehicle age`, `Driver age`, `Bonus-Malus score`, `Vehicle brand`, `Fuel type`, `Geographic area`, `Population density`, `Geographic region`).
   Or point to your own copy:
   ```bash
   export INSURANCE_DATA_PATH=/path/to/freMTPL2freq.csv
   ```
3. Paste `src/claim_risk_analytics.py` into one Jupyter cell and run it (or `python src/claim_risk_analytics.py`). Outputs (tables, charts, model bundle, model card) are written to `insurance_risk_outputs_enhanced/`.

Settings such as `MAX_ROWS`, `THRESHOLD_METHOD` (`f1` / `cost` / `capacity`) and cost ratios are in the CONFIG block of Part 9.

## Methodology highlights

- **Leakage control:** all preprocessing fitted on training rows; model, calibration and threshold chosen on validation; test scored once.
- **Imbalance-aware evaluation:** Average Precision, Brier score, lift/gains and review-capacity tables instead of accuracy; bootstrap 95% confidence intervals.
- **Business thresholds:** F1, cost-based (FP vs FN cost) and capacity-based alert rules compared side by side.
- **Actuarial track:** Poisson frequency with exposure weights, giving claims per policy-year and P(≥1 claim in t years) = 1 − e^(−λt).
- **Governance:** versioned model bundle, model card JSON, prediction log, out-of-range and unseen-category warnings.

## Limitations

- The claim models are trained on **French** MTPL data and are **not validated for any Indian insurer**. This is an educational prototype, not an underwriting, pricing or fraud system.
- Random (not time-based) split; no claim cost, dates or fraud labels.
- Anomaly flags are statistical oddities, not fraud findings.
- Portfolio results are in-sample and window-dependent.

## Tech stack

Python (pandas, NumPy, scikit-learn, SciPy, Matplotlib, ipywidgets) · Advanced Excel with OpenSolver
