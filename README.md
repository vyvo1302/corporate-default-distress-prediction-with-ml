# Corporate Default & Distress Prediction with Machine Learning

Predicting next-year corporate default as a rare-event classification problem: 78,682 firm-years from 8,262 firms (fiscal years 1999–2018), with only 609 defaults (0.77%). The project compares six models from logistic regression to XGBoost. It also studies class imbalance, evaluates risk ranking and calibration, and analyzes model behavior in the dot-com and 2008 financial-crisis stress periods.

**Author:** Thanh Vy Vo

## Key results (test set: 2015–2018 fiscal years, 119 defaults)

| Model | ROC-AUC | PR-AUC | Defaults captured in top risk decile | Top-decile lift |
|---|---:|---:|---:|---:|
| Logistic regression | 0.721 | 0.036 | 36.1% | 3.6× |
| LASSO logistic | 0.742 | 0.048 | 38.7% | 3.9× |
| **Ridge logistic** (deployed) | 0.738 | 0.046 | 38.7% | 3.9× |
| Decision tree | 0.691 | 0.020 | 24.4% | 2.4× |
| Random forest | 0.734 | 0.048 | 35.3% | 3.5× |
| XGBoost | 0.735 | 0.038 | 32.8% | 3.3× |

- **Linear models hold their own.** Under a firm-clustered bootstrap, LASSO, Ridge, the random forest and XGBoost cannot be told apart on test ROC-AUC. Ridge is deployed under the rule fixed on validation, with XGBoost as the challenger.
- **Rebalancing didn't help.** Neither class weighting nor undersampling beat unweighted training under the pre-specified validation rule.
- **Calibration matters.** Raw Ridge probabilities underestimate risk (mean predicted 0.57% vs 0.97% observed). Platt scaling fitted on validation corrects the level (0.93%) without changing the ranking.
- **Stress periods.** Default rates peak in the 2008 financial crisis (1.40% vs 0.78% in other years). Return and volatility shift the most in both stress periods.

## How to read this project

| If you have… | Read |
|---|---|
| 5 minutes | [`default_prediction_report.pdf`](default_prediction_report.pdf): a 5-page summary of the findings, figures and recommendation, with no code |
| 30+ minutes | [`default_prediction_analysis.ipynb`](default_prediction_analysis.ipynb): the full analysis with code and outputs. It renders on GitHub, and the outputs are already saved, so you don't need to run it |
| Specific numbers | [`results_summary.json`](results_summary.json): every metric the notebook computes, as machine-readable output |

The notebook has seven sections:

1. **Data audit and exploration**: integrity checks, how rare defaults are, predictor distributions
2. **Linear benchmarks**: logistic regression, LASSO and Ridge, tuned on validation ROC-AUC
3. **Trees, bagging and boosting**: a decision tree, a random forest and XGBoost
4. **Class imbalance experiment**: class weighting and undersampling compared with unweighted training
5. **Out-of-sample evaluation**: a single locked test evaluation, ROC/PR curves and risk deciles
6. **Calibration and interpretation**: Platt and isotonic calibration, permutation and native importance
7. **Stress-period analysis**: the dot-com bust and the 2008 financial crisis, plus a deployment recommendation

## Methodology highlights

- **Chronological split, never shuffled.** Training covers fiscal years 1999–2011, validation 2012–2014 and test 2015–2018. Predictors from year *t* forecast default in year *t*+1.
- **Test lock.** All hyperparameters, thresholds, calibration maps and the deployment rule are frozen and fingerprinted before the test outcomes are released. The test set is scored once.
- **Honest comparisons.** Each model comparison uses a firm-clustered bootstrap with 2,000 resamples. A difference is called "higher" or "lower" only if its 95% interval excludes zero.
- **No leakage.** Scaling and encoding are fitted on training rows only, and undersampling never touches validation or test data.
- **Reproducible.** Every random step uses a fixed seed. Re-running the notebook reproduces `results_summary.json` byte for byte.

## Data

The data are an anonymized firm-year panel with 21 predictors in five groups: liquidity, leverage, profitability, operating efficiency and market information (market-to-book, market value, annual return, volatility and beta). The panel also has an industry code. Amounts are on a synthetic scale, so the results illustrate methods rather than real-world magnitudes. **The dataset is not included in this repository.**

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place the four data files in data/:
#   Corporate_Default_Model_Data_1999_2018.csv, Corporate_Default_Raw_FirmYear_1999_2018.csv,
#   Corporate_Default_Annual_Counts.csv, data_dictionary.csv
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 default_prediction_analysis.ipynb   # ~2 min
tectonic report/default_prediction_report.tex --outdir .   # optional: rebuild the PDF report
```

## Repository layout

```
├── default_prediction_analysis.ipynb   # main analysis (executed)
├── default_prediction_report.pdf       # 5-page written report
├── results_summary.json                # all computed metrics
├── figures/                            # charts produced by the notebook
├── report/default_prediction_report.tex
├── data/                               # data files go here (not tracked)
└── requirements.txt
```
