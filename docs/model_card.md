# Model Card: Credit-Card Fraud Detector (XGBoost)

## Model details
- **Type:** gradient-boosted trees, `XGBClassifier` (`tree_method="hist"`, native categorical support).
- **Version:** identified by the first 12 characters of the model file's MD5 (`model_version` in every API
  response and in `prediction_log`), currently `72b46678548f`.
- **Training settings:** `scale_pos_weight` computed from train (normal ÷ fraud ≈ 170.4); up to 2,000 trees with
  early stopping (50 rounds, on validation PR-AUC) → **86 trees**; `random_state=42`. Everything else is XGBoost's
  defaults. Two runs with the same data and settings produce a byte-identical model file.
- **Inputs (9 features):** `amt`, `category` (14 values), `hour`, `age`, `secs_since_last`, `is_first_txn`,
  `card_avg_amt_before`, `amt_ratio`, `txn_count_24h`. Card features use only each card's **past** purchases.
- **Output:** a fraud score from 0 to 1, a decision (**flag / review / approve**), and up to 3 reason codes.

## Intended use
- Scoring card transactions in real time, so that **human analysts review the flagged ones**. It's a decision-support
  tool: it is not meant to block cards automatically without human review.
- **Not intended for:** real customers' data without retraining and validation; any decision about a person other
  than "should an analyst look at this transaction".

## Training data
- ~1.85 million **simulated** card transactions (2019–2020, ~1,000 cardholders), from a public synthetic dataset.
- **Time-based split:** train (Jan 2019 – 20 Mar 2020, 1,070,966 rows, 0.584% fraud), validation (21 Mar – 21 Jun 2020,
  225,709 rows, 0.556%), test (21 Jun – 31 Dec 2020, 555,719 rows, 0.386%; opened once, at the end).
- **Dropped:** personal identifiers and per-person columns (names, street, city, zip, job, location), merchant
  location (random noise), merchant (no signal beyond category), and **gender** (a protected attribute, kept only
  to check fairness).

## Performance

| | Validation | Test |
|---|---|---|
| PR-AUC | 0.940 | 0.906 |
| ROC-AUC | 0.999 | 0.998 |
| Recall at cutoff 0.90 | 90.4% | 87.6% |
| Precision at cutoff 0.90 | 78.4% | 72.9% |
| Alerts per month | 479 | 406 |

- Baseline (logistic regression): validation PR-AUC 0.221.
- The cutoff (0.90) was chosen on validation from analyst capacity (~1,500 alerts per 3 months). The cost-minimising
  cutoff on validation was 0.59 ($35k vs $44k for 3 months), but it needs ~43% more analyst capacity.
- Test PR-AUC is lower mainly because of **prior shift** (fraud fell from 0.556% to 0.386%); ROC-AUC held.

## Explainability
- Every decision returns **reason codes**: the top features that pushed the score toward fraud (SHAP values computed
  by XGBoost, `pred_contribs=True`), as sentences with context, e.g. *"Amount $1,077.69 is 19.2x this card's usual
  spend", "Purchase at 22:00 (high-risk night window)"*.
- Overall drivers (mean |SHAP|): amount, category, hour, then age and the card-history features.
  `is_first_txn` has zero importance (the model reads "no history" from the empty `secs_since_last`).

## Fairness (validation, cutoff 0.90)
Compared per group: **false alarm rate** (innocent customers flagged) and **recall** (frauds caught).

| Group | False alarm rate | Recall |
|---|---|---|
| Women | 0.164% | 86.9% |
| Men | 0.108% | 94.0% |
| Ages 25–45 | 0.196% | 81–85% |
| Ages 55+ | 0.06–0.075% | 95–98% |

Women and ages 25–45 get **more false alarms and less protection**. Gender is **not** a model input, so the gap comes
through proxies (e.g. spending categories); removing a protected column doesn't remove the bias. Small groups are
noisy (under-25 has only 71 frauds). **Not yet mitigated.** Next steps: retrain without `age` and measure the gap and
the PR-AUC cost on validation; investigate gender proxies; review with a model-risk team.

## Limitations and guardrails

| Limitation | Effect | Guardrail |
|---|---|---|
| Trees don't extrapolate (largest training fraud: $1,372) | Scores freeze above ~$1,200: $2,000 and $90,000 score the same | Amount above the training maximum ($28,948.90) → **review** |
| "First purchases" in training are the dataset's start, not new cards | Brand-new cards are under-flagged (no history, weakest signals) | New card with amount above the training 99th percentile ($546.72) → **review** |
| Synthetic data | Cleaner patterns than real life | Retrain and re-validate on real data before any real use |

Decision order: **flag** if the score ≥ 0.90 (the model is confident), else **review** if a rule fires, else
**approve**. The limits are computed from the train split (`models/guardrails.json`). On validation the rules add
**1 review in 3 months** (it was a real fraud the model missed), far within capacity.

## Monitoring (planned)
Every prediction is stored in `prediction_log` (model version, the 9 inputs, score, decision, which rule fired, and
the true label once confirmed). Watch: the fraud rate (prior shift), feature drift, precision from analyst decisions,
alert and review volume vs capacity, and the fairness metrics above, monthly.
