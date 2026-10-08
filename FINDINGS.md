# Fraud Detection: Findings Log

One place for everything we found so far. For each finding:
- **Why we checked**: the question we were asking
- **What we found**: the numbers
- **How we use it**: the decision it led to

Data: `fraudTrain.csv` (Jan 2019 – Jun 2020) and `fraudTest.csv` (Jun – Dec 2020).
Learning details and code are in [LEARNING_NOTES.md](LEARNING_NOTES.md).

**Status:** Step 1 ✅ · Step 2 ✅ · Step 3a ✅ · Step 3b ✅ · Step 3c ✅ · Step 3d ✅ · Step 3e ✅ · Step 4 ✅ · Step 5 prep ✅ · Step 5 baseline ✅ · Step 6 part 1 ✅ · Step 6 part 2 ✅ · Step 6 part 3 ✅ · Step 6 ✅ · Step 7 ✅ (final test done) · Step 8 ✅ · **Notebook phase done**. The decisions summary is the last cell of `notebooks/05_explain.ipynb`. The pipeline must reproduce val PR-AUC ≈ 0.940. · Step 3d ⬜ · Step 3e ⬜

---

## Step 1: Data inventory

| # | Why we checked | What we found | How we use it |
|---|---|---|---|
| 1 | How rare is fraud? | Train 0.579% (7,506 of 1,296,675). Test 0.386% (2,145 of 555,719). | Accuracy is useless (99.4% by always saying "not fraud"). Use precision, recall and PR-AUC. |
| 2 | Is the fraud rate the same in train and test? | No. It dropped from 0.579% to 0.386%, too big to be luck. The cause is unknown. | This is **prior shift**: a threshold tuned on train will be wrong on test. The threshold needs checking and monitoring. |
| 3 | Do the train and test dates overlap? | No. Test starts 48 seconds after train ends. | It's a clean time split, like real life. Never shuffle across time. |
| 4 | Are the rows sorted by time? | Train is **not** fully sorted. | Sort by card and time before any "history" feature. |
| 5 | Do the same cards appear in both files? | 908 of 924 test cards (98%) are in train. 16 are new, and 75 train cards disappear. | Card-history features will work on test. New cards need a sensible "no history" value. |
| 6 | Are there missing values or duplicates? | 0 nulls, 0 duplicate rows. `trans_num` is unique for every row. | No cleaning is needed for nulls. |
| 7 | Do the column types match their meaning? | `trans_date_trans_time` and `dob` are text but really dates. `cc_num` and `zip` are numbers but really IDs. | Convert the dates. Never do maths on IDs. |
| 8 | Is `Unnamed: 0` useful? | It's the old CSV row number. | Dropped. It won't exist for a live transaction. |
| 9 | Do `unix_time` and `trans_date_trans_time` agree? | Same clock time, but 7 years earlier. The gap goes 2557 → 2556 → 2557 days because of leap days. No transactions are dated 2020-02-29. | `unix_time` is a relabelled copy, so it was dropped. `trans_date_trans_time` is the only time column. |
| 10 | How much memory does the data use? | Train 1,085 MB, test 465 MB. 78% of that is text columns. | Convert repeated text columns to `category` later to save memory. |

---

## Step 2: Exploring the data (train only)

| # | Why we checked | What we found | How we use it |
|---|---|---|---|
| 11 | Do fraud amounts differ? | Median fraud $397 vs normal $47 (8.4×). Fraud never goes above $1,376. | `amt` is a strong feature. |
| 12 | What shape are the fraud amounts? | There are two groups: 21% are $50 or less, and 76% are $200–$1,500. | "Test small, then spend big." This motivates card-history features (Step 3c). |
| 13 | Is `amt` lopsided? | Skewness 42.3 → −0.3 after log1p. Kurtosis 4,546 → −0.5. | Use log1p for linear models. Trees don't need it, because log keeps the order. |
| 14 | Are there zero or negative amounts? | None. The minimum is $1.00. | log1p is safe. Keep it as the default anyway. |
| 15 | Which categories are risky? | Highest: shopping_net 1.76%, misc_net 1.45%, grocery_pos 1.41%. Lowest: health_fitness 0.15%. That's an 11.3× spread. | `category` is a strong feature. |
| 16 | Does the time of day matter? | 22:00–23:59 runs at about 2.9% (5× average). 00:00–03:59 runs at about 1.5% (2.5×). Daytime is about 0.1%. | Added `hour` (Step 3a). Night is a strong signal. |
| 17 | Does age matter? | It's U-shaped: 75+ is 0.90%, 35–45 is 0.43%. But each age group is only about 90–210 people. | Added `age` (Step 3a). Treat it as a medium signal. |
| 18 | Is the fraud rate stable month to month? | No. It ranges from 0.38% to 1.04%. There's no clear trend. Seasonality is possible but can't be proven with 18 months. | Test's 0.386% is within normal monthly movement. The base rate moves, so a production model needs monitoring. |
| 19 | Anything strange in the data? | 13 cardholders are under 18 (the youngest is 13). The hour pattern switches sharply at 04:00. Fraud stops at $1,376. | It's generated data. Some patterns may not hold in real life. These are noted, not removed. |

---

## Step 3: Building features

| # | Why we checked | What we found | How we use it |
|---|---|---|---|
| 20 | Which columns can a model use? | `Unnamed: 0`, `unix_time`, `trans_num`, `first`, `last` and `street` say nothing about fraud, or won't exist for a new transaction. | Dropped (3a). Added `age` and `hour`. 23 → 19 columns. |
| 21 | Does each merchant have a fixed location? | No. Each merchant has 727 to 4,403 different locations (median 1,863), and 0 of 693 have one fixed spot. Every transaction has its own location. | The merchant location is random, not a real address. |
| 22 | Is the distance from home a fraud signal? | No. The mean is 76.27 km for fraud vs 76.11 km for normal, and the medians are 77.93 vs 78.23 km. | Dropped `distance_km`, `merch_lat` and `merch_long` (3b). A noise column lets trees make random splits that fit train by luck (mild overfitting). In a bank, every feature costs effort to build, monitor and justify, so a useless one is pure cost. |
| 23 | Does a merchant change category? | Every merchant has exactly 1 category. | Use this in 3d, when deciding how to handle `merchant` and `category`. |
| 24 | Does a stolen card get used fast? | Yes. The median time since the card's last purchase is 4,908 s (1.4 h) for fraud vs 16,623 s (4.6 h) for normal, so fraud is about 3.4× faster. | Added `secs_since_last`. It was computed on train + test combined and sorted by card and time, so test cards can use their train history (the past is allowed). |
| 25 | What about a card's first-ever purchase? | Train has 983 empty rows (one per card) and test has 16 (the new cards). Both match the prediction. | Left empty (NaN), because "no history" is unknown and filling with the median would be a lie. Added `is_first_txn` (1 = first purchase). Trees handle NaN. Linear models would need a fill value. **Caution:** in train, the 983 "first purchases" are not new cards. They're just the first transaction since the data starts (Jan 2019), and those cards existed before. Only the 16 in test are really new. So in train the flag mostly means "start of the dataset", not "new card". |
| 26 | Why does `secs_since_last` matter beyond the model? | (Mentor's notes) | (1) Model: a strong split for the trees. (2) Explaining decisions (SHAP): "flagged because the card was used again 2 minutes after the last purchase". (3) Serving: the live API must know each card's last transaction time, which is why a card-history table in Neon is planned. (4) Drift: if fraudsters change speed, this feature's distribution shifts and the drift monitor catches it. |
| 27 | Is a big amount big *for this person*? | Yes. The median `amt_ratio` is 5.23 for fraud vs 0.66 for normal. 24% of frauds are below 1 (the small "testing" purchases). Normal is below 1 because a few big purchases pull a card's average above its typical purchase. | Added `card_avg_amt_before` (the card's average over **previous** transactions only; using all transactions would include the future, which is leakage) and `amt_ratio` = amt ÷ that average. Empty for first purchases (983 train, 16 test). |
| 28 | Is the card used in bursts? | Weakly. The mean `txn_count_24h` is 5.23 for fraud vs 4.88 for normal (medians 5 vs 4), only 7% higher. Busy and quiet cardholders have very different normal counts. | Added `txn_count_24h` (a time-based 24-hour window per card, minimum 1). Kept, because trees can combine it with other features. Idea: compare it to the card's usual daily count. |

| 29 | Which text and person columns are groups, and which are nametags? | Unique values in train: category 14, gender 2, state 51, job 494, merchant 693, city_pop 879, city 894, lat 968, long 969, zip 970 (vs 983 cards). Everything except category and merchant is **fixed per card**. | Nametags (about 1–2 people per value) are dropped: city, zip, job, lat, long, city_pop. They only say "which person", so the model would memorise. |
| 30 | Is `state` a real signal? | No. States with 20+ cards range only 0.49–0.69%. The "risky" states are tiny: DE = 1 card, 100% fraud (9 transactions). RI = 1 card, AK = 3 cards. | Dropped. Keeping it would teach "DE = fraud", which is memorising one person. |
| 31 | Does `merchant` add anything beyond `category`? | No. The spread of merchant rates within a category (middle 50%: 0.74–1.22× the category rate) matches pure chance (0.74–1.24×). The median is only 5 frauds per merchant. | Dropped. `category` already captures "where scams happen". |
| 32 | Should `gender` be used? | It's a group (2 values), not a nametag. | Dropped as a **protected attribute**: using it in fraud decisions is a fairness and legal risk for a bank. |
| 33 | Which text column is kept? | `category`: 14 values, describes the purchase, 11.3× risk spread. | Kept. It gets converted to numbers in the next step. |

| 34 | What goes into the final files? | 9 features: `amt`, `category`, `hour`, `age`, `secs_since_last`, `is_first_txn`, `card_avg_amt_before`, `amt_ratio`, `txn_count_24h`. Target: `is_fraud`. Helpers: `trans_date_trans_time` (time split), `cc_num` (trace to a card), `gender` (fairness checks only). | 13 columns in both files. Saved to `data/interim/features_train.parquet` and `features_test.parquet`. |
| 35 | Is Parquet worth it? | Train 351.2 MB CSV → 41.5 MB Parquet. Test 150.4 MB → 17.7 MB. Both 8.5× smaller. Dates stay dates and category stays category after reloading. | All later steps read Parquet. Note: rows are sorted by card, then time, so Step 4 must sort by time before splitting. |

---

## Step 4: Split by time

| # | Why we checked | What we found | How we use it |
|---|---|---|---|
| 36 | We need a "mock exam" set that's honest about the future | The train file was cut at 2020-03-21. Train: 2019-01-01 → 2020-03-20, 1,070,966 rows, 0.584% fraud. Validation: 2020-03-21 → 2020-06-21, 225,709 rows, 0.556%. Test: 2020-06-21 → 2020-12-31, 555,719 rows, 0.386%. The rows add up, and there's no overlap. | Train = learn, validation = compare models and tune, test = look **once** at the end. Cut by time, never randomly, because the model always predicts the future. |
| 37 | Do the three sets have the same fraud rate? | Train and validation are close (0.584% vs 0.556%), but test is much lower (0.386%). | Validation won't fully warn us about test's lower rate, so a threshold picked on validation may over-flag on test. This is prior shift: monitor the threshold after deployment. |
| 38 | How do later steps get the same sets every time? | Redoing the cut in every notebook risks a different date or a forgotten sort, which quietly gives different sets. | Saved once: `data/processed/train.parquet` (1,070,966), `val.parquet` (225,709), `test.parquet` (555,719). Every later step reads these. The cut date `2020-03-21` is written in a markdown cell, and later moves to `params.yaml`. The Step 3 output lives in `data/interim/features_*.parquet`, so the split never overwrites its own input. |

---

## Step 5 prep: Data for logistic regression

| # | Why we checked | What we found | How we use it |
|---|---|---|---|
| 39 | Logistic regression can't handle empty values | Train-only medians: `secs_since_last` 16,469 s (4.6 h), `card_avg_amt_before` $65.02, `amt_ratio` 0.666. | Filled train, val and test with these **train** medians (learn on train, apply everywhere). `is_first_txn` still marks the originally empty rows. Done on in-notebook copies only; trees use the original data with its empty values. |
| 40 | Logistic regression needs numbers, not text | `category` → 14 one-hot columns (0/1). Val and test were reindexed to train's columns. | 22 feature columns (9 − 1 + 14). 0 empty values and the same columns in the same order across all three sets. Helpers and the target are kept out of X. |
| 41 | Are the long-tailed columns still lopsided? | Skewness on train, before → after log1p: `amt` 41.59 → −0.30, `secs_since_last` 4.33 → −0.75, `card_avg_amt_before` 9.55 → 0.21, `amt_ratio` 57.77 → 1.60 (still mildly skewed). | log1p applied to all 3 sets. It's a fixed formula that learns nothing, so it's safe everywhere. Only logistic regression needs it; trees don't. |
| 42 | Are the columns on the same scale? | StandardScaler **fitted on train only** (22 columns). `amt` after scaling: train mean 0.000 / std 1.000, val mean 0.002 / std 1.001. | Val being slightly off 0 and 1 **proves** the scaler learned only from train (val is measured with train's ruler). For the live API, the medians, log step and fitted scaler must be **saved** and reused exactly. |

---

## Step 5: Baseline model (logistic regression)

| # | Why we checked | What we found | How we use it |
|---|---|---|---|
| 43 | How good is a simple model? | On val: accuracy 0.910, ROC-AUC 0.933, **PR-AUC 0.221** (random = 0.0056, so about 40× better). | This is the **baseline** that the tree models must beat. Accuracy (0.910) is below "always not fraud" (0.994), which proves accuracy is misleading. |
| 44 | Which features drive it? | Top weights: `amt_ratio` +3.34, `amt` −3.06, `card_avg_amt_before` +1.33. | The three overlap (after the log, ratio ≈ log amt − log average), so individual signs can't be read alone. **Together** they say "an amount high for this card = fraud", which matches Step 3c. |
| 45 | What can't a linear model learn? | `hour` weight is only +0.42. The 22:00–03:59 risk window wraps around midnight, so a straight line can't capture it. `txn_count_24h` (−0.11) and `age` (+0.05) are near zero. | Expect tree models to do better: they can learn "hour ≥ 22 or hour ≤ 3". |

---

## Step 6: Tree models

| # | Why we checked | What we found | How we use it |
|---|---|---|---|
| 46 | Do trees beat the baseline? | First XGBoost (defaults + `scale_pos_weight` 170.4, `hist`, `enable_categorical`): **val PR-AUC 0.939**, val ROC-AUC 0.999. That's about 4× the logistic regression (0.221). | Trees learn what a line can't: the midnight-wrapping night window, category effects, and "big amount for this card". It uses the original 9 features with no filling, log, scaling or one-hot. |
| 47 | Is XGBoost memorising? | Train PR-AUC 0.990 vs val 0.939: a gap of 0.051. | A small gap, so mild overfitting. Tuning (for example tree depth) can shrink it. Keep watching the gap. |
| 48 | What does XGBoost rely on? | Top 5 by gain: `amt` 5,047, `category` 1,310, `hour` 708, `card_avg_amt_before` 173, `txn_count_24h` 158. `amt_ratio` is not in the top 5 (it overlaps with `amt` + `card_avg_amt_before`). | The top 3 are the dataset's **generator rules** spotted in Step 2 (amount cap, risky categories, night window). Gain is the average improvement per split, so read it as a ranking. |
| 49 | How much do the card-history features add? | Without the 5 card features (only `amt`, `category`, `hour`, `age`): val PR-AUC **0.907** vs **0.939** with them. | A small drop (0.032): most of the power comes from the simple rules. But the card features cut the remaining error (1 − PR-AUC) from 0.093 to 0.061, about a third. On real bank data, where the rules are less clean, card history would likely matter more. |
| 50 | How many trees does XGBoost really need? | Early stopping (`n_estimators=2000`, `early_stopping_rounds=50`, `eval_metric="aucpr"`, val as `eval_set`): best_iteration 85, so **86 trees**. Val PR-AUC 0.940 (vs 0.939), train 0.987 (vs 0.990), gap 0.047 (vs 0.051). | The default 100 trees was already near the best. The gap barely moved, so memorising comes from **how** each tree learns (depth, learning rate), not how many trees there are. Val now helped choose, so it's slightly optimistic: **test stays sealed**. |
| 51 | Does LightGBM work with the same settings? | Not at its default learning rate (0.1). With `scale_pos_weight` 170, val PR-AUC jumped around (0.27 → 0.26 → 0.40) and early stopping quit after **4 trees** (PR-AUC 0.17). With `learning_rate=0.05` it climbs smoothly. | LightGBM needs a gentler learning rate with a big class weight. That's a **stability** difference worth reporting. |
| 52 | XGBoost or LightGBM? | XGBoost: 86 trees, val 0.940, train 0.987, 15 s to train, **0.12 s** to score all val (0.55 µs per transaction). LightGBM (lr 0.05): 471 trees, val 0.941, train 0.987, 17 s to train, **1.64 s** (7.3 µs per transaction). | The scores are the same (0.001 apart) and so are the gaps. XGBoost scores about **13× faster** (5× fewer trees) and was stable at its defaults, which favours XGBoost for the API. **Be honest:** the comparison wasn't perfectly fair (LightGBM got a lower learning rate), but the scores tied anyway. **Speed in production:** the model takes under 1 µs per transaction, so the slow part of the real API will be looking up the card's history in the database to build the card features. |

---

## Step 7: Choosing the cutoff

| # | Why we checked | What we found | How we use it |
|---|---|---|---|
| 53 | What does each cutoff cost? (champion XGBoost, val, 1,256 frauds) | 0.50: 2,350 flagged, recall 96.0%, 1,144 false alarms, precision 51.3%. 0.70: 1,919 / 94.4% / 733 / 61.8%. **0.90: 1,447 / 90.4% / 312 / 78.4%**. 0.95: 1,297 / 87.7% / 195 / 85.0%. 0.99: 1,059 / 80.3% / 51 / 95.2%. | Higher cutoff = fewer false alarms and higher precision, but lower recall. 0.5 isn't automatically right, because `scale_pos_weight` pushes scores up. |
| 54 | Which cutoff for a team that can review about 1,500 alerts in 3 months? | 0.90 is the lowest cutoff under capacity (1,447 alerts): it catches 90.4% of frauds, and 78% of alerts are real. | Choose **0.90**, with 0.95 as the fallback (1,297 alerts) because there's little headroom. The cutoff is a **business** decision and must be monitored, because volume and fraud rate shift (test 0.386% vs val 0.556%). |
| 55 | Which cutoff is cheapest? (cost = alerts × $10 + `amt` of missed frauds, cutoffs 0.50–0.99) | **0.59 → $35,211** (2,144 alerts, 60 missed frauds, 948 false alarms). 0.90 → $44,406 (1,447 alerts, 121 missed, 312 false alarms). No model → $671,623. | 0.59 saves $9,196 vs 0.90, and the model saves about 95% vs no model. Missed frauds average about $247 each (about 25 alerts' worth), so cost pushes the cutoff down. But 0.59 needs about 43% more analyst capacity and gives 3× the false alarms. Cheapest only if the bank adds analysts; with today's team, 0.90 is realistic. Chosen on val, so slightly optimistic. |
| 56 | **Final exam:** how does the champion do on unseen test (opened once, cutoff 0.90)? | Test (6.3 months, 0.386% fraud): **PR-AUC 0.906** (val 0.940), ROC-AUC 0.998 (val 0.999), recall 87.6% (val 90.4%), precision 72.9% (val 78.4%), 698 false alarms, cost $85,477. **406 alerts/month** (val 479), cost $13,482/month (val $14,693). | **The honest score: PR-AUC 0.906.** ROC-AUC held, so the ranking holds; PR-AUC fell because fraud is rarer in test (prior shift), which cuts precision. The workload (406/month) fits the capacity of about 500/month. Nothing changes after seeing test. |

---

## Step 8: Explaining the model (SHAP)

| # | Why we checked | What we found | How we use it |
|---|---|---|---|
| 57 | What drives the model overall? (SHAP on all 1,256 val frauds + 5,000 random normals) | Mean push size: amt 4.44, category 3.26, hour 1.41, age 0.69, secs_since_last 0.67, card_avg_amt_before 0.53, txn_count_24h 0.45, amt_ratio 0.39, **is_first_txn 0.00**. The receipt check adds up exactly (6.0297 = 6.0297). | The top 3 agree with gain (amt, category, hour). Places 4–5 differ: `age` differs most (#4 by SHAP, not in the gain top 5), because it gives small pushes on many transactions. `is_first_txn` is never used, since trees read the empty `secs_since_last` directly, so it's a candidate to drop. |
| 58 | Can single decisions be explained? (reason codes, cutoff 0.90) | **Top fraud** (score 1.000): amt $702 ↑, 23:00 ↑, 20.8 h since the last purchase ↑ (11.6× the card's usual). **Top false alarm** (score 0.9995): amt $269 ↑, grocery_pos ↑, 03:00 ↑. | The false alarm matches the fraud pattern (a risky category, a few hundred dollars, at night), but it's only 2.9× the card's usual (vs 11.6×). An analyst seeing the reasons can check it quickly. That's why a human reviews alerts. |
| 59 | Is the model fair by **gender**? (val, cutoff 0.90) | F: false alarm rate 0.164%, recall 86.9% (642 frauds). M: false alarm rate 0.108%, recall 94.0% (614 frauds). | **Women do worse on both**: about 1.5× the false alarm rate **and** 7 points less fraud protection. Gender isn't a feature, so the gap must come through **proxies** (for example spending categories). Worth flagging to model risk. |
| 60 | Is the model fair by **age**? | 25–45: the highest false alarm rates (0.196%) and the lowest recall (80.8–85.2%). 55+: low false alarm rates (0.06–0.075%) and high recall (94.9–98.4%). Under 25 has only 71 frauds (noisy). | The model serves **older customers better** on both measures. `age` is a feature (SHAP #4). Next question: does dropping it close the gap, and at what cost to PR-AUC? Decide on val, never on test. |

**Column count check (keep these exact):** 23 raw → 19 after 3a → 20 with `distance_km` → 17 after dropping the 3 distance columns → 19 with `secs_since_last` and `is_first_txn` → 21 with `card_avg_amt_before` and `amt_ratio` → 22 with `txn_count_24h` (+ the `dataset` helper = 23) → **final file: 13 columns** (9 features + 1 target + 3 helpers).

---

## Feature status

| Feature | Status | Why |
|---|---|---|
| `amt` | Keep | Fraud median is 8.4× normal |
| `category` | Keep (convert to numbers next) | Up to 11.3× risk spread |
| `hour` | Added | Night is 2.5–5× riskier |
| `age` | Added | 75+ is about 2× the risk of 35–45 |
| `secs_since_last` | Added (3c) | Fraud median gap 1.4 h vs 4.6 h normal |
| `is_first_txn` | Added (3c) | Marks the 999 "no history" rows (in train this means "start of the data", not "new card") |
| `card_avg_amt_before`, `amt_ratio` | Added (3c) | Fraud median ratio 5.23 vs 0.66 normal |
| `txn_count_24h` | Added (3c) | Weak alone: 5.23 vs 4.88 mean |
| `gender` | **Drop** (3d) | Protected attribute |
| `state`, `merchant` | **Drop** (3d) | Checked: no real signal beyond tiny groups or chance |
| `city`, `zip`, `job`, `lat`, `long`, `city_pop` | **Drop** (3d) | Nametags: about 1–2 people per value |
| `cc_num`, `trans_date_trans_time` | Helper only | Needed for sorting and card history, then dropped in 3e |
| `dob` | Helper only | Already turned into `age`, so it gets dropped |
| `distance_km`, `merch_lat`, `merch_long` | **Dropped** | Random locations, no signal |
| `Unnamed: 0`, `unix_time`, `trans_num`, `first`, `last`, `street` | **Dropped** | Useless, or won't exist when a new transaction arrives |
