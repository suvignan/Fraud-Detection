# Fraud Detection: Findings Log

One place for everything we found so far. For each finding:
- **Why we checked**: the question we were asking
- **What we found**: the numbers
- **How we use it**: the decision it led to

Data: `fraudTrain.csv` (Jan 2019 – Jun 2020) and `fraudTest.csv` (Jun – Dec 2020).
Learning details and code are in [LEARNING_NOTES.md](LEARNING_NOTES.md).

**Status:** Step 1 ✅ · Step 2 ✅ · Step 3a ✅ · Step 3b ✅ · Step 3c ✅ · Step 3d ✅ · Step 3e ✅ · **Step 3 done** · Step 3d ⬜ · Step 3e ⬜

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

| 34 | What goes into the final files? | 9 features: `amt`, `category`, `hour`, `age`, `secs_since_last`, `is_first_txn`, `card_avg_amt_before`, `amt_ratio`, `txn_count_24h`. Target: `is_fraud`. Helpers: `trans_date_trans_time` (time split), `cc_num` (trace to a card), `gender` (fairness checks only). | 13 columns in both files. Saved to `data/processed/train.parquet` and `test.parquet`. |
| 35 | Is Parquet worth it? | Train 351.2 MB CSV → 41.5 MB Parquet. Test 150.4 MB → 17.7 MB. Both 8.5× smaller. Dates stay dates and category stays category after reloading. | All later steps read Parquet. Note: rows are sorted by card, then time, so Step 4 must sort by time before splitting. |

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
