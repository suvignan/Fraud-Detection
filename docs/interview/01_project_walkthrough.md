# Part 1: Explaining the Project

*Interview preparation for the credit-card fraud detection project.*

**How to use the three files:**

| File | Use it for |
|---|---|
| **01: Project walkthrough** (this file) | "Tell me about your project." The 30-second, 2-minute and 10-minute versions. |
| **02: Questions and answers** | Every likely question with a model answer, an analogy, and the follow-up ("cross") questions. |
| **03: How to stand out** | What strong interviewers look for, phrases to use, mistakes to avoid, and a practice plan. |
| **04: Pipeline and MLOps** | The modular pipeline, DVC, MLflow, tests and CI. |
| **05: API and production** | FastAPI, Neon, running totals, guardrails, Docker, deployment, scaling follow-ups. |

**Golden rule:** never memorise word for word. Learn the **story** and the **numbers**, then say it in your own words. Interviewers can hear a script.

---

## 1. The 30-second version (the elevator pitch)

> "I built a credit-card fraud detection system on 1.85 million transactions, where only about half a percent are fraud. The main challenges were the extreme rarity of fraud and avoiding **data leakage**, so I split the data **by time** and built per-card history features using **only the past**. A tuned XGBoost model reached a **PR-AUC of 0.906 on an untouched test set**, against **0.22** for a logistic-regression baseline. I then chose the alert cutoff from **analyst capacity and dollar cost**, explained every alert with **SHAP reason codes**, and ran a **fairness check** by gender and age. Then I took it to production: a tested, reproducible pipeline with DVC and MLflow, and a **live FastAPI service** in Docker that scores a transaction in about **90 ms** of server time, using running totals per card in Postgres and **guardrails** for inputs the model can't judge."

**Why it works:** it gives the problem, the hard part, the result with a number, the business side and the responsible-AI side in about 30 seconds. Every sentence invites a follow-up question you're ready for.

---

## 2. The 2-minute version

Follow this order. Each line is one or two sentences.

1. **Problem:** banks must catch fraud in real time, but fraud is about **1 in 200** transactions. A model that always says "not fraud" is 99.4% accurate and useless.
2. **Data:** 1.85 million simulated card transactions (2019–2020), about 1,000 cardholders, 23 raw columns.
3. **Exploration:** fraud is **bigger amounts** (median $397 vs $47), in **online-shopping and grocery categories**, at **night** (10pm–4am).
4. **Features:** I built **card-history features**: time since the card's last purchase, the card's past average spend, how unusual this amount is *for this card*, and purchases in the last 24 hours. All are computed **from the past only**.
5. **Split:** by **time**: train (until March 2020), validation (March–June), test (June–December). Never random, because the model always predicts the future.
6. **Models:** baseline logistic regression **PR-AUC 0.22**. XGBoost **0.94** on validation. LightGBM tied, but scored predictions about 9–13× slower, so XGBoost won.
7. **Decision:** a **0.90 cutoff** fits the analyst team (about 500 alerts a month), catching about **90% of fraud** with **78% of alerts real**.
8. **Final test (opened once):** **PR-AUC 0.906**. The drop from 0.94 is mostly because fraud got rarer in the test period (**prior shift**), and the ranking quality held (ROC-AUC 0.998).
9. **Trust:** **SHAP reason codes** for every alert, and a **fairness check** that found gaps by gender and age, logged as next experiments.
10. **Pipeline:** six tested components; every setting in `params.yaml`; DVC reruns only changed stages; MLflow records every run; it reproduces the notebooks exactly (0.9401 / 0.9058).
11. **Live API:** FastAPI in Docker on Render, next to Neon Postgres. Card state as **running totals** (storage 411 MB → 9 MB), card numbers stored only as an **HMAC**, a **parity test** proving the API's features equal training's, and about **90 ms** server-side per prediction.
12. **Stress test → guardrails:** extreme inputs exposed two blind spots (new cards, amounts above the training range), so those go to a third decision, **review**, adding only about 1 review per 3 months on validation.

---

## 3. The full walkthrough (5–10 minutes)

Use this when the interviewer says "walk me through it" or when you share your screen.

### The big picture

```
 RAW DATA (2 CSV files, 1.85M rows)
      |
      v
 1. INVENTORY + EXPLORATION ---- what is in the data? what does fraud look like?
      |
      v
 2. FEATURES ------------------- 9 features, card history built from the PAST only
      |
      v
 3. TIME SPLIT ----------------- train | validation | test   (no shuffling)
      |
      v
 4. MODELS --------------------- baseline LR (0.22)  ->  XGBoost (0.94)  vs LightGBM
      |
      v
 5. CUTOFF + COST -------------- turn scores into alerts the team can handle
      |
      v
 6. FINAL TEST (once) ---------- honest score: PR-AUC 0.906
      |
      v
 7. EXPLAIN + FAIRNESS --------- SHAP reason codes, checks by gender and age
      |
      v
 8. NEXT: PRODUCTION PIPELINE -- config, modules, tests, API, monitoring
```

### Step 1: Understanding the data

**What I did:** checked size, types, empty values, duplicates, dates and the fraud rate before building anything.

**Key facts:**
- Train file: 1,296,675 rows. Test file: 555,719 rows. **No empty values, no duplicates.**
- Fraud: **0.58%** in the train file, **0.39%** in the test file. Fraud got rarer, and that matters later.
- The files **don't overlap in time**: test starts 48 seconds after train ends. That's a clean time split, like real life.
- **98% of test cards also appear in train**, so card-history features will work. 16 cards are brand new.
- `unix_time` turned out to be the same clock time with the year shifted, so it's a duplicate and was dropped.

**What fraud looks like (exploration on train only):**

| Signal | Fraud | Normal |
|---|---|---|
| Median amount | $397 | $47 |
| Riskiest categories | shopping_net 1.76%, misc_net 1.45%, grocery_pos 1.41% | health_fitness 0.15% |
| Time of day | 22:00–23:59 about 2.9% (5x average) | daytime about 0.1% |
| Time since card's last purchase (median) | 1.4 hours | 4.6 hours |

**The story to tell:** *"Fraud follows a recipe: test the card with a small purchase, then spend big, online or at grocery stores, late at night. My features were designed around that recipe."*

### Step 2: Features (9 in total)

| Feature | What it means | Why it helps |
|---|---|---|
| `amt` | Purchase amount | Fraud is bigger |
| `category` | Type of shop (14 kinds) | Some categories are 11x riskier |
| `hour` | Hour of day | Night is risky |
| `age` | Cardholder age at purchase | Medium signal |
| `secs_since_last` | Seconds since this card's previous purchase | Stolen cards get used fast |
| `is_first_txn` | 1 if no earlier purchase | Marks "no history" |
| `card_avg_amt_before` | Card's average spend, **past purchases only** | "What is normal for this person?" |
| `amt_ratio` | Amount ÷ that average | "Is this big **for this card**?" (median 5.2 for fraud vs 0.66 for normal) |
| `txn_count_24h` | Card's purchases in the last 24 hours | Bursts of use |

**The most important idea: point-in-time correctness.** Every feature uses only information that **existed at the moment of the purchase**.

```
Card A's timeline:   $20 ---- $35 ---- $25 ---- [$900 ?]
                                                    ^
               "card's average before" = (20+35+25)/3 = $26.7
               amt_ratio = 900 / 26.7 = 33.7  -> very unusual -> suspicious
               (the $900 itself and anything after it are NOT used)
```

**Columns I dropped and why** (interviewers love this):
- **Personal "nametag" columns** (name, street, city, zip, job, exact home location): about 1–2 people per value, so the model would **memorise people**, not learn fraud.
- **Merchant location:** each merchant appeared in up to 4,400 random places. It's noise, and the distance from home had no signal (76.3 km vs 76.1 km).
- **merchant name:** added nothing beyond category (the differences matched pure chance).
- **state:** the "risky" states were tiny (one state = 1 card that was all fraud). Keeping it means memorising one person.
- **gender:** a **protected attribute**, kept only as a helper for the fairness check, never given to the model.

### Step 3: Splitting by time

```
 Jan 2019 ------------------- 21 Mar 2020 -------- 21 Jun 2020 ------------ Dec 2020
 |          TRAIN             |    VALIDATION      |          TEST          |
 |  1,070,966 rows  0.58%     |  225,709   0.56%   |  555,719 rows  0.39%   |
 |  model learns here         |  I make choices    |  opened ONCE at end    |
```

- **Why time, not random?** In real life the model always predicts the **future**. A random split lets the model peek at "future" patterns, so scores look better than reality.
- **Why three sets?** Train = study material. Validation = practice tests (to choose models and settings). Test = the final exam, taken once.

### Step 4: Models

| Model | Val PR-AUC | Notes |
|---|---|---|
| Random guessing | 0.006 | equals the fraud rate |
| Logistic regression (baseline) | **0.221** | needed filling, log, scaling, one-hot. Can't learn the night window (it wraps around midnight) |
| XGBoost (defaults + class weight) | **0.939** | no preprocessing needed |
| XGBoost + early stopping (**champion**) | **0.940** | 86 trees, train/val gap 0.047 (mild overfitting only) |
| LightGBM | 0.941 | tied, but about 9–13x slower to score, and unstable at its default learning rate |

**Why such a jump from 0.22 to 0.94?** Trees can learn **cuts** ("hour ≥ 22 or hour < 4") and **combinations** ("big amount AND online shopping AND night"). A straight-line model can only say "more of X = more risk".

**Handling rare fraud:** `scale_pos_weight = 170.4` (normal ÷ fraud in train), so each fraud counts 170 times in training and the model can't ignore fraud.

**Proof that the card features matter:** retraining **without** the 5 card-history features dropped PR-AUC from 0.939 to 0.907. That removes about **one third of the remaining error**.

### Step 5: Choosing the cutoff (the business decision)

The model gives a **score** from 0 to 1. The bank must decide **at what score to raise an alert**.

| Cutoff | Alerts (3 months) | Fraud caught | Precision |
|---|---|---|---|
| 0.50 | 2,350 | 96.0% | 51% |
| **0.90 (chosen)** | **1,447** | **90.4%** | **78%** |
| 0.99 | 1,059 | 80.3% | 95% |

- **Capacity:** the team can review about 1,500 alerts per 3 months. **0.90** is the lowest cutoff that fits.
- **Cost:** assuming **$10 per alert** and a missed fraud costs its amount, the model saves about **95%** of fraud losses ($671,623 with no model → $44,406). The cheapest cutoff was **0.59** ($35,211), but it needs 43% more analysts and triples false alarms. That's a choice for the business.

### Step 6: The final exam (test, opened once)

| | Validation | Test |
|---|---|---|
| Fraud rate | 0.556% | 0.386% |
| **PR-AUC** | 0.940 | **0.906** |
| ROC-AUC | 0.999 | 0.998 |
| Recall | 90.4% | 87.6% |
| Precision | 78.4% | 72.9% |
| Alerts per month | 479 | 406 (fits capacity) |

**The explanation:** ROC-AUC held, so the model's **ranking skill** held. PR-AUC fell mainly because **fraud became rarer** (prior shift): the same number of false alarms became a bigger share of alerts. Nothing was changed after seeing test.

### Step 7: Explaining and checking fairness

**SHAP** gives every prediction an "itemised receipt": a starting value plus each feature's push equals the score exactly (I verified this: 6.0297 = 6.0297).

- Overall drivers: **amount, category, hour**, then age and card history.
- Example reason codes for the top fraud: *"Amount $702.24, at 23:00, 20.8 h since the card's last purchase."*
- `is_first_txn` had zero importance (XGBoost already learns "no history" from the empty `secs_since_last`), so it's a candidate to drop.

**Fairness (validation, cutoff 0.90):** compared the **false alarm rate** (harm to innocent people) and **recall** (protection) per group.
- Women: 0.164% false alarms vs 0.108% for men, and 86.9% recall vs 94.0%, **even though gender is not a feature** (proxies).
- Ages 25–45: the highest false alarm rate (0.196%) and the lowest recall (81–85%).
- These are logged as next experiments (retrain without `age`, find the proxies), not hidden.

### Step 8: Production (built and deployed)

```
 TRAINING:  params.yaml --> ingest --> validate --> features --> split --> train --> evaluate (quality gate)
            dvc repro reruns only changed stages · MLflow logs every run · pytest + CI on every push

 SERVING:   transaction --> FastAPI (Docker, Render, Ohio) --> card state from Neon (running totals)
                        --> same feature code as training --> XGBoost --> flag / review / approve
                        --> SHAP reason codes --> logged to prediction_log
```
- **Proof of correctness:** the pipeline's features are identical to the notebook's; the API's features equal the
  pipeline's on 2,000 test transactions.
- **Engineering wins:** storage 411 MB → 8.9 MB; per-request latency from ~1.3 s (first version, from India) to
  ~90 ms server-side once deployed next to the database.
- **Details:** see files 04 and 05.

---

## 4. Numbers to know by heart

| What | Number |
|---|---|
| Total transactions | ~1.85 million (1,296,675 + 555,719) |
| Fraud rate train / val / test | 0.584% / 0.556% / 0.386% |
| Baseline LR val PR-AUC | 0.221 |
| XGBoost champion val PR-AUC | 0.940 (86 trees) |
| **Test PR-AUC (headline)** | **0.906** |
| Test ROC-AUC | 0.998 |
| Class weight | 170.4 |
| Cutoff | 0.90 |
| Recall / precision at 0.90 (val) | 90.4% / 78.4% |
| Recall / precision at 0.90 (test) | 87.6% / 72.9% |
| Money saved vs no model (val) | about 95% |
| Without card features | 0.907 (vs 0.939) |
| Tests | 23 (feature invariants, parity, guardrails, ...) |
| Neon storage: full history → running totals | 411 MB → 8.9 MB |
| Server-side time per prediction (live) | ~90 ms |
| Guardrail limits | $28,948.90 (train max) · $546.72 (train 99th percentile) |
| Reviews added on validation | 1 in 3 months (vs ~500/month capacity) |
| Docker image | 757 MB, non-root |

---

## 5. Honest limitations (say these before you're asked)

1. **The data is simulated.** Some patterns are cleaner than real life (for example, fraud never exceeds $1,376, and the hour pattern switches sharply at 4am). On real bank data I'd expect lower scores and more value from card-history features.
2. **The cost model is simple:** $10 per alert is an assumption, and it ignores customer annoyance from blocked cards.
3. **Validation was used for several choices** (early stopping, cutoff), so its 0.940 is slightly optimistic. That's why test exists.
4. **Fairness gaps were found but not fixed yet.** They're documented as the next experiments.
5. **The LightGBM comparison wasn't perfectly equal on tuning.** It got a lower learning rate, but the scores tied anyway.
6. **Two model blind spots, found by stress-testing the live API:** scores freeze above ~$1,200 (trees don't extrapolate past the largest training fraud, $1,372), and brand-new cards are under-flagged (no history; the training data's "first purchases" were just where the dataset began). **Mitigated** with guardrails that send those cases to human review.
7. **Engineering limits:** no idempotency yet (a retried request counts twice in a card's totals); transactions must arrive in time order; the model files are committed to git as a deploy shortcut (the clean design pulls them with DVC in CI).

> Saying limitations **before** you're asked shows maturity. Interviewers trust a candidate who knows where their work is weak.
