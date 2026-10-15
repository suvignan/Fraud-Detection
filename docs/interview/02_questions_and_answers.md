# Part 2: Questions and Answers

*Likely interview questions on the fraud detection project, with model answers.*

**How each question is laid out:**
- **Answer:** what to say (short first, then detail if they want more).
- **Analogy:** a simple picture to make it stick (for you and for the interviewer).
- **Cross-questions:** the follow-ups they're likely to ask, with short answers.

**Tip:** answer in this order: **the decision → why → the evidence (a number) → the trade-off.** Stop and let them ask more.

---

## Section A: The project in general

### A1. Why did you choose fraud detection?

**Answer:** It has every hard part of real machine learning in one problem:
- the target is **extremely rare** (about 0.5%)
- data arrives **over time**, so leakage is easy
- every mistake has a **real cost** on both sides (missed fraud vs blocked customers)
- decisions must be **explainable and fair**

That forced me to think beyond "train a model and report accuracy".

**Cross-questions:**
- *"Is this data real?"* → No, it's a widely used **simulated** dataset. I treat that as a limitation: some patterns are cleaner than real life, so real-world scores would likely be lower.
- *"What would change with real bank data?"* → Messier labels (fraud is often confirmed weeks later), more features (device, location, merchant history), concept drift (fraudsters adapt), and strict latency and regulatory requirements.

### A2. What was the hardest part?

**Answer:** **Avoiding data leakage in the card-history features.** My first version of "the card's average spend before this purchase" used a shift that **crossed from one card into the next**, so some cards' first purchases were using another card's history. I caught it by **checking a card in the middle of the data**, not just the first one, and fixed it by calculating per card: (running total − current amount) ÷ number of earlier purchases.

**Analogy:** like a teacher's spreadsheet where one student's grades slipped into the next student's row. Everything looks fine at the top of the sheet; the bug only shows in the middle.

**Cross-questions:**
- *"How would you prevent that in the future?"* → Write **invariant checks** (rules that must always hold): for example, "every card's first purchase must have an empty history value", and run them as automated tests in the pipeline.

### A3. What is the result, in one line?

**Answer:** "On an untouched test set covering six months, the model reached **PR-AUC 0.906**. At the chosen cutoff it catches **88% of fraud**, and **73% of alerts are real**, with about **400 alerts a month**, which fits the analyst team."

**Cross-questions:**
- *"Is 0.906 good?"* → A random model scores 0.004 on test (the fraud rate), and my baseline scored 0.22 on validation. So it's very strong, but on simulated data. I'd quote it alongside the baseline, not alone.

---

## Section B: The data

### B1. Describe the dataset.

**Answer:** About **1.85 million** card transactions from 2019–2020, around **1,000 cardholders**, 23 columns: amount, merchant, category, time, cardholder details (age, gender, location, job) and the label `is_fraud`. It comes as two files that are already split by time.

### B2. What checks did you do before modelling?

**Answer:** An **inventory** before any modelling:
- size and memory
- column types (dates stored as text, IDs stored as numbers)
- empty values and duplicates (none)
- the fraud rate per file
- whether the files overlap in time (no)
- whether cards appear in both files (98% do)
- whether rows were sorted by time (train wasn't)

**Analogy:** a chef checks the ingredients before cooking. You don't find out the milk is off after the cake is baked.

**Cross-questions:**
- *"Why does it matter that train wasn't sorted?"* → Any "history" feature (time since last purchase, running average) assumes time order. Unsorted data gives wrong history, silently.
- *"Why care that IDs are stored as numbers?"* → A model would treat card number 4000 as "bigger" than 3000, which is meaningless. IDs are labels, not quantities.

### B3. The fraud rate differs between train (0.58%) and test (0.39%). Why does that matter?

**Answer:** It's called **prior shift**: the base rate of the target changes over time. It matters because:
1. **Precision and PR-AUC depend on the base rate.** With rarer fraud, the same model produces a bigger share of false alarms.
2. **A cutoff chosen on one period may over-flag in another.**

So in production you must **monitor the fraud rate and precision** and re-check the cutoff.

**Analogy:** a metal detector at the beach is just as good every day. On a day with fewer coins buried, more of its beeps are bottle caps. The detector didn't get worse; the beach changed.

**Cross-questions:**
- *"Did you correct for it?"* → No. I chose not to adjust anything using test. I measured its effect: ROC-AUC (which ignores the base rate) held at 0.998 while PR-AUC fell. In production I'd re-calibrate the cutoff on recent data.

### B4. Why did you drop so many columns?

**Answer:** Each was dropped for a **tested reason**, not a feeling:

| Column(s) | Reason |
|---|---|
| name, street, city, zip, job, home lat/long, city population | **Nametags:** about 1–2 people per value, so the model would memorise people |
| merchant lat/long, distance | Random per transaction; distance had **no signal** (76.3 vs 76.1 km) |
| merchant | Added nothing beyond category; differences matched pure chance |
| state | "Risky" states were tiny (one state = one card, 100% fraud) |
| unix_time | A copy of the timestamp with the year relabelled |
| transaction ID, row number | Random IDs; won't help on new data |
| gender | **Protected attribute**: kept only for fairness checks |

**Analogy:** packing for a trip. Every item you carry costs effort. In a bank, every feature must be built, monitored and justified, so a useless one is pure cost.

**Cross-questions:**
- *"Couldn't trees just ignore useless columns?"* → Mostly, but noise columns let trees make **lucky splits** that fit train and fail on new data. They also add cost and risk in production.
- *"How did you test that merchant added nothing?"* → Within each category I compared how much the merchants' fraud rates spread out against how much they'd spread by pure chance given their small sizes. The spreads matched, so the differences were noise.

### B5. Anything strange in the data?

**Answer:** Yes, signs that it's simulated:
- 13 cardholders are under 18 (the youngest is 13)
- fraud never goes above $1,376
- the hour pattern switches sharply at exactly 4am

I **noted** these instead of removing them, and I mention them as a reason real-world performance would be lower.

---

## Section C: Features and leakage

### C1. What is data leakage?

**Answer:** When the model gets information during training that it **won't have at prediction time**, usually information from the future or from the answer itself. Scores look great in the notebook and then fail in production.

**Analogy:** a student who sees the answer key before the exam. They get 100% in practice and fail the real exam.

**Cross-questions:**
- *"Give an example from your project."* → (1) A card's average spend using **all** its purchases would include future purchases, so I used only **earlier** purchases. (2) Filling empty values with a median from validation or test would leak, so I used the **train** median. (3) A random split would leak future patterns, so I split by time.

### C2. What is point-in-time correctness?

**Answer:** Every feature for a transaction is computed using **only information that existed at that exact moment**. For each card I sorted by time and used running totals that exclude the current and future purchases.

```
time --->   p1     p2     p3     p4 (now)    p5 (future)
history for p4 uses:  p1 p2 p3      never p4's own label, never p5
```

### C3. You built card features on train and test combined. Isn't that leakage?

**Answer:** No, because each row only looks **backwards in time**. A test transaction in July 2020 can use the same card's purchases from 2019, and in real life the bank **would** have that history. What would be leakage is a train row looking **forward** into test, and the per-card, time-sorted, past-only calculation prevents that.

**Analogy:** your doctor uses your full medical history from previous years when diagnosing you today. That's allowed. Using next year's test results is not.

### C4. Why `amt_ratio`? Isn't `amt` enough?

**Answer:** $300 is normal for one person and very unusual for another. `amt_ratio` = amount ÷ the card's past average says "**is this big for this person?**". The median ratio is **5.2 for fraud vs 0.66 for normal**.

**Cross-questions:**
- *"Why is the normal median below 1?"* → A few large purchases pull a card's average above its typical purchase, so a typical purchase sits below the average.
- *"What about the first purchase, with no history?"* → The ratio is left **empty**. I didn't invent a value; trees handle empty values directly.

### C5. How did you handle missing values?

**Answer:** The only empty values are a card's first purchase (no history). The choice depended on the model:
- **XGBoost:** left empty. At every split it **learns which side empty values should go** (the default direction).
- **Logistic regression:** filled with the **train median**, plus an `is_first_txn` flag so the model still knows the value was originally empty.

**Cross-questions:**
- *"Why not fill with 0?"* → 0 seconds since the last purchase would mean "used again instantly", the most suspicious value possible. That's a lie that looks like fraud.

### C6. Why did `is_first_txn` end up useless?

**Answer:** Its SHAP importance was **0.000**. XGBoost already learns "no history" from the empty `secs_since_last`, so the flag repeats information. It's redundant for trees, though still useful for linear models. It's on the list to drop in the pipeline, after confirming on validation that the score doesn't change.

### C7. Why Parquet instead of CSV?

**Answer:** Files are about **8.5x smaller** (351 MB → 42 MB), faster to load, and they **keep column types**: dates stay dates and categories stay categories. CSV turns everything back into text.

---

## Section D: Splitting

### D1. Why split by time and not randomly?

**Answer:** The model will always predict the **future**. A random split mixes future transactions into training, so the model learns patterns from "tomorrow" and scores look better than they will be in production.

**Analogy:** predicting next week's weather after peeking at a random half of next week's actual weather.

**Cross-questions:**
- *"Doesn't that waste data?"* → Validation and test are still used for evaluation. For the final production model, you can retrain on train + validation with the settings you already chose.
- *"What about cross-validation?"* → Standard k-fold shuffles time. I'd use **time-series cross-validation** (an expanding window: train on months 1–6, validate on 7; train on 1–7, validate on 8, and so on).

### D2. Why three sets? What is each for?

**Answer:**
- **Train** = the model learns from it.
- **Validation** = *I* learn from it: comparing models, early stopping, choosing the cutoff.
- **Test** = touched once, at the end, for the honest score.

**Analogy:** textbook (train), practice tests (validation), final exam (test).

**Cross-questions:**
- *"Why not just train and test?"* → Every decision made by looking at a set makes that set's score optimistic. If you choose models using test, there's no honest number left.

### D3. Why was the validation cut on 21 March 2020?

**Answer:** It leaves about **3 months** of validation at the end of the train file, enough frauds (1,256) for stable scores, while keeping most data (about 83%) for training. It's recorded as a setting so it's never hidden in code.

---

## Section E: Metrics

### E1. Why not accuracy?

**Answer:** With 99.4% normal transactions, a model that **always says "not fraud"** is 99.4% accurate and catches nothing. My baseline scored 0.910 accuracy, *lower* than that useless model, while catching most frauds. Accuracy rewards doing nothing on rare-event data.

**Analogy:** a smoke alarm that never rings is "right" 99.9% of the time, until the fire.

### E2. Explain precision and recall.

**Answer:**
- **Precision:** of the transactions I **flagged**, how many were really fraud? (*"Am I wasting the analysts' time?"*)
- **Recall:** of **all** real frauds, how many did I catch? (*"What slipped through?"*)

```
                  REAL fraud         REAL normal
 FLAGGED          caught (TP)        false alarm (FP)    precision = TP / (TP + FP)
 NOT flagged      missed (FN)        correct (TN)
                  recall = TP / (TP + FN)
```

**Analogy:** fishing with a net. Recall = how many of the fish in the lake you caught. Precision = how much of your net is fish rather than old boots.

### E3. Why PR-AUC and not ROC-AUC?

**Answer:** ROC-AUC measures false alarms as a share of **all normal transactions**, and there are hundreds of thousands of them, so even thousands of false alarms look tiny. PR-AUC focuses on **precision**, which is exactly what the analyst feels. My baseline had ROC-AUC 0.93 (looks great) but PR-AUC 0.22 (the truth).

**Cross-questions:**
- *"What does a random model score on PR-AUC?"* → The fraud rate itself (about 0.0056 on validation), because random flags are fraud only as often as fraud exists. On ROC-AUC, random is 0.5.
- *"So when is ROC-AUC useful?"* → It measures **ranking skill** and ignores the base rate. That's why I used it to show the model's skill held on test while PR-AUC dropped from prior shift.

### E4. What is a threshold or cutoff?

**Answer:** The model outputs a score from 0 to 1. The cutoff is the score above which we raise an alert. **The model doesn't choose it; the business does**, based on analyst capacity and the cost of a miss vs a false alarm.

---

## Section F: Models

### F1. Why start with logistic regression?

**Answer:** As a **baseline**: simple, fast and explainable. It proves that the complex model is worth it. Without a baseline, "0.94" has no context.

**Cross-questions:**
- *"What preparation did it need?"* → Fill empty values (train medians), one-hot encode category, log the long-tailed amounts, and scale with StandardScaler **fitted on train only**.
- *"Why did it do so badly?"* → It draws a weighted sum, a straight-line boundary. Fraud risk by hour is high at **both ends of the clock** (10pm–4am), and a straight line can't represent "high at both ends, low in the middle". It also can't learn combinations like "big amount AND online AND night".

### F2. What does `fit` vs `transform` mean, and why fit only on train?

**Answer:** `fit` = **learn** numbers from data (for example, the mean and spread of each column). `transform` = **apply** those numbers. Fit on train only, then transform train, validation and test with the **same** fitted scaler. Fitting on validation or test would leak their statistics into training.

**Analogy:** calibrate the weighing scale once, then weigh everything on it. You don't recalibrate for each item.

**Cross-questions:**
- *"How can you prove you didn't fit on validation?"* → After scaling, train's mean is exactly 0 and its std exactly 1, while validation's are slightly off (0.002, 1.001). If validation were exactly 0 and 1, someone fitted on it.

### F3. Why XGBoost?

**Answer:** Gradient-boosted trees are the strongest general choice for **tabular data**:
- they learn cuts and combinations automatically
- they handle empty values and categories natively
- they don't need scaling
- they're fast to score

It went from 0.22 to **0.94** PR-AUC with **no preprocessing at all**.

**Cross-questions:**
- *"Why no scaling for trees?"* → Trees ask "is the value above X?". Only the **order** of values matters, and scaling or log doesn't change the order.
- *"How does it handle the category column?"* → With `enable_categorical=True` and the pandas category type, a split can group categories directly ("category in {shopping_net, misc_net}?"), so no one-hot is needed.

### F4. Explain boosting vs bagging.

**Answer:**
- **Bagging** (Random Forest) builds many trees **independently** on random samples and **averages** them. That mainly reduces **variance**.
- **Boosting** (XGBoost) builds trees **one after another**, each one correcting the errors the previous ones still make, and **adds** their scores. That mainly reduces **bias**.

```
 BAGGING:  tree  tree  tree  tree   ->  average vote
 BOOSTING: tree -> tree -> tree -> tree  (each fixes what is still wrong)
```

**Analogy:** bagging = asking 100 people separately and averaging. Boosting = golf: each shot corrects the remaining distance to the hole.

**Cross-questions:**
- *"So how do you control variance in boosting?"* → A lower learning rate, shallower trees, row and column subsampling (borrowed from bagging), regularisation, and **early stopping**.

### F5. How did you handle class imbalance?

**Answer:** With `scale_pos_weight = 170.4` (normal count ÷ fraud count in train), so each fraud counts about 170 times in training. In logistic regression I used `class_weight="balanced"`: the same idea under a different name.

**Analogy:** in a class with 170 quiet students and one who rarely speaks, the teacher weights that one voice 170x so it's heard.

**Cross-questions:**
- *"Why not oversample with SMOTE?"* → Class weighting gets the same effect without inventing fake transactions, is simpler, and avoids leakage risks from synthetic rows. I'd only try SMOTE if weighting failed.
- *"Side effect?"* → Weighting pushes scores **up**, so 0.5 is no longer a sensible cutoff. That's one reason I chose the cutoff separately.

### F6. What is early stopping and what did it give you?

**Answer:** I allowed up to 2,000 trees but stopped when validation PR-AUC hadn't improved for **50 trees** in a row. It picked **86 trees**. The score stayed the same (0.940), but:
- the tree count was chosen by data instead of guessed
- prediction got faster
- the train/validation gap shrank from 0.051 to 0.047

**Analogy:** practising until your practice-test score stops improving, then going back to your best version.

**Cross-questions:**
- *"Any downside?"* → Validation chose the stopping point, so validation's score is slightly optimistic. That's why the untouched test score (0.906) is the headline. A cleaner setup cuts a separate early-stopping slice from the end of train.

### F7. How do you detect overfitting?

**Answer:** Compare train and validation scores. Train PR-AUC 0.987 vs validation 0.940 is a **small gap (0.047)**, so only mild overfitting.

```
 both low          -> underfitting (high bias)    -> more capacity
 train >> val      -> overfitting (high variance) -> simplify / regularise
 both high, close  -> good  <- my model
```

### F8. Why XGBoost over LightGBM if LightGBM scored slightly higher?

**Answer:** 0.941 vs 0.940 is a **tie**: never pick a winner on a 0.001 difference. The decision came from other factors:
- **Speed:** XGBoost used 86 trees vs 471 and scored predictions roughly **9–13x faster** across my runs.
- **Stability:** XGBoost worked at default settings. LightGBM overshot at its default learning rate with the 170x class weight, and early stopping quit after 4 trees, until I lowered the learning rate.

**Cross-questions:**
- *"Was the comparison fair?"* → Not perfectly: they had different learning rates. I compared each at a setting that worked. A full tuning search on both would be the rigorous next step.
- *"Does 1 vs 8 microseconds matter in an API?"* → Barely. Network and the card-history lookup take milliseconds. But fewer trees means cheaper, simpler serving, and the stability advantage matters more.

### F9. What does the learning rate do?

**Answer:** It's the size of each tree's correction. A large rate learns fast but can overshoot; a small rate is smoother but needs more trees. XGBoost at 0.3 needed 86 trees; LightGBM at 0.05 needed 471.

---

## Section G: Feature importance

### G1. Which features matter most?

**Answer:** By SHAP: **amount, category and hour** dominate, then age and the card-history features. Gain importance agreed on the same top 3.

### G2. How did you test whether the card-history features were worth it?

**Answer:** With a **removal (what-if) test**: I retrained with identical settings **without** the 5 card features. PR-AUC fell from **0.939 to 0.907**. That looks small, but the remaining error (1 − PR-AUC) went from 0.093 to 0.061, so the card features remove about **a third of the mistakes**.

**Analogy:** you only know a football player's real value by playing a match without them.

**Cross-questions:**
- *"Why change only the features?"* → A **controlled experiment**: change one thing so the difference has one cause.
- *"Why not trust the importance chart?"* → Gain shows how much the model **used** a feature, not how much it would **lose** without it. Overlapping features (like `amt` and `amt_ratio`) split the credit.

### G3. Logistic regression gave `amt` a negative weight. Does a bigger amount mean safer?

**Answer:** No. `amt` and `amt_ratio` **overlap**, because the ratio is built from the amount. The model gave a big **+3.3** to one and **−3.1** to the other, and they partly cancel. Together they say "an amount high **for this card** is risky". This is **multicollinearity**, and it's why I don't read linear weights one at a time.

**Analogy:** two friends sharing one bill: one pays +$30, the other gets $30 back. Looking at one of them alone tells you nothing.

---

## Section H: Cutoff and cost

### H1. How did you choose the cutoff?

**Answer:** From **analyst capacity**. The team can review about 1,500 alerts per 3 months. **0.90** is the lowest cutoff under that (1,447 alerts), catching **90.4%** of fraud with **78%** of alerts real. 0.95 is the fallback if volume grows.

**Cross-questions:**
- *"Why not 0.5?"* → Class weighting pushes scores up, so 0.5 isn't neutral. At 0.5, half the alerts were false.

### H2. What was the cost analysis?

**Answer:** Total cost = alerts × **$10** (analyst time, an assumption) + the amount of every **missed** fraud.

| | Cost on validation (3 months) |
|---|---|
| No model | $671,623 |
| Cutoff 0.90 | $44,406 |
| Cheapest cutoff (0.59) | $35,211 |

The model saves about **95%**. 0.59 is cheaper, but it needs about 43% more analyst time and triples false alarms. So I'd present both options and **let the business decide** based on staffing.

```
 total cost
   high |*                                   *
        |  *                              *       right: missed fraud grows
        |     *  *                  *  *
        |           *  *  *  *  *              left: alert cost grows
        +---------------------------------------- cutoff
       0.50         0.59              0.90       0.99
```

**Cross-questions:**
- *"What does your cost model ignore?"* → Customer annoyance and churn from blocked cards, chargeback fees, and the cost of investigating. Including customer annoyance would push the cutoff **higher**.
- *"Why does cost favour a low cutoff?"* → A missed fraud averages about $247, the same as about 25 alerts, so missing is expensive.

---

## Section I: The final test

### I1. How did you use the test set?

**Answer:** **Once**, at the very end, with **prediction only**: no training, no tuning. Nothing was changed after seeing the result. Otherwise test becomes another validation set and there's no honest number left.

### I2. Why did the test score drop from 0.940 to 0.906?

**Answer:** Three reasons, the biggest first:
1. **Prior shift:** fraud fell from 0.556% to 0.386%. Per month, false alarms stayed about the same (104 vs 111) but there were fewer frauds (419 vs 340), so precision fell, and PR-AUC with it.
2. **Validation optimism:** early stopping and the cutoff were chosen on validation.
3. **Time drift:** test is later (July–December, including the holidays).

The key evidence: **ROC-AUC barely moved (0.999 → 0.998)**, so the ranking skill held.

### I3. Is the model "worse" on test?

**Answer:** "It depends." Its **ranking skill held**. The **world changed** (fraud got rarer), which lowered precision. Operationally it still works: **406 alerts a month** fits capacity and it catches **87.6%** of fraud. It's not a worse model, it's a harder period, and that's exactly why production needs monitoring.

### I4. Why did you compare per month?

**Answer:** Validation covers 3 months and test 6.3 months, so raw totals aren't comparable. Per month, test was actually **lighter** (406 vs 479 alerts).

---

## Section J: Explainability

### J1. What is SHAP?

**Answer:** A method that splits each prediction into **contributions from each feature**. Base value + all pushes = the model's score, exactly. It comes from game theory (Shapley values): the fair way to share credit among players.

**Analogy:** an itemised restaurant bill. The total is split by what each person ordered, and the items add up to the total.

**Cross-questions:**
- *"How do you know it's correct?"* → I checked the receipt: base value + pushes = **6.0297**, and the model's raw score = **6.0297**.
- *"Why are values in log-odds, not probability?"* → Only raw scores add up neatly. The sigmoid squashes, so pushes on the probability scale don't add.
- *"Which explainer did you use?"* → `TreeExplainer`: exact and fast for tree models.

### J2. Why sample all frauds plus 5,000 normals for SHAP?

**Answer:** A random 6,256 rows would contain only about **35 frauds**, too few to explain fraud. All 1,256 frauds plus 5,000 normals (with a fixed random seed, so it's reproducible) gives plenty of both. The caveat: importance then leans toward what drives **fraud** scores.

### J3. What are reason codes?

**Answer:** The top 3 SHAP pushes for one alert, turned into plain English for analysts and customers. For example: *"Amount $702.24, at 23:00, 20.8 h since the card's last purchase."*

**Cross-questions:**
- *"What did the top false alarm look like?"* → $269 at a grocery store at 3am. It matched the fraud recipe. But its amount was only **2.9x** the card's usual, versus **11.6x** for the real fraud, and that's what an analyst would use to clear it quickly.
- *"Does SHAP show causes?"* → No. It explains the **model's reasoning**, not real-world causation.

---

## Section K: Fairness

### K1. How did you check fairness?

**Answer:** On validation at cutoff 0.90, per gender and age group, I compared:
- **False alarm rate** (innocent people flagged ÷ all innocent people): the **harm**
- **Recall** (frauds caught ÷ all frauds in the group): the **protection**

Equal on both is called **equalized odds**.

**Cross-questions:**
- *"Why not compare how often each group is flagged?"* → Groups have different real fraud rates (75+ has 1.06% vs 0.34% for 35–45). Equal flag rates would mean ignoring real fraud or inventing it.
- *"Why validation, not test?"* → If the result leads to dropping a feature, that's a model decision, and decisions must never come from test.

### K2. What did you find?

**Answer:**
- **Gender:** women had **0.164%** false alarms vs **0.108%** for men (1.5x), and **86.9%** recall vs **94.0%**. So more harm and less protection, **even though gender is not a feature**.
- **Age:** 25–45 had the highest false alarm rate (**0.196%**, about 3x that of 55+) and the lowest recall (81–85%).

### K3. How can a model be unfair on gender if gender isn't a feature?

**Answer:** Through **proxies**: features that correlate with gender, like spending categories, amounts and purchase times. Removing a protected column ("fairness through unawareness") doesn't remove the bias.

**Analogy:** removing the "neighbourhood" field from a loan form doesn't help if the zip code still tells you the neighbourhood.

### K4. What would you do about it?

**Answer:** Investigate before fixing:
1. Retrain **without `age`** on validation and measure both the fairness gap and the PR-AUC cost.
2. Find which features act as gender proxies.
3. Report to the model-risk or compliance team with numbers.

Using **separate cutoffs per group** might equalise rates, but using a protected attribute directly in decisions can itself be **illegal**, so it needs legal sign-off.

**Cross-questions:**
- *"The under-25 group looks fine. Is it?"* → It has only 71 frauds, so one fraud moves its recall by 1.4 points. Small groups are **noisy**; always report the counts.

---

## Section L: Production and MLOps

> **Update:** this section was written before the production phase. It's all built now, with real numbers: see **Part 4 (pipeline and MLOps)** and **Part 5 (API and production)**. Use those answers first.

### L1. How would you deploy this?

**Answer:**

```
 card swipe --> API --> look up card history (last time, running avg, 24h count)
                         |
                         v
                  build the 9 features (same code as training)
                         |
                         v
                  model (loaded from file) --> score
                         |
                         v
          score >= 0.90 ? --> alert + 3 reason codes --> analyst queue
                         |
                         v
          update card history; log everything for monitoring
```

**Cross-questions:**
- *"What's the slowest part?"* → Not the model (about 1 microsecond per transaction). It's the **card-history lookup** in the database. That needs a fast store keyed by card.
- *"How do you keep training and serving identical?"* → One shared feature module used by both. Otherwise you get **training-serving skew**: the model sees features in production that are calculated differently from training.

### L2. What is training-serving skew?

**Answer:** When features are calculated differently in production than in training, for example a different average window or a different empty-value rule. The model silently gets worse. The fix is **one feature code path** for both, plus tests that compare offline and online feature values.

### L3. How would you monitor the model?

**Answer:**
1. **Data drift:** are feature distributions changing (amounts, hours, time between purchases)?
2. **Prior shift:** the fraud rate over time.
3. **Performance:** precision from analyst decisions (fast feedback), recall from confirmed chargebacks (slow feedback).
4. **Operations:** alert volume vs capacity, latency, errors.
5. **Fairness:** the same group metrics, monthly.

**Cross-questions:**
- *"Labels arrive late. How do you monitor recall?"* → Use **analyst decisions** as an early signal, and confirmed fraud reports (weeks later) for the true recall.

### L4. When would you retrain?

**Answer:** On a schedule (for example, monthly) **and** when monitoring triggers it (drift or a precision drop). Always compare the new model against the current one on the same recent data before switching (a champion vs challenger comparison).

### L5. How do you make the pipeline reproducible?

**Answer:**
- Every decision lives in one config file (`params.yaml`): cut date, features, model settings, cutoff.
- Fixed random seeds, pinned library versions (`requirements.txt`), saved data splits, and the model saved to a file.
- The **first automatic test**: the pipeline must reproduce the notebook's validation PR-AUC of **about 0.940**. If it doesn't, there's a bug.

---

## Section M: Behaviour and reflection

### M1. What was a mistake you made and how did you fix it?

**Answer (pick one, tell it as a story):**
- **Leakage in the card average** (see A2): found by checking a card in the middle, fixed with a per-card calculation, and the lesson became "write invariant checks".
- **The split overwrote its own input file:** fixed by separating folders (interim features vs processed splits), so no step writes over what it reads.
- **LightGBM quit after 4 trees:** I diagnosed overshooting from the large class weight and fixed it with a lower learning rate.

**How to tell it:** situation → what went wrong → how I found it → the fix → what I do differently now.

### M2. What would you do differently?

**Answer:**
- Write **automated checks from day one** (for example, "no feature uses future data").
- Use **time-series cross-validation** instead of a single validation period.
- Keep a separate early-stopping slice so validation stays cleaner.
- Write my prediction **before** opening the test set.

### M3. Did you use help or AI tools?

**Answer (be honest):** "Yes, I worked with a mentor and used AI assistance for parts of the code. But I went back through every notebook line by line until I could explain and rewrite each step myself, and that's what I'm showing you today."

> Honesty here is a strength. Interviewers test understanding with follow-up questions anyway.

### M4. What did you learn?

**Answer:** "That the model is the easy part. Most of the value came from **not fooling myself**: time splits, past-only features, an untouched test set. And from turning scores into **decisions** a business can act on: the cutoff, the cost, explanations and fairness."
