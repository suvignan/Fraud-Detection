# Part 5: The API and Production

*Questions about serving the model live: FastAPI, Neon, running totals, guardrails, Docker, deployment, and
system-design follow-ups.*

**Layout:** **Answer** → **Analogy** (where it helps) → **Cross-questions**.

---

## The 30-second version of this phase

> "The model is served by a FastAPI app in Docker on Render, next to a Neon Postgres database in US East.
> Each request is validated, the card's state is read from Neon as running totals (one row per card plus the
> last 24 hours), the same feature recipes as training build the inputs, XGBoost scores it, and the API returns
> approve, review or flag with SHAP reason codes. Card numbers are stored only as an HMAC. A parity test proves the
> API's features equal the pipeline's. Moving from full history to running totals cut storage from 411 MB to 9 MB,
> and a prediction takes about 90 ms of server-side work."

```
client ──► FastAPI (Docker, Render, Ohio)
             1. schema check ──────────────────────────► 422 if invalid
             2. card state ◄── Neon (Ohio): card_stats (1 row/card) + recent_transactions (last 24 h)
             3. shared feature recipes (same code as training)
             4. XGBoost score ──► decision: flag (≥0.90) > review (guardrail) > approve
             5. SHAP reason codes
             6. write prediction_log + update the card's totals (one database transaction)
           ◄── {"fraud_score", "decision", "reason_codes", "guardrail", "model_version"}
```

**Live:** https://fraud-detection-ozfn.onrender.com/docs

---

## Section A: API design

### A1. Walk me through a `/predict` request.

**Answer:** see the diagram above. The key design points:
- **Validation at the door:** Pydantic rejects bad input with **HTTP 422** before it touches the model (amount must
  be > 0; category must be one of the 14, read from the **same feature spec the model was trained with**).
- **Load once:** the model, spec, cutoff and guardrail limits load **once at startup** (FastAPI's *lifespan*) and
  live on `app.state`. Loading per request would slow every transaction.
- **Traceability:** every response carries `model_version`, the first 12 characters of the model file's MD5, so each
  prediction can be traced to the exact model that made it.
- **`/health`** for the platform's health check.

**Analogy:** a coffee shop turns the espresso machine on once in the morning, not once per customer.

### A2. How does the API get a card's history?

**Answer:** through one method, `get_card_state(card, before)`. Two implementations plug in behind it: a **file
store** (offline development and the parity test) and a **Neon store** (production), chosen by an environment
variable. The API code doesn't change when the storage changes. That's what made the later switch to running
totals a change to one class.

**Analogy:** a phone charger: the phone only cares about the plug shape, not whether it's a wall socket or a power bank.

### A3. How do you prevent training-serving skew?

**Answer:** two layers:
1. **Shared code:** the API uses the **same feature functions** (`build_features.py`) and the same spec as training.
2. **A parity test:** it takes test-set transactions, builds their features through the **API's** code, and compares
   them with the pipeline's. They match on **2,000 transactions** (the largest difference is 2.8e-14, rounding from
   running sums).

**Cross-question:** *"Did the parity check ever fail?"* → Yes, on 1 row in 1,000: two purchases on the same card **in
the same second**. That's 88 rows out of 1.85M (0.005%). The live Neon store handles them correctly (the current
transaction isn't stored yet, so "up to and including this second" is safe); the offline file store can't, because
its files contain the transaction being scored. I documented it as a known edge case.

---

## Section B: The database (Neon Postgres)

### B1. How do you store card numbers safely?

**Answer:** never in plain text. The database stores **`card_key = HMAC-SHA256(card number, secret)`**, with the
secret only in environment variables.

**Why not plain SHA-256?** Card numbers have a small, structured value space (16 digits, a bank prefix, a check
digit), so an attacker could **hash every possible number** and build a lookup table. HMAC needs the secret, so the
table can't be built.

**Analogy:** a common lock anyone can buy and practise on at home, versus a lock made from your private mould.

**Cross-questions:**
- *"What if the secret changes?"* → Every card gets a new key, all history becomes unreachable, and every customer
  looks brand new (and with the guardrails, everyone goes to review). So the secret must be backed up, the same
  secret must be used everywhere (it was a deploy checklist item), and rotation needs a planned migration.
- *"How do you test it?"* → Tests use a fake secret (pytest's `monkeypatch`): the same card gives the same key,
  different cards different keys, a different secret a different key, and a missing secret fails loudly.

### B2. Why running totals instead of storing every transaction?

**Answer:** the first version stored full history: **1.3M rows, 411 MB of the 512 MB free tier**, with each request
fetching ~1,300–2,500 rows, which took **~1.3 s**. But the 5 card features only need:

| Feature | Needs |
|---|---|
| time since last purchase | the last purchase time |
| first purchase? | count = 0 |
| average spend before | total ÷ count |
| amount vs that average | the amount ÷ the average |
| purchases in 24 h | the last 24 h of purchase times |

So each card has **one row of totals** (count, total, last time) plus only its **last 24 hours** of purchases.
**Storage: 411 MB → 8.9 MB**, and each request is **one small query**.

**Analogy:** your bank balance vs keeping every receipt you ever got.

**Cross-questions:**
- *"Isn't that separate code from training?"* → Yes, which is why the parity tests exist: one proves the totals equal
  full history row by row, the other compares against the pipeline on 2,000 transactions.
- *"How is a card updated after a prediction?"* → An **upsert** (insert a new card, or add to an existing card's totals),
  plus inserting the purchase into `recent_transactions` and deleting rows older than 24 hours, all in **one database
  transaction** together with the prediction log row: both are saved, or neither.
- *"Any assumption?"* → Transactions must arrive **in time order** (true live, and in a sorted replay); the code logs
  a warning if one arrives out of order.

### B3. Why an ORM?

**Answer:** SQLAlchemy ORM models are readable Python classes and the single source of truth for the schema, with
constraints (amount > 0, valid decisions) and a composite index on (card_key, time). I know the trade-off: for two
flat tables, plain SQL would also be fine, and once tables hold real data, changes go through **migrations
(Alembic)** instead of dropping and recreating tables.

### B4. Walk me through your latency analysis.

**Answer:** the problem was distance, not work:
```
one round trip India → Neon (Ohio), doing NO work:   256 ms
first version: ~5 round trips per lookup              ≈ 1.3 s
fix 1: one combined query, no separate BEGIN/COMMIT   2 round trips ≈ 500 ms (from India)
fix 2: deploy the API in the SAME region as Neon      → a round trip is a few ms
measured live: /health ~300 ms, /predict ~390 ms from India → ~90 ms of server-side work per prediction
```
**Cross-questions:**
- *"What's `pool_pre_ping`?"* → Neon's free tier puts idle databases to sleep, so the connection pool tests a
  connection before using it. It costs one round trip, which is cheap in the same region.
- *"Why put the API next to the database rather than next to the user?"* → The API makes **two** database trips per
  request; the user makes **one** trip to the API. Minimise the most frequent trips.

---

## Section C: Decisions, reasons and guardrails

### C1. What are reason codes, and how are they computed?

**Answer:** the top features that pushed **this** transaction toward fraud, from SHAP values computed by XGBoost
itself (`pred_contribs=True`), turned into sentences **with context**:
*"Amount $1,077.69 is 19.2x this card's usual spend", "Purchase at 22:00 (high-risk night window)".*
Rules: only pushes **toward** fraud; at most **one** amount reason (amount and amount ratio describe the same
thing); no repeated sentences.

**Cross-question:** *"Does SHAP show causes?"* → No: it explains the **model's reasoning**, not real-world causation.

### C2. What blind spots did you find when stress-testing the API?

**Answer:** I sent extreme inputs on purpose:

| Input | New card | Known card (usual spend ~$56) |
|---|---|---|
| $900 at night | approve 0.04 | **flag 0.998** |
| $2,000 / $10,000 / $90,000 | approve 0.05 (identical) | flag 0.999 (identical) |

1. **Scores freeze above ~$1,200:** trees don't extrapolate. The largest training fraud was **$1,372** and every
   training purchase above $5,000 was legitimate, so everything above the last split lands in the same leaf.
2. **Brand-new cards are under-flagged:** with no history, the strongest signals are missing, and the training data's
   "first purchases" were just where the dataset began, not genuinely new cards.

**Analogy:** a cashier trained on notes up to $100 who's handed a $10,000 note: they've never seen one.

### C3. How did you fix it?

**Answer:** **guardrails**: rules alongside the model, with a **third decision, review** (a human looks at it):
- amount **above the training maximum** ($28,948.90) → review
- **new card** and amount above the **training 99th percentile** ($546.72) → review
- **order:** the model's confident flag wins; otherwise a fired rule means review; otherwise approve
- limits are **computed from the train split** and saved with the model; each rule has its own reason code, and the
  rule is logged in `prediction_log`

**Why review, not flag?** In these cases the model doesn't say "fraud", it says "I don't know". Blocking would punish
real customers for the model's blind spot.

**Capacity check before shipping:** on validation the rules add **1 review in 3 months** (0.3 a month vs ~500 analyst
capacity), and that one was a real fraud the model had missed.

**Live:** a new card spending $10,000 → `review`; repeating it twice more minutes later → `flag` (the card now has
history: three large purchases in minutes).

**Cross-question:** *"Why not retrain instead?"* → The data has no genuinely new cards and no huge frauds to learn
from. Rules handle what the model has never seen: model plus rules is the standard pattern in bank fraud systems.

---

## Section D: Docker

### D1. Walk me through your Dockerfile.

**Answer:**
1. **Slim Python base image** (small, fewer vulnerabilities)
2. **System library** XGBoost needs (OpenMP)
3. **Dependencies first, code last:** Docker caches layers, so a code change rebuilds in seconds instead of
   reinstalling every package
4. **The model is baked in:** the image *is* the model version
5. **Non-root user:** a break-in doesn't get admin rights
6. **`uvicorn` on `0.0.0.0`**, port from **`$PORT`** (set by the platform)

Plus a **`.dockerignore`** that keeps `.env`, data, notebooks and the virtual environment out of the build.

**Analogy:** a shipping container: a standard box that every ship and crane handles the same way.

### D2. Why is `.env` never copied into the image?

**Answer:** anyone who can pull an image can read every file inside it, and images get copied to registries and
servers, so a secret baked into an image is effectively published. Secrets are given to the container only when it
runs (`--env-file` locally, environment settings on the platform).

### D3. How did you keep the image small, and what went wrong?

**Answer:** a lean `requirements-api.txt` with only what serving needs (no DVC, MLflow, SHAP or notebook tools),
**`xgboost-cpu`** (the standard Linux package pulls in GPU libraries a CPU server never uses), and I removed pyarrow
(156 MB) once I saw the container never reads Parquet. **Result: 757 MB.**

**What went wrong:** the first Render deploy crashed at startup with *"sklearn needs to be installed"*.
`XGBClassifier` is XGBoost's scikit-learn-style wrapper, so it needs sklearn **when the object is created**, even
though imports work fine. My check had only tested imports. I fixed the requirements and added a test that
**creates** the model object, which CI runs against the lean requirements. **Lesson: test that the app can start,
not just that it imports.**

---

## Section E: Deployment and operations

### E1. How is it deployed?

**Answer:** Render builds the Docker image from GitHub and runs it in **US East (Ohio)**, next to Neon (`us-east-2`).
Three environment variables only: the database URL, the HMAC secret, and the history backend. **Least privilege:**
the server doesn't get the MLflow or DVC tokens it doesn't need. `/health` is the health check.

**Cross-questions:**
- *"How does the model get into the image?"* → For the first deploy, the 850 KB model files are committed to git, a
  documented shortcut. The cleaner design: a CI job runs `dvc pull` with a token from GitHub Secrets, builds the
  image, and pushes it to a registry; the platform deploys that image, and the image digest identifies the model.
- *"What about the free tier?"* → It sleeps after ~15 idle minutes, so the first request takes ~30–60 s. Fine for
  a demo; production would use an always-on instance.

### E2. How would you monitor it?

**Answer:** every prediction is logged in `prediction_log`: model version, the 9 inputs (JSON), score, decision, the
rule that fired, and the true label once confirmed. From it I'd watch:
1. **data drift** in the inputs (amounts, hours, time between purchases)
2. **prior shift**: the fraud rate over time
3. **precision** from analyst decisions (fast feedback) and **recall** from confirmed fraud (slow feedback)
4. **alert and review volume** vs analyst capacity
5. **latency and errors**
6. **fairness** metrics by group, monthly

### E3. What are the known limitations of the live system?

**Answer:**
- the guardrails cover the blind spots, but the model itself still can't judge new cards or huge amounts
- **no idempotency**: a retried request is counted twice in the card's totals (fix: a unique transaction ID per
  request, ignoring duplicates)
- transactions must arrive in time order for the running totals
- synthetic training data; fairness gaps documented but not fixed

---

## Section F: System-design follow-ups ("how would you scale this?")

| They ask | Answer |
|---|---|
| *10,000 requests per second?* | Run many API containers behind a load balancer (the API is stateless: all state is in the database). Move card state to an in-memory store (e.g. Redis) or a feature store, with Postgres as the system of record. |
| *The log write slows responses?* | Write the prediction log asynchronously (a queue), but keep the card-state update in order, or features drift. |
| *Two requests for the same card at once?* | Row-level locking or atomic increments on the card's totals; process a card's transactions in order (partition by card). |
| *Retries create duplicates?* | Idempotency keys: each transaction carries a unique ID, and repeats are ignored. |
| *Shipping a new model safely?* | Model registry + versioned images; **shadow mode** (score live traffic without acting), then a **canary** (a small % of traffic), compare metrics, then roll out; keep the old image for instant rollback. |
| *When to retrain?* | On a schedule and on drift alerts; always compare the new model against the current one on the same recent data (champion vs challenger) before switching. |
| *Where do labels come from?* | Analyst decisions (fast) and chargebacks/confirmed fraud (weeks later); join them back to `prediction_log` by transaction. |

---

## Quick numbers for this phase

| What | Number |
|---|---|
| Storage: full history → running totals | 411 MB → **8.9 MB** |
| Lookup: first version (from India) | ~1.3 s (5 round trips × 256 ms) |
| Live: `/health` / `/predict` from India | ~300 ms / ~390 ms → **~90 ms server-side** |
| API parity | identical on **2,000** test transactions |
| Same-second edge case | 88 rows (0.005%) |
| Guardrail limits | $28,948.90 (train max), $546.72 (train p99) |
| Reviews on validation | **1 in 3 months** (a real fraud), vs ~500/month capacity |
| Docker image | **757 MB**, non-root |
| Tests | 23 |
