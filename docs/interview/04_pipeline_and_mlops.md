# Part 4: The Production Pipeline (MLOps)

*Questions about turning the notebooks into a modular, tested, reproducible training pipeline.*

**Layout of each question:** **Answer** → **Analogy** (where it helps) → **Cross-questions**.
Answer in the order **decision → why → evidence (a number) → trade-off**, then stop.

---

## The 30-second version of this phase

> "I turned five notebooks into a modular Python package: six components (ingestion, validation, features, split,
> training, evaluation), each taking the previous step's output 'receipt' and returning its own. Every setting
> lives in `params.yaml`. The pipeline reproduces the notebook exactly: the feature files are byte-identical,
> validation PR-AUC is 0.9401, test 0.9058, and two training runs produce an identical model file.
> DVC versions the data and reruns only the stages whose inputs changed; MLflow on DagsHub records every run;
> pytest guards the leakage rules; and GitHub Actions lints and tests every push."

```
params.yaml ──► ingest ──► validate ──► features ──► split ──► train ──► evaluate
                  │           │            │           │          │          │
               parquet   report.json   features +   train/val/  model.json  metrics.json
                                       feature_spec    test     guardrails  (quality gate)
                                                                  .json
   dvc repro: reruns only what changed        MLflow: logs every run        pytest + CI: every push
```

---

## Section A: Structure

### A1. Why turn notebooks into a package at all?

**Answer:** Notebooks are great for exploring, but they run top to bottom by hand, hide state between cells,
and can't be tested, scheduled or reused by an API. A package gives:
- **one place for each step**, testable on its own
- **settings outside the code** (`params.yaml`)
- **the same feature code for training and the API**
- **a pipeline that a machine can run**: CI, a scheduler, a retraining job

**Evidence it was a correct refactor, not a rewrite:** the pipeline's feature files are **identical to the
notebook's** (`pd.testing.assert_frame_equal` on 1.85 million rows), and the model scores the same 0.940 / 0.906.

**Analogy:** a recipe scribbled in a notebook vs a restaurant kitchen with stations. Same dish, but now anyone
can cook it, the same way, every time.

### A2. Walk me through how a component is built.

**Answer:** every component follows the same pattern:
```
class DataSplit:
    __init__(previous_step_artifact, my_config)     ← inputs: what came before + my settings
    initiate_data_split() -> DataSplitArtifact       ← does the work, returns a "receipt"
        try: ... except Exception as e: raise FraudException(e) from e
```
- **Config (dataclass):** what the step needs (paths, settings), built from `params.yaml`.
- **Artifact (dataclass):** what the step produced (paths, row counts).
- Both are **frozen** dataclasses: a typo in a field name fails loudly, and nothing can change them mid-run.

**Cross-questions:**
- *"Why return an artifact instead of the next step reading a fixed folder?"* → The artifact is a **contract**:
  the next step uses the paths the previous step **actually wrote**, and if a step fails there's no artifact,
  so the next step can't start on stale files.
- *"Why dataclasses, not dicts?"* → Fixed fields (typos fail immediately), types as documentation, editor
  autocomplete, and `frozen=True` makes config read-only during a run.

### A3. What goes in `params.yaml`, and what doesn't?

**Answer:** every **choice** the pipeline makes: paths, the cut date, the 9 features and the dropped columns,
XGBoost settings, early stopping, the cutoff, the alert cost, the quality-gate minimum. Code never hard-codes them.

What **doesn't** go in: values that **depend on the data**. `scale_pos_weight` is **computed** from train at training
time (expected ~170.4, written as a comment), because a hard-coded value silently goes wrong when the data changes.
And **facts the data must obey** (column types, allowed values) go in a separate **data contract**, `schema.yaml`.

### A4. How do you handle errors and logging?

**Answer:**
- **One logger** for the whole package: every line has the time, level and module, and goes to the screen and a
  timestamped file. Pipelines run unattended, so the log file is the only witness when something fails at 3am.
  **Never log personal data** (no raw card numbers: a card number in a log is a PCI-DSS breach).
- **One custom exception** that adds the **file and line in our own code** where the error happened: not just
  `KeyError: 'amt'`, but `KeyError: 'amt' in feature_engineering.py line 42`. It skips frames inside pandas, so
  it points at the line you wrote.

---

## Section B: Data validation

### B1. What does the validation step do, and why "fail fast"?

**Answer:** it checks the ingested data against a **contract** (`schema.yaml`): all 22 columns present, no extra
ones, the right types, no empty values in the columns the pipeline uses, and value rules (amount > 0,
`is_fraud` only 0 or 1, category only from the 14 known values). If anything fails, **the pipeline stops**.

**Why:** data usually comes from another team. One day a column is renamed, or amounts arrive in cents. Without a
contract, the pipeline keeps running, builds garbage features and trains a broken model, and nobody notices
until fraud gets through.

**Analogy:** the goods-in check at a restaurant: the delivery is compared to the order sheet before anything
reaches the kitchen.

**Cross-questions:**
- *"Why save the report even when it fails?"* → It's the **audit trail**: proof of what was checked and what broke.
- *"Why collect all problems instead of stopping at the first?"* → One run shows everything that's wrong.
- *"Any gotcha?"* → pandas 3 stores text as `str`, not `object`. Checking exact type names would reject valid
  data, so the schema uses simple words (`int`, `float`, `str`, `datetime`) checked with pandas' `is_*_dtype`
  functions.
- *"How did you test it?"* → I added a fake column (`card_pin`) to the contract: the pipeline stopped with
  *"missing columns: ['card_pin']"* for train and test.

---

## Section C: Features and the "shared recipes" idea

### C1. Why are the feature calculations in a separate file?

**Answer:** the features are needed in **two places**: training (1.85M rows) and the API (one transaction).
If each had its own copy, a bug fixed in one copy would stay in the other, and the model would get different
inputs live than in training: **training-serving skew**, one of the most common real-world ML failures, because it
fails silently. So `build_features.py` holds **pure functions** (DataFrame in, DataFrame out), used by both.

**Analogy:** one master recipe book for the restaurant kitchen and the catering van.

### C2. How did you prove the modular code matched the notebook?

**Answer:** the notebook's outputs were my **answer key**. I renamed them before running the new code (so they
weren't overwritten), then compared with `pd.testing.assert_frame_equal`: **identical** for train and test.
Quick checks too: 983 / 16 empty `secs_since_last` values (first purchases), and the minimum 24-hour count is 1.

### C3. Why does feature engineering combine train and test?

**Answer:** so a test-period transaction can use the same card's **earlier** history from the train period, which
is what a bank would actually have. Each feature only looks **backwards in time** per card, so there's no leakage.
The data is labelled train/test, combined, sorted by card then time, processed, then split back.

### C4. Why fix the category list in a "feature spec" file?

**Answer:** pandas numbers categories by the values it happens to see. In training all 14 are present, so
`grocery_pos` might be code 4; for a single live transaction it would be code 0, which is wrong. The spec stores the
**14 categories in a fixed order**, and both training and the API use it. The spec also stores the 9 feature names in
order.

**Analogy:** a fixed seating chart: seat 4 is always the same person, whoever shows up.

---

## Section D: Training and evaluation

### D1. How do you save the model, and why not pickle?

**Answer:** XGBoost's own **`save_model` → JSON**. Pickle stores the Python object itself: it can break when library
versions change, and **loading a pickle can run hidden code** (a security risk). XGBoost's JSON format is portable,
readable and stable.

### D2. How do you make training reproducible?

**Answer:** fixed `random_state` in `params.yaml`, pinned library versions, data versioned with DVC, and settings in
config. **Proof:** I trained twice and both model files had the **same MD5 fingerprint**. In a bank you must be able
to rebuild any past model exactly, for audits and disputes.

### D3. What is the quality gate?

**Answer:** after scoring, if **validation** PR-AUC is below a minimum (0.90 in `params.yaml`), the pipeline raises
an error and the model is **never marked ready**. The metrics file is saved **before** the gate, so a rejected model
still leaves its numbers behind.

**Cross-question:** *"Why validation and not test?"* → Accepting a model is a **decision**, and decisions are never
made on test. Test numbers are only reported.

### D4. What does the evaluation report contain?

**Answer:** for validation and test: PR-AUC, ROC-AUC, frauds, alerts, recall and precision at the cutoff, false
alarms, total cost, and the guardrail **reviews per month** (capacity check). The numbers matched the notebook
exactly: val 0.9401 / test 0.9058; test at cutoff 0.90 = 2,576 alerts, 1,878 caught, 698 false alarms, $85,477.

---

## Section E: Orchestration, DVC and MLflow

### E1. How do the steps connect?

**Answer:** `training_pipeline.py` runs the six components in order, **passing each artifact to the next**: a relay
race where the artifact is the baton. If any step raises, the chain stops there. Reading the six lines shows
which step depends on which.

### E2. What do DVC's `dvc.yaml` and `dvc.lock` do?

**Answer:**
- **`dvc.yaml`** = the **recipe** (you write it): for each stage, its command, **deps** (input data **and code**),
  **params** (the exact `params.yaml` keys it reads), and **outs**.
- **`dvc.lock`** = the **receipt** (DVC writes it): the fingerprints of every dep, param and output from the last run.
- `dvc repro` compares them and **reruns only the stages whose inputs changed**, plus everything after them.

**Evidence:** I changed the cutoff from 0.90 to 0.85 and **only `evaluate` reran** (one log file instead of six).
Changing it back, DVC said *"evaluate is cached"* and restored the old result **without running anything** (the run
cache).

**Analogy:** a cook who keeps receipts: if none of the ingredients changed, they don't cook the dish again.

**Cross-questions:**
- *"Why must `build_features.py` be a dependency of the features stage?"* → If it isn't, editing a feature
  calculation doesn't make DVC rerun the stage. DVC says "up to date" and the model trains on **old features
  without any error**. Stale results that look fine are worse than a crash.
- *"How do separate DVC stages share data if nothing stays in memory?"* → Each stage rebuilds its inputs from
  files on disk. Even the validation verdict is passed as a file: the features stage reads `validation_status` from
  the validation report, so the safety gate still works across separate commands.
- *"Why does `params:` list individual keys?"* → Precision: the evaluate stage lists only `evaluation.*` keys, which
  is why changing the cutoff reruns only evaluation.

### E3. How is data versioned and stored?

**Answer:** git stores code plus a tiny **`.dvc` ticket** (a fingerprint); the data itself lives in remote storage on
**DagsHub** (S3-compatible). `dvc pull` restores the exact data for any commit. Git can't hold 500 MB of CSVs (it keeps
every version forever, and GitHub blocks files over 100 MB).

**Analogy:** a coat check: git carries the ticket, DVC holds the coat.

### E4. What does MLflow record?

**Answer:** every pipeline run: **params** (XGBoost settings, cut date, cutoff, the computed class weight),
**metrics** (val and test, flattened like `val_pr_auc`) and **artifacts** (the model, feature spec, metrics,
validation report). Comparing two runs side by side showed the cutoff changes precision, recall and cost but
**not PR-AUC**, because a cutoff doesn't change the model's ranking.

**Cross-questions:**
- *"Where do credentials go?"* → In an untracked `.env`. MLflow reads its address, username and token from
  environment variables, so the **same code** logs locally (no `.env`) or to DagsHub (with `.env`).
- *"Any production surprise?"* → MLflow prints an emoji, and Windows output to a pipe uses an old encoding without
  emojis, so the run crashed **after** finishing its work. Fixed by forcing UTF-8 at the entry point. Lesson: code
  that works in your terminal can fail in automation.

### E5. How do you handle secrets?

**Answer:** names only in a committed template (`.env.example`); values in an untracked `.env`; DVC's token in
`.dvc/config.local` (never committed). I checked git history for leaked tokens, and when a token had been exposed
in a chat, I **rotated** it and updated every consumer (DVC and MLflow), then verified both.

---

## Section F: Testing and CI

### F1. What do your tests cover?

**Answer:** **23 tests**, mostly on **tiny hand-made data** where the right answer can be worked out on paper:
- **Feature invariants:** each card's first purchase has empty history; the card average **never includes the
  current purchase** (the leakage rule as a test); values never mix between cards; the 24-hour count; an age that
  rounds **down** the day before a birthday.
- **Running totals == full history**, row by row.
- **Validation** stops a file with missing columns (and still writes its report).
- **Time split:** every train row is before every validation row, which is before every test row.
- **Model:** the spec has 9 features and 14 categories; val PR-AUC passes the gate.
- **API parity** (Part 5), **HMAC card keys**, **guardrails**, and that the API's requirements can create the model.

**Analogy:** smoke detectors: quiet when all is well, and loud the moment someone reintroduces a leak.

**Cross-questions:**
- *"Why tiny data instead of the real 1.8M rows?"* → Speed (CI runs on every push), and on 6 rows you **know** the
  right answer. The values are chosen to test traps: card 2 starts after card 1, so cross-card leakage shows up.
- *"How do tests that need the pipeline's outputs run in CI?"* → `pytest.mark.skipif`: they skip with a visible
  reason when the files aren't there, because those files live in DVC, not git.
- *"Any testing mistake you learned from?"* → Three tests silently never ran because their names didn't start with
  `test_`. pytest said "2 passed" with no warning. Now I always run with `-v` and check every test name appears.

### F2. What does CI do?

**Answer:** GitHub Actions on every push and pull request: a fresh Ubuntu machine checks out the code, installs the
API requirements plus test tools, runs **ruff** (lint: unused imports, undefined names, import order) and **pytest**.

**Cross-questions:**
- *"Did CI catch anything?"* → Before the first push, I **rehearsed CI locally** on a copy containing only
  git-tracked files. It caught a test that imported a module which reads the model spec **at import time**, so it
  crashed before `skipif` could skip it. I moved the import inside the test.
- *"Why isn't the Docker build in CI?"* → The image needs the model file; the clean design is a CI job that runs
  `dvc pull` with the DagsHub token from GitHub Secrets and pushes the image to a registry. For the first deploy, I
  committed the 850 KB model to git instead, as a documented shortcut.

---

## Section G: Stories to have ready

| Story | Use it for |
|---|---|
| **Proving the refactor:** notebook outputs as the answer key, `assert_frame_equal` identical on 1.85M rows | "How do you know your code is correct?" |
| **The silent test bug:** 3 tests never ran (no `test_` prefix) | "A mistake you learned from" |
| **The cutoff test:** only evaluate reran; then "cached" | "Explain DVC" |
| **CI rehearsal:** a fresh copy caught an import-time crash | "How do you prevent surprises?" |
| **Token rotation:** exposed token → rotated → updated DVC and MLflow → verified | "Security practices" |

---

## Quick numbers for this phase

| What | Number |
|---|---|
| Components | 6 (+ a pipeline class) |
| Feature files vs notebook | identical (1.85M rows) |
| Val / test PR-AUC from the pipeline | 0.9401 / 0.9058 |
| Trees / class weight | 86 / 170.4 |
| Reproducibility | same MD5 on two training runs |
| Tests | 23 |
| Full pipeline run | ~75 s |
| Cutoff change | only `evaluate` reruns |
