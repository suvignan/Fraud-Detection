# Fraud Detection: My Learning Notes

These are my notes for this project. For every task they show:
- **The task**: what I was asked to do
- **The idea**: the concept, in simple words
- **How**: the steps
- **Code**: what I actually ran (from `notebooks/01_data_inventory.ipynb`)
- **Result**: the numbers I got
- **What it means**: how to understand the numbers
- **Remember**: a one-line memory hook

Data: `data/raw/fraudTrain.csv` (train) and `data/raw/fraudTest.csv` (test).
In the code, `df` = train and `df2` = test.

---

## Step 1: Data inventory (getting to know the data)

Think of it like a detective opening two evidence boxes. Ask five questions in order:
**How much? When? Who? What shape? Do the clocks agree?**

### 1.1 How many rows, and what is the fraud rate?

**The idea:** `is_fraud` holds only 0 and 1. The average of a 0/1 column is the share of 1s.
So the mean of `is_fraud` *is* the fraud rate.

**Code:**
```python
train_mean = df['is_fraud'].mean()
test_mean = df2['is_fraud'].mean()
```

**Result:**

| File | Rows | Fraud rows | Fraud rate |
|---|---|---|---|
| Train | 1,296,675 | 7,506 | 0.579% |
| Test | 555,719 | 2,145 | 0.386% |

**What it means:**
- Fraud is very rare: about 1 in 170 transactions.
- A model that always says "not fraud" would be 99.4% accurate and catch zero fraud. **So accuracy is useless here.** Use precision, recall and PR-AUC instead.
- Test has less fraud than train. With 2,145 test frauds, this drop is too big to be luck, so the change is real. We do **not** know why it happened.
- The correct way to say it: *"The base rate shifted over time. That is prior shift, and it means a threshold tuned on train will be miscalibrated on test."*

**Remember:** *The mean of a 0/1 column is a percentage in disguise.*

### 1.2 When? (time ranges and overlap)

**The idea:** the date column was stored as text. Always convert it to a real date first.
Two time ranges overlap only if **each one starts before the other ends.**

**Code:**
```python
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
df2['trans_date_trans_time'] = pd.to_datetime(df2['trans_date_trans_time'], errors='coerce')

min_date_train = df['trans_date_trans_time'].min()
max_date_train = df['trans_date_trans_time'].max()
min_date_test = df2['trans_date_trans_time'].min()
max_date_test = df2['trans_date_trans_time'].max()

overlap = (min_date_test <= max_date_train) and (max_date_test >= min_date_train)
```

**Result:**

| File | Start | End |
|---|---|---|
| Train | 2019-01-01 00:00:18 | 2020-06-21 12:13:37 |
| Test | 2020-06-21 12:14:25 | 2020-12-31 23:59:34 |

Overlap = **False**. Test starts 48 seconds after train ends.

**What it means:**
- The data is split by time: we train on the past and test on the future, just like real life.
- The train rows are not perfectly sorted by time. **Sort them before making any time-based feature.**

**Remember:** *Text sorts like a dictionary, dates sort like a calendar.*

### 1.3 Who? (cards and merchants)

**The idea:** "unique" means how many *different* values there are. Use a set intersection (like a Venn diagram) to find the cards that appear in both files.

**Code:**
```python
df["cc_num"].nunique(), df2["cc_num"].nunique()
df["merchant"].nunique(), df2["merchant"].nunique()

shared_cards = set(df["cc_num"]).intersection(set(df2["cc_num"]))
len(shared_cards)
```

**Result:**

| | Train | Test | In both |
|---|---|---|---|
| Unique cards | 983 | 924 | 908 |
| Unique merchants | 693 | 693 | 693 |

- 16 test cards are new (not in train).
- 75 train cards never show up in test. Cards come **and** go over time.

**What it means:**
- 98% of test cards have history in train, so "card history" features will work at test time.
- Same customers plus different dates means the split is by **time**, not by customer.

**Remember:** *Shared people = intersection of sets.*

### 1.4 What shape? (nulls, dtypes, junk columns)

**Result:** 0 nulls and 0 duplicate rows in both files.

| Column | Stored as | Really is | What to do |
|---|---|---|---|
| `trans_date_trans_time` | text | date | Convert to datetime |
| `dob` | text | date | Convert, then make age |
| `cc_num` | number | ID | Never do maths on it |
| `zip` | number | category | Treat as a label |
| `unix_time` | number | shifted date | Drop (see 1.5) |
| `Unnamed: 0` | number | old row number | Drop |
| `first`, `last`, `street`, `trans_num` | text | personal IDs | Don't use as features |

**Two tests to use on every column:**
1. **The adding test:** would adding two values make sense? If not, it's a label, not a number. Adding two zip codes means nothing.
2. **The scoring-time test:** will this value exist when a *new, live* transaction comes in? A CSV row number won't, and `trans_num` is a random new ID every time. If the answer is no, the column can't be a feature.

**Memory used (how much space the data takes in RAM):**

| File | With text counted | Numbers only |
|---|---|---|
| Train | 1,085 MB | 239 MB |
| Test | 465 MB | 102 MB |

About 78% of the memory is text columns. Converting repeated text such as `merchant`, `category`, `state` and `job` to pandas `category` type would save a lot.

```python
df.memory_usage(deep=True).sum() / 1e6   # MB, counting the text too
df = df.drop(columns=["Unnamed: 0"])
df2 = df2.drop(columns=["Unnamed: 0"])
```

**Remember:** *If you'd never add it, it's a label, not a number.*

### 1.5 Do the clocks agree? (unix_time vs trans_date_trans_time)

**Code:**
```python
df["unix_datetime"] = pd.to_datetime(df["unix_time"], unit="s")
df["time_difference"] = df["unix_datetime"] - df["trans_date_trans_time"]
df["time_difference"].value_counts()
```

**Result:** the clock time is the same, but the year is 7 years earlier (2019 shows as 2012).
The gap is not always the same:

| Period | Gap |
|---|---|
| 2019-01-01 to 2019-02-27 | −2557 days |
| 2019-02-28 | mixed |
| 2019-03-01 to 2020-02-28 | −2556 days |
| 2020-03-01 onward (and all of test) | −2557 days |

**What it means:**
- The data generator just relabelled the years. It didn't shift time by a fixed amount, so leap days make the gap wobble by one day.
- There are even **no transactions on 2020-02-29**, because the relabelled calendar had no 29 February that year.
- `unix_time` adds nothing new, so **drop it** and keep `trans_date_trans_time`.

**Remember:** *Same clock, wrong year.*

### 1.6 Merchant locations

**Result:**
- No merchant has a fixed location: every train row has its own `merch_lat` and `merch_long`.
- The merchant point is always within 1 degree of the cardholder's home, spread randomly.
- The distance from home to merchant is the same for fraud (76.3 km on average) and legit (76.1 km).

**What it means:** this is fake, randomly generated data. **A "distance to merchant" feature will not help in this dataset.** We checked before building it, which saves wasted work.

---

## Step 1: Questions a panel will ask (my practice answers)

The goal is to say these in **my own words**. Use the memory hook to rebuild each answer.

**Q1. The ranking is perfect, but the fraud rate changed from train to test. What breaks?**
The threshold breaks. It was set on train (0.58% fraud), but test has only 0.39%. The model's probabilities still assume the higher rate, so the same threshold flags more innocent transactions and precision drops. That is prior shift: the *order* survives, the *cutoff* doesn't.
*Hook: the order survives, the cutoff doesn't.*

**Q2. 98% of test cards are in train. Is that leakage?**
No. A real bank knows a card's past purchases, so using card history is allowed at scoring time. It would be leakage only if a feature used that card's *future* transactions. The catch: our test score says little about brand-new customers, because only 16 test cards are new.
*Hook: the past is allowed, the future is cheating.*

**Q3. A tree memorises `trans_num`. What happens with a new transaction?**
`trans_num` is a random ID that's different every time. The tree can memorise "this ID was fraud" and look perfect on train, but a new transaction has an ID it has never seen, so the prediction is random noise. That's overfitting, so drop the column.
*Hook: memorising receipt numbers isn't learning.*

**Q4. You count "transactions on this card in the last 24 hours" on the unsorted train file. What goes wrong?**
The count assumes the rows are in time order. They aren't, so the "previous rows" can include *future* transactions. That leaks the future and gives wrong counts. Sort by card and time first.
*Hook: shuffled pages, wrong past.*

**Q5. Explain unix_time in two sentences.**
`unix_time` has the same clock time as the real timestamp but is 7 years earlier, and the gap switches between 2557 and 2556 days around leap days. So the generator relabelled the years, the column adds nothing, and I dropped it.
*Hook: same clock, wrong year.*

---

## Step 2: Exploring the data (EDA)

All of Step 2 uses **train only**. Test stays unseen.

**The one pattern behind tasks 4–7:** group by something, then take the mean of `is_fraud`.

### 2.1 Amount summary: fraud vs legit

**The idea:** line up all the amounts from small to big.
- **Median** = the one in the middle.
- **p95** = only 5% are bigger.
- **p99** = only 1% are bigger.

Use the median, not the average, because a few huge amounts pull the average up.

**Code:**
```python
fraud = df[df['is_fraud'] == 1]
legit = df[df['is_fraud'] == 0]

statistics = {
    "min":    [fraud['amt'].min(), legit['amt'].min()],
    "median": [fraud['amt'].median(), legit['amt'].median()],
    "p95":    [fraud['amt'].quantile(0.95), legit['amt'].quantile(0.95)],
    "p99":    [fraud['amt'].quantile(0.99), legit['amt'].quantile(0.99)],
    "max":    [fraud['amt'].max(), legit['amt'].max()],
}
stats_table = pd.DataFrame(statistics, index=["Fraud", "Legit"])
```

**Result:**

| | Min | Median | p95 | p99 | Max |
|---|---|---|---|---|---|
| Fraud | $1.06 | $396.51 | $1,083.99 | $1,179.69 | $1,376.04 |
| Legit | $1.00 | $47.28 | $189.90 | $486.30 | $28,948.90 |

**What it means:**
- A typical fraud is about **8 times bigger** than a typical normal purchase ($397 vs $47).
- But fraud **never goes above $1,376**, while normal purchases go up to $28,949. Fraudsters stay under some limit, probably to avoid getting blocked.
- Fraud amounts come in **two groups**:
  - Many small ones under $50 (about 1,600 frauds). These look like "testing the card" to see if it works.
  - A big group from $200 to $1,500 (about 5,700 frauds). These are the real cash-outs.
- Pattern: **test small, then spend big.**

**Remember:** *Line them up and point at a position.*

### 2.2 Skewness and kurtosis, before and after log1p

**The idea:**
- **Skewness** = is the data lopsided? 0 means balanced. A big positive number means a long tail of huge values on the right.
- **Kurtosis** = how extreme are the outliers? Pandas uses 0 for a normal bell curve. A big number means a few monster values.
- **log1p** = log(1 + x). It squashes big numbers much more than small ones, so the long tail shrinks.

**Code:**
```python
raw_skew = df['amt'].skew()
raw_kurtosis = df['amt'].kurtosis()

df['amt_log'] = np.log1p(df['amt'])
skew_log = df['amt_log'].skew()
kurtosis_log = df['amt_log'].kurtosis()
```

**Result:**

| | Before log | After log1p |
|---|---|---|
| Skewness | 42.28 | −0.30 |
| Kurtosis | 4,545.64 | −0.53 |

**What it means:**
- Before: very lopsided, with a few giant amounts.
- After log1p: almost balanced, and the giants are tamed.
- **Does it matter for tree models?** Barely. A tree only asks questions like "is amt > 200?", so it only cares about the **order** of the values. Log keeps the order the same (the biggest is still the biggest), so the tree makes the same splits.
- Log helps models that care about distances or straight lines: logistic regression, neural networks and KNN.

**Remember:** *Log changes the ruler, not the line-up.*

### 2.3 Are any amounts 0 or negative?

**The idea:** log(0) is minus infinity, which breaks things. log1p(0) = 0 is safe.

**Code:**
```python
(df['amt'] == 0).sum()   # 0
(df['amt'] < 0).sum()    # 0
df['amt'].min()          # 1.0
```

**Result:** no zero amounts and no negative amounts. The smallest amount is $1.00.

**What it means:** both log and log1p are safe on this data. Still use **log1p** out of habit, because a future transaction could be $0.

**Remember:** *Log can't eat zero. log1p can.*

### 2.4 Fraud rate by category

**The idea:** use the **rate**, not the count. A count only shows how popular a category is. A rate shows how risky it is.

**Code:**
```python
category_stats = df.groupby('category')['is_fraud'].agg(
    fraud_rate='mean',
    count='count'
).sort_values("fraud_rate", ascending=False)

overall_rate = df['is_fraud'].mean()
category_stats["relative_to_overall"] = category_stats["fraud_rate"] / overall_rate
```

**Result:**

| | Category | Fraud rate | vs overall |
|---|---|---|---|
| Top 1 | shopping_net | 1.76% | 3.0× |
| Top 2 | misc_net | 1.45% | 2.5× |
| Top 3 | grocery_pos | 1.41% | 2.4× |
| Bottom 3 | food_dining | 0.17% | 0.28× |
| Bottom 2 | home | 0.16% | 0.28× |
| Bottom 1 | health_fitness | 0.15% | 0.27× |

The riskiest category is about **11 times** riskier than the safest (1.76% ÷ 0.155%).

**What it means:**
- Online (`_net`) shopping is risky, because no physical card is needed.
- `grocery_pos` is a surprise: it's an in-store category but still very risky.
- Everyday spending (food, home, health) is very safe.
- So `category` will be a useful feature.

**Remember:** *A count measures popularity. A rate measures risk.*

### 2.5 Fraud rate by hour of day

**Code:**
```python
df["hour"] = df["trans_date_trans_time"].dt.hour
hour_stat = df.groupby('hour')['is_fraud'].mean()

plt.figure(figsize=(12, 5))
plt.bar(hour_stat.index, hour_stat.values)
plt.axhline(overall_rate, color="red", linestyle="--", label="Overall fraud rate (0.58%)")
plt.xlabel("Hour of day"); plt.ylabel("Fraud rate"); plt.title("Fraud Rate by Hour of Day")
plt.legend(); plt.show()
```

**Result:**

| Hours | Fraud rate | vs overall (0.58%) |
|---|---|---|
| 22:00–23:59 (10pm–midnight) | about 2.85% | about **5× higher** |
| 00:00–03:59 (midnight–4am) | about 1.5% | about **2.5× higher** |
| 04:00–21:59 (daytime) | about 0.1% | about 5× **lower** |

**What it means:**
- Fraud happens at **night**, from 10pm to 4am, when the real owner is asleep and won't notice.
- The jump is sharp. At 3am the rate is 1.4%, and at 4am it suddenly falls to 0.1%. Real life is rarely that clean, which tells us the data was generated by a program.
- Hour (or simply "is it night?") will be a strong feature.
- Real systems must know **which time zone** the hour is in.

**Remember:** *Fraudsters work while you sleep.*

### 2.6 Age at transaction

**The idea:** age **at the moment of the purchase**, not today.

**Code:**
```python
df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
age_days = (df["trans_date_trans_time"] - df["dob"]).dt.days
df['age'] = np.floor(age_days / 365.25).astype('int')

bins = [0, 25, 35, 45, 55, 65, 75, np.inf]
labels = ["Under 25", "25-35", "35-45", "45-55", "55-65", "65-75", "75+"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

age_stats = df.groupby("age_group", observed=True)['is_fraud'].agg(
    fraud_rate='mean',
    count='count'
)
```

**Result:** ages run from 13 to 95.

| Age group | Fraud rate | Transactions | About how many cards |
|---|---|---|---|
| 75+ | 0.90% | 93,274 | 105 |
| 55–65 | 0.77% | 164,087 | 186 |
| Under 25 | 0.63% | 121,688 | 93 |
| 65–75 | 0.59% | 100,384 | 106 |
| 45–55 | 0.58% | 256,553 | 213 |
| 25–35 | 0.48% | 287,749 | 207 |
| 35–45 | 0.43% | 272,940 | 192 |

**What it means:**
- **Older people (75+) are targeted most**, about 2 times more than people aged 35–45.
- Young people (under 25) are also above average. The middle ages (25–45) are the safest.
- The shape is like a **U**: high at both ends, low in the middle.
- **Be careful:** each group has many transactions but only about 100–200 **people**. A few unlucky cards can move a group's rate a lot. Tiny groups lie.
- **Odd value:** 13 cards (15,568 transactions) belong to people **under 18**, and the youngest is 13. That's strange for credit cards. It's fake data, but note it and don't trust it blindly.

**Remember:** *Age at the moment, not age today.*

### 2.7 Monthly fraud rate (18 train months)

**The idea:** a fraud rate over time can be:
- **Stable**: a flat line
- **Trending**: steadily going up or down
- **Seasonal**: the same pattern repeating each year

**Code:**
```python
df['year_month'] = df['trans_date_trans_time'].dt.to_period('M')
monthly_stats = df.groupby("year_month")["is_fraud"].agg(fraud_rate="mean", count="count")

plt.figure(figsize=(12, 5))
plt.plot(monthly_stats.index.to_timestamp(), monthly_stats["fraud_rate"], marker="o")
plt.axhline(overall_rate, color="red", linestyle="--", label="Overall fraud rate (0.58%)")
plt.xlabel("Month"); plt.ylabel("Fraud rate"); plt.title("Monthly Fraud Rate")
plt.legend(); plt.show()
```

**Result:**

| Month | Rate | Transactions |
|---|---|---|
| 2019-01 | 0.96% | 52,525 |
| 2019-02 | **1.04%** (highest) | 49,866 |
| 2019-03 | 0.70% | 70,939 |
| 2019-04 | 0.55% | 68,078 |
| 2019-05 | 0.56% | 72,532 |
| 2019-06 | 0.41% | 86,064 |
| 2019-07 | **0.38%** (lowest) | 86,596 |
| 2019-08 | 0.44% | 87,359 |
| 2019-09 | 0.59% | 70,652 |
| 2019-10 | 0.66% | 68,758 |
| 2019-11 | 0.55% | 70,421 |
| 2019-12 | 0.42% | 141,060 |
| 2020-01 | 0.66% | 52,202 |
| 2020-02 | 0.70% | 47,791 |
| 2020-03 | 0.61% | 72,850 |
| 2020-04 | 0.45% | 66,892 |
| 2020-05 | 0.71% | 74,343 |
| 2020-06 | 0.58% | 57,747 |

**What it means:**
- **Not stable.** The rate moves between 0.38% and 1.04%, almost 3 times from lowest to highest.
- **No clear trend.** It doesn't keep going up or down.
- **Maybe seasonal, but not proven.**
  - Jan–Feb were the highest months in both years, and summer (Jun–Aug 2019) was low.
  - December 2019 had double the usual transactions (holiday shopping) but a low rate. Lots of normal shopping "dilutes" the fraud.
  - We only have 1.5 years, so only Jan–Jun repeat. That's not enough to prove seasonality.
- **Link to the test set:** test covers Jun–Dec 2020, and its rate of 0.39% is the same as the low months of Jun–Aug and Dec 2019. So the low test rate is not strange. It's **inside the normal monthly ups and downs.**
- **Big lesson:** the base rate already moves a lot *inside* train, so it will move in production too. That's why a live fraud model needs **monitoring** and a threshold that gets checked and reset. This is what the drift project will do.

**Remember:** *If the base rate moves inside train, it'll move in production too.*

---

## Notebook fixes to make (from the review)

1. **Risk ratio uses the wrong row.** `bottom_3['fraud_rate'].iloc[0]` is `food_dining`, but the lowest category is the *last* row, `health_fitness`. Use `.iloc[-1]`. The correct ratio is about **11.3×**, not 10.6×.
2. **Changing a slice gives a warning.** `top_3 = category_stats.head(3)` followed by changing `top_3` can give a `SettingWithCopyWarning`. Write `category_stats.head(3).copy()`.
3. **Don't type the overall rate by hand.** Use `overall_rate = df['is_fraud'].mean()` instead of `0.0058`, so it is always exact.
4. **Delete the empty plot cells.** The cells that only call `plt.axhline(...)` on their own make a blank chart.
5. **Convert dates once.** `trans_date_trans_time` is converted 3 times. Once, at the top, is enough.
6. **Write findings in markdown cells.** Under each chart, add 1–2 lines saying what you learned. A panel reads conclusions, not just numbers.
7. **Later: split into two notebooks.** Step 1 stays in `01_data_inventory.ipynb`, and Step 2 moves to `02_eda.ipynb`.

---

## Feature ideas collected so far

| Idea | Why | Signal seen? |
|---|---|---|
| `amt` (or log of amt) | Fraud median is $397 vs $47 | Strong |
| `category` | Online shopping is up to 11× riskier | Strong |
| `hour` or "is night" (10pm–4am) | Night is 2.5–5× riskier | Strong |
| `age` | 75+ and under 25 are riskier | Medium |
| Card history (count and amount in the last 24 hours) | Fraud = "test small, then spend big" | To check (sort by time first!) |
| Distance to merchant | Locations are random | **None, dropped in Step 3b** |

---

## Step 3: Building features (`notebooks/02_features.ipynb`)

### 3a. Remove useless columns, add age and hour

**The idea:** a column stays only if (1) it says something about fraud **and** (2) it will exist when a new transaction arrives.

**Code:**
```python
df['dob'] = pd.to_datetime(df['dob'])
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

df['age'] = np.floor((df['trans_date_trans_time'] - df['dob']).dt.days / 365.25).astype(int)
df['hour'] = df['trans_date_trans_time'].dt.hour

columns_to_drop = ['Unnamed: 0', 'unix_time', 'trans_num', 'first', 'last', 'street']
df = df.drop(columns=columns_to_drop)
# ...exactly the same for df2 (test)
```

**Result:** 23 → 19 columns in both files. Train age 13–95, test age 15–96, hour 0–23.

**Remember:** *Whatever you do to train, do to test.* (In one cell, the "Test hour" check printed `df` instead of `df2`, which is exactly the mistake this rule catches.)

### 3b. Distance from home

**The idea:** a card used far from home can be fraud. But a real shop doesn't move, so first check whether the merchant locations are real.

**Code:**
```python
merchant_lat_counts = df.groupby("merchant")["merch_lat"].nunique()
merchant_lat_counts.min(), merchant_lat_counts.median(), merchant_lat_counts.max()

lat1, long1 = np.radians(df['lat']), np.radians(df['long'])
lat2, long2 = np.radians(df['merch_lat']), np.radians(df['merch_long'])
dlat, dlon = lat2 - lat1, long2 - long1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
df['distance_km'] = 2 * 6371 * np.arcsin(np.sqrt(a))

df.groupby('is_fraud')['distance_km'].agg(['mean', 'median'])
```

**Result:**
- Each merchant has 727–4,403 different locations, and 0 merchants have a fixed one.
- The distance is the same for fraud and normal: mean 76.27 vs 76.11 km.

**Decision:** drop `distance_km`, `merch_lat` and `merch_long`.
- **Data science reason:** a noise column gives trees chances to make random splits that fit train by luck.
- **Banking reason:** every feature has to be built, monitored and justified. A useless one is pure cost.

**Remember:** *A real shop doesn't move. Check before you trust a feature.*

### 3c part 1. Time since the card's last transaction

**The idea:** a thief uses a stolen card fast, before it gets blocked. So "how long since this card was last used?" can reveal fraud.

**The 4 steps:**
1. **Label** each row "train" or "test" (like name tags on luggage).
2. **Stack** both tables, so a test card can see its history in train. *The past is allowed, the future is cheating.*
3. **Sort** by card, then time. *Shuffled pages, wrong past.*
4. **For each card**, subtract the previous purchase time from this one and convert to seconds. Then split the tables back apart using the labels.

**Code:**
```python
train_data = df.copy(); train_data["dataset"] = "train"
test_data = df2.copy(); test_data["dataset"] = "test"

all_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
all_data = all_data.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
all_data["secs_since_last"] = (
    all_data.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds()
)

train = all_data[all_data["dataset"] == "train"].copy()
test = all_data[all_data["dataset"] == "test"].copy()

train["is_first_txn"] = train["secs_since_last"].isna().astype(int)
test["is_first_txn"] = test["secs_since_last"].isna().astype(int)

train.groupby("is_fraud")["secs_since_last"].median()
```

**Result:**
- Empty rows: train 983 (= 983 cards) and test 16 (= 16 new cards). Both match the prediction.
- Median gap: fraud 4,908 s (1.4 h) vs normal 16,623 s (4.6 h). Fraud is **about 3.4× faster**.

**Decision on the empty rows:** leave them empty and add the `is_first_txn` flag. "No history" is unknown, and filling with the median would pretend it's normal. Trees handle empty values, and the flag makes it clear.

**Mistake caught:** the 3 distance columns were dropped from train only, so they came back as empty values after combining. Always drop from both files.

**Mentor's corrections:**
- The column count went **20 → 17**, not 18 → 17, because `distance_km` had been added first (19 + 1 = 20). Keep counts exact: when the notebook becomes Python files, a wrong column count is where bugs hide.
- In train, `is_first_txn` = 1 doesn't mean "new card". It means "first transaction we can see", because the data starts in Jan 2019 and those cards existed before. Only the 16 test cards are really new.

**Why this feature matters later:** it's a strong model signal, an easy-to-explain reason ("used again 2 minutes after the last purchase"), the reason the live API needs a card-history table (each card's last transaction time), and something the drift monitor can watch.

**Remember:** *Thieves are in a hurry.*

### 3c part 2. Amount compared to the card's usual

**The idea:** $400 is normal for some people and shocking for others. So ask "is it big **for this person**?"

**The trap:** the card's average over **all** its purchases includes future purchases, which is leakage. At the card's first purchase, it would already "know" money that hasn't been spent yet.

| Purchase | amt | Previous-only average (right) | All-purchases average (wrong) |
|---|---|---|---|
| 1st | $10 | empty | $80 |
| 2nd | $20 | $10 | $80 |
| 3rd | $30 | $15 | $80 |
| 4th | $260 (fraud) | $20, so the ratio is 13× | $80, so the ratio is only about 3× |

**Correct code (per card, can't mix cards):**
```python
g = all_data.groupby("cc_num")["amt"]
all_data["card_avg_amt_before"] = (g.cumsum() - all_data["amt"]) / g.cumcount()
all_data["amt_ratio"] = all_data["amt"] / all_data["card_avg_amt_before"]
```
The idea is (running total minus this amount) ÷ (number of previous purchases). A first purchase gives 0 ÷ 0, which is empty.

**Result:** median `amt_ratio` is 5.23 for fraud vs 0.66 for normal. Empty rows: 983 train and 16 test. 24% of frauds are below 1 ("test small, then spend big").

**Mistake caught:** the first version did `.expanding().mean().shift(1)`, and the shift ran across the whole column instead of inside each card. Each card's first purchase got the **previous card's** average. The giveaway was 1 empty row instead of 999. The hand check missed it because it used the very first card. **Always check a card from the middle.**

**Remember:** *Big for whom?*

### 3c part 3. How busy is the card? (`txn_count_24h`)

**The idea:** a thief uses a stolen card many times in a short burst. Count the card's purchases in the last 24 hours. Banks call this a **velocity** feature.

**Why a time window, not "last 5 rows":** 5 rows can be 10 minutes for a busy shopper and 2 weeks for a quiet one. A 24-hour window means the same for everyone.

**Code:**
```python
rolling_txn_count = (
    all_data.groupby("cc_num", sort=False)
    .rolling("24h", on="trans_date_trans_time")["amt"]
    .count()
    .reset_index(level=0, drop=True)
)
all_data["txn_count_24h"] = rolling_txn_count.to_numpy()
```

**Checks:** minimum 1; 0 empty values; every card's first purchase = 1; a middle-card hand count matched (5 = 5); an independent recount of all 1.85M rows matched.

**Result:** mean 5.23 for fraud vs 4.88 for normal (medians 5 vs 4). It's **weak**, only 7% higher, because busy and quiet cardholders have very different normal counts. It's the same "big for whom?" problem. Comparing to the card's usual count could make it stronger.

**Remember:** *Thieves race the clock, so measure by the clock.*

### 3d. Text columns: group or nametag?

**The idea:** a column that describes a **kind** of thing ("online shopping", "female") lets the model learn a pattern. A column with about one value per person is a **nametag**: the model can only memorise people, the same problem as `trans_num`.

**Code:**
```python
cols = ["category", "gender", "state", "city", "zip", "job",
        "merchant", "lat", "long", "city_pop"]

print(train[cols].nunique())                              # different values in total
print(train.groupby("cc_num")[cols].nunique().max())      # 1 = never changes for a person
```

**Result:**

| Column | Values | Per card | People per value | Decision | Reason |
|---|---|---|---|---|---|
| category | 14 | up to 14 | (describes the purchase) | **Keep** | 11× risk spread |
| gender | 2 | 1 | about 490 | Drop | Protected attribute (fairness and legal risk) |
| state | 51 | 1 | about 19 | Drop | High rates only in 1–3-person states (DE = 1 person, 100%) |
| job | 494 | 1 | about 2 | Drop | Nametag |
| merchant | 693 | up to 678 | (describes the purchase) | Drop | Differences between merchants are no bigger than luck; category covers it |
| city_pop | 879 | 1 | about 1.1 | Drop | Nametag |
| city | 894 | 1 | about 1.1 | Drop | Nametag |
| lat | 968 | 1 | about 1.0 | Drop | Nametag (distance already gone) |
| long | 969 | 1 | about 1.0 | Drop | Nametag |
| zip | 970 | 1 | about 1.0 | Drop | Nametag |

**Lessons:**
- "It has nothing to do with fraud" is a **guess**. Use the number, or run a check.
- 19 people per state means it's a group, not a nametag. But the fraud-rate check showed the "risky" states were tiny (1–3 people). *Tiny groups lie.*
- Merchant: compare the real spread with what pure luck would give. If they match, there's no signal.

**Remember:** *A group is a pattern. A nametag is memorising. Check, don't guess.*

### 3e. Final column list and save

**The idea:** every column goes into one of three groups.

| Group | Columns | Why |
|---|---|---|
| **Features** (9) | `amt`, `category`, `hour`, `age`, `secs_since_last`, `is_first_txn`, `card_avg_amt_before`, `amt_ratio`, `txn_count_24h` | What the model learns from |
| **Target** | `is_fraud` | What the model predicts |
| **Helpers / audit** | `trans_date_trans_time`, `cc_num`, `gender` | Kept in the file, **never** given to the model: time for the Step 4 split, card ID to trace decisions, gender for fairness checks |

Dropped: `merchant`, `city`, `state`, `zip`, `lat`, `long`, `city_pop`, `job`, `dob`, `dataset`.

**Code:**
```python
features = ["amt", "category", "hour", "age", "secs_since_last", "is_first_txn",
            "card_avg_amt_before", "amt_ratio", "txn_count_24h"]
target = "is_fraud"
helpers = ["trans_date_trans_time", "cc_num", "gender"]
keep = helpers + features + [target]

train_final = train[keep].copy()
test_final = test[keep].copy()
train_final["category"] = train_final["category"].astype("category")
test_final["category"] = pd.Categorical(test_final["category"],
                                        categories=train_final["category"].cat.categories)

train_final.to_parquet("../data/interim/features_train.parquet", index=False)
test_final.to_parquet("../data/interim/features_test.parquet", index=False)
```

**Why Parquet:** it keeps column types (dates stay dates) and is much smaller and faster than CSV.

| File | CSV | Parquet | Smaller by |
|---|---|---|---|
| Train | 351.2 MB | 41.5 MB | 8.5× |
| Test | 150.4 MB | 17.7 MB | 8.5× |

**Checks:** both files have 13 columns with the same names. Reloading gives the same rows and fraud rates (0.579%, 0.386%). Dates stay datetime, and category stays category.

**Remember:** *Features teach, helpers explain, the target is the answer.*

---

## Step 4: Split by time (`notebooks/03_split.ipynb`)

**The idea (like studying for an exam):**
- **Train** = practice questions. The model learns from these.
- **Validation** = a mock exam. Compare models and choose settings, as often as you like.
- **Test** = the final exam. Look **once**, at the very end. Checking it while making choices is secretly studying for the final, and the score stops being honest.

**Why cut from the end, not random rows:** in real life the model always predicts the future from the past. Random rows would leak future days into training.

```
|<------ TRAIN (Jan 2019 – 20 Mar 2020) ------>|<-- VALIDATION (21 Mar – 21 Jun 2020) -->|<-- TEST (21 Jun – 31 Dec 2020) -->|
```

**Code:**
```python
train = pd.read_parquet("../data/interim/features_train.parquet")
test = pd.read_parquet("../data/interim/features_test.parquet")

train = train.sort_values("trans_date_trans_time").reset_index(drop=True)
cut_date = pd.to_datetime("2020-03-21 00:00:00")

new_train = train[train["trans_date_trans_time"] < cut_date].copy()     # strictly before
validation = train[train["trans_date_trans_time"] >= cut_date].copy()   # on or after
```

**Result:**

| Set | Start | End | Rows | Fraud rate |
|---|---|---|---|---|
| Train | 2019-01-01 00:00:18 | 2020-03-20 23:58:44 | 1,070,966 | 0.584% |
| Validation | 2020-03-21 00:00:20 | 2020-06-21 12:13:37 | 225,709 | 0.556% |
| Test | 2020-06-21 12:14:25 | 2020-12-31 23:59:34 | 555,719 | 0.386% |

**Checks:** 1,070,966 + 225,709 = 1,296,675, with no overlap. Use `<` for one set and `>=` for the other, so no row is lost or doubled.

**What it means:** validation looks like train (0.584% vs 0.556%), but test is lower (0.386%). So validation won't fully warn us about the future. A threshold tuned on validation may over-flag on test, which is prior shift again.

**No need to recompute features:** they only looked at each card's past, so they're still correct.

**Saving the split (so every step uses the same sets):**
```python
new_train.to_parquet("../data/processed/train.parquet", index=False)
validation.to_parquet("../data/processed/val.parquet", index=False)
test.to_parquet("../data/processed/test.parquet", index=False)
```
- **Why save it:** if every notebook redoes the cut, one small mistake (a different date, a forgotten sort) quietly gives different sets, and results stop being comparable. Real pipelines have a "split" step that writes files, and the next step reads them.
- **The cut date `2020-03-21`** is written in a markdown cell. Later it moves to `params.yaml`, so it's never hidden in code.
- **The trap avoided:** the split reads `features_train.parquet` and writes `train.parquet`. If it read and wrote the same file, a second run would cut an already-cut file and validation would be empty. **Never overwrite your own input.**

**Data flow:**
```
data/raw/*.csv  -> 02_features ->  data/interim/features_{train,test}.parquet
                -> 03_split    ->  data/processed/{train,val,test}.parquet
```

**Remember:** *Practise on the past, mock exam on the recent past, final exam once.*
