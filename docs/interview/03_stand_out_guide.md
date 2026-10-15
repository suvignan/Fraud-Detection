# Part 3: How to Stand Out in Interviews

*How to present this project at a top-tier technology company.*

---

## 1. What strong interviewers are really checking

They aren't checking whether you can call `XGBClassifier()`. Thousands of candidates can. They check these six things:

| They check | What it sounds like when you have it |
|---|---|
| **1. You don't fool yourself** | "I split by time and kept the test set closed until the end." |
| **2. You think in trade-offs** | "0.59 is cheaper, but it needs 43% more analysts. That's a business choice." |
| **3. You back claims with numbers** | "Removing the card features cut PR-AUC from 0.939 to 0.907." |
| **4. You connect ML to the business** | "At 0.90 we get about 400 alerts a month, which fits the team." |
| **5. You know your limits** | "The data is simulated, so real scores would likely be lower." |
| **6. You think about production** | "The slow part isn't the model, it's the card-history lookup." |

**Analogy:** a driving test. Anyone can press the pedal. The examiner watches whether you check your mirrors (leakage), read the road (business) and know your car's limits (honesty).

---

## 2. The answer formula

For almost every "why did you...?" question, use **D-W-E-T**:

```
 D - Decision    "I split the data by time."
 W - Why         "Because the model always predicts the future."
 E - Evidence    "Train ends March 2020, validation is March to June, test is June to December."
 T - Trade-off   "It leaves fewer recent rows for training, so for production I'd retrain on train plus validation."
```

**Then stop talking.** Let them choose the next question. Short, complete answers beat long speeches.

For "tell me about a time..." (behavioural) questions, use **STAR**:

```
 S - Situation   "My card-average feature looked fine."
 T - Task        "I had to make sure it used only past data."
 A - Action      "I checked a card in the middle of the data and found it leaking across cards. I rewrote it per card."
 R - Result      "Fixed. Now I write invariant checks for every history feature."
```

---

## 3. Weak answers vs strong answers

| Weak (sounds junior) | Strong (sounds senior) |
|---|---|
| "My model has 99% accuracy." | "Accuracy is misleading at 0.5% fraud. I used PR-AUC: 0.906 on test vs 0.22 for the baseline." |
| "I used XGBoost because it's the best." | "It beat the baseline by 4x and tied LightGBM, and I chose it for speed and stability at default settings." |
| "I removed useless columns." | "I dropped merchant location because distance showed no signal: 76.3 vs 76.1 km." |
| "The test score dropped, so the model overfit." | "ROC-AUC held at 0.998, so ranking held. PR-AUC fell from prior shift: fraud dropped from 0.56% to 0.39%." |
| "I used a 0.5 threshold." | "I chose 0.90 from analyst capacity, and I modelled cost to show the cheapest option too." |
| "The feature importance shows amount matters." | "Gain and SHAP agreed on the top 3, and a removal test confirmed it." |
| "It's fair because I removed gender." | "Removing gender isn't enough. Women still had 1.5x the false alarm rate through proxies." |
| "I'd deploy it with an API." | "It's live: FastAPI in Docker next to its database, ~90 ms server-side, with a parity test proving its features equal training's." |
| "The model works for all inputs." | "Stress-testing showed two blind spots (new cards, amounts above the training range); I added guardrails that send them to review, and checked the review volume fits analyst capacity." |
| "I stored the card history in the database." | "Full history took 411 MB and 1.3 s per lookup; I switched to running totals per card (9 MB, one query) and proved the features identical on 2,000 transactions." |
| "I hashed the card numbers." | "HMAC with a secret, not plain SHA-256: the card number space is small enough to brute-force a plain hash." |
| "My refactor is correct." | "The pipeline's feature files are identical to the notebook's on 1.85M rows, and two training runs give the same model file." |

---

## 4. Ten lines that make you stand out

Learn the **idea** of each one, not the exact words.

1. "The model is the easy part. Most of the work was **not fooling myself**."
2. "I treat the **cutoff as a business decision**, not a model setting."
3. "When ROC holds and PR drops, I check for **prior shift** before blaming the model."
4. "Feature importance shows what the model **used**. A **removal test** shows what it would **lose**."
5. "I verified SHAP adds up: the base value plus the pushes equals the raw score exactly."
6. "I checked fairness as **equalized odds**: false alarm rate and recall per group, not flag rates."
7. "**Fairness through unawareness** doesn't work. Proxies carry the removed information."
8. "Validation chose the stopping point and the cutoff, so its score is optimistic. **Test is my headline.**"
9. "The slow part in production is the **feature lookup**, not the model."
10. "The pipeline's first test is to **reproduce the notebook's validation score**. If it doesn't, there's a bug."

**Five more, for the production phase:**

11. "**Training-serving skew fails silently**, so I don't trust shared code alone: a **parity test** proves it."
12. "The model says 'I don't know' more often than it says 'fraud'. **Guardrails turn 'I don't know' into a human review.**"
13. "I measured before optimising: **256 ms per round trip** was the real cost, not the queries. So I cut round trips and put the API **next to the database**."
14. "**Test that the app can start, not just that it imports.** That's the lesson from my first failed deploy."
15. "A secret baked into an image is a published secret. **Secrets arrive at run time; only names live in git.**"

---

## 5. Red flags to avoid

| Never say or do | Why it hurts |
|---|---|
| Quote accuracy as your main metric | Shows you don't understand imbalance |
| "I shuffled the data and split 80/20" | Leakage on time-based data |
| "I tuned until test looked good" | Test is no longer honest |
| Claim things you didn't do ("it's in production") | Follow-up questions will expose it |
| Say "the model learned that X causes fraud" | SHAP and importance aren't causation |
| Blame the data ("the data was bad") | Sounds like excuses; say what you did about it |
| Give a 5-minute answer to a yes/no question | Shows poor communication |
| Freeze when you don't know | See the next section |

---

## 6. When you don't know the answer

Never bluff. Interviewers respect clear reasoning more than a memorised fact.

```
 1. Say it plainly:      "I haven't used that directly."
 2. Reason out loud:     "But I'd expect it to work like X, because..."
 3. Connect to your work: "In my project, the closest thing was Y, where I..."
 4. Say how you'd check: "I'd verify by testing Z on validation."
```

**Example:** *"Have you used time-series cross-validation?"*
> "Not in this project. I used a single time-based validation period. But I'd use an expanding window: train on the first months, validate on the next one, then grow the window. It gives several validation scores instead of one, so I'd know how stable 0.94 really is."

---

## 7. Explaining it on a whiteboard

Practise drawing this in under 2 minutes, talking while you draw:

```
   [ raw data ] --> [ checks ] --> [ 9 features, past only ]
                                          |
                     +--------------------+--------------------+
                     |                    |                    |
                 [ TRAIN ]          [ VALIDATION ]          [ TEST ]
                 learn              choose model,           once, at end
                                    cutoff                   -> 0.906
                     |
                [ XGBoost ] --> score --> [ cutoff 0.90 ] --> alert + reasons
                                                    |
                                       [ fairness + monitoring ]
```

**Tip:** draw the **time arrow** under train, validation and test. It shows you understand leakage without saying a word.

---

## 8. Questions to ask the interviewer at the end

Asking good questions shows you think like a team member:

1. "How do your teams decide alert cutoffs: is it capacity, cost, or something else?"
2. "How quickly do confirmed labels come back, and how do you monitor models in the meantime?"
3. "How do you handle training-serving skew for real-time features?"
4. "What does a fairness review look like before a model goes live here?"
5. "What does success look like for someone in this role after six months?"

---

## 9. Practice plan

| Day | Practice |
|---|---|
| 1 | Read file 01. Say the **30-second pitch** out loud 5 times **without looking**. Record yourself once. |
| 2 | Say the **2-minute version** without looking. Check that you used at least 5 numbers. |
| 3 | File 02, sections A–E. Cover the answer, say yours, then compare. |
| 4 | File 02, sections F–I. Same method. |
| 5 | File 02, sections J–M. Same method. |
| 6 | Draw the **whiteboard diagram** from memory. Explain the notebooks while scrolling through them. |
| 6b | Files **04 and 05**: the pipeline and the live API. Open the live `/docs` page and **demo a prediction** while explaining each step (the strongest thing you can show). |
| 7 | **Mock interview:** someone asks random questions and cross-questions. Note every hesitation and revise those parts. |

**The self-test:** pick any cell in any notebook and explain **what it does, why it's there, and what would go wrong without it**. If you can do that for every cell, you're ready.

---

## 10. The day before

- [ ] Can I say the 30-second pitch without notes?
- [ ] Do I know the 10 key numbers (file 01, section 4)?
- [ ] Can I explain leakage, the time split, PR-AUC vs ROC-AUC, and the cutoff choice in plain words?
- [ ] Can I tell one mistake story with STAR?
- [ ] Can I list 3 honest limitations?
- [ ] Is the repository clean: README, notebooks run top to bottom, `requirements.txt` up to date?
- [ ] Do I have 2–3 questions ready for the interviewer?

> **Remember:** you aren't trying to look perfect. You're showing that you can be **trusted with a real problem**: careful, honest, clear and focused on the business. That's what stands out.
