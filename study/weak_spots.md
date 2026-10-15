# Weak spots: my real study guide

Start every session with the open questions. Answer them out loud in my own words, without notes.
Mark a question ✅ only when I can answer it cleanly twice in a row.

## Open (carried over from the build phase)
- [ ] **Guardrails:** what is the ONE situation where a guardrail changes the decision? (Hint: what does the model say first?)
- [ ] **Running totals vs full history:** explain in 2–3 sentences, with the bank-balance analogy and the numbers (411 MB → 9 MB).
- [ ] **Parity test:** what does it compare, and why is it the proof against training-serving skew?
- [ ] **DVC caching:** how does `dvc repro` decide what to rerun? What happened when I changed the cutoff to 0.85 and back?
- [ ] **Docker secrets:** why is `.env` never copied into the image? (my own words, still owed to my mentor)
- [ ] **Multicollinearity:** logistic regression gave `amt` a weight of −3.06. Does a bigger amount mean safer? Why not?
- [ ] **YAML:** why is `"Unnamed: 0"` in quotes, and why `yaml.safe_load` instead of `yaml.load`?
- [ ] **Python structure:** why did `initiate_data_ingestion` not belong to the class in my first version? Why must `return` be inside the `try`?
- [ ] **Score meaning:** why isn't a fraud score of 0.72 a 72% chance of fraud?

## Answered well ✅
(move questions here)
