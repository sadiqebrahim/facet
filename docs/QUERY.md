# The query and ranking engine (Phase 9)

Turns an index into ranked, explained results.

```bash
python scripts/search.py --index facet.db \
    --age 25 35 --age-weight 0.3 \
    --gender female --gender-weight 0.2 \
    --attractiveness-percentile 0.8 --attractiveness-weight 0.5 \
    --limit 20 --explain

python scripts/search.py --index facet.db --query configs/query/example.yaml
```

## The score

```
relevance = Σ_c  w_c · match_c · confidence_c  /  Σ_c w_c
```

Multiplying match by confidence is the decision that makes it behave: an uncertain criterion
contributes little in either direction rather than a confident-looking wrong answer. It also
means every term in the sum is a row in the "why this result" panel — the ranking explains
itself rather than being taken on trust.

Nothing here is learned. `§9.1` deliberately separates *ranking faces by attractiveness*
(machine learning) from *ranking results by query match* (a transparent function), because a
learned end-to-end scorer could not tell a user why a result placed where it did.

Real output, from the brief's own example:

```
  #    rel   attr   pct   age   gender  qual  file
  1  0.777   3.87  0.99    29   female  0.49  008626.jpg
      age             match=1.000 x conf=0.653 x w=0.30 = 0.1960   (predicted 29, wanted 25-35)
      gender          match=0.997 x conf=0.997 x w=0.20 = 0.1988   (female (p=1.00), wanted female)
      attractiveness  match=0.973 x conf=0.786 x w=0.50 = 0.3823   (percentile 0.99, wanted top 20%)
```

## Three things the research phase changed

**Attractiveness is a percentile, not a threshold.** E7 found absolute scores do not survive a
domain change — two reasonable rater pools disagreed on 80 % of a top-100 — so
`attractiveness > 4.0` is not a meaningful request. `min_percentile: 0.8` is, and it is
label-free, which matters because a photo directory has no ground truth to calibrate against.
Percentiles are computed over the **whole collection**, not the filtered subset, so "top 20 %"
does not shift meaning when filters change.

**Gender is soft by default.** `§8`: a hard filter on a classifier at 0.96 (E4) discards true
matches unevenly across groups. `mode: strict` exists and *reports what it removed* — in a
3,000-image test it excluded 15 faces for low confidence, and said so.

**Unknown is not unmatched.** Age and gender come from a lazy pass (E4: MiVOLO is ~190× the
cost of everything else), so faces legitimately have no prediction. Those criteria contribute
**nothing** rather than scoring zero, and the count is surfaced. Treating "not yet predicted"
as "does not match" would be exactly the invisible failure `§13.2` warns about.

## Diagnostics are part of the answer

Every response reports what it did to the candidate set:

```
candidates_before_filters: 4181
missing_age: 815              ← lazy pass hasn't reached these
missing_gender: 815
collapsed_near_duplicates: 42
excluded_ood: 2349            ← when --exclude-ood is set
excluded_low_gender_confidence: 15
excluded_required_missing: 815
total_matched: 4139
```

A result set without these numbers would hide the difference between "your directory has few
matches" and "the pipeline dropped them".

## Uncertainty in the scoring

- **Age** — the shoulder widens with that face's predicted age uncertainty, taken from E4's
  measured per-bucket error profile (2.7 years at 20–29, 6.9 at 70+). A face the model is
  unsure about is not penalised as sharply for falling outside the requested range.
- **Attractiveness** — E12 suppressed confidence for out-of-distribution faces at predict
  time, so here they contribute on match alone rather than on a number we do not trust.
- **Gender** — the classifier's own probability is the confidence.

## Honest limitations

- **The demographic skew E11 measured is in these rankings.** On a balanced set the model
  over-selects White faces 2.2× and under-selects Southeast Asian faces 4.3×. Ranking across
  groups as if the scores were comparable is not supportable without disclosing that, and
  every response carries a note saying so.
- **Percentiles are relative to what you indexed.** "Top 20 %" of a directory of one person is
  not a meaningful statement.
- Age uncertainty uses a fixed measured profile rather than a per-face estimate; MiVOLO
  produces a point prediction and TTA-based spread (E12) is not yet wired into this pass.
- Sorting and scoring happen in Python over all matched candidates. Fine at the scale tested
  (4,181 faces, instant); a million-face index would want the filter pushed further into SQL
  and a top-k heap instead of a full sort.
