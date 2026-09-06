# The API and UI (Phases 10–11)

```bash
python scripts/serve.py --index facet.db --features feats/
# → http://127.0.0.1:8000   (UI)      http://127.0.0.1:8000/docs   (OpenAPI)
```

## Endpoints

| | | |
|---|---|---|
| `GET` | `/api/about` | name + the disclaimers every client must surface |
| `GET` | `/api/stats` | image/face/prediction counts, per-status breakdown, recent runs |
| `POST` | `/api/search` | a `QuerySpec`; returns ranked results, per-criterion explanations, diagnostics |
| `GET` | `/api/face/{id}` | full detail: all predictions, quality signals, duplicates |
| `GET` | `/api/image/{id}` | original image, scaled server-side |
| `GET` | `/api/crop/{face_id}` | face crop at a requested size |
| `POST` | `/api/index` | start a background indexing run |
| `GET` | `/api/index/status` | live progress; survives page reloads |
| `POST` | `/api/feedback` | like / dislike / hide / "prediction wrong" |
| `GET`/`POST`/`DELETE` | `/api/searches` | saved searches |
| `POST` | `/api/export?fmt=csv\|json` | export the current result set |

## Two properties the API enforces rather than documents

**Provenance travels with the numbers.** Every search response, every face detail and the
first two lines of every CSV export carry the SCUT-FBP5500 source string and the measured
demographic skew. A client cannot render a bare attractiveness score without also having the
caveats in hand. Tests assert this for both the JSON and the CSV path — a spreadsheet of
context-free numbers is precisely how an estimate turns into an apparent measurement.

**Nothing leaves the machine.** There is no outbound request anywhere in the API. Images are
read from local disk and served to a local browser, and `serve.py` binds to `127.0.0.1`,
warning if asked to do otherwise. Face embeddings are biometric data (`LICENSING.md §4`), so
local-only is the default rather than an option.

## The UI

One file, `web/index.html`, no build step. Vanilla JS against the API above.

- **Index tab** — add a directory, watch live progress, see per-status counts and error logs.
  Re-running is incremental and safe.
- **Search tab** — age range, gender, attractiveness percentile, each with an importance
  slider; quality floor, minimum face size, OOD and duplicate toggles.
- **Results** — grid of face crops with match %, estimate, age, gender, quality, and an
  out-of-distribution chip where it applies.
- **Detail** — original image with the detection box drawn, the full rating distribution as a
  bar chart, aleatoric vs. epistemic uncertainty separately, the conformal interval (or a
  clear "suppressed" marker), quality warnings, and **the arithmetic that produced the rank**.
- Favourites, saved searches, CSV export, dark/light theme, keyboard navigation
  (`←` `→` `F` `Esc`).

### Deliberate honesty in the interface

The research phase's conclusions are visible in the UI rather than buried in a doc:

- A standing banner states these are estimates, not measurements, and quotes the measured
  top-100 skew (White 2.2× over-selected, Southeast Asian 4.3× under).
- Attractiveness is labelled **"est. X / 5"** and the control is a **percentile of your
  collection**, with a note explaining why an absolute threshold is not meaningful (E7).
- Confidence reads **"suppressed"** for out-of-distribution faces rather than showing a
  number the model has not earned (E12).
- Faces awaiting the lazy age/gender pass show **"age —"** and the diagnostics line says how
  many — "not yet predicted" is never rendered as "does not match".
- Gender is described as **"presenting as"**, and strict filtering reports how many faces it
  excluded.

## Limitations

- Single-user, single-process; no auth, because it binds to localhost by design.
- Feedback is stored but **not yet used for ranking**. E14 showed personalisation only pays
  off with a diverse rater pool, and then only as a residual model gated on population fit;
  wiring that in is Phase 12.
- Batch selection and a full-screen viewer are not implemented.
- The UI was verified by static analysis (JS syntax, every referenced element id, every API
  path returning 200) and by exercising the API directly, not by visual regression testing.

---

# Personalisation — teaching it your taste

Beauty is subjective. The dataset behind the population model has an inter-rater correlation
of only **0.77**, and E7 found two reasonable rater pools disagreeing on **80 % of a top-100**.
So a single "attractiveness" ranking is always somebody's taste — by default, 60 Chinese
undergraduates in 2017. This lets it be yours instead.

## Two ways to teach it

| | How | When it helps |
|---|---|---|
| **Reference faces** | Point at an image or a folder of faces you find attractive | Works from **one** example — solves cold start |
| **Like / reject** | ♥ My type / ✕ Not for me on any result | Reflects real judgements on real candidates |

Both reduce to labelled feature vectors and train one model. Rating a face **re-ranks
immediately** — fitting takes milliseconds because it runs over the cached embeddings, which
is the payoff of the encode-once architecture.

## How it blends (and why it never takes over)

```
final_percentile = (1 − α) · population + α · your_taste
α = min(0.85, n / (n + 25)) × strength
```

E14 measured why this shape is necessary: training on a user's labels **alone** is
catastrophic in the cold-start regime (Spearman 0.21 vs 0.51 at 10 labels). So the population
model carries the ranking until you have taught it enough, α rises with your label count, and
it is **capped at 0.85** — your labels will always be far fewer than the training set's, and
E14's own failure case was a personal model confidently fitting noise.

The model also switches representation as it learns: **centroid similarity** below 3 examples
per class (a fitted linear model on 2 points is meaningless), then **ridge** once it can
actually separate your likes from your dislikes.

## Measured on 386 real faces

Teaching it the five faces the population model ranked **worst**:

| | Before | After |
|---|---|---|
| those 5 reference faces | positions 382–386 of 386 | **positions 80–260** |
| top results | population's favourites | faces scoring 2.84–3.15 population, **0.84–0.96 yours** |

Rejecting the population model's **top 8**:

| | Before | After |
|---|---|---|
| those 8 faces | positions **1–8** | positions **257–333** of 371 |

The model switched from centroid to ridge automatically at that point, and α rose to 0.34.

## In the UI

The **Your taste** panel shows a live bar of how much of the ranking is yours, the current
method, and how many labels it has. Every result card carries a **you N%** chip so you can see
where the personal model and the population model disagree, and the detail view spells out the
blend in the "why it ranked here" breakdown. **↺** forgets everything learned.

## Honest limits

- Personalisation shifts *ranking within what was indexed*. It cannot surface a face the
  detector missed, and it does not retrain the underlying encoder.
- E14 found gains of ~+0.025 Spearman at 100 labels on a diverse rater pool — real and
  consistent, but a refinement rather than a transformation. It does **not** undo the
  demographic skew measured in E11; it reweights, it does not repair.
- The personal model is per `user` string and stored in the same local index. There is no
  multi-user isolation beyond that key.
