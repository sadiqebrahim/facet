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
