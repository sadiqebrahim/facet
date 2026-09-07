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

**Outbound requests are enumerable, and there are none by default.** A local install still
makes zero: images are read from disk and served to a local browser, and `serve.py` binds to
`127.0.0.1`. Hosting adds exactly two, both opt-in and both named in `api/app.py`'s imports —
Google's token endpoint during sign-in (`FACET_GOOGLE_CLIENT_ID`), and fetching an image a
user pasted a URL for. Face embeddings are biometric data (`LICENSING.md §4`), so this stays
a property you can check rather than a promise. See [`HOSTING.md`](HOSTING.md).

**Accounts do not share data.** Every image row carries an `owner` and every read path joins
through it — search, media, face detail, statistics, saved searches. Administrators are not
exempt; there is no endpoint that opens somebody else's library. `HOSTING.md` §2 is the whole
mechanism in one page.

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

## Accounts

With **no accounts**, the app runs open under a shared profile — a fresh local install works
without a login step. The moment anyone registers, **authentication becomes mandatory**,
because from then on there are separate taste models to keep apart. The first account
inherits whatever the open profile had already learned, so nothing is lost.

Each account keeps its own likes, rejects and reference faces. That is not a nicety: E14's
result is that preference is personal, so sharing one bucket between people would blend
incompatible tastes and quietly degrade everyone's ranking. **The server overrides any
client-supplied user** on every request — a browser cannot read another account's profile by
naming it in the request body, and there is a test asserting exactly that.

Passwords are PBKDF2-SHA256 with a per-user salt and a constant-time comparison; sessions are
random 256-bit tokens with a 30-day expiry.

**Authentication is enforced server-side, not by the UI.** The sign-in panel is only
decoration — deleting it in devtools reveals an empty shell, because every endpoint that
returns face data, imagery, predictions or statistics requires a valid session. Media
(`/api/crop`, `/api/image`) additionally accepts the token as `?t=` because an `<img>` tag
cannot set an `Authorization` header; it is the same token with the same lifetime, not a
weaker side door. Only `/api/about` and `/api/auth/*` are reachable unauthenticated.

**There is no TLS.** On an untrusted network, passwords and tokens cross the wire in the
clear. Put it behind a VPN or an HTTPS reverse proxy — which is what `serve.py` prints when
you bind beyond localhost.

### Roles, resets and the admin panel

The **first account created becomes the administrator**. There is no mail server here, so
"forgot password" cannot send a link — instead the login screen's *Forgot your password?*
names the administrator(s) to ask, and the admin panel lets them set a new one. A password
the admin resets is marked *must change*, so the user picks their own at next sign-in, and
**all of that user's live sessions are invalidated** — a reset the user did not perform
themselves should not leave a session behind.

The Admin tab (visible only to admins) lists every account with how much each has taught its
model, and allows: create user, reset password, promote/demote, delete. Guard rails prevent
deleting your own account or removing the last administrator. **Deleting an account also
deletes its likes, rejects and reference faces** — `LICENSING.md §4.2` requires that deletion
actually deletes.

Password fields have a reveal toggle, since a typo in a masked field is the most common
reason a correct password appears to fail.

## Reaching it from another device

```bash
python scripts/serve.py --host 0.0.0.0 --index facet.db --features feats/
```

It prints every address the machine is reachable on, so a VPN interface is easy to pick out,
and warns before doing anything risky:

```
  Binding to 0.0.0.0 — reachable from other machines on this network.
  Face embeddings are biometric data (docs/LICENSING.md §4).
  ⚠ NO ACCOUNTS EXIST, so the app is in open mode: anyone who can reach
    this port gets full access. Create an account in the UI to require a login.
  There is no TLS here: on an untrusted network, passwords and session
  tokens cross the wire in the clear. Use a VPN or an HTTPS proxy.
```

**Create an account before binding to `0.0.0.0`.**

## Undo, and when judgements take effect

A like or reject does two different things at two different times:

| | when |
|---|---|
| the face leaves your results | **immediately** — the point of rejecting is not to see it again |
| the preference model learns from it | **after the undo window** (default 10 s, adjustable 0–60 s) |

That split is the whole design. Undoing inside the window means the model **never saw** the
judgement, so there is nothing to unlearn — as opposed to teaching it something and then
trying to teach the opposite. The toast shows a countdown ring and an **Undo** button;
undoing restores the face to results at once. Set the window to 0 to commit immediately.

## Browser support

Tested against Chromium and Firefox feature sets. Range sliders carry both `-webkit-` and
`-moz-` pseudo-element rules — Firefox ignores the WebKit tree entirely and would otherwise
render default platform sliders — and scrollbar styling uses `scrollbar-width` /
`scrollbar-color` alongside `::-webkit-scrollbar`. `<dialog>`, `aspect-ratio`, `dvh` and
`backdrop-filter` are all supported in current Firefox and Chromium; the translucent overlays
keep an opaque-enough fallback colour where `backdrop-filter` is unavailable.

## Responsive behaviour

One layout, three tiers, and the important part is not the column count — it is that **every
hover-only affordance has a touch equivalent**. The favourite star and the ♥/✕ judge buttons
were originally revealed by `:hover`, which made them unreachable on a phone.

| width | layout |
|---|---|
| > 1180 px | persistent 288 px rail + fluid grid |
| ≤ 1180 px | narrower 250 px rail, denser grid |
| ≤ 820 px | rail becomes an **off-canvas drawer**: ☰ in the top bar, a floating **⚙ Filters** button, backdrop, Esc to close, and it closes itself after a search so you land on the results |
| ≤ 420 px | fixed 2-column grid, Export moves into the drawer |
| landscape phone | detail view returns to two columns so the image pane does not vanish |

Under `@media (hover:none)` every control grows to a ~40 px tap target — range thumbs, toggles,
buttons — and the card affordances become permanently visible.

Other mobile specifics:

- **Full-screen detail view** on small screens, stacked image-over-panel, with **swipe left/right**
  to move between results and a sticky action bar above the home indicator.
- `100dvh` rather than `100vh`, so the collapsing mobile URL bar does not clip the sheet.
- `viewport-fit=cover` plus `env(safe-area-inset-*)` for notches and home indicators.
- Zoom is **not** disabled — no `user-scalable=no`.
- `prefers-reduced-motion` honoured.
- Re-layout is debounced on resize, and the detail view re-fits on orientation change.

**Device-aware image sizing**, with a caveat worth stating because it is counter-intuitive: a
retina phone showing 2 columns needs *more* pixels per card than a 1× laptop showing 7, so
sizing to the device does not by itself reduce mobile bandwidth. Grid thumbnails cap DPR at
1.5 (they are small and heavily cropped, so the sharpness cost is marginal), which puts a
60-card page at ~0.8–0.9 MB on a phone instead of ~1.3 MB. The detail view uses full DPR,
where fidelity actually shows.

## Limitations

- Single-user, single-process; no auth, because it binds to localhost by design.
- Batch selection is not implemented.
- Sessions have no refresh; after 30 days you sign in again.
- Undo is a single-step, most-recent-action affair — there is no history stack.
- The UI was verified by static analysis (JS syntax, every referenced element id, every API
  path returning 200, balanced CSS at every breakpoint) and by exercising the API directly.
  **It has not been viewed in a real browser** — the Claude in Chrome extension was not
  connected during development, so there is no visual confirmation and no real-device testing
  of the responsive tiers.

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
