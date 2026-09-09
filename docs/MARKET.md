# Facet: market, positioning and go-to-market

Research date: 7 September 2026. Every figure below is sourced; where I am estimating rather
than citing, it says so.

This document answers the brief — market, users, selling points, ads, revenue — but it does
not answer it in the order it was asked, because the research turned up one finding that
reorders everything else.

---

## 0. The finding that comes first

**"Rank your friends for fun" cannot ship on iOS, and is the single most legally exposed
thing this codebase could be used for.**

Apple's App Review Guidelines, §1.2:

> Apps with user-generated content or services that end up being used primarily for
> pornographic content, Chatroulette-style experiences, random or anonymous chat,
> **objectification of real people (e.g. "hot-or-not" voting)**, making physical threats, or
> bullying do not belong on the App Store and may be removed without notice.

That is not a grey area or an appeal you win. It names the mechanic.

The legal exposure is worse than the distribution problem. Under Illinois BIPA, consent must
come from **the person whose face is scanned**, not the person holding the phone. Uploading a
photo of a friend collects *their* biometric identifier without *their* written release.
BIPA has statutory damages with no injury requirement (*Rosenbach v. Six Flags*, 2019), and
the settlements are not symbolic: Meta $650M, Google Photos $100M, Clearview AI $51.75M, with
107 new class actions filed in Illinois in 2025 alone. Texas CUBI and Washington's HB 1493
are comparable; GDPR Art. 9 treats face embeddings as special-category data requiring
explicit consent from the data subject.

So the constraint the brief already stated — *the product must not offend people* — turns out
to be the same constraint as *the product must be shippable*. That is unusually convenient.
The rest of this document treats them as one requirement.

**The rule that follows: never rate a person who did not ask to be rated.** Everything below
is built on it.

---

## 1. The market that exists

Three adjacent markets, with very different sizes and very different risk.

### 1.1 Self-rating ("looksmaxxing") — proven, crowded, reputationally toxic

| | |
|---|---|
| **Reference** | Umax / LooksMax AI |
| **Scale** | 3.5M+ downloads, ~$6M annual revenue (Apptopia); later reporting puts it past 7M downloads |
| **Price** | $3.99/week — roughly $200/year |
| **Channel** | TikTok, almost entirely organic at launch |
| **Why it is allowed** | It rates *your own* face. Self-improvement, not objectification of a third party. |

This market is real and it converts. It is also under sustained clinical criticism: a 2025
review in *Facial Plastic Surgery & Aesthetic Medicine* found looksmaxxing "reduces beauty to
mathematical symmetry" and cultivates a body schema resembling body dysmorphic disorder;
Michigan Medicine, PBS and Baylor have all published on its links to BDD and eating-disorder
symptomology in teenage boys. Platform policy has not caught up yet, which is a description
of a gap that is going to close, not a durable advantage.

**Verdict:** the money is real, the category is a reputational liability, and entering it
means owning that. There is a defensible way in — §3.2 — but it is a narrow one.

### 1.2 Consensual social ranking — the pattern that actually worked

| | |
|---|---|
| **Reference** | tbh (acquired by Facebook, ~$100M, 2017); Gas (acquired by Discord, 2023) |
| **Scale** | Gas: #1 on the US App Store Nov 2022, 7.4M downloads, ~$7M in consumer spend in under a year |
| **Mechanic** | Anonymous **polls among friends**, with **positive-only options** |

This is the closest existing product to "a ranking for fun among friends", and the reason it
survived App Review while hot-or-not apps do not is the design, not the lawyers: nobody
receives a score, everybody in the poll opted in by joining, and every possible outcome is a
compliment. Gas's founder has said the constraint was deliberate.

Note also that the exit was the business model. Neither company built a durable subscription;
both were acquired for the network.

### 1.3 Consented photo selection — small, boring, and the one with real willingness to pay

| | |
|---|---|
| **References** | Photofeeler (human panel + Photofeeler-D3 model), Roast, Tinder's built-in Photo Selector (2024, US then international) |
| **Demand signal** | Tinder's own survey: 52% of singles find it hard to pick a profile photo; 68% said an AI photo-selection feature would help |

Everyone here rates photos **you own, of yourself, that you asked to have rated**. No consent
problem, no App Review problem. The market is smaller and less viral — and it is the only one
of the three where a paying customer has a concrete outcome they can attribute to the product.

---

## 2. What Facet actually has that the market does not

This matters more than the category choice, so it goes before the product options.

| Asset | Why competitors do not have it | Evidence |
|---|---|---|
| **A per-user taste model** | Umax and every clone output one universal number. Facet fits a personal model over frozen embeddings in milliseconds and blends it with the population prior (E14's residual formulation). | `models/preference.py`; alpha grows with labels, capped at 0.85 |
| **Honest uncertainty** | The category sells false precision — "your PSL score is 6.2". Facet reports a calibrated interval, separates rater disagreement from model disagreement, and suppresses the number entirely when the face is out-of-distribution. | E12; conformal coverage 0.92 in-domain |
| **A published bias measurement** | Nobody in this category publishes one. | E11: White faces selected into the top 100 at 2.2× their share, Southeast Asian at 0.23× |
| **Ranking a whole library, not scoring one selfie** | Different product class. Weighted multi-criteria search over ~12k faces with per-result explanation of the arithmetic. | `query/engine.py`, `docs/QUERY.md` |
| **Runs locally / self-hostable** | Privacy as an architectural fact, not a policy page. A local install makes zero outbound requests. | `docs/HOSTING.md` §1 |
| **A research record that contradicts its own priors** | Eleven experiments, several of which falsified the hypothesis they tested. | `docs/RESEARCH.md` |

**The single sharpest selling point, and the one to build the brand on:**

> The dataset behind every attractiveness model in this category has an inter-rater
> correlation of **0.77**. The models report 0.93. That gap is not accuracy — it is the model
> learning to predict the *average of a crowd* better than the crowd predicts itself. A score
> that claims more precision than the raters had is not a measurement. It is a rounding error
> with a marketing department.
>
> Facet is the only one that says so, and the only one that then learns what *you* think
> instead.

That is a genuinely differentiated, genuinely true claim, and it converts the biggest
technical liability in the project into the reason to choose it.

---

## 3. Three products, ranked

### 3.1 **Recommended: "Which of these should I use?" — consented photo triage**

**What it is.** Upload a batch of photos *of yourself* (or that you have rights to). Facet
ranks them, explains why, and learns which look you actually want to lead with. The
personalisation is the product: two users with the same 200 photos get different orders.

**Who buys it.**

| Segment | Job to be done | Willingness to pay |
|---|---|---|
| Dating-app users | "I have 300 camera-roll photos and six slots" | Proven — Tinder built this in-house because demand was there |
| Creators / models | Pick the shot for the post, the portfolio, the comp card | High; time is money |
| Casting, modelling and talent agencies | Triage 2,000 submissions against a brief (age band, presentation, photo quality) | Highest — this is Facet's existing multi-criteria search, unchanged |
| Photographers | Cull a shoot: sharpest, best-lit, best-expression frames per subject | Medium; adjacent to existing culling tools |

**Why it wins.** It is the only option where the technology is used at full strength, the
consent story is clean, and there is a buyer with a budget. The B2B casting/agency line is
the one that could carry a real business — it is a search-and-filter problem over a
proprietary library, which is exactly what is already built.

**Risk:** less viral. Accept that; buy the growth in §6 or sell it directly in §5.

### 3.2 **Second: self-analysis, but honest**

Enter the Umax market as the anti-Umax. Same input (a selfie), opposite output: a range not a
number, a stated confidence, an explicit "this model is predicting one 2017 rating panel of
60 people aged 18–27, not the truth about your face", and the measured demographic skew shown
in-product rather than buried.

**Why it might work:** the category's own users increasingly distrust it. "The face rating app
that tells you the score is meaningless" is a real position, and the clinical criticism in
§1.1 is free air cover for the brand that takes it.

**Why it might not:** you are still monetising insecurity, and a more honest product in a
harmful category is still in the category. If the answer to "would I be comfortable if a
journalist covering teen BDD found this" is no, do not ship it.

### 3.3 **Third: the friends game, restructured so it is legal and kind**

The original idea survives if the mechanic changes. Concretely:

* **Everyone uploads their own photo.** Nobody is entered by somebody else. This alone fixes
  BIPA, GDPR and App Review §1.2.
* **No scores, no ordering of people.** Output *superlatives*, drawn from the taste model:
  "the group agrees you photograph best in candids", "your friends' taste is more similar to
  Priya's than anyone else's".
* **The comparison is between tastes, not between faces.** This is the genuinely novel thing
  Facet can do that Gas could not: a taste-similarity graph across a group. "You and Sam
  agree 81% of the time. You and Alex agree 12%." That is a fun, shareable, harmless artefact
  that no competitor can produce because none of them model individual taste.
* Positive-only outcomes, as tbh and Gas both proved is necessary.

**This is a good product.** It keeps the social energy the brief wanted, drops the part that
gets you removed and sued, and it is differentiated by the same asset as §3.1.

---

## 4. Positioning

**Name.** `Facet` is good and I would keep it: faceted search, facets of a face, and it
carries no claim about beauty. If a consumer app needs a warmer name, the taste angle gives
you `Taste`, `Bias` (self-aware, risky), or `Squint`.

**One-liner, by audience:**

* Consumer (§3.1): *"Your taste, not a score."*
* B2B (casting/agency): *"Search 5,000 headshots the way you'd brief a casting director."*
* Technical / privacy-led: *"Face ranking that runs on your machine and admits what it
  doesn't know."*

**Positioning statement.** For people who have more photos than they can judge, Facet ranks
them by criteria you set and taste it learns from you — unlike face-rating apps, which output
one number from one 2017 rating panel and present it as a fact.

**Three claims, in priority order.** Each is defensible with an artefact already in the repo:

1. **It learns you.** 5 likes and it is already re-ranking; ~100 and it is mostly your model.
2. **It shows its work.** Every result explains the arithmetic that ranked it.
3. **It admits doubt.** Intervals, not scores; and it goes quiet on faces it does not
   understand.

**What never to say:** "objective beauty", "accurate", "your rating", any single number
without a range, and any comparison between two named people.

---

## 5. Revenue

### 5.1 Model

**Recommended: freemium web app with a hard paywall on the second library, plus a B2B tier.**

| Tier | Price | What it gates |
|---|---|---|
| Free | £0 | 100 photos, population ranking, 20 taste labels |
| Plus | **£6.99/mo or £49/yr** | Unlimited library, full taste model, export, saved searches |
| Studio (B2B) | **£99–£399/mo per seat** | Bulk import, API, shared briefs, team libraries, audit export |
| Self-host | £0 / source-available | The whole thing, on your own box. A funnel, not a loss. |

Deliberately **not** copying Umax's $3.99/week. Weekly plans are 55.6% of subscription revenue
in the graphics-and-design category, so it works — but weekly pricing on a self-esteem product
is the mechanic the critics in §1.1 are pointing at, and it is inconsistent with the honesty
positioning. Annual with a real trial is the on-brand choice and it costs you less in refunds.

### 5.2 What the benchmarks say to expect

From RevenueCat's and Adapty's 2026 subscription data:

* Healthy subscription apps convert **15–30% of installs into trials** and **40–65% of trials
  into paid**.
* **Hard paywall: 10.7% median D35 trial-to-paid. Freemium: 2.1%.** Five times the difference.
* **55% of trial cancellations happen on day zero** — the first session is the entire game.
* AI apps monetise at **2× pre-AI ARPU** but retain ~20% worse; LTV still comes out ahead.
* Photo & video has the *lowest* trial-to-paid rate of any category. Assume you are on the
  wrong side of that and price accordingly.

### 5.3 A unit-economics model to argue with

Illustrative, not a forecast. Consumer (§3.1), UK/US, paid acquisition:

```
CPI (TikTok, UGC creative)                          $2.00      §6, benchmark $0.70–4.50
Install → trial                        20%      →   $10.00 per trial
Trial → paid (hard paywall)            11%      →   $90.91 per subscriber (CAC)
Price                                  £49/yr   ≈   $62
Gross margin (inference is cheap; encode once)  85%  →  $52.70/yr
Year-1 retention                       55%
LTV (2.1 yr average life)                       ≈   $110
LTV : CAC                                       ≈   1.2 : 1
```

**That does not work.** Paid acquisition alone does not fund this at a $49 price point, which
is exactly why every app in §1.1 grew on organic TikTok instead. Two levers move it:

* **Organic-first** (§6.1) takes CPI toward $0 and the ratio to ~4:1.
* **B2B** at £199/mo with a 6-seat account is $14k/yr against a sales cost of maybe $2k. That
  is where the business is, if there is one.

Be honest about the third possibility: **consumer here is a marketing channel for the B2B
product, not a business.**

---

## 6. Growth and advertising

### 6.1 Channel priority

1. **Organic short-form video (TikTok, Reels, Shorts).** Non-negotiable. Umax and Gas both got
   to millions of downloads on it with near-zero spend, and it is the only channel where this
   product's demo is inherently watchable.
2. **Paid amplification of what already worked organically.** UGC-style creative cuts CPI by
   **25–40%** versus polished creative. Do not commission an agency film.
3. **Earned/press on the honesty angle.** §2's inter-rater-correlation claim is a story, and
   the looksmaxxing backlash means journalists are actively looking for a counterweight.
4. **Communities**: r/photography and r/datingoverthirty for §3.1 consumer; casting-director
   and modelling-agency Slack/LinkedIn groups for Studio.
5. **Self-host / open source as top-of-funnel.** Hacker News and r/selfhosted respond well to
   "here is the research document and here are the results that contradicted my hypothesis".

**Operating constraints from the data:** UGC creative fatigues in **~7.6 days** — plan a
rotation, not a campaign. Ad groups running **5–7 creatives beat those running fewer than 3 by
22% on CPI**. TikTok wants 9:16, 15–21s; Meta Reels wants 20–30s in both 4:5 and 9:16.

### 6.2 Creative concepts

Six, in rough order of expected performance. All avoid showing a real person being scored.

---

**1. "The model is lying to you" — 18s, TikTok/Reels.**
Hook (0–2s): *"Every face-rating app gives you a number. Here's why the number is fake."*
Screen recording: the same face scored by the population model — a wide interval appears.
Text on screen: `raters agreed with each other 77% of the time. the model claims 93%.`
Turn: *"It's not measuring your face. It's guessing what a crowd would say — and the crowd
didn't agree either."*
CTA: *"Facet shows the range. Link in bio."*
Why it works: attacks the category leader without naming it, and the claim is true and
citable.

---

**2. "Teach it your type" — 21s, the core demo.**
Screen record of the grid. Five taps on ♥. The grid visibly re-orders. Counter in the corner:
`your taste: 0% → 17% → 38%`.
VO: *"I didn't tell it what I like. I just showed it, five times."*
End card: `it learns you. it doesn't rate you.`
Why it works: this is the actual product, it is visually legible in under 20 seconds, and no
competitor can copy the footage.

---

**3. "Taste twins" — 15s, the shareable one (product feature ships with the ad).**
Two friends each rate 20 photos. Output: a similarity score and one line — *"You and Maya
agree 81% of the time. You and Josh: 12%."*
Why it works: it is a *result about a friendship*, not about a face. Infinitely postable, zero
consent problem. **Build this feature specifically because the ad works.**

---

**4. "300 photos, 6 slots" — 20s, the dating-app cut.**
Camera roll scroll (fast, chaotic) → drop into Facet → six photos surface with one-line
reasons: *sharpest · you're actually laughing · nobody else in frame*.
CTA: *"Stop asking the group chat."*
Why it works: names a specific, felt, weekly problem with a measurable outcome.

---

**5. "What it refuses to answer" — 12s, trust builder.**
An out-of-distribution face. The score is replaced with `not confident enough to say`.
VO: *"Every other app would have given you a 6.4."*
Why it works: restraint as a feature is genuinely novel here and it screenshots well.

---

**6. "It's biased and we measured it" — 25s, the risky one.**
The E11 chart on screen. VO: *"This model over-picks white faces by 2.2×. We measured it,
published it, and put it in the app. Everyone else's model does this too — they just haven't
looked."*
Why it works: enormous trust yield, real press potential.
**Caveat:** this invites the headline *"Face app admits it's racist"*. Have the full research
page live before it runs, and do not run it as the first creative.

---

### 6.3 Static and post formats

* **Carousel: "Five things a beauty score can't know about you."** Ends on the interval.
* **Screenshot-native**: a single result card with its explanation panel. The explanation *is*
  the ad — no competitor has one to show.
* **The research document as a post.** "I ran 11 experiments to build a face-ranking app. Six
  of them proved me wrong." Long-form on LinkedIn/X/HN. This is the highest-credibility asset
  in the project and it is already written.
* **Comparison table** vs the category: number-vs-range, universal-vs-personal, bias
  published Y/N, runs locally Y/N. Do not name competitors.

### 6.4 Design direction

The whole category is dark, neon, aggressive, "PSL 6.4" in a Impact-adjacent font, arrows
pointing at jawlines. Go the exact opposite way, because the positioning is honesty:

* **Light, calm, editorial.** Generous whitespace, one accent colour, real typographic
  hierarchy. The current light theme is right; push it further.
* **Show ranges, never single numbers,** in every marketing surface. The interval bar should
  be as recognisable as a brand mark.
* **Never show a face with a number stamped on it.** Show the *interface* — the grid, the
  explanation panel, the taste meter.
* **Photography**: hands holding a phone, contact sheets, a photographer's cull — the language
  of *choosing between photos*, not of *judging a person*.

---

## 7. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Apple removes the app under §1.2 | **Fatal to iOS** | Never rate a third party. §3.1/§3.3 mechanics are compliant by construction; ship web-first regardless |
| BIPA / CUBI / GDPR Art. 9 claim | **Existential** — statutory damages, no injury needed | Self-photos only; explicit consent at upload; per-account isolation (already built); self-service export and deletion (already built); geofence Illinois/Texas until counsel signs off |
| EU AI Act | High | Art. 5(1)(g) prohibits biometric categorisation inferring race, ethnicity, political opinion, religion, sex life or orientation. Facet infers age and gender presentation — adjacent, not on the list, but do not add ethnicity inference under any framing. High-risk obligations landed August 2026; get an opinion before an EU launch |
| Press coverage tying the product to teen BDD | High | §3.1 positioning avoids it entirely; §3.2 does not. 18+ gate; no streaks, no daily score, no push notification about your face |
| A minor uses it | High | 18+, enforced at signup; no Kids-category adjacency; honest App Store age rating |
| Someone uploads a photo of a person who did not consent | Certain — it will happen | Consent copy at the upload point (shipped); reporting route; fast takedown; retention limits |
| The model's demographic skew becomes the story | Medium | It is already published. Own it first, in-product, before someone else finds it |
| Category commoditisation | Medium | The taste model is the moat, and it compounds per user — a user with 200 labels cannot switch cheaply |

**One thing to fix before any launch:** the attractiveness head is trained on SCUT-FBP5500,
which is **non-commercial research use only** (`docs/LICENSING.md`). Charging money for its
output is a licence breach. Retraining the head on a commercially-licensed or
self-collected-with-consent dataset is a prerequisite for revenue, not a nice-to-have. Budget
for it.

---

## 8. First 90 days

**Days 1–30 — decide and de-risk.**
Pick one of §3.1/§3.2/§3.3 (recommendation: 3.1, with 3.3's taste-similarity as the viral
feature). Get a licensing opinion on SCUT and start the replacement dataset. Ship the 18+
gate, consent copy and a privacy notice naming a controller. Build "taste twins" — it is one
endpoint over machinery that already exists.

**Days 31–60 — prove the hook.**
Post 20 organic short-form videos across the six concepts in §6.2. No spend. The only metric
that matters is watch-through past three seconds; two of the six will carry everything. In
parallel, ten customer-development calls with casting directors and agency bookers — that
conversation decides whether Studio is a real product or a slide.

**Days 61–90 — commit.**
Paid amplification only behind organic winners, 5–7 creatives per ad group, rotated weekly.
Instrument the funnel against §5.2's benchmarks. Decide at day 90 on evidence: consumer
subscription, B2B, or open-source-plus-consulting.

---

## 9. Other things worth doing

Ordered by expected value, engineering and product mixed.

1. **Taste-similarity between accounts.** One cosine similarity between two preference
   vectors, gated behind mutual consent. It is the viral feature, the §6.2/3 ad, and a
   genuine research result — nobody has published what taste agreement looks like at scale.
2. **Retrain the attractiveness head on a licensed dataset.** Blocks revenue. See §7.
3. **"Why this photo" for a single subject.** Given 40 photos of one person, say which
   *photograph* is better rather than which *face* — expression, sharpness, crop, gaze. It
   moves the claim from a person to an image, which is both more useful and immune to every
   risk in §7. This is probably the best product idea in the document.
4. **A public research page.** `docs/RESEARCH.md` is a marketing asset that is already
   written. Publish it.
5. **Per-user model export.** "Your taste model, as a file you own." Strong privacy story,
   trivial to build (`PreferenceModel.save` exists).
6. **Cohort calibration.** E7 showed absolute scores do not transfer between collections.
   Report percentiles within *the library being searched* — already done — and say so
   loudly, because it is the correct behaviour and the competition gets it wrong.
7. **On-device inference for the free tier.** ArcFace + a linear head is small. It would make
   "your photos never leave your phone" literally true and remove the biggest privacy
   objection at a stroke.
8. **Team/brief mode for Studio.** Shared libraries, a written brief that maps to a
   `QuerySpec`, and an audit trail of who shortlisted whom. This is the B2B product.
9. **Kill the single number in the UI entirely.** Show the interval and the percentile, never
   the mean. It is the positioning made structural, and it costs one line in `card()`.

---

## Sources

- Apple, *App Review Guidelines* §1.2 — https://developer.apple.com/app-store/review/guidelines/
- Apptopia, *LooksMax AI / Umax Face Rating* — https://apptopia.com/ios/app/6498628699/about
- *The Conversation*, "Looksmaxxing isn't just a TikTok trend" — https://theconversation.com/looksmaxxing-isnt-just-a-tiktok-trend-it-often-reflects-severe-body-image-issues-in-teen-boys-and-young-men-280567
- Michigan Medicine Health Lab, looksmaxxing and body image — https://www.michiganmedicine.org/health-lab/looksmaxxing-isnt-just-tiktok-trend-it-often-reflects-severe-body-image-issues-boys-and-young-men
- PBS NewsHour, looksmaxxing and mental health — https://www.pbs.org/newshour/health/looksmaxxing-may-point-to-deeper-body-image-issues-in-young-men-mental-health-expert-says
- Baylor College of Medicine, "How looksmaxxing breeds body dysmorphia" — https://blogs.bcm.edu/2026/06/04/how-looksmaxxing-breeds-body-dysmorphia/
- American Bar Association, "Historic Biometric Privacy Suit Settles for $650 Million" — https://www.americanbar.org/groups/business_law/resources/business-law-today/2021-february/historic-biometric-privacy-settlement/
- Epstein Becker Green, "Biometric Backlash: The Rising Wave of Litigation Under BIPA and Beyond" — https://www.commerciallitigationupdate.com/biometric-backlash-the-rising-wave-of-litigation-under-bipa-and-beyond
- European Commission, *AI Act* regulatory framework — https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- Deloitte, "Unacceptable AI practices: analysis of the EU AI Act prohibitions" — https://www.deloitte.com/lt/en/services/legal/perspectives/unacceptable-ai-practices-comprehensive-analysis-eus-ai-act-prohibitions.html
- Business Today, "This social media app for giving compliments is going viral among teens" (Gas) — https://www.businesstoday.in/technology/news/story/this-social-media-app-for-giving-compliments-is-going-viral-among-teens-366815-2023-01-20
- CBC *The Current*, "This app allows teens to compliment each other anonymously" — https://www.cbc.ca/radio/thecurrent/gas-app-teens-high-school-1.6718229
- Tinder press room, "Tinder Unveils 'Photo Selector' AI" — https://www.tinderpressroom.com/Tinder-R-Unveils-Photo-Selector-AI-Feature-to-Make-Choosing-Profile-Pictures-Easier
- Photofeeler, "AI Trained on 100 Million Opinions" — https://blog.photofeeler.com/photofeeler-d3/
- RevenueCat, *State of Subscription Apps 2026* — https://www.revenuecat.com/blog/growth/subscription-app-trends-benchmarks-2026
- Business of Apps, *App Subscription Trial Benchmarks (2026)* — https://www.businessofapps.com/data/app-subscription-trial-benchmarks/
- Adapty, *Graphics & Design subscription benchmarks 2026* — https://adapty.io/blog/graphics-design-app-subscription-benchmarks/
- Admiral Media, *Mobile App Marketing Benchmarks 2026* — https://admiral.media/mobile-app-marketing-benchmarks-2026/
- Stackmatix, *TikTok UGC Ads Strategy 2026* — https://www.stackmatix.com/blog/tiktok-ugc-ads-strategy
