# Hosting Facet

Facet began as a single-user tool on a laptop. Everything in this document exists because
"a tool on your own machine" and "a service strangers can reach" have almost nothing in
common, and several of the differences are not obvious until they bite.

Read section 2 even if you skip the rest. It is the one that stops your users seeing each
other's photographs.

---

## 1. Configuration

| Variable | Default | What it does |
|---|---|---|
| `FACET_PUBLIC_URL` | *(unset)* | The https origin users reach you on, e.g. `https://facet.example.com`. **Setting it switches the app into hosted mode**: open mode is disabled and every request needs a session. Also the base for Google's redirect URI. |
| `FACET_GOOGLE_CLIENT_ID` | *(unset)* | OAuth client. With `FACET_GOOGLE_CLIENT_SECRET` and `FACET_PUBLIC_URL` all set, the "Continue with Google" button appears. Any missing and it does not. |
| `FACET_GOOGLE_CLIENT_SECRET` | *(unset)* | |
| `FACET_REQUIRE_AUTH` | *(inferred)* | `1` forces a login even without a public URL; `0` allows open mode even with one. Leave unset unless you know why. |
| `FACET_ALLOW_SERVER_PATHS` | off | `1` re-enables indexing directories on the server by path. **Never set this on a shared host** — see section 3. |
| `FACET_UPLOAD_DIR` | `data/uploads` | Where uploaded originals live. One directory per account. |
| `FACET_USER_QUOTA_MB` | `2048` | Per-account storage ceiling, checked before each write. |
| `FACET_TRUST_PROXY` | off | `1` if a reverse proxy sets `X-Forwarded-For`. Off means the rate limiter keys on the socket address, which is correct when nothing is in front of you and wrong when something is. |
| `FACET_FORCE_CPU` | off | `1` to keep every model off the GPU. Set it when you share a card with a training job. |
| `FACET_SIGNUP_LIMIT` | `10` | Registrations per IP per hour. |

Minimum viable hosted deployment:

```bash
export FACET_PUBLIC_URL=https://facet.example.com
export FACET_UPLOAD_DIR=/var/lib/facet/uploads
python scripts/serve.py --host 127.0.0.1 --port 8000 \
  --index /var/lib/facet/facet.db --features /var/lib/facet/features
```

…behind a TLS-terminating reverse proxy. Bind to localhost and let the proxy be the only
thing on a public interface.

---

## 2. Tenancy: how accounts are kept apart

**Every image row carries an `owner`, and every read path joins through it.** That is the
whole mechanism, and it is deliberately boring — one predicate, applied in one place, with
no role that bypasses it.

* `images.owner` is set at index time by `Indexer.index_directories(owner=…)`. It is never
  applied retroactively, because a half-finished run that left rows unattributed would be
  a run whose images belong to nobody — or, far worse, to everybody.
* `SearchEngine.search` puts `i.owner = ?` first in its `WHERE` clause, before any filter.
  `QuerySpec.owner` is set by the API from the session and overwritten if a client sends
  one.
* Media routes (`/api/image`, `/api/crop`, `/api/preference/thumb`) and `/api/face` re-check
  ownership and answer **404, not 403**, for somebody else's id. A 403 would confirm that
  the id exists, which is itself a fact about their library.
* Percentiles are computed per owner. "Top 20%" is a claim about the collection being
  searched; pooling everyone's photos would make your threshold move when a stranger
  uploaded theirs.
* Feedback, reference faces and saved searches are all keyed by user. Two accounts may hold
  a saved search of the same name.
* **Administrators are not exempt.** `/api/admin/users` reports counts; there is no endpoint
  that lets an admin open another account's library. An operator with shell access can of
  course read the disk — no application-level control changes that — but nothing in the app
  offers it as a feature.

What is shared: the feature store (a flat file of vectors, addressable only through face
rows that carry an owner), the trained attractiveness head, and the SQLite file itself.

### Upgrading an existing index

`Index.__init__` migrates automatically. The `images` table is rebuilt, because the old
`UNIQUE(path)` constraint made a shared server one shared library — two accounts uploading
the same photo would have collided on one row.

* Row ids are preserved, so every `faces.image_id` still resolves.
* Pre-existing images are attributed to **the earliest administrator**, on the grounds that
  they are the account that ran the indexing. Assigning them to nobody would be safer in the
  abstract but would silently empty the library of the person who built it.
* A copy is written to `<index>.pre-owner-migration.bak` first. Keep it until you have
  looked at the result.

---

## 3. What is turned off, and why

**Indexing server directories** (`POST /api/index`) and **reference faces by server path**
(`POST /api/preference/references`) both return 403 unless `FACET_ALLOW_SERVER_PATHS=1`.

On one trusted box these were conveniences. Exposed to accounts you do not control they are
an arbitrary filesystem read for anyone who can register: `{"paths": ["/etc"]}` was a
directory listing, and `/api/index` would walk anything the process could open. No amount of
path validation fixes that, because the problem is not which path — it is that the caller
gets to name one at all.

The replacement is uploads, which cannot express a path: files are written under
`FACET_UPLOAD_DIR/<account>/` and named by the SHA-256 of their contents, so no
client-supplied filename ever reaches the filesystem.

**Open mode** — no accounts, one shared profile, no login — switches itself off when
`FACET_PUBLIC_URL` is set. It is a first-run convenience for a laptop, and on a host it
would hand a signed-in session to anyone who finds the URL.

---

## 4. Sign-in

Two routes, and the deployment picks which exist.

**Password.** PBKDF2-SHA256, 200k rounds, per-user salt; sessions are 256-bit random tokens
with a 30-day expiry. Registration and login are rate-limited (10 login attempts per
IP+username per 5 minutes; `FACET_SIGNUP_LIMIT` registrations per IP per hour). There is no
mail server, so "forgot password" means asking an administrator — the sign-in panel names
them.

**Google.** Standard OpenID Connect authorization-code flow. Notes on the parts that are
easy to get wrong:

* Matching is on `(provider, sub)` — Google's own immutable subject id — never on the email
  address alone. Emails get reassigned, and treating one as a key is how federated logins
  hand a stranger somebody else's account.
* An *existing password account* is linked when the verified email matches. An **unverified**
  email is discarded (`email_verified` false → empty string), because otherwise anybody who
  can set a Google profile address could claim an account.
* `state` is single-use with a 10-minute TTL and lives in memory, so a restart invalidates
  in-flight sign-ins rather than leaving them replayable.
* The ID token's signature is not re-verified locally. It is fetched by this server directly
  from Google's token endpoint over TLS in exchange for a code and the client secret, which
  OpenID Connect Core §3.1.3.7 rule 6 says is sufficient. `iss`, `aud`, `exp` and `nonce`
  *are* checked — those defend against a valid token issued for a different application
  being replayed here, which TLS does nothing about.
* The session token comes back in the URL **fragment**, not a query parameter. Browsers do
  not send fragments to servers or put them in `Referer`, so it stays out of access logs.

Configure the redirect URI in Google Cloud Console as exactly
`${FACET_PUBLIC_URL}/api/auth/google/callback`.

---

## 5. Request forgery, uploads and quotas

The URL importer is the only place in the app that fetches an address a user chose. It is
also the only place a request-forgery bug can live, so:

* http/https only.
* The hostname is resolved and every returned address is checked; private, loopback,
  link-local, multicast, reserved and unspecified ranges are refused. `169.254.169.254` —
  cloud instance metadata — is covered by link-local.
* Redirects are followed **by hand**, three at most, and every hop is re-validated. A
  permitted URL that 302s to `http://127.0.0.1:8000/api/admin/overview` would otherwise walk
  straight past the check.
* Responses are capped at 12MB and 12 seconds.

Known residual: there is a TOCTOU window between resolving the name here and the HTTP client
resolving it again, which a DNS-rebinding attacker could in principle use. Closing it
properly means pinning the socket to the validated address. It blocks every non-adversarial
mistake and the ordinary attacker; if you are hosting this somewhere with a sensitive
internal network, put the process on a segment that cannot reach it.

Uploads are validated by magic number *and* by actually decoding the image — an extension
and a `Content-Type` are both attacker-supplied and neither says what a file is. Per-file
cap 25MB, 200 files per request, per-account quota checked before the write.

---

## 6. Data rights

Face embeddings are biometric data (`docs/LICENSING.md` §4). A hosted service that cannot
answer "what do you hold on me" has no defensible answer to the question either, so:

* `GET /api/account/export` returns the account, library, judgements, reference faces and
  saved searches as one JSON document.
* `POST /api/account/delete` erases the account, the uploaded originals, every image and
  face row, the derived predictions, the feedback and the taste model. Both are reachable
  from the Account panel in the UI without asking anybody.
* Deleting an account through the admin panel does the same thing.

Embeddings themselves are not exported: they are large binary vectors, and they are deleted
with the face rows. Feature-store rows are left in place and become unaddressable once no
face row points at them — if you need them physically overwritten, compact the store.

---

## 7. Operating notes

**GPU sharing.** Every model stage asks whether the card has room before claiming it
(`models/device.py`) and falls back to the CPU rather than failing. The motivating incident:
an import died with `OutOfMemoryError: Tried to allocate 90.00 MiB` *after* detection,
encoding and scoring had all succeeded, because a co-tenant process held 42 of 48 GB. Stages
are also independent now — a failure in age estimation is reported as a warning and the run
still completes.

**Concurrency.** One import slot per account. A second upload while the first is still
running is queued and folded into the same job, not refused — by the time the request
arrives the bytes are already stored, so a 409 would leave them permanently unindexed.

**Backups.** One SQLite file plus the feature store directory plus the upload directory.
Take them together; a feature store without its index is unaddressable, and an index without
its uploads is a catalogue of missing files.

**Headers.** The app sets `X-Frame-Options: DENY`, a CSP with `frame-ancestors 'none'`,
`nosniff` and `no-referrer`. The framing rules matter most: every click in this UI is a
judgement recorded against a real person's photograph, and clickjacking one is not a
theoretical harm.

---

## 8. Things this does not do

Stated plainly, because a deployment guide that only lists strengths is not much use:

* **No TLS of its own.** Put a proxy in front. `serve.py` warns when you bind externally
  without `FACET_PUBLIC_URL` set to an https origin.
* **No horizontal scaling.** SQLite plus an in-process job runner plus in-memory OAuth state
  means one process. That is adequate to a few hundred users; it is not a cluster.
* **No virus scanning** of uploads beyond "it decodes as an image".
* **No audit log.** You can see counts per account, not a history of who looked at what.
* **No age verification.** If you operate this publicly you are responsible for deciding
  whether that is acceptable where you are.
* **No content moderation.** Users can upload anything that decodes. If strangers can reach
  your instance, you need a plan for that before they do, not after.
