#!/usr/bin/env python
"""Query an index.

Implements the brief's example directly:

    Age: 25-35, importance 30%
    Gender: Female, importance 20%
    Attractiveness: High, importance 50%

    python scripts/search.py --index facet.db \
        --age 25 35 --age-weight 0.3 \
        --gender female --gender-weight 0.2 \
        --attractiveness-percentile 0.8 --attractiveness-weight 0.5 \
        --limit 20

Queries can also be given as YAML/JSON (`--query q.yaml`), which is what the API will pass.
Every result carries the per-criterion arithmetic that produced its score, so the ranking can
explain itself rather than being taken on trust.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.query.engine import SearchEngine  # noqa: E402
from facet.query.spec import QuerySpec  # noqa: E402


def build_spec(a) -> QuerySpec:
    if a.query:
        import yaml
        text = Path(a.query).read_text()
        return QuerySpec.from_dict(yaml.safe_load(text))
    prefs: dict = {}
    if a.age:
        prefs["age"] = {"range": [a.age[0], a.age[1]], "weight": a.age_weight,
                        "required": a.age_required}
    if a.gender:
        prefs["gender"] = {"value": a.gender, "weight": a.gender_weight,
                           "mode": "strict" if a.gender_strict else "soft"}
    if a.attractiveness_percentile is not None:
        prefs["attractiveness"] = {"min_percentile": a.attractiveness_percentile,
                                   "weight": a.attractiveness_weight}
    return QuerySpec.from_dict({
        "preferences": prefs,
        "filters": {"min_quality": a.min_quality, "min_face_px": a.min_face_px,
                    "exclude_ood": a.exclude_ood,
                    "exclude_near_duplicates": not a.keep_duplicates},
        "sort_by": a.sort_by, "limit": a.limit, "offset": a.offset,
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--query", help="YAML/JSON query file")
    ap.add_argument("--age", nargs=2, type=float, metavar=("MIN", "MAX"))
    ap.add_argument("--age-weight", type=float, default=0.30)
    ap.add_argument("--age-required", action="store_true")
    ap.add_argument("--gender", choices=["female", "male"])
    ap.add_argument("--gender-weight", type=float, default=0.20)
    ap.add_argument("--gender-strict", action="store_true")
    ap.add_argument("--attractiveness-percentile", type=float,
                    help="0.8 = top 20%% of the collection (E7: absolute thresholds do not "
                         "survive a domain change)")
    ap.add_argument("--attractiveness-weight", type=float, default=0.50)
    ap.add_argument("--min-quality", type=float, default=0.0)
    ap.add_argument("--min-face-px", type=float, default=0.0)
    ap.add_argument("--exclude-ood", action="store_true")
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--sort-by", default="relevance",
                    choices=["relevance", "attractiveness", "confidence", "quality",
                             "age_match", "random"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--explain", action="store_true", help="show per-criterion arithmetic")
    args = ap.parse_args()

    eng = SearchEngine(args.index)
    resp = eng.search(build_spec(args))

    if args.json:
        print(json.dumps(resp.as_dict(), indent=2, default=str))
        eng.close()
        return 0

    d = resp.diagnostics
    print(f"{resp.total_matched} matched, showing {len(resp.results)} "
          f"(offset {args.offset}) sorted by {args.sort_by}\n")
    hdr = f"{'#':>3} {'rel':>6} {'attr':>6} {'pct':>5} {'age':>5} {'gender':>8} {'qual':>5}  file"
    print(hdr); print("-" * len(hdr))
    for i, r in enumerate(resp.results, args.offset + 1):
        flag = "!" if r.ood else " "
        print(f"{i:>3} {r.relevance:>6.3f} "
              f"{(f'{r.attractiveness:.2f}' if r.attractiveness is not None else '  -  '):>6} "
              f"{(f'{r.attractiveness_percentile:.2f}' if r.attractiveness_percentile is not None else '  - '):>5} "
              f"{(f'{r.age:.0f}' if r.age is not None else '  -'):>5} "
              f"{(r.gender or '-'):>8} {r.quality:>5.2f}{flag} {Path(r.path).name}")
        if args.explain:
            for c in r.contributions:
                print(f"      {c.criterion:<15} match={c.match:.3f} x conf={c.confidence:.3f} "
                      f"x w={c.weight:.2f} = {c.contribution:.4f}   ({c.detail})")

    print(f"\ndiagnostics:")
    for k, v in d.items():
        if v:
            print(f"  {k}: {v}")
    if d.get("missing_age") or d.get("missing_gender"):
        print("  note: age/gender come from a lazy pass; missing is not the same as "
              "non-matching, so those criteria contributed nothing rather than scoring zero")
    print(f"\n{__import__('facet.query.engine', fromlist=['x']).BEAUTY_SOURCE_NOTE}")
    eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
