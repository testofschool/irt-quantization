# Package Status

This is a **proposal / methods paper** package. It makes no empirical claims.

## Present and complete
- main.tex / main.pdf        proposal paper (no findings claimed)
- code/*.py, scripts/*.py    runnable reference implementation
- example_findings.json      OUTPUT FORMAT ONLY (placeholder values, not measurements)
- figures/*.png              illustrative schematics (banner-labeled "not measured data")
- data/README.md, results/README.md   input schemas
- README, LICENSE, DATA_LICENSE, CITATION, requirements

## Intentionally NOT included
- data/items.json and results/*.jsonl are NOT included because no experiment was run.
  The paper does not claim they exist. Supplying them lets a user execute the protocol
  and generate real findings.

## To execute the protocol yourself
  python scripts/check_manifest.py   # READY once you add items.json + results/*.jsonl
  python code/analyze_all.py         # writes real findings.json
  python code/make_figs.py           # real figures replace the schematics
