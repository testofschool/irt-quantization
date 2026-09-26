# Data License

No data ships in this repository yet (`data/items.json` and `results/*.jsonl` are not
included). When the data are released, they will be released as two kinds of data with
**different** licenses, because the ARC source is share-alike and cannot be relicensed.

## ARC-derived content (questions, choices, answer keys)
The item content in `data/items.json` is derived from the AI2 Reasoning Challenge
(ARC; Clark et al., 2018), which is licensed **CC BY-SA 4.0**. Under share-alike,
this derived content **remains CC BY-SA 4.0**. It is not, and cannot be, relicensed.

## Original scored model-output fields
The fields produced by this project's scoring runs — per-item `correct`,
`lp_correct`, and `opt_lp` in `results/*.jsonl`, and the aggregate statistics in
`findings.json` — will be released with the data under **CC BY 4.0** where they are legally separable
from the ARC item text. Where a file interleaves ARC item content with scored
outputs, the more restrictive **CC BY-SA 4.0** governs that file as a whole.

## Code
All code in `code/` is released under the MIT License (see `LICENSE`).

## Summary
| Artifact | License |
|----------|---------|
| `data/items.json` (ARC-derived) | CC BY-SA 4.0 |
| `results/*.jsonl` (scored outputs; may include ARC ids/text) | CC BY-SA 4.0 if interleaved, else CC BY 4.0 |
| `findings.json` (aggregate stats) | CC BY 4.0 |
| `code/*.py` | MIT |
| Paper text (`main.tex`, `main.pdf`) | CC BY 4.0 |
