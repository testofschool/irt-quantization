# data/ — inputs (ADD before running)

Place the ARC item bank here:

- `items.json` — 100 ARC items (50 Easy + 50 Challenge)

Schema (one JSON array; each element):
```json
{
  "id": "Mercury_7081673",
  "question": "Which property of a mineral can be determined by...",
  "texts": ["luster", "mass", "weight", "color"],
  "answer_idx": 0,
  "strat": "easy"
}
```
`strat` is "easy" or "challenge". Derived from allenai/ai2_arc
(Clark et al. 2018, CC BY-SA 4.0). Consumed by `code/run_eval.py`.
