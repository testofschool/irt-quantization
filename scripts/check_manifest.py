#!/usr/bin/env python3
"""check_manifest.py — verify the repo has everything needed for full reproduction.

Exit code 0 = ready to run the full pipeline from raw data.
Exit code 1 = missing inputs (still fine for code review / reconstruction figures).
"""
import os, sys, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = ["f16","Q8_0","Q6_K","Q5_K_M","Q4_0","Q4_K_M","Q3_K_M","Q2_K",
         "1.5B_Q8_0","1.5B_Q4_K_M","1.5B_Q2_K","llama1B_Q8_0","llama1B_Q4_K_M"]
REQUIRED_RESULTS = [f"{s}.jsonl" for s in PANEL] + ["ext_3B.jsonl"]

def check():
    ok = True
    items = os.path.join(ROOT, "data", "items.json")
    if not os.path.exists(items):
        print("MISSING  data/items.json"); ok = False
    else:
        try:
            n = len(json.load(open(items)))
            print(f"OK       data/items.json ({n} items)")
        except Exception as e:
            print(f"INVALID  data/items.json ({e})"); ok = False
    rd = os.path.join(ROOT, "results")
    for fn in REQUIRED_RESULTS:
        p = os.path.join(rd, fn)
        print(f"{'OK      ' if os.path.exists(p) else 'MISSING '} results/{fn}")
        ok = ok and os.path.exists(p)
    return ok

if __name__ == "__main__":
    ready = check()
    print("\n" + ("READY: full pipeline reproducible." if ready
          else "NOT READY: add the missing files above (figures fall back to reconstruction)."))
    sys.exit(0 if ready else 1)
