#!/usr/bin/env python3
"""run_eval.py — Score a GGUF model on ARC items via next-token option-letter logprobs.

Usage:
    python3 run_eval.py <subject_name> <gguf_path> [n_items]

Environment:
    FAMILY=qwen|llama   selects the chat template (default: qwen)

SCORING METHOD (prompt-only; no generation):
  We evaluate ONLY the prompt and read the model's next-token log-probabilities
  directly from the logits, then score each option letter as a candidate
  continuation. We never sample or generate, so we never risk reading a generated
  continuation token (the documented pitfall with create_completion + echo, where
  the returned token_logprobs are offset by one position).

  Concretely, for prompt p:
      llm.reset(); llm.eval(tokenize(p))
      logprobs = log_softmax(llm.scores[len(p)-1])      # next-token distribution
  For a single-token option letter L with id t:  score(L) = logprobs[t].
  For a multi-token option letter, we extend the context token-by-token and sum
  conditional next-token logprobs (teacher-forced), which is the exact
  log P(option | prompt). The tokenizer diagnostic records whether each letter is
  single- or multi-token and whether the leading-space variant differs, so the
  chosen tokenization is auditable.

  Requires logits_all=True (needed for per-position logits) and temperature 0.

Outputs results/<subject>.jsonl (id, strat, correct, lp_correct, opt_lp) and
results/<subject>.meta.json (tokenizer diagnostics + scoring config).
"""
import json, os, sys, math

LETTERS = ["A", "B", "C", "D", "E"]

def log_softmax(logits):
    import numpy as np
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max()
    return x - math.log(float(np.exp(x).sum()))

def build_prompt(it, family):
    n = len(it["texts"])
    body = "\n".join(f"{LETTERS[i]}. {it['texts'][i]}" for i in range(n))
    user = f"{it['question']}\n{body}\nAnswer with the letter only."
    if family == "llama":
        return ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

def pick_letter_tokens(llm, letter):
    """Return the token id sequence for an option letter, preferring the variant
    the model would actually emit after the prompt. We try the bare letter and the
    leading-space letter and keep whichever is shorter (usually a single token)."""
    cand = []
    for variant in (letter, " " + letter):
        toks = llm.tokenize(variant.encode("utf-8"), add_bos=False)
        cand.append((len(toks), toks))
    cand.sort(key=lambda x: x[0])
    return cand[0][1]

def seq_logprob(llm, prompt_tokens, cont_tokens):
    """Teacher-forced log P(cont | prompt) by extending context token-by-token.
    Reads next-token logits from llm.scores; no sampling/generation."""
    import numpy as np
    llm.reset()
    llm.eval(prompt_tokens)
    total = 0.0
    ctx_len = len(prompt_tokens)
    for t in cont_tokens:
        lp = log_softmax(llm.scores[ctx_len - 1])   # dist for next token
        total += float(lp[t])
        llm.eval([t])                                # extend context by the gold token
        ctx_len += 1
    return total

def main():
    if len(sys.argv) < 3:
        print("usage: run_eval.py <subject> <gguf_path> [n_items]"); sys.exit(1)
    subject, gguf = sys.argv[1], sys.argv[2]
    n_items = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    family = os.environ.get("FAMILY", "qwen")

    from llama_cpp import Llama
    llm = Llama(model_path=gguf, logits_all=True, n_ctx=4096, verbose=False)

    items = json.load(open("data/items.json"))[:n_items]
    os.makedirs("results", exist_ok=True)

    # Tokenizer diagnostic + scoring config (auditable)
    diag = {"scoring": "prompt-only next-token logits; teacher-forced multi-token sum",
            "temperature": 0.0, "family": family, "letters": {}}
    for L in LETTERS:
        diag["letters"][L] = {
            "bare": llm.tokenize(L.encode(), add_bos=False),
            "space": llm.tokenize((" " + L).encode(), add_bos=False),
            "chosen": pick_letter_tokens(llm, L)}
    json.dump(diag, open(f"results/{subject}.meta.json", "w"), indent=2)

    outpath = f"results/{subject}.jsonl"
    done = {json.loads(l)["id"] for l in open(outpath)} if os.path.exists(outpath) else set()
    with open(outpath, "a") as f:
        for k, it in enumerate(items):
            if it["id"] in done:
                continue
            prompt = build_prompt(it, family)
            ptoks = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
            n = len(it["texts"])
            lps = [seq_logprob(llm, ptoks, pick_letter_tokens(llm, LETTERS[i])) for i in range(n)]
            pred = max(range(n), key=lambda i: lps[i])
            gold = it["answer_idx"]
            rec = {"id": it["id"], "strat": it["strat"],
                   "correct": int(pred == gold),
                   "lp_correct": round(lps[gold], 6),
                   "opt_lp": [round(x, 6) for x in lps]}
            f.write(json.dumps(rec) + "\n"); f.flush()
            if k % 10 == 0:
                print(f"[{subject}] {k+1}/{len(items)}")
    print(f"[{subject}] done -> {outpath} (+ {subject}.meta.json)")

if __name__ == "__main__":
    main()
