# Eval fixtures

`pii_corpus.jsonl` is a **synthetic** labeled set. Every name, number, key, and
address is a reserved test value or an obvious placeholder. There is no client
or operator personal data here.

```bash
python -m scripts.eval_pii
python -m scripts.eval_pii --write-results
```

Face / body / residual-OCR quality is **not** scored here. That gate is
`python -m scripts.run_image_validation` on a GPU host with a private photo set
that must never be committed.
