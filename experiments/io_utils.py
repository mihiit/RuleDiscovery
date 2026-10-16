import os, csv, json, time
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results")


def save_rows(name, rows, meta):
    os.makedirs(OUT, exist_ok=True)
    keys = sorted({k for r in rows for k in r}); first = [k for k in ("experiment", "condition", "agent", "seed") if k in keys]
    keys = first + [k for k in keys if k not in first]
    with open(os.path.join(OUT, name + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    json.dump(dict(meta, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), n_rows=len(rows)), open(os.path.join(OUT, name + ".json"), "w"), indent=2, default=str)
