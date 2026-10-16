import json


def load_traces(path="data/results/recovery_traces.json"):
    return json.load(open(path))


def pair_series(trace, truth):
    """Return {label: (steps, probs)} for true key->door pairs and identity (same-colour) pairs."""
    series = {}
    for step, d in trace:
        for k, p in d.items():
            _, pr = k.split(":")
            key, door = pr.split(">")
            lab = ("TRUE " if truth.get(key) == door else "same-colour " if key == door else None)
            if lab is None:
                continue
            series.setdefault(lab + pr, ([], []))
            series[lab + pr][0].append(step); series[lab + pr][1].append(p)
    return series
