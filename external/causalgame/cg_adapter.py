"""Adapter between non-LLM agents and the real CausalGame simulation backend (https://causalgame.github.io; MIT-licensed code by the
CausalGame authors, which is NOT vendored here). Set CAUSALGAME_DIR to a checkout and start `uvicorn api.app:app --port 8000` from it.

Scope: this evaluates the *survival* part of the benchmark only (exploration with <=10 deployments / 200 drones, then one 1,000-drone
evaluation against the scenario threshold). The benchmark's LLM-judged causal-reasoning report rubric cannot be applied to non-language agents.
Agents see only what the API shows for a batch: the number deployed and the number that survived."""
import json, os
import numpy as np
import requests

CG_DIR = os.environ.get("CAUSALGAME_DIR", "/home/claude/cg/CausalGame-main")
URL = os.environ.get("CAUSALGAME_URL", "http://localhost:8000")
SCENARIOS = sorted(d for d in os.listdir(os.path.join(CG_DIR, "experiments")) if os.path.isdir(os.path.join(CG_DIR, "experiments", d)))


class Spec:
    """Factor space of one scenario, read from experiments/<name>/action_space.json."""

    def __init__(self, name):
        a = json.load(open(os.path.join(CG_DIR, "experiments", name, "action_space.json")))
        self.name = name
        self.num = {k: (v["min"], v["max"], v["default"]) for k, v in a.get("numerical", {}).items()}
        self.disc = {k: list(v["options"]) for k, v in a.get("discrete", {}).items()}
        self.disc_default = {k: v.get("default", list(v["options"])[0]) for k, v in a.get("discrete", {}).items()}
        self.boo = {k: bool(v.get("default", False)) for k, v in a.get("boolean", {}).items()}
        self.budget = a.get("constraints", {}).get("total_def_budget")
        self.names = list(self.num)

    def default(self):
        return dict(num={k: v[2] for k, v in self.num.items()}, disc=dict(self.disc_default), boo=dict(self.boo))

    def project(self, d):
        """Clip to bounds and scale the numeric DEF values down proportionally until they respect the total-DEF budget."""
        num = {k: int(min(max(round(d["num"][k]), lo), hi)) for k, (lo, hi, _) in self.num.items()}
        if self.budget is not None and sum(num.values()) > self.budget:
            f = self.budget / sum(num.values())
            num = {k: int(np.floor(v * f)) for k, v in num.items()}
        return dict(num=num, disc=dict(d["disc"]), boo=dict(d["boo"]))

    def random(self, rng):
        hi_cap = self.budget / 3 if self.budget else 50
        num = {}
        for k, (lo, hi, _) in self.num.items():
            num[k] = lo if rng.random() < 0.3 else rng.uniform(lo, min(hi, hi_cap))
        return self.project(dict(num=num, disc={k: v[rng.integers(len(v))] for k, v in self.disc.items()}, boo={k: bool(rng.random() < 0.5) for k in self.boo}))

    def perturb(self, d, rng, k_changes=2):
        out = dict(num=dict(d["num"]), disc=dict(d["disc"]), boo=dict(d["boo"]))
        keys = [("n", k) for k in self.num] + [("d", k) for k in self.disc] + [("b", k) for k in self.boo]
        for i in rng.choice(len(keys), size=min(k_changes, len(keys)), replace=False):
            t, k = keys[i]
            if t == "n":
                lo, hi, _ = self.num[k]
                out["num"][k] = out["num"][k] + rng.choice([-1, 1]) * rng.uniform(4, 20)
            elif t == "d":
                out["disc"][k] = self.disc[k][rng.integers(len(self.disc[k]))]
            else:
                out["boo"][k] = not out["boo"][k]
        return self.project(out)

    def features(self, d):
        """intercept, scaled numeric, scaled numeric squared, discrete one-hot, booleans"""
        f = [1.0]
        sc = [(d["num"][k] - lo) / max(hi - lo, 1) for k, (lo, hi, _) in self.num.items()]
        f += sc + [s * s for s in sc]
        for k, opts in self.disc.items():
            f += [1.0 if d["disc"][k] == o else 0.0 for o in opts]
        f += [1.0 if d["boo"][k] else 0.0 for k in self.boo]
        return np.array(f)

    def prior_precision(self):
        n, m, c = len(self.num), 0, sum(len(v) for v in self.disc.values()) + len(self.boo)
        return np.array([1e-2] + [1 / 0.5 ** 2] * n + [1 / 0.25 ** 2] * n + [1 / 0.3 ** 2] * c)

    def _design(self, d):
        return {k: int(v) for k, v in d["num"].items()}

    def _equip(self, d):
        """discrete choices, plus boolean switches as the strings the API accepts in `equipment`"""
        return {**d["disc"], **{k: "true" if v else "false" for k, v in d["boo"].items()}}

    def payload(self, d, count=None):
        p = {"design": self._design(d), "equipment": self._equip(d)}
        if count is not None:
            p["count"] = count
        return p


class Session:
    """One benchmark session (10 deployments, 200 drones, one submission)."""

    def __init__(self, scenario, name="agent"):
        self.spec = Spec(scenario)
        r = requests.post(URL + "/api/agent/register", json={"model_name": name, "agent_name": name, "experiment": scenario, "execution_mode": "hybrid"}, timeout=30)
        r.raise_for_status()
        self.h = {"X-Session-ID": r.json()["session_id"]}
        s = requests.get(URL + "/api/agent/status", headers=self.h, timeout=30).json()
        self.deploys_left, self.drones_left, self.threshold = s["deployments_remaining"], s["drones_remaining"], s["victory_threshold"]

    def deploy(self, d, count):
        count = min(count, self.drones_left, 50)
        r = requests.post(URL + "/api/agent/deploy", headers=self.h, json=self.spec.payload(d, count), timeout=60)
        r.raise_for_status()
        j = r.json()
        self.deploys_left -= 1
        self.drones_left = j.get("drones_remaining", self.drones_left - count)
        return int(j["survived"]), int(j["deployed"])

    def submit(self, d):
        r = requests.post(URL + "/api/agent/submit", headers=self.h, json={"design": self.spec._design(d), "equipment": self.spec._equip(d)}, timeout=60)
        r.raise_for_status()
        j = r.json()
        return float(str(j["survival_rate"]).rstrip("%")) / 100.0, bool(j["victory"])
