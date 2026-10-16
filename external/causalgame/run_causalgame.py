"""Runs every agent on every scenario against a running CausalGame backend and writes data/results/causalgame.csv (one row per session)."""
import os, sys, time, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cg_adapter import Session, SCENARIOS
from cg_agents import AGENTS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "results", "causalgame.csv")


def main(n_sessions=40, scenarios=None, agents=None):
    """Resumable: finished (scenario, agent) cells are kept in the CSV and skipped."""
    import pandas as pd
    scenarios, agents = scenarios or SCENARIOS, agents or list(AGENTS)
    rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []
    done = {(r["scenario"], r["agent"]) for r in rows}
    for sc in scenarios:
        for an in agents:
            if (sc, an) in done:
                continue
            for k in range(n_sessions):
                rng = np.random.default_rng((abs(hash(sc)) % 100003) * 1000 + sorted(AGENTS).index(an) * 100 + k)
                sess = Session(sc, an)
                design = AGENTS[an]().run(sess, rng)
                surv, win = sess.submit(design)
                rows.append(dict(scenario=sc, agent=an, session=k, survival=surv, win=int(win), threshold=sess.threshold, deployments_used=10 - sess.deploys_left))
            print(sc, an, round(float(np.mean([x["survival"] for x in rows if x["scenario"] == sc and x["agent"] == an])), 3), flush=True)
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
