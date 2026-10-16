"""Second environment family: combination vault. Train sizes L in {4,5}, k in {2,3}; test on larger codes."""
import os, sys, glob, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from environment.vault import Vault
from agents.vault_agents import RandomVault, ReactiveVault, HypothesisVault, VaultPPOAgent
from io_utils import save_rows

COND = {"in_dist": ([4, 5], [2, 3]), "larger": ([6, 7], [4, 5])}
T = 150


def run_ep(agent, L, k, seed):
    env = Vault(L, k, T, seed); o = env.reset(); agent.reset(o)
    while not o["done"]:
        o = env.step(agent.act(o))
    return o["success"], env.t, env.resets


def main(seeds=30, envs=30):
    if os.environ.get("AGENT_SET") == "rl":
        return
    mp = lambda m: sorted(q for q in glob.glob(f"data/models/vault_{m}_s[0-9]*.pkl") if os.path.exists(q[:-4] + ".done"))
    AG = {"random": lambda s: RandomVault(s), "reactive": lambda s: ReactiveVault(s), "hypothesis": lambda s: HypothesisVault(s)}
    for m in ("ff", "gru"):
        if mp(m):
            print("vault", m, "models:", len(mp(m)))
            AG[f"vault_{m}"] = (lambda mm: (lambda s: VaultPPOAgent(s, mp(mm)[s % len(mp(mm))], f"vault_{mm}")))(m)
    rows = []
    for cname, (Ls, ks) in COND.items():
        for name, f in AG.items():
            for s in range(seeds):
                rng = np.random.default_rng(10_000 + s); ag = f(s); res = []
                for i in range(envs):
                    L, k = int(rng.choice(Ls)), int(rng.choice(ks))
                    ok, st, rs = run_ep(ag, L, k, 5_000_000 + s * 1000 + i)
                    res.append((ok, st, rs, sum((L - p - 1) / 2 for p in range(k))))
                a = np.array(res, float)
                rows.append(dict(experiment="vault", condition=cname, agent=name, seed=s, success_rate=a[:, 0].mean(), mean_episode_len=a[:, 1].mean(),
                                 resets=a[:, 2].mean(), pred_elim_resets=a[:, 3].mean()))
            g = [r for r in rows if r["condition"] == cname and r["agent"] == name]
            print(cname, name, "success", round(np.mean([r["success_rate"] for r in g]), 3), "resets", round(np.mean([r["resets"] for r in g]), 2),
                  "pred", round(np.mean([r["pred_elim_resets"] for r in g]), 2), flush=True)
    save_rows("vault", rows, dict(seeds=seeds, envs=envs, T=T, train="L in {4,5}, k in {2,3}"))


if __name__ == "__main__":
    main()
