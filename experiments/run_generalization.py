"""Generalization regimes A, B, C (chain length), C2 (size only), D (unseen colours), E."""
from common import *


def main(seeds=20, envs=30, n_train=40):
    rows = []
    for rname, reg in REGIMES.items():
        for name in ("random", "reactive", "psrl", "hypothesis", "hyp_adaptive", "hyp_transfer", "macro_ff", "macro_gru", "macro_gruB"):
            if not keep(name):
                continue
            f = AGENTS.get(name) or rl_factory(name, rname)
            if f is None:
                continue
            ns, ne = budget(name, seeds, envs)
            for s in range(ns):
                ag = f(s)
                if name == "hyp_transfer":
                    train_agent(ag, reg["train"], s, n_train)
                for split, tm, base in (("train_heldout", reg["train"], EVAL_BASE), ("test", reg["test"], TEST_BASE)):
                    eps = eval_episodes(ag, tm, base, s, ne)
                    rows.append(dict(experiment="generalization", condition=rname, agent=name, split=split, seed=s, **aggregate(eps)))
            sr = lambda sp: np.mean([r["success_rate"] for r in rows if r["condition"] == rname and r["agent"] == name and r["split"] == sp])
            print(rname, name, "train", round(sr("train_heldout"), 3), "test", round(sr("test"), 3), flush=True)
    save("generalization", rows, dict(seeds=seeds, envs=envs, n_train=n_train, rl_seeds=RL_SEEDS, rl_envs=RL_ENVS,
                                      regimes={k: {"train": str(v["train"]), "test": str(v["test"])} for k, v in REGIMES.items()}))


if __name__ == "__main__":
    main()
