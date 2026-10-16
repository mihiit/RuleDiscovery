
"""Full pipeline: tests -> non-RL experiments -> learned-agent training ->
evaluation -> vault -> statistics -> figures -> LaTeX paper.

Usage:
  python scripts/run_all.py
  python scripts/run_all.py --skip-training
  python scripts/run_all.py --skip-rl
  python scripts/run_all.py --with-primitive-ppo

On Windows, Bash scripts use Git Bash when it is installed.
Only this repository's tests under ./tests are collected.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")


def merged_env(**overrides):
    """Return a copy of the current environment with overrides applied."""
    result = os.environ.copy()
    result.update({key: str(value) for key, value in overrides.items()})
    return result


def run_process(args, *, env=None, cwd=None):
    """Run a command without shell=True so paths containing spaces work."""
    printable = " ".join(
        f'"{arg}"' if " " in str(arg) else str(arg)
        for arg in args
    )
    print(f"\n>>> {printable}", flush=True)

    completed = subprocess.run(
        [str(arg) for arg in args],
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
    )

    if completed.returncode != 0:
        raise SystemExit(
            f"Command failed with exit code {completed.returncode}: "
            f"{printable}"
        )


def run_python(*args):
    """Run a script with the active Python interpreter."""
    run_process([sys.executable, *args])


def run_bash_script(script, *args, env=None):
    """Run a repository Bash script using Git Bash on Windows."""
    if os.name == "nt" and GIT_BASH.is_file():
        bash = str(GIT_BASH)
    else:
        bash = shutil.which("bash")

    if not bash:
        raise SystemExit(
            f"Cannot run {script}: Bash was not found. "
            "Install Git for Windows and check "
            r"C:\Program Files\Git\bin\bash.exe."
        )

    script_path = (ROOT / script).resolve()

    if not script_path.is_file():
        raise SystemExit(f"Bash script does not exist: {script_path}")

    # Convert the Windows script path to a Git Bash-compatible path.
    if os.name == "nt" and Path(bash).resolve() == GIT_BASH.resolve():
        relative_script = os.path.relpath(script_path, ROOT).replace("\\", "/")
        bash_script = f"./{relative_script}"
    else:
        bash_script = str(script_path)

    run_process(
        [bash, bash_script, *args],
        env=env,
        cwd=ROOT,
    )


def build_pdf():
    """Compile paper/main.tex without shell command chaining."""
    paper_dir = ROOT / "paper"

    commands = (
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
    )

    for command in commands:
        run_process(command, cwd=paper_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Reuse trained models already present in data/models/",
    )
    parser.add_argument(
        "--skip-rl",
        action="store_true",
        help="Skip all learned-agent stages",
    )
    parser.add_argument(
        "--with-primitive-ppo",
        action="store_true",
        help="Also train primitive-action PPO baselines",
    )

    args = parser.parse_args()
    start = time.time()

    try:
        # Do not collect tests from external repositories.
        run_python("-m", "pytest", "-q", "tests")

        (ROOT / "data" / "results").mkdir(
            parents=True, exist_ok=True
        )
        (ROOT / "data" / "models").mkdir(
            parents=True, exist_ok=True
        )

        # Run non-RL experiments.
        run_bash_script("scripts/run_stages.sh", "nonrl")

        # Train and evaluate learned agents unless explicitly skipped.
        if not args.skip_rl:
            if not args.skip_training:
                run_bash_script("scripts/train_macro_seeds.sh")
                run_bash_script("scripts/train_generic_all.sh")
                run_bash_script("scripts/train_vault_all.sh")

                if args.with_primitive_ppo:
                    run_bash_script("scripts/train_baselines.sh")

            run_bash_script(
                "scripts/run_stages.sh",
                "rl",
                env=merged_env(AGENT_SET="rl"),
            )

        # Analysis, figures, and manuscript generation.
        run_python("experiments/run_vault.py")
        run_python("scripts/analyze.py")
        run_python("visualization/plots.py")
        run_python("paper/build_paper.py")

        # Compile the manuscript.
        build_pdf()

    except FileNotFoundError as exc:
        raise SystemExit(
            f"Required executable or file was not found: {exc}. "
            "Check that Bash, pdflatex, and bibtex are installed "
            "and available. On Windows, this script prefers Git Bash."
        ) from exc

    elapsed = time.time() - start
    print(
        f"\nPipeline completed in {elapsed:.0f} seconds. "
        f"Expected PDF: {ROOT / 'paper' / 'main.pdf'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
