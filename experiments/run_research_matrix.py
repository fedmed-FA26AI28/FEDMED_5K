"""List or execute the 5K validation-only pilot/sensitivity matrix."""

import argparse
import subprocess
import sys


def jobs(phase):
    if phase == "pilot":
        for model, size, augment in (
            ("tiny_cnn", 28, False),
            ("tiny_cnn", 64, False),
            ("tiny_cnn", 64, True),
            ("mobilenet_v3_small", 64, True),
        ):
            yield {"model": model, "size": size, "augment": augment,
                   "strategy": "fedavg", "alpha": 0.3, "seed": 42}
    elif phase == "sensitivity":
        for seed in (42, 43, 44):
            for strategy in ("fedavg", "coverage"):
                yield {"model": "tiny_cnn", "size": 64, "augment": True,
                       "strategy": strategy, "alpha": 0.1, "seed": seed}
    else:
        raise ValueError(phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pilot", "sensitivity"])
    parser.add_argument("--execute", action="store_true", help="Run every job; default only prints commands")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--client_gpus", type=float, default=0.0)
    args = parser.parse_args()
    for job in jobs(args.phase):
        command = [sys.executable, "-m", "experiments.run_simulation",
                   "--num_clients", "10", "--train_samples", "5000",
                   "--rounds", str(args.rounds), "--local_epochs", str(args.local_epochs),
                   "--model", job["model"], "--size", str(job["size"]),
                   "--strategy", job["strategy"], "--alpha", str(job["alpha"]),
                   "--seed", str(job["seed"]), "--client_gpus", str(args.client_gpus)]
        if job["augment"]:
            command.append("--augment")
        print(" ".join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
