# Optional vacant-class distillation experiment

This branch adds `vacant_distill` (raw CE plus distillation) and
`coverage_distill` (existing coverage objective plus distillation). They are
opt-in; FedAvg and existing coverage remain unchanged. This is **FedVLS-inspired**,
not full FedVLS: its logit-suppression component is not reproduced.

At each active client, a frozen copy of the received global model teaches the
student on that client's training images only. Classes with at most
`--distill_max_count` local training examples (default zero) are retained in
the distillation target; all other classes form one aggregate comparison bin.
This works even if exactly one class is vacant. The teacher is never uploaded.
The server still applies sample-weighted FedAvg. Validation and test flows are
unchanged, and test is not loaded by development runs.

Round 1 is skipped by default (`--distill_warmup_rounds 1`) so the initially
random global model is not used as a teacher. `--distill_mu 0.1` and
`--distill_temperature 2` are pilot settings, not tuned values. Active clients
pay for a second model copy and forward pass; record training time and peak
memory on the real Jetsons before claiming an edge benefit.

From this project directory, run a small validation-only smoke:

```powershell
python -m experiments.run_simulation --strategy vacant_distill --num_clients 2 --rounds 2 --local_epochs 1 --train_samples 160 --model tiny_cnn --size 28 --alpha 0.3 --distill_max_count 10
```

The paired 5K sensitivity study uses three seeds, ten virtual clients, 64x64
TinyCNN, the same training/validation partitions, and one local epoch. It
compares FedAvg, coverage, distillation alone, coverage plus distillation, and
a near-uniform-head control (`--coverage_kappa 1e12`). The uniform control is
not matched for total penalty strength; tune fairly using validation before
claiming a coverage-specific effect.

```powershell
python -m experiments.run_research_matrix distillation
python -m experiments.run_research_matrix distillation --execute --rounds 30 --local_epochs 1
```

Choose using validation macro-F1 **and** per-class/worst-class recall. Once
locked, test one selected configuration; do not retune based on test:

```powershell
python -m experiments.run_simulation --final_test --locked_config path/to/development/run_spec.json
```

The full-training comparison is in the matching FedMed_20K branch. On each
Jetson, compare the same train shard with these train-only benchmarks. Check
`teacher_model_copy` and `distill_active_classes` in the output: a client
with no qualifying classes has no distillation overhead or effect.

```bash
python -m experiments.benchmark_jetson --strategy fedavg --client_id 0 --output results/jetson_fedavg.json
python -m experiments.benchmark_jetson --strategy vacant_distill --client_id 0 --output results/jetson_distill.json
```

This Git branch is `experiment/vacant-class-distillation`. The branch
`baseline/pre-distillation` retains the exact pre-experiment working baseline.
After the worktree is clean, `git switch baseline/pre-distillation` removes
the experimental method from your active files without deleting its branch.
`main` remains untouched and may be older than this preserved baseline.

Related paper: [FedVLS, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33864).
