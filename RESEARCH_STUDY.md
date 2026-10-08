# FedMed 5K research workflow

This project uses a stratified 5,000-image subset of the official BloodMNIST **training** split. It classifies eight normal blood-cell types, not disease. Virtual clients receive indices into separate training partitions; only weights, sample counts, and metrics are exchanged during FL. The official test split is reserved for locked final runs.

From this project directory:

```powershell
python -m unittest discover -s test -p 'test*.py'
python -m experiments.verify_resolution_alignment
python -m experiments.research_eda --size 64 --train_samples 5000 --num_clients 10 --alpha 0.1
python -m experiments.run_research_matrix pilot
python -m experiments.run_research_matrix pilot --execute --rounds 30 --local_epochs 1
python -m experiments.run_research_matrix sensitivity --execute --rounds 30 --local_epochs 1
```

The matrix command prints jobs by default. Add `--client_gpus 0.5` only if CUDA works and Ray can share your GPU. Pilot jobs compare 28/64 resolution, mild training-only augmentation, and TinyCNN/MobileNetV3-Small. Sensitivity jobs use three seeds and compare FedAvg with the coverage objective at α=0.1. `tiny_cnn` is architecturally identical to the full-train project's `tiny_cnn`; `legacy` preserves the former 5K CNN.

All commands above are **development** runs: results contain `final_validation_metrics` (exact pooled monitor-set macro-F1, worst-class recall, calibration), a `run_spec.json`, and no final-test file. Select a configuration using validation, then freeze it by using its run spec for one final run:

```powershell
python -m experiments.run_simulation --final_test --locked_config path/to/development/run_spec.json
```

The final command reloads the frozen settings, retrains, calibrates on the separate validation-calibration subset, then evaluates test once. Do not use test results to change a configuration. A different seed needs its own development and final run spec. An exploratory `test/main.py` now reads published split metadata without loading test images.

On each physical Jetson, benchmark using the **training** split only:

Use `requirements-jetson.txt` for supporting packages after installing a PyTorch/torchvision build compatible with that Jetson's JetPack; the PC `requirements.txt` is not a Jetson wheel recipe.

```bash
python -m experiments.benchmark_jetson --model tiny_cnn --size 64 --client_id 0 --output results/orin_tiny64.json
python -m experiments.benchmark_jetson --model tiny_cnn --size 64 --client_id 1 --output results/nano_tiny64.json
```

Run each command on its respective device; also benchmark `mobilenet_v3_small`. The benchmark records batch-one p50/p95 inference latency, local training time for up to 256 assigned samples, process/CUDA memory, and parameter bytes. Record JetPack, power mode, and clock settings alongside results. Physical benchmarks have **not** been run by the implementation agent.

For a two-device FL check, generate a two-client training partition with `python -m experiments.run_partition --num_clients 2 --alpha 0.3 --train_samples 5000`, start `python -m experiments.run_server --num_clients 2 --rounds 2 --model tiny_cnn --size 64` on the server, and start `python -m experiments.run_jetson_client --server_ip SERVER_IP --client_id 0` and `--client_id 1` with matching `--num_clients 2 --alpha 0.3 --model tiny_cnn --size 64` on Orin and Nano. Pass `--device_type jetson_nano` on Nano if its registry ID differs. The physical clients use validation for monitoring, never test. Both devices must have the public benchmark and partition manifest available locally before FL; no images are transmitted by Flower rounds. This is **not** evidence of genuinely private hospital silos.
