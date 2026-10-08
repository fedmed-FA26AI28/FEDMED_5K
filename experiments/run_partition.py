"""Script kiểm tra và chạy Dirichlet partition cho mọi tổ hợp num_clients × alpha."""
import argparse
import numpy as np
from datasets.medmnist_code import get_bloodmnist_dataset
from datasets.partition import (
    dirichlet_partition, save_partition, visualize_partition, stratified_subsample_indices
)
# Các giá trị theo plan
DEFAULT_NUM_CLIENTS = [3, 5, 10]
DEFAULT_ALPHAS      = [1.0, 0.3, 0.1]
DEFAULT_SEED        = 42
def run_one(train_dataset, num_clients: int, alpha: float, seed: int,
            train_samples: int = 5000):
    """Chạy partition cho 1 tổ hợp (num_clients, alpha)."""
    print(f"\n{'='*55}")
    print(f"  Partitioning | num_clients={num_clients} | alpha={alpha} | seed={seed}")
    print(f"{'='*55}")
    selected = stratified_subsample_indices(train_dataset, train_samples, seed=seed)
    partition = dirichlet_partition(train_dataset, num_clients, alpha, seed,
                                    eligible_indices=selected)
    # In thống kê
    sizes = [len(idx) for idx in partition]
    print(f"  Samples per client: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
    total = sum(sizes)
    print(f"  Total samples: {total}")
    assert total == len(selected), "Partition must cover only the selected train budget"
    # Kiểm tra không có index trùng
    all_indices = [idx for sublist in partition for idx in sublist]
    assert len(all_indices) == len(set(all_indices)), "Có indices trùng lặp!"
    print("  Validation PASSED: no duplicates, total matches.")
    # Lưu JSON
    save_partition(partition, train_dataset, alpha, seed, num_clients,
                   save_dir="data/partitions")
    # Lưu plot
    plot_path = f"data/partitions/plots/dist_alpha{alpha}_clients{num_clients}.png"
    visualize_partition(partition, train_dataset, alpha, num_clients, save_path=plot_path)
def main():
    parser = argparse.ArgumentParser(description="Run Dirichlet non-IID partition.")
    parser.add_argument("--num_clients", type=int, default=None,
                        help="Số client (mặc định: chạy [3, 5, 10])")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Tham số Dirichlet (mặc định: chạy [1.0, 0.3, 0.1])")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed (mặc định: 42)")
    parser.add_argument("--all", action="store_true",
                        help="Chạy toàn bộ 9 tổ hợp theo plan")
    parser.add_argument("--train_samples", type=int, default=5000)
    args = parser.parse_args()
    print("Loading BloodMNIST train dataset...")
    train_dataset, _ = get_bloodmnist_dataset("train", download=True)
    print(f"Train size: {len(train_dataset)} samples")
    if args.all or (args.num_clients is None and args.alpha is None):
        # Chạy toàn bộ matrix: 3 × 3 = 9 tổ hợp
        for nc in DEFAULT_NUM_CLIENTS:
            for al in DEFAULT_ALPHAS:
                run_one(train_dataset, nc, al, args.seed, args.train_samples)
    else:
        # Chạy 1 tổ hợp cụ thể
        nc = args.num_clients if args.num_clients else 10
        al = args.alpha if args.alpha else 0.3
        run_one(train_dataset, nc, al, args.seed, args.train_samples)
    print("\nDone! Check data/partitions/ for JSON files and plots.")
if __name__ == "__main__":
    main()  # python -m experiments.run_partition --all
