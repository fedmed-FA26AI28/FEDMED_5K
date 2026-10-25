"""
Triển khai các thuật toán phân chia dữ liệu cho Federated Learning:
  1. Dirichlet non-IID (BloodMNIST)
  2. Practical non-IID (FedSLD ISBI 2022 - 80 rounds)
  3. Pathological non-IID (FedSLD ISBI 2022 / McMahan 2017 - 160 rounds)
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np


def _extract_labels(dataset) -> np.ndarray:
    """Trích xuất nhãn nhanh từ MedMNIST dataset hoặc PyTorch Dataset thông thường."""
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels).squeeze()
    return np.array([dataset[i][1] for i in range(len(dataset))]).squeeze()


# ─────────────────────────────────────────────────────────────
# 1. Dirichlet Non-IID Partition (Dùng cho BloodMNIST)
# ─────────────────────────────────────────────────────────────

def dirichlet_partition(dataset, num_clients: int, alpha: float, seed: int = 42) -> List[List[int]]:
    """
    Phân chia dataset theo phân phối Dirichlet (non-IID).
    """
    np.random.seed(seed)
    labels = _extract_labels(dataset)
    num_classes = len(np.unique(labels))

    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (proportions * len(class_indices)).astype(int)

        diff = len(class_indices) - proportions.sum()
        proportions[-1] += diff

        start = 0
        for cid, count in enumerate(proportions):
            end = start + count
            client_indices[cid].extend(class_indices[start:end].tolist())
            start = end

    return client_indices


def save_partition(client_indices: List[List[int]], dataset,
                   alpha: float, seed: int, num_clients: int,
                   save_dir: str = "data/partitions") -> Path:
    """Lưu kết quả Dirichlet partition ra file JSON (BloodMNIST)."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    labels = _extract_labels(dataset)
    num_classes = len(np.unique(labels))

    clients_data = {}
    for client_id, indices in enumerate(client_indices):
        client_labels = labels[indices]
        class_dict = {
            str(c): int(np.sum(client_labels == c))
            for c in range(num_classes)
        }
        clients_data[str(client_id)] = {
            "num_samples": len(indices),
            "indices": indices,
            "class_distribution": class_dict
        }

    result = {
        "dataset": "bloodmnist",
        "alpha": alpha,
        "seed": seed,
        "num_clients": num_clients,
        "total_samples": sum(len(idx) for idx in client_indices),
        "clients": clients_data
    }

    filename = f"partition_seed{seed}_alpha{alpha}_clients{num_clients}.json"
    filepath = save_path / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print(f"Partition saved: {filepath}")
    return filepath


def load_partition(json_path: str) -> List[List[int]]:
    """Load partition từ file JSON đã lưu."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_clients = data["num_clients"]
    client_indices = []
    for i in range(num_clients):
        client_indices.append(data["clients"][str(i)]["indices"])
    return client_indices


def load_partition_with_test(json_path: str):
    """
    Load cả train_indices và test_indices cho từng client theo partition FedSLD.
    Returns: (train_partition, test_partition)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_clients = data["num_clients"]
    train_indices = []
    test_indices = []
    for i in range(num_clients):
        cdata = data["clients"][str(i)]
        train_indices.append(cdata.get("train_indices", cdata.get("indices", [])))
        test_indices.append(cdata.get("test_indices", []))
    return train_indices, test_indices


# ─────────────────────────────────────────────────────────────
# 2. FedSLD Partitions (Practical & Pathological)
# ─────────────────────────────────────────────────────────────

def practical_noniid_partition(dataset, num_clients: int = 12,
                               seed: int = 42) -> List[List[int]]:
    """
    Practical Non-IID theo FedSLD paper (ISBI 2022) - Chạy 80 rounds.

    Với mỗi class c:
      - Chia samples thành 12 shards: 10 shards x 1%, 1 shard x 10%, 1 shard x 80%.
      - Shuffle gán cho 12 clients.
    Mỗi client có đủ tất cả class nhưng tỉ lệ chênh lệch lớn giữa các client.
    """
    np.random.seed(seed)
    labels = _extract_labels(dataset)
    num_classes = int(labels.max()) + 1
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_idxs = np.where(labels == class_id)[0]
        np.random.shuffle(class_idxs)
        n = len(class_idxs)

        # 10 x 1% + 1 x 10% + 1 x 80% = 100%
        proportions = np.array([0.01] * (num_clients - 2) + [0.10, 0.80])
        np.random.shuffle(proportions)

        sizes = (proportions * n).astype(int)
        sizes[-1] += n - sizes.sum()   # Bù sai số làm tròn để đủ 100% samples

        start = 0
        for cid in range(num_clients):
            end = start + sizes[cid]
            client_indices[cid].extend(class_idxs[start:end].tolist())
            start = end

    return client_indices


def pathological_noniid_partition(dataset, num_clients: int = 12,
                                  num_classes_per_client: int = 2,
                                  seed: int = 42) -> List[List[int]]:
    """
    Pathological Non-IID theo FedSLD paper (ISBI 2022 / McMahan 2017) - Chạy 160 rounds.

    - Mỗi client nhận đúng 2 classes ngẫu nhiên.
    - Toàn bộ dữ liệu của mỗi class được chia đều cho các client đã chọn class đó.
    - Sử dụng 100% dữ liệu train, không làm thất thoát mẫu.
    """
    np.random.seed(seed)
    labels = _extract_labels(dataset)
    num_classes = int(labels.max()) + 1
    client_indices = [[] for _ in range(num_clients)]

    # Gán mỗi client 2 class ngẫu nhiên, đảm bảo toàn bộ classes đều xuất hiện
    while True:
        client_classes = [
            list(np.random.choice(num_classes, size=num_classes_per_client, replace=False))
            for _ in range(num_clients)
        ]
        all_chosen = set(c for classes in client_classes for c in classes)
        if len(all_chosen) == num_classes:
            break

    # Phân phối toàn bộ samples của từng class cho các client được phân class đó
    for class_id in range(num_classes):
        class_idxs = np.where(labels == class_id)[0]
        np.random.shuffle(class_idxs)
        assigned_cids = [cid for cid in range(num_clients) if class_id in client_classes[cid]]

        shards = np.array_split(class_idxs, len(assigned_cids))
        for cid, shard in zip(assigned_cids, shards):
            client_indices[cid].extend(shard.tolist())

    return client_indices


def save_fedsld_partition(client_indices: List[List[int]], dataset,
                          dataset_name: str, partition_method: str,
                          num_clients: int, seed: int,
                          save_dir: str = "data/partitions") -> Path:
    """Lưu kết quả phân chia FedSLD (Practical hoặc Pathological) ra file JSON."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    labels = _extract_labels(dataset)
    num_classes = int(labels.max()) + 1

    clients_data = {}
    for cid, indices in enumerate(client_indices):
        cl = labels[np.array(indices)]
        clients_data[str(cid)] = {
            "num_samples": len(indices),
            "indices": [int(i) for i in indices],
            "class_distribution": {
                str(c): int(np.sum(cl == c)) for c in range(num_classes)
            }
        }

    result = {
        "dataset": dataset_name,
        "partition_method": partition_method,
        "num_clients": num_clients,
        "seed": seed,
        "total_samples": sum(len(idx) for idx in client_indices),
        "clients": clients_data
    }

    filename = f"{partition_method}_{dataset_name}_seed{seed}_clients{num_clients}.json"
    filepath = save_path / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print(f"Saved: {filepath}")
    return filepath


# ─────────────────────────────────────────────────────────────
# 3. Visualization (Giữ lại & hỗ trợ đa dataset)
# ─────────────────────────────────────────────────────────────

def visualize_partition(partition: List[List[int]], dataset,
                        alpha: Optional[float] = None,
                        num_clients: Optional[int] = None,
                        title: Optional[str] = None,
                        class_names: Optional[List[str]] = None,
                        save_path: Optional[str] = None):
    """
    Vẽ stacked bar chart thể hiện phân bổ class của từng client.
    Trục X = Client ID, Trục Y = Số samples, Màu = Class.
    Tương thích với cả BloodMNIST (alpha), PathMNIST, OrganMNIST, Practical và Pathological.
    """
    if num_clients is None:
        num_clients = len(partition)

    labels = _extract_labels(dataset)
    num_classes = len(np.unique(labels))

    # Tên class mặc định nếu không truyền vào
    if class_names is None or len(class_names) < num_classes:
        if num_classes == 8:
            class_names = ["Basophil", "Eosinophil", "Erythroblast", "IG",
                           "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"]
        else:
            class_names = [f"Class {c}" for c in range(num_classes)]

    # Ma trận phân phối: [num_clients x num_classes]
    dist_matrix = np.zeros((num_clients, num_classes), dtype=int)
    for client_id, indices in enumerate(partition):
        client_labels = labels[indices]
        for c in range(num_classes):
            dist_matrix[client_id, c] = int(np.sum(client_labels == c))

    # Vẽ stacked bar chart
    fig, ax = plt.subplots(figsize=(max(8, num_clients * 1.0), 5))
    x = np.arange(num_clients)
    bottom = np.zeros(num_clients)
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

    for c in range(num_classes):
        label_name = class_names[c] if c < len(class_names) else f"Class {c}"
        ax.bar(x, dist_matrix[:, c], bottom=bottom,
               label=label_name, color=colors[c])
        bottom += dist_matrix[:, c]

    if title is None:
        if alpha is not None:
            title = f"Data Distribution | alpha={alpha} | {num_clients} Clients"
        else:
            title = f"Data Distribution | {num_clients} Clients"

    ax.set_title(title)
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Number of Samples")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in range(num_clients)])
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    plt.close()