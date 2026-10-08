"""Trien khai cac thuat toan chia nho dataset (IID va non-IID qua phan phoi Dirichlet) cho cac client."""
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from typing import List, Dict

# Duong dan mac dinh den file config
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


# ─────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────

def resolve_device_type(client_id: int) -> str:
    """
    Tu dong xac dinh device_type dua tren client_id.
    Tra cuu trong client_registry cua configs/jetson.yaml.

    Args:
        client_id: ID cua client (0-9).

    Returns:
        device_type: 'jetson_orin' | 'jetson_nano' | 'pc'

    Vi du:
        resolve_device_type(0)  -> 'jetson_orin'  (client 0-4 la Orin)
        resolve_device_type(7)  -> 'jetson_nano'  (client 5-9 la Nano)
    """
    hw_path = _CONFIG_DIR / "jetson.yaml"
    with open(hw_path, "r", encoding="utf-8") as f:
        hw_cfg = yaml.safe_load(f)

    registry = hw_cfg.get("client_registry", {})
    if client_id not in registry:
        raise ValueError(
            f"client_id={client_id} khong co trong client_registry cua jetson.yaml. "
            f"Cac ID hop le: {sorted(registry.keys())}"
        )
    return registry[client_id]


def load_hardware_config(client_id: int = None, device_type: str = None) -> Dict:
    """
    Doc cau hinh phan cung tu configs/jetson.yaml.

    Uu tien theo thu tu:
      1. client_id  -> tu resolve device_type qua client_registry
      2. device_type -> dung truc tiep
      3. Ca 2 deu None -> doc device_type tu experiment.yaml (mac dinh: 'pc')

    Args:
        client_id  : ID cua client (0-9). Tu dong tra cuu Orin/Nano.
        device_type: 'pc' | 'jetson_orin' | 'jetson_nano' (override thu cong).

    Returns:
        Dict chua batch_size, num_workers, pin_memory, device_type.
    """
    hw_path = _CONFIG_DIR / "jetson.yaml"
    with open(hw_path, "r", encoding="utf-8") as f:
        hw_cfg = yaml.safe_load(f)

    # 1. Uu tien: client_id -> tra registry
    if client_id is not None:
        device_type = resolve_device_type(client_id)

    # 2. Fallback: doc device_type tu experiment.yaml
    elif device_type is None:
        exp_path = _CONFIG_DIR / "experiment.yaml"
        with open(exp_path, "r", encoding="utf-8") as f:
            exp_cfg = yaml.safe_load(f)
        device_type = exp_cfg.get("device_type", "pc")

    if device_type not in hw_cfg:
        raise ValueError(
            f"device_type '{device_type}' khong ton tai trong jetson.yaml. "
            f"Chon 1 trong: {[k for k in hw_cfg if k != 'client_registry']}"
        )

    return hw_cfg[device_type]


# ─────────────────────────────────────────────────────────────
# Partition
# ─────────────────────────────────────────────────────────────

def _dataset_labels(dataset) -> np.ndarray:
    return np.asarray(
        [int(np.asarray(dataset[index][1]).squeeze()) for index in range(len(dataset))],
        dtype=np.int64,
    )


def stratified_subsample_indices(
    dataset, num_samples: int = None, seed: int = 42
) -> List[int]:
    """Select a reproducible class-stratified training budget."""
    if num_samples is None or num_samples >= len(dataset):
        return list(range(len(dataset)))
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than zero")
    labels = _dataset_labels(dataset)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    raw_targets = counts / counts.sum() * num_samples
    targets = np.floor(raw_targets).astype(int)
    for position in np.argsort(-(raw_targets - targets))[: num_samples - int(targets.sum())]:
        targets[position] += 1
    selected: List[int] = []
    for class_id, target in zip(classes, targets):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        selected.extend(class_indices[:target].tolist())
    rng.shuffle(selected)
    return selected


def stratified_holdout_indices(
    dataset, holdout_fraction: float, seed: int = 42
) -> tuple[List[int], List[int]]:
    """Split validation indices into local-monitor and calibration subsets."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    labels = _dataset_labels(dataset)
    rng = np.random.default_rng(seed)
    monitor: List[int] = []
    holdout: List[int] = []
    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        holdout_size = max(1, int(round(len(class_indices) * holdout_fraction)))
        holdout_size = min(holdout_size, len(class_indices) - 1)
        holdout.extend(class_indices[:holdout_size].tolist())
        monitor.extend(class_indices[holdout_size:].tolist())
    rng.shuffle(monitor)
    rng.shuffle(holdout)
    return monitor, holdout


def dirichlet_partition(
    dataset,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    eligible_indices: List[int] = None,
) -> List[List[int]]:
    """
    Phan chia dataset theo phan phoi Dirichlet (non-IID).

    Args:
        dataset    : Dataset goc (BloodMNIST train).
        num_clients: So luong client (vi du: 3, 5, 10).
        alpha      : Tham so Dirichlet. Nho -> rat non-IID. Lon -> gan IID.
        seed       : Random seed de reproducibility.

    Returns:
        List gom num_clients danh sach, moi danh sach chua cac sample index.
    """
    if num_clients <= 0 or alpha <= 0:
        raise ValueError("num_clients and alpha must be greater than zero")
    rng = np.random.default_rng(seed)
    labels = _dataset_labels(dataset)
    eligible = np.asarray(
        list(range(len(dataset))) if eligible_indices is None else eligible_indices,
        dtype=np.int64,
    )
    if len(eligible) == 0 or len(np.unique(eligible)) != len(eligible):
        raise ValueError("eligible_indices must be non-empty and unique")
    if eligible.min() < 0 or eligible.max() >= len(dataset):
        raise ValueError("eligible_indices contains an out-of-range index")
    num_classes = len(np.unique(labels))

    # danh sach cac client
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_indices = eligible[labels[eligible] == class_id]
        rng.shuffle(class_indices)

        # ty le phan phoi theo dirichlet
        proportions = rng.dirichlet(alpha * np.ones(num_clients))

        # tinh so luong sample base on alpha
        proportions = (proportions * len(class_indices)).astype(int)

        # Dieu chinh de tong bang dung len(class_indices) (tranh mat mau do lam tron)
        diff = len(class_indices) - proportions.sum()
        proportions[-1] += diff

        # phan chia indices
        start = 0
        for cid, count in enumerate(proportions):
            end = start + count
            client_indices[cid].extend(class_indices[start:end].tolist())
            start = end

    return client_indices


def stratified_validation_partition(
    dataset, num_clients: int, seed: int = 42, eligible_indices: List[int] = None
) -> List[List[int]]:
    """Create deterministic, disjoint validation subsets for virtual clients."""
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than zero")

    labels = _dataset_labels(dataset)
    eligible = np.asarray(
        list(range(len(dataset))) if eligible_indices is None else eligible_indices,
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]

    for class_id in np.unique(labels):
        class_indices = eligible[labels[eligible] == class_id]
        rng.shuffle(class_indices)
        for client_id, indices in enumerate(
            np.array_split(class_indices, num_clients)
        ):
            client_indices[client_id].extend(indices.tolist())

    for indices in client_indices:
        rng.shuffle(indices)
    return client_indices


def save_partition(client_indices: List[List[int]], dataset,
                   alpha: float, seed: int, num_clients: int,
                   save_dir: str = "data/partitions") -> Path:
    """
    Luu ket qua partition ra file JSON de reproducibility.
    Ten file: partition_seed{seed}_alpha{alpha}_clients{num_clients}.json
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    labels = np.array([dataset[i][1] for i in range(len(dataset))]).squeeze()
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
        "split": "train",
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
    """Load partition tu file JSON da luu."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_clients = data["num_clients"]
    client_indices = []
    for i in range(num_clients):
        client_indices.append(data["clients"][str(i)]["indices"])
    return client_indices


# ─────────────────────────────────────────────────────────────
# DataLoader
# ─────────────────────────────────────────────────────────────

def get_client_dataloader(dataset, client_indices: List[int],
                          client_id: int = None,
                          batch_size: int = None,
                          shuffle: bool = True,
                          num_workers: int = None) -> DataLoader:
    """
    Tao DataLoader cho 1 client dua tren danh sach indices.

    Args:
        client_id   : ID cua client (0-9). Tu dong tra cuu device_type -> batch_size, num_workers.
                      Neu None -> fallback ve experiment.yaml (mac dinh: pc).
        batch_size  : Override thu cong neu muon. Mac dinh doc tu config.
        num_workers : Override thu cong neu muon. Mac dinh doc tu config.
    """
    hw_cfg      = load_hardware_config(client_id=client_id)
    batch_size  = batch_size  if batch_size  is not None else hw_cfg["batch_size"]
    num_workers = num_workers if num_workers is not None else hw_cfg["num_workers"]
    pin_memory  = hw_cfg.get("pin_memory", False)

    subset = Subset(dataset, client_indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def get_all_client_dataloaders(dataset, partition: List[List[int]],
                               batch_size: int = None,
                               num_workers: int = None) -> List[DataLoader]:
    """
    Tao DataLoader cho tat ca clients (thuong dung cho simulation).
    Doc active_clients tu experiment.yaml de map dung client_id.
    """
    exp_path = _CONFIG_DIR / "experiment.yaml"
    with open(exp_path, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)
    
    active_clients = exp_cfg.get("active_clients")
    # Neu khong set active_clients, mac dinh danh so 0, 1, 2...
    if not active_clients or len(active_clients) != len(partition):
        active_clients = list(range(len(partition)))

    loaders = []
    for i, indices in enumerate(partition):
        cid = active_clients[i]
        loader = get_client_dataloader(dataset, indices,
                                       client_id=cid,
                                       batch_size=batch_size,
                                       num_workers=num_workers)
        loaders.append(loader)
    return loaders


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def visualize_partition(partition: List[List[int]], dataset,
                        alpha: float, num_clients: int,
                        save_path: str = None):
    """
    Ve stacked bar chart the hien phan bo class cua tung client.
    Truc X = Client ID, truc Y = so samples, mau = class.
    """
    labels = np.array([dataset[i][1] for i in range(len(dataset))]).squeeze()
    num_classes = len(np.unique(labels))
    class_names = ["Basophil", "Eosinophil", "Erythroblast", "IG",
                   "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"]

    # Ma tran: [num_clients x num_classes]
    dist_matrix = np.zeros((num_clients, num_classes), dtype=int)
    for client_id, indices in enumerate(partition):
        client_labels = labels[indices]
        for c in range(num_classes):
            dist_matrix[client_id, c] = int(np.sum(client_labels == c))

    # Ve stacked bar
    fig, ax = plt.subplots(figsize=(max(8, num_clients * 1.2), 5))
    x = np.arange(num_clients)
    bottom = np.zeros(num_clients)
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))

    for c in range(num_classes):
        ax.bar(x, dist_matrix[:, c], bottom=bottom,
               label=class_names[c], color=colors[c])
        bottom += dist_matrix[:, c]

    ax.set_title(f"Data Distribution | alpha={alpha} | {num_clients} Clients")
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
