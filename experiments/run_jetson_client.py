"""
FL Client thực - Chạy trên Jetson, kết nối tới PC Server qua Wi-Fi.

Cách dùng (trên Jetson):
    python run_jetson_client.py --server_ip 192.168.1.X --client_id 0 --alpha 1.0
"""

import argparse
import sys
import json
import glob
import time
import threading
import subprocess
import numpy as np
import torch
import flwr as fl
from pathlib import Path
from datetime import datetime

from models.cnn import build_model, get_parameters, set_parameters
from client.train import train
from client.evaluate import evaluate
from algorithms.coverage import count_client_classes, snapshot_classifier_head
from datasets.sampling import balanced_loader
from torch.utils.data import DataLoader, Subset
from datasets.medmnist_code import get_bloodmnist_dataset, build_transform
from datasets.partition import load_partition, stratified_holdout_indices, stratified_validation_partition

import yaml
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
with open(CONFIG_DIR / "experiment.yaml", "r", encoding="utf-8") as f:
    EXP_CFG = yaml.safe_load(f)
with open(CONFIG_DIR / "jetson.yaml", "r", encoding="utf-8") as f:
    JETSON_CFG = yaml.safe_load(f)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "real_federated"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─── HÀM LẤY THÔNG TIN PHẦN CỨNG JETSON ─────────────────────────────────────
def get_hardware_info() -> dict:
    """Thu thập thông tin tĩnh phần cứng của Jetson khi khởi động."""
    info = {"timestamp": datetime.now().isoformat()}

    # Tên model thiết bị (chỉ có trên Jetson Linux)
    try:
        with open("/proc/device-tree/model", "r") as f:
            info["device_model"] = f.read().strip("\x00").strip()
    except Exception:
        info["device_model"] = "Unknown"

    # CPU + RAM
    try:
        import psutil
        info["cpu_count"]    = psutil.cpu_count(logical=True)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        info["cpu_count"]    = -1
        info["ram_total_gb"] = -1

    # GPU (PyTorch CUDA)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"]      = props.name
        info["vram_total_mb"] = props.total_memory // (1024 ** 2)
    else:
        info["gpu_name"]      = "No CUDA"
        info["vram_total_mb"] = 0

    return info


def get_temperature() -> dict:
    """
    Đọc nhiệt độ từ thermal zones của Linux (Jetson).
    Fallback sang nvidia-smi nếu chạy trên PC.
    """
    # Cách 1: Đọc từ /sys/class/thermal (Jetson / Linux chung)
    temps = {}
    for zone_path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        zone_name = zone_path.split("/")[-2]
        try:
            with open(zone_path, "r") as f:
                # Giá trị tính bằng milli-Celsius → chia 1000
                temps[zone_name] = round(float(f.read().strip()) / 1000.0, 1)
        except Exception:
            pass
    if temps:
        return temps

    # Cách 2: nvidia-smi (PC thường)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return {"gpu": float(result.stdout.strip())}
    except Exception:
        pass

    return {}


def get_params_size_mb(params: list) -> float:
    """Tính dung lượng (MB) của bộ tham số model (upload hoặc download)."""
    total_bytes = sum(p.nbytes for p in params)
    return round(total_bytes / (1024 ** 2), 4)


# ─── BACKGROUND MONITOR NHIỆT ĐỘ & GPU ──────────────────────────────────────
class HardwareMonitor:
    """Theo dõi nhiệt độ và GPU util trong background, ghi nhận peak."""
    def __init__(self, interval: float = 3.0):
        self.interval       = interval
        self._stop_event    = threading.Event()
        self._thread        = None
        self.temp_records   = []   # [{zone: temp_C}]
        self.gpu_util_records = [] # [%]
        self.vram_used_records = [] # [MB]

    def _loop(self):
        while not self._stop_event.is_set():
            self.temp_records.append(get_temperature())

            # VRAM used
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated(0) // (1024 ** 2)
                self.vram_used_records.append(vram_used)

            self._stop_event.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        """Trả về thống kê peak nhiệt độ và VRAM."""
        result = {}
        # Peak VRAM
        if self.vram_used_records:
            result["peak_vram_used_mb"] = max(self.vram_used_records)
            result["avg_vram_used_mb"]  = round(
                sum(self.vram_used_records) / len(self.vram_used_records), 1)

        # Peak nhiệt độ theo từng zone
        if self.temp_records:
            all_zones = set(k for rec in self.temp_records for k in rec)
            peak_temps = {}
            for zone in all_zones:
                vals = [rec[zone] for rec in self.temp_records if zone in rec]
                if vals:
                    peak_temps[zone] = max(vals)
            result["peak_temperatures_c"] = peak_temps

        return result


# ─── FLOWER CLIENT ────────────────────────────────────────────────────────────
class JetsonFlowerClient(fl.client.NumPyClient):
    def __init__(self, net, train_loader, val_loader,
                 local_epochs, lr, device, client_id=0, seed=42):
        self.net          = net
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.client_id = client_id
        self.seed = seed
        self.class_counts = count_client_classes(train_loader.dataset, net.fc.out_features)
        self.local_epochs = local_epochs
        self.lr           = lr
        self.device       = device
        # Lưu lại thống kê từng round
        self.upload_mb_per_round   = []  # Dung lượng gửi lên Server
        self.download_mb_per_round = []  # Dung lượng nhận từ Server
        self.fit_time_per_round    = []  # Thời gian train mỗi round
        self.eval_time_per_round   = []  # Thời gian eval mỗi round

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        """Nhận trọng số Server → train cục bộ → gửi về."""
        round_seed = self.seed + 1009 * int(config.get("server_round", 0)) + self.client_id
        torch.manual_seed(round_seed)
        np.random.seed(round_seed % (2 ** 32 - 1))
        # Đo download (Server gửi xuống)
        download_mb = get_params_size_mb(parameters)
        self.download_mb_per_round.append(download_mb)

        set_parameters(self.net, parameters)
        current_lr = float(config.get("lr", self.lr))
        optimizer  = torch.optim.Adam(self.net.parameters(), lr=current_lr)
        head_mu = float(config.get("head_mu", 0.0))
        local_train = self.train_loader
        if bool(config.get("balanced_sampling", False)):
            local_train = balanced_loader(
                self.train_loader, self.class_counts,
                self.seed + 1009 * int(config.get("server_round", 0)) + self.client_id,
            )

        t0 = time.time()
        train_metrics = train(
            model=self.net,
            train_loader=local_train,
            optimizer=optimizer,
            epochs=self.local_epochs,
            device=self.device,
            val_loader=self.val_loader,
            class_counts=self.class_counts,
            logit_tau=float(config.get("logit_tau", 0.0)),
            prior_smoothing=float(config.get("prior_smoothing", 1.0)),
            head_mu=head_mu,
            coverage_kappa=float(config.get("coverage_kappa", 32.0)),
            global_head=snapshot_classifier_head(self.net) if head_mu > 0 else None,
            proximal_mu=float(config.get("proximal_mu", 0.0)),
            global_params=(
                [parameter.detach().clone() for parameter in self.net.parameters()]
                if float(config.get("proximal_mu", 0.0)) > 0 else None
            ),
        )
        self.fit_time_per_round.append(round(time.time() - t0, 2))

        # Đo upload (Client gửi lên)
        updated_params = get_parameters(self.net)
        upload_mb = get_params_size_mb(updated_params)
        self.upload_mb_per_round.append(upload_mb)

        return updated_params, len(self.train_loader.dataset), {
            "train_loss": float(train_metrics["final_loss"]),
            "train_accuracy": float(train_metrics["final_accuracy"]),
            "upload_mb": float(upload_mb),
        }

    def evaluate(self, parameters, config):
        """Nhận trọng số Server → đánh giá → gửi metrics về."""
        set_parameters(self.net, parameters)

        t0 = time.time()
        loss, eval_metrics = evaluate(self.net, self.val_loader, self.device)
        self.eval_time_per_round.append(round(time.time() - t0, 2))

        return loss, len(self.val_loader.dataset), {
            "accuracy":  float(eval_metrics["accuracy"]),
            "precision": float(eval_metrics["precision"]),
            "recall":    float(eval_metrics["recall"]),
            "f1_score":  float(eval_metrics["f1_score"]),
        }


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FedMedAI - Jetson FL Client")
    parser.add_argument("--server_ip",   type=str, required=True)
    parser.add_argument("--port",        type=int, default=8080)
    parser.add_argument("--client_id",   type=int, default=0)
    parser.add_argument("--alpha",       type=float, default=1.0)
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--size", type=int, choices=[28, 64], default=28)
    parser.add_argument("--model", choices=["legacy", "tiny_cnn", "mobilenet_v3_small"], default="legacy")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--partition_path", default=None)
    parser.add_argument("--device_type", type=str, default=None,
                        help="Ghi đè: jetson_orin hoặc jetson_nano")
    args = parser.parse_args()

    # ─── Hardware config ─────────────────────────────────────────────────────
    registry    = JETSON_CFG.get("client_registry", {})
    device_type = args.device_type or registry.get(args.client_id, "jetson_orin")
    hw_cfg      = JETSON_CFG.get(device_type, JETSON_CFG["jetson_orin"])
    batch_size  = hw_cfg["batch_size"]
    num_workers = hw_cfg["num_workers"]
    pin_memory  = hw_cfg["pin_memory"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Thu thập hardware info ngay khi khởi động ───────────────────────────
    hw_info = get_hardware_info()
    hw_info["device_type"]  = device_type
    hw_info["torch_device"] = str(device)
    hw_info["batch_size"]   = batch_size

    print(f"  Device    : {hw_info.get('device_model', 'Unknown')}")
    print(f"  GPU       : {hw_info.get('gpu_name', 'N/A')} | VRAM: {hw_info.get('vram_total_mb', 0)} MB")
    print(f"  RAM total : {hw_info.get('ram_total_gb', 'N/A')} GB")
    print(f"  Batch size: {batch_size}")

    # ─── Load partition ──────────────────────────────────────────────────────
    num_clients = args.num_clients
    partition_path = Path(args.partition_path) if args.partition_path else (
        Path(EXP_CFG["partition_save_dir"]) /
        f"partition_seed{args.seed}_alpha{args.alpha}_clients{num_clients}.json"
    )
    if not partition_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Partition file not found: {partition_path}\n"
            f"  Chay tren PC: python -m scripts.distribute_data --client_id {args.client_id}"
        )
    with open(partition_path, "r", encoding="utf-8") as f:
        partition_metadata = json.load(f)
    if partition_metadata.get("split") != "train":
        raise ValueError("Jetson partition must contain training indices only")
    if int(partition_metadata.get("total_samples", -1)) != args.train_samples:
        raise ValueError("partition size differs from --train_samples; regenerate the 5K partition")
    client_indices = load_partition(str(partition_path))[args.client_id]
    print(f"  Client {args.client_id}: {len(client_indices)} training samples")

    # ─── Dataset & Loader ───────────────────────────────────────────────────
    train_full, _ = get_bloodmnist_dataset("train", download=True, size=args.size)
    if any(index < 0 or index >= len(train_full) for index in client_indices):
        raise ValueError("partition contains an out-of-range training index")
    train_full.transform = build_transform("train", augment=args.augment)
    val_dataset, _ = get_bloodmnist_dataset("val", download=True, size=args.size)
    local_val, _ = stratified_holdout_indices(val_dataset, 0.5, seed=args.seed)
    val_indices = stratified_validation_partition(
        val_dataset, num_clients, seed=args.seed, eligible_indices=local_val
    )[args.client_id]
    train_subset = Subset(train_full, client_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset,  batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=pin_memory)

    # ─── Model ──────────────────────────────────────────────────────────────
    net = build_model(args.model, num_classes=8).to(device)

    # ─── Khởi động hardware monitor ─────────────────────────────────────────
    monitor = HardwareMonitor(interval=3.0)
    monitor.start()

    # ─── Kết nối Server và chạy FL ──────────────────────────────────────────
    server_address = f"{args.server_ip}:{args.port}"
    print(f"\n  Dang ket noi den Server: {server_address}")

    fl_client = JetsonFlowerClient(
        net=net,
        train_loader=train_loader,
        val_loader=val_loader,
        local_epochs=EXP_CFG["local_epochs"],
        lr=EXP_CFG["learning_rate"],
        device=device,
        client_id=args.client_id,
        seed=args.seed,
    )

    t_total_start = time.time()
    fl.client.start_numpy_client(server_address=server_address, client=fl_client)
    total_time = round(time.time() - t_total_start, 1)

    # ─── Dừng monitor và thu thập kết quả ───────────────────────────────────
    monitor.stop()
    hw_peak = monitor.summary()

    # ─── Tính thống kê truyền thông ─────────────────────────────────────────
    total_upload_mb   = round(sum(fl_client.upload_mb_per_round), 4)
    total_download_mb = round(sum(fl_client.download_mb_per_round), 4)
    total_comm_mb     = round(total_upload_mb + total_download_mb, 4)
    num_rounds        = len(fl_client.fit_time_per_round)

    # ─── Lưu kết quả client ─────────────────────────────────────────────────
    client_stats = {
        "client_id":       args.client_id,
        "server_ip":       args.server_ip,
        "alpha":           args.alpha,
        "num_rounds":      num_rounds,
        "total_time_s":    total_time,
        "hardware_info":   hw_info,
        "hardware_peak":   hw_peak,
        # Thời gian
        "fit_time_per_round_s":  fl_client.fit_time_per_round,
        "eval_time_per_round_s": fl_client.eval_time_per_round,
        "avg_fit_time_s":  round(sum(fl_client.fit_time_per_round) / max(num_rounds, 1), 2),
        # Truyền thông
        "upload_mb_per_round":   fl_client.upload_mb_per_round,
        "download_mb_per_round": fl_client.download_mb_per_round,
        "total_upload_mb":       total_upload_mb,
        "total_download_mb":     total_download_mb,
        "total_comm_mb":         total_comm_mb,
        "comm_per_round_mb":     round(total_comm_mb / max(num_rounds, 1), 4),
    }

    run_name = f"real_clients1_alpha{args.alpha}_client{args.client_id}"
    out_dir  = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "client_stats.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(client_stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"  HOAN THANH! {num_rounds} rounds")
    print(f"  Tong thoi gian  : {total_time}s")
    print(f"  Upload total    : {total_upload_mb} MB")
    print(f"  Download total  : {total_download_mb} MB")
    print(f"  Comm total      : {total_comm_mb} MB")
    if "peak_temperatures_c" in hw_peak:
        print(f"  Peak nhiet do   : {hw_peak['peak_temperatures_c']}")
    if "peak_vram_used_mb" in hw_peak:
        print(f"  Peak VRAM dung  : {hw_peak['peak_vram_used_mb']} MB")
    print(f"  Da luu: {out_file}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
