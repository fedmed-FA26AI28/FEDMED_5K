"""
Custom SCAFFOLD Strategy cho Flower (Karimireddy et al., ICML 2020).
Duy trì và cập nhật Global Control Variate (c_global) sau mỗi vòng liên kết,
đồng bộ trạng thái với các Client thông qua file pickle trên disk.
"""

import pickle
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)

SCAFFOLD_STATE_DIR = Path("data/scaffold_states")


class SCAFFOLD(FedAvg):
    """
    SCAFFOLD Strategy kế thừa từ FedAvg:
      - Quản lý c_global
      - Cập nhật c_global += (1 / N) * sum(delta_c_i)
    """

    def __init__(self, num_clients: int = 12, reset_state: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_clients = num_clients
        self.c_global: Optional[List[torch.Tensor]] = None

        if reset_state:
            self._reset_scaffold_dir()
        else:
            SCAFFOLD_STATE_DIR.mkdir(parents=True, exist_ok=True)

    def _reset_scaffold_dir(self):
        """Xóa sạch thư mục trạng thái cũ khi bắt đầu một run mới."""
        if SCAFFOLD_STATE_DIR.exists():
            try:
                shutil.rmtree(SCAFFOLD_STATE_DIR)
            except Exception:
                pass
        SCAFFOLD_STATE_DIR.mkdir(parents=True, exist_ok=True)

    def _save_c_global(self):
        """Lưu c_global ra disk để tất cả Client đọc."""
        if self.c_global is not None:
            state_file = SCAFFOLD_STATE_DIR / "c_global.pkl"
            with open(state_file, "wb") as f:
                pickle.dump(self.c_global, f)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Đảm bảo c_global được ghi ra disk trước khi Client bắt đầu round huấn luyện."""
        self._save_c_global()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[object, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Tổng hợp trọng số mô hình và cập nhật c_global từ delta_c của các client."""
        # 1. Tổng hợp trọng số mô hình theo FedAvg
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        if not results:
            return aggregated_parameters, metrics_aggregated

        # 2. Đọc delta_c của các client tham gia
        delta_c_list = []
        for _, fit_res in results:
            client_id = fit_res.metrics.get("client_id", None)
            if client_id is not None:
                delta_file = SCAFFOLD_STATE_DIR / f"delta_c_{int(client_id)}.pkl"
                if delta_file.exists():
                    try:
                        with open(delta_file, "rb") as f:
                            delta_c = pickle.load(f)
                            # Kiểm tra NaN / Inf để tránh làm ô nhiễm c_global
                            if not any(torch.isnan(t).any() or torch.isinf(t).any() for t in delta_c):
                                delta_c_list.append(delta_c)
                    except Exception as e:
                        print(f"   [SCAFFOLD Server] Cảnh báo không đọc được {delta_file}: {e}")

        # 3. Khởi tạo c_global nếu round đầu tiên
        if delta_c_list:
            if self.c_global is None:
                self.c_global = [torch.zeros_like(t) for t in delta_c_list[0]]

            # 4. Cập nhật: c_global += (1 / N) * sum(delta_c_i)
            # Theo paper: delta_c = c_i_new - c_i
            # c_global_new = c_global_old + (1 / N_total) * sum_{i in S} delta_c_i
            scale = 1.0 / float(self.num_clients)
            for delta_c in delta_c_list:
                for j, dt in enumerate(delta_c):
                    self.c_global[j].add_(dt * scale)

            # Norm clipping: Giới hạn chuẩn norm của c_global <= 5.0 để ngăn runaway drift
            for j in range(len(self.c_global)):
                c_norm = torch.norm(self.c_global[j]).item()
                if c_norm > 5.0:
                    self.c_global[j].mul_(5.0 / c_norm)

            # Lưu lại c_global mới
            self._save_c_global()

        return aggregated_parameters, metrics_aggregated
