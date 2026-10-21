"""
Custom Strategy kết hợp FedNova + SCAFFOLD (SCAFFOLD-Nova).
- Chuẩn hóa cập nhật trọng số toàn cục theo FedNova (giải quyết Objective Inconsistency)
- Hiệu chỉnh gradient và cập nhật c_global theo SCAFFOLD (giải quyết Client Drift)
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
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

SCAFFOLD_STATE_DIR = Path("data/scaffold_states")


class FedNovaScaffold(FedAvg):
    """
    Chiến lược kết hợp FedNova (normalized aggregation) + SCAFFOLD (control variates).
    """

    def __init__(self, num_clients: int = 12, reset_state: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_clients = num_clients
        self.c_global: Optional[List[torch.Tensor]] = None
        if self.initial_parameters is not None:
            self.current_weights: Optional[List[np.ndarray]] = parameters_to_ndarrays(
                self.initial_parameters
            )
        else:
            self.current_weights = None

        if reset_state:
            self._reset_scaffold_dir()
        else:
            SCAFFOLD_STATE_DIR.mkdir(parents=True, exist_ok=True)

    def _reset_scaffold_dir(self):
        if SCAFFOLD_STATE_DIR.exists():
            try:
                shutil.rmtree(SCAFFOLD_STATE_DIR)
            except Exception:
                pass
        SCAFFOLD_STATE_DIR.mkdir(parents=True, exist_ok=True)

    def _save_c_global(self):
        if self.c_global is not None:
            state_file = SCAFFOLD_STATE_DIR / "c_global.pkl"
            with open(state_file, "wb") as f:
                pickle.dump(self.c_global, f)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        self._save_c_global()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[object, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # 1. Trích xuất thông tin từ client: weights, num_samples, tau, client_id
        weights_list = []
        num_samples_list = []
        tau_list = []
        client_ids = []

        for _, fit_res in results:
            w_i = parameters_to_ndarrays(fit_res.parameters)
            n_i = fit_res.num_examples
            tau_i = float(fit_res.metrics.get("tau", 1.0))
            cid = fit_res.metrics.get("client_id", None)

            weights_list.append(w_i)
            num_samples_list.append(n_i)
            tau_list.append(tau_i if tau_i > 0 else 1.0)
            if cid is not None:
                client_ids.append(int(cid))

        total_samples = sum(num_samples_list)
        if total_samples == 0:
            return None, {}

        # 2. FedNova: Chuẩn hóa theo tau
        p_list = [n / total_samples for n in num_samples_list]
        if self.current_weights is None:
            self.current_weights = [np.copy(layer) for layer in weights_list[0]]

        w0 = self.current_weights
        tau_eff = sum(p * tau for p, tau in zip(p_list, tau_list))

        w_new = [np.copy(layer) for layer in w0]
        num_layers = len(w0)

        for p_i, tau_i, w_i in zip(p_list, tau_list, weights_list):
            weight_factor = (p_i * tau_eff) / tau_i
            for layer_idx in range(num_layers):
                w_new[layer_idx] -= weight_factor * (w0[layer_idx] - w_i[layer_idx])

        self.current_weights = [np.copy(layer) for layer in w_new]

        # 3. SCAFFOLD: Cập nhật c_global từ delta_c
        delta_c_list = []
        for cid in client_ids:
            delta_file = SCAFFOLD_STATE_DIR / f"delta_c_{cid}.pkl"
            if delta_file.exists():
                try:
                    with open(delta_file, "rb") as f:
                        delta_c_list.append(pickle.load(f))
                except Exception as e:
                    print(f"   [FedNovaScaffold] Cảnh báo không đọc được {delta_file}: {e}")

        if delta_c_list:
            if self.c_global is None:
                self.c_global = [torch.zeros_like(t) for t in delta_c_list[0]]

            scale = 1.0 / float(self.num_clients)
            for delta_c in delta_c_list:
                for j, dt in enumerate(delta_c):
                    self.c_global[j].add_(dt * scale)

            self._save_c_global()

        # 4. Metrics phụ
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return ndarrays_to_parameters(w_new), metrics_aggregated
