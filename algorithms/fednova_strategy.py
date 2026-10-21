"""
Custom FedNova Strategy cho Flower (Li et al., NeurIPS 2020).
Federated Normalized Averaging (FedNova) giải quyết vấn đề Objective Inconsistency
bằng cách chuẩn hóa gradient cục bộ theo số bước cập nhật (tau).
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


class FedNova(FedAvg):
    """
    Chiến lược FedNova kế thừa từ Flower FedAvg.
    Công thức tổng hợp chuẩn:
      - p_i = n_i / sum(n_j)
      - d_i = (w_0 - w_i) / tau_i   (Normalized gradient direction)
      - tau_eff = sum(p_i * tau_i)  (Effective number of steps)
      - D = sum(p_i * d_i)
      - w_new = w_0 - tau_eff * D
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.initial_parameters is not None:
            self.current_weights: Optional[List[np.ndarray]] = parameters_to_ndarrays(
                self.initial_parameters
            )
        else:
            self.current_weights = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[object, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Tổng hợp trọng số từ các client theo thuật toán chuẩn hóa FedNova."""
        if not results:
            return None, {}

        # 1. Trích xuất trọng số, num_samples, tau từ từng client
        weights_list = []
        num_samples_list = []
        tau_list = []

        for _, fit_res in results:
            w_i = parameters_to_ndarrays(fit_res.parameters)
            n_i = fit_res.num_examples
            # Lấy tau từ metrics (mặc định 1.0 nếu thiếu)
            tau_i = float(fit_res.metrics.get("tau", 1.0))
            if tau_i <= 0.0:
                tau_i = 1.0

            weights_list.append(w_i)
            num_samples_list.append(n_i)
            tau_list.append(tau_i)

        total_samples = sum(num_samples_list)
        if total_samples == 0:
            return None, {}

        p_list = [n / total_samples for n in num_samples_list]

        # 2. Nếu chưa có current_weights (ví dụ round đầu tiên chưa khởi tạo),
        # fallback dùng weighted average của w_i
        if self.current_weights is None:
            w0 = weights_list[0]
            self.current_weights = [np.copy(layer) for layer in w0]

        w0 = self.current_weights

        # 3. Tính tau_eff = sum(p_i * tau_i)
        tau_eff = sum(p * tau for p, tau in zip(p_list, tau_list))

        # 4. Tính normalized gradient direction D = sum(p_i * (w0 - w_i) / tau_i)
        # và w_new = w0 - tau_eff * D
        w_new = [np.copy(layer) for layer in w0]
        num_layers = len(w0)

        for p_i, tau_i, w_i in zip(p_list, tau_list, weights_list):
            weight_factor = (p_i * tau_eff) / tau_i
            for layer_idx in range(num_layers):
                # Delta w_i = (w0 - w_i)
                # w_new = w0 - sum( weight_factor * (w0 - w_i) )
                w_new[layer_idx] -= weight_factor * (w0[layer_idx] - w_i[layer_idx])

        # 5. Lưu lại w_new làm trọng số global cho round tiếp theo
        self.current_weights = [np.copy(layer) for layer in w_new]

        # 6. Tổng hợp metrics phụ (nếu có)
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return ndarrays_to_parameters(w_new), metrics_aggregated
