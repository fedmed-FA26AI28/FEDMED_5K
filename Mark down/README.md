# FedMedAI: Federated Learning for Medical Image Diagnostics

This research project focuses on building and evaluating a Federated Learning (FL) system using the **Flower** framework and **PyTorch**. The primary task is medical image classification using the [BloodMNIST](https://medmnist.com/) dataset (classifying blood cells into 8 distinct categories).

The system is designed to train a lightweight CNN (`SimpleCNN`) across three different environments:
1. **Centralized Baseline**: Traditional centralized training (used as an upper-bound benchmark).
2. **Simulated Federated Learning**: Simulating an FL network locally on a PC using Ray.
3. **Real-world Deployment**: Deploying physical FL over a network between a PC (Server) and an NVIDIA Jetson (Client).

---

## Part 1: Centralized Baseline

Centralized training gathers all data onto a single machine. The purpose of this phase is to establish the upper-bound performance that the `SimpleCNN` model can achieve on the BloodMNIST dataset without data fragmentation.

### Usage
```bash
python -m experiments.train_centralized
```
Hyperparameters (epochs, learning rate, etc.) are managed globally in `configs/experiment.yaml`.

### Results Achieved
*Training Setup: 100 Epochs, Early Stopping, Learning Rate Scheduler*

* **Training Time:** ~68.12 seconds (On a high-performance PC)
* **Accuracy:** 90.41%
* **Final Loss:** 0.2789

**Precision, Recall, and F1-Score (Macro Avg):**
* **Precision:** 89.28%
* **Recall:** 89.47%
* **F1-Score:** 89.28%

**Class-wise Metrics:**
| Label | Cell Type | Precision | Recall | F1-Score |
|---|---|---|---|---|
| 0 | Basophil | 92.17% | 87.60% | 89.83% |
| 1 | Eosinophil | 92.83% | 94.46% | 93.63% |
| 2 | Erythroblast | 90.72% | 88.08% | 89.38% |
| 3 | Immature Granulocytes | 92.20% | 88.75% | 90.44% |
| 4 | Lymphocyte | 86.81% | 86.44% | 86.63% |
| 5 | Monocyte | 88.29% | 93.91% | 91.01% |
| 6 | Neutrophil | 89.41% | 92.42% | 90.89% |
| 7 | Platelet | 81.82% | 84.11% | 82.95% |

---

## Part 2: Simulated Federated Learning

This phase uses the Flower Virtual Client Engine (backed by Ray) to spin up virtual clients. Data is partitioned in a Non-IID fashion using a Dirichlet distribution (controlled by the $\alpha$ parameter) to simulate real-world medical data imbalances across different hospitals.

Supported FL Strategies: `FedAvg`, `FedProx`, `FedBN`.

### Usage
Generate data partitions first (creates `.json` partition files):
```bash
python -m datasets.partition
```

Run all experimental configurations defined in `experiment.yaml`:
```bash
python -m experiments.run_simulation --all
```
Run a specific configuration:
```bash
python -m experiments.run_simulation --num_clients 10 --alpha 0.3 --strategy fedprox
```
A comprehensive evaluation report is automatically generated as a CSV at `results/simulate_federated/comparison_table.csv`.

### Results (To be updated)
*Simulation results will be populated here once experiments are completed.*

---

## Part 3: Real-world NVIDIA Jetson Deployment

Deploying the physical FL architecture over a LAN/Wi-Fi connection:
* A PC acts as the Global Server, aggregating model weights via gRPC.
* NVIDIA Jetson boards (Nano/Orin) act as Edge Clients performing local training.

*(For detailed instructions on IP configuration and data distribution, see `JETSON_GUIDE.md`)*

### Usage
**Step 1: Start the Server on the PC**
```bash
python -m experiments.run_server --strategy fedavg --num_clients 1 --alpha 1.0
```

**Step 2: Start the Client on the Jetson**
Wait for the server to display a listening status, then execute on the Jetson:
```bash
python experiments/run_jetson_client.py --server_ip <PC_IP_ADDRESS> --client_id 0
```
The process runs autonomously while tracking communication costs (Upload/Download MB) and hardware utilization (Thermal/GPU peaks).

### Results (To be updated)
*Empirical metrics regarding edge hardware utilization, bandwidth consumption, and FL convergence will be added here.*
