# FedACM: Attack Concept Memory for Class-Incremental Intrusion Detection in IoT/IIoT Environments

Public implementation of **FedACM**, a federated class-incremental intrusion-detection method
built around a compact server-side **Attack Concept Memory**.

Clients keep their traffic local. Alongside the usual model update, each client returns a
class-level summary of its local evidence — a latent prototype with its support, dispersion
and predictive uncertainty. The server aggregates the shared encoder conventionally and
updates the memory class by class through **Reliability-Aware Concept Aggregation (RACA)**,
which favours well-supported, confident, compact and temporally consistent concepts. A
drift-compensation step then realigns stored prototypes with the evolving encoder. Detection
is nearest-prototype matching over the accumulated concepts, without task identifiers and
without storing historical traffic.

This repository contains the minimum code needed to run and verify FedACM on the three
benchmarks of the paper. It is deliberately small: it is not a replica of the full
experimental campaign. Section and equation numbers below refer to the manuscript.

---

## Repository layout

```
.
├── fedacm_edgeiiot/          Edge-IIoTset
├── fedacm_rtiot22/           RT-IoT2022
├── fedacm_xiiotid/           X-IIoTID
│   ├── pyproject.toml        run configuration, commented key by key
│   └── fedacm_ids/
│       ├── task.py           data pipeline, backbone, local objective, metrics
│       ├── strategy.py       server aggregation, RACA, drift compensation
│       ├── config.py         method presets and ablation flags
│       ├── client_app.py     Flower ClientApp
│       ├── server_app.py     Flower ServerApp
│       └── metrics.py        client-level metric aggregation
├── requirements.txt
└── LICENSE
```

One entry point per dataset, which keeps each configuration explicit and reviewable. The
three apps share an identical `fedacm_ids/` package except for `task.py`, which differs only
in the target column, the label taxonomy and the default class-to-stage assignment.

## Method and code map

| Manuscript | Where it lives |
|---|---|
| Stage↔round mapping, Eq. (1) | `strategy.py::FedACMStrategy._stage_for_round` |
| Local objective, Eq. (2) | `task.py::train_fedacm` |
| Prototype alignment / margin, Eqs. (3a)–(3b) | `task.py::_prototype_alignment_loss`, `_prototype_margin_loss` |
| Evidence tuple, Eqs. (4a)–(4c) | `task.py::compute_local_concepts` |
| RACA reliability score, Eq. (5) | `strategy.py::FedACMStrategy._update_memory_raca` |
| Normalised weights and candidate, Eqs. (6a)–(6b) | `strategy.py::FedACMStrategy._commit_memory_update` |
| Memory smoothing, Eq. (7) | `strategy.py::FedACMStrategy._commit_memory_update` |
| Prototype drift compensation, Eqs. (8)–(10) | `strategy.py::FedACMStrategy._compensate_prototype_drift` |
| Nearest-prototype inference, Eq. (11) | `task.py::evaluate_fedacm` |
| Dirichlet partitioning, Eq. (12) | `task.py::_dirichlet_client_indices` |
| Macro/weighted/new/old F1, Eqs. (13)–(16) | `task.py::evaluate_fedacm` |
| Forgetting, Eqs. (17)–(18) | `strategy.py::FedACMStrategy.aggregate_evaluate` |
| Global vs. client-level evaluation, §4.5 | `task.py::evaluate_global` (`cen_*` metrics) vs. `metrics.py::weighted_average` |
| Backbone and hyperparameters, §4.6 | `task.py::TabularFedACM`, `pyproject.toml` |

---

## Installation

Python 3.10+ (3.11 was used for the reported runs).

```bash
git clone https://github.com/Minivipiu/FedACM-for-Class-Incremental-IDS-in-IoT-IIoT-Environments.git
cd FedACM-for-Class-Incremental-IDS-in-IoT-IIoT-Environments
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `flwr[simulation]==1.19.0`, the version used for the reported runs.
A CUDA-capable GPU is optional; the code falls back to CPU.

---

## Datasets

The three benchmarks are public and are **not redistributed here**. Download them from the
original sources and prepare them as described in Section 4.2 of the manuscript. Each dataset
is distributed under the licence set by its own authors; please cite the original papers.

| Dataset | Source | DOI | App |
|---|---|---|---|
| Edge-IIoTset | [IEEE DataPort](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications) · [Kaggle](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot) | `10.21227/mbc1-1h68` | `fedacm_edgeiiot/` |
| RT-IoT2022 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/942/rt-iot2022) | `10.24432/C5P338` | `fedacm_rtiot22/` |
| X-IIoTID | [IEEE DataPort](https://ieee-dataport.org/documents/x-iiotid-connectivity-and-device-agnostic-intrusion-dataset-industrial-internet-things) | `10.21227/mpb6-py55` | `fedacm_xiiotid/` |

### Expected input

Each app reads a single CSV, given by the `dataset-path` key of its `pyproject.toml`. **Edit
that path before the first run**, or override it per run with
`--run-config "dataset-path='/your/path.csv'"`.

The three apps expect the CSV already prepared, as described in Section 4.2: every column
except the target must be numeric, and the target column must hold integer class ids. The
loader only replaces non-finite values and performs the stratified 80/20 split, so no
transformation is fitted at run time. On first use it caches the arrays next to the CSV as
`<name>_X.npy` / `<name>_y.npy`, which speeds up later runs; delete those two files if you
regenerate the CSV.

The target column is `Attack` for Edge-IIoTset, `Attack_type` for RT-IoT2022 and `class2`
for X-IIoTID, the sub-category attack label level.

Shapes after preparation (manuscript Table 3). Each app prints these at startup, so you can
use them to check your copy:

| Dataset | Samples | Features | Classes |
|---|---|---|---|
| Edge-IIoTset | 201,393 | 80 | 15 |
| RT-IoT2022 | 123,117 | 94 | 12 |
| X-IIoTID | 203,400 | 48 | 10 |

### Class identifiers

These are the ids used by `class-stages` and by manuscript Table 4. Edge-IIoTset and
RT-IoT2022 names come from `LABEL_NAMES` in the corresponding `task.py`:

| Edge-IIoTset | | RT-IoT2022 | |
|---|---|---|---|
| 0 Backdoor · 1 DDoS_HTTP · 2 DDoS_ICMP | 3 DDoS_TCP · 4 DDoS_UDP · 5 Fingerprinting | 0 ARP_poisioning · 1 DDOS_Slowloris | 2 DOS_SYN_Hping · 3 MQTT_Publish |
| 6 MITM · 7 Normal · 8 Password | 9 Port_Scanning · 10 Ransomware | 4 Metasploit_Brute_Force_SSH · 5 NMAP_FIN_SCAN | 6 NMAP_OS_DETECTION · 7 NMAP_TCP_scan |
| 11 SQL_injection · 12 Uploading | 13 Vulnerability_scanner · 14 XSS | 8 NMAP_UDP_SCAN · 9 NMAP_XMAS_TREE_SCAN | 10 Thing_Speak · 11 Wipro_bulb |

For X-IIoTID the ids are the integer values of `class2` itself; `LABEL_NAMES` carries
placeholder names, which you can replace with the taxonomy of your copy for readable
per-class metrics.

---

## Running FedACM

One command per dataset, using the configuration that produced the reported results:

```bash
cd fedacm_edgeiiot && flwr run .
cd fedacm_rtiot22  && flwr run .
cd fedacm_xiiotid  && flwr run .
```

Any key of `pyproject.toml` can be overridden without editing the file, for example:

```bash
flwr run . --run-config "seed=7"
flwr run . --run-config "dataset-path='/data/EdgeIIoT_cleaned.csv' partition-alpha=0.7"
```

A run covers the full five-stage protocol over ten communication rounds and takes roughly
33–41 s on Edge-IIoTset on the reference machine of manuscript Section 4.1 (Intel Core
i7-13700H, 32 GB RAM, NVIDIA RTX 4060 Laptop, Ubuntu 24.04, Python 3.11).

Quick check that the installation works, without touching the paper protocol:

```bash
flwr run . --run-config "num-stages=1 rounds-per-stage=1 local-epochs=1 \
  min-available-clients=2 max-train-samples-per-client=64 max-test-samples-per-client=64" \
  --federation-config "options.num-supernodes=2"
```

### What a run reports

Flower's history contains the per-round federated metrics (the client-level view of §4.5) and,
prefixed with `cen_`, the centralized metrics computed on the pooled global test set covering
all classes seen so far (the global view). `mean_forgetting` and `cen_mean_forgetting` are the
forgetting scores of Eqs. (17)–(18).

---

## Parameters

Defaults shared by the three datasets (manuscript §4.3 and §4.6):

| Key | Value | Meaning |
|---|---|---|
| `num-stages` / `rounds-per-stage` | 5 / 2 | ten communication rounds over the incremental sequence |
| `local-epochs` / `batch-size` | 3 / 64 | local optimisation per round |
| `learning-rate` / `embedding-dim` | 0.001 / 32 | Adam, latent dimension |
| `partition-alpha` | 0.3 | Dirichlet concentration `alpha_D`, Eq. (12) |
| `recurrent-class-prob` | 0.35 | probability that a previously seen class reappears |
| `min-available-clients` | 10 | federation size (match `options.num-supernodes`) |
| `lambda-proto` / `lambda-margin` / `lambda-kd` | 0.25 / 0.05 / 0.05 | `lambda_P`, `lambda_M`, `lambda_KD` of Eq. (2) |
| `prototype-margin` / `kd-temperature` | 1.0 / 2.0 | `m`, `T_d` |
| `memory-eta` / `drift-tau` | 0.65 / 1.0 | `eta` of Eq. (7), `tau` of Eq. (9a) |
| `raca-alpha` / `raca-epsilon` | 1.0 / 1e-6 | support exponent, numerical stability |
| `max-train-samples-per-client` | 0 | no sampling cap: each client uses its full local partition |

Dataset-specific values, selected on a validation subset (manuscript Table 5):

| Dataset | `raca-beta` | `raca-gamma` | `raca-delta` | `fedprox-mu` |
|---|---|---|---|---|
| Edge-IIoTset | 0.25 | 0.10 | 0.02 | 0.0100 |
| RT-IoT2022 | 1.00 | 1.00 | 0.10 | 0.0025 |
| X-IIoTID | 1.00 | 1.00 | 0.10 | 0.0100 |

Class-to-stage assignment, fixed across methods (manuscript Table 4):

| Dataset | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 |
|---|---|---|---|---|---|
| Edge-IIoTset | 7,1,2,3,4 | 5,9,13 | 0,8,10 | 11,12,14 | 6 |
| RT-IoT2022 | 3,10,11,1,2 | 6,7,8 | 5,9 | 0 | 4 |
| X-IIoTID | 4,5 | 6,1 | 3,8 | 0,2 | 7,9 |

### Seeds

The apps ship with a single default seed, `seed = 42`, changed with
`--run-config "seed=<n>"`. **The results reported in the paper are averaged over 15
independent runs with 15 distinct random seeds** (manuscript §4.6); this repository does not
automate that campaign. The seed drives the train/test split, the Dirichlet partition, the
client exposure sequence and the `random`/`numpy`/`torch` state, so one seed yields the same
data realization for every method, which is what makes the runs paired.

`set_random_seed` also sets `cudnn.deterministic = True`. Exact bitwise reproduction across
different GPUs, driver versions or PyTorch builds is not guaranteed; run-to-run variation on
the same machine is what the reported standard deviations cover.

### Model selection

Hyperparameter selection never touches the test partition. `validation-fraction=0.16` carves a
stratified validation subset out of the 80 % training partition (a 64/16/20 split) and
`evaluation-split='validation'` scores the run on it:

```bash
flwr run . --run-config "validation-fraction=0.16 evaluation-split='validation' raca-gamma=0.10"
```

With the default `validation-fraction=0.0` the loader performs exactly one split, so the
default path is the one that produced the reported numbers. The validation subset is carved
from the training indices alone, so the test partition is identical in both modes and
`train + validation` recomposes the original 80 % exactly.

---

## Scope

The `method` key selects FedACM (the default) or one of the compared methods of §4.4, and the
`ablate-*` keys switch off one FedACM mechanism at a time; both are documented in each
`pyproject.toml` and resolved in `config.py`. They are included because they share this exact
code path, which makes the FedACM components verifiable, but the repository does not automate
the experimental campaign of the paper: it contains no sweep, ablation, scalability,
statistical-analysis, figure-generation or log-parsing tooling, and no experimental outputs.
<!-- 
## Citation

```bibtex
@article{garciasaez_fedacm,
  title   = {{FedACM}: Attack Concept Memory for Consistent Class-Incremental
             Intrusion Detection across Heterogeneous Edge {IoT}/{IIoT} Environments},
  author  = {Garc{\'i}a-S{\'a}ez, Luis Miguel and Mora, Alessio and
             Romandini, Nicol{\`o} and Rold{\'a}n-G{\'o}mez, Jos{\'e} and
             Mart{\'i}nez, Jos{\'e} Luis},
  year    = {2026}
}
```
-->
## License

MIT. See [`LICENSE`](LICENSE).
