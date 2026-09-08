"""Data pipeline, backbone, local objective and evaluation for RT-IoT2022.

- ``load_and_preprocess_dataset``  stratified 80/20 split, plus the optional
  64/16/20 model-selection protocol of Section 4.6.
- ``parse_class_stages`` / ``_client_visible_classes``  the five-stage
  class-incremental protocol of Section 4.3.
- ``_dirichlet_client_indices``  class-wise Dirichlet partitioning, Eq. (12).
- ``TabularFedACM``  the shared MLP encoder plus auxiliary head.
- ``train_fedacm``  the composite local objective of Eq. (2).
- ``compute_local_concepts``  the evidence tuple of Eqs. (4a)-(4c).
- ``evaluate_fedacm``  nearest-prototype inference, Eq. (11), and the metrics of
  Eqs. (13)-(16).
- ``evaluate_global``  the pooled global evaluation view of Section 4.5.

Only this module differs between the three dataset apps, and only in the target
column, the label taxonomy and the default class-to-stage assignment.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


TARGET_COLUMN = "Attack_type"

LABEL_NAMES: Dict[int, str] = {
    0: "ARP_poisioning",
    1: "DDOS_Slowloris",
    2: "DOS_SYN_Hping",
    3: "MQTT_Publish",
    4: "Metasploit_Brute_Force_SSH",
    5: "NMAP_FIN_SCAN",
    6: "NMAP_OS_DETECTION",
    7: "NMAP_TCP_scan",
    8: "NMAP_UDP_SCAN",
    9: "NMAP_XMAS_TREE_SCAN",
    10: "Thing_Speak",
    11: "Wipro_bulb",
}

DEFAULT_CLASS_STAGES: Tuple[Tuple[int, ...], ...] = (
    (3, 10, 11, 1, 2),
    (6, 7, 8),
    (5, 9),
    (0,),
    (4,),
)



@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    class_ids: Tuple[int, ...]
    num_features: int
    num_outputs: int
    train_indices_by_class: Dict[int, np.ndarray]
    test_indices_by_class: Dict[int, np.ndarray]
    # Validation subset carved out of the training partition. Empty unless
    # ``validation-fraction`` is set, i.e. unless a tuning run is in progress.
    val_indices_by_class: Dict[int, np.ndarray] = field(default_factory=dict)
    # The map every evaluation path reads. It is ``test_indices_by_class`` for a
    # normal run and ``val_indices_by_class`` for a hyperparameter-search run, so
    # that the held-out test partition is never touched during model selection.
    eval_indices_by_class: Dict[int, np.ndarray] = field(default_factory=dict)


dataset_cache: dict[tuple[str, float, float, str, int], DatasetBundle] = {}
partition_cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = {}
num_classes: int | None = None
num_features: int | None = None
class_ids_global: Tuple[int, ...] | None = None


def set_random_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_int(config: dict, key: str, default: int) -> int:
    return int(config.get(key, default))


def _as_float(config: dict, key: str, default: float) -> float:
    return float(config.get(key, default))


def _load_arrays(dataset_path: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(dataset_path)
    x_cache = path.with_name(f"{path.stem}_X.npy")
    y_cache = path.with_name(f"{path.stem}_y.npy")

    if x_cache.exists() and y_cache.exists():
        X = np.load(x_cache).astype(np.float32, copy=False)
        y = np.load(y_cache).astype(np.int64, copy=False)
        return np.nan_to_num(X), y

    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET_COLUMN}'")

    X = df.drop(columns=[TARGET_COLUMN]).to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int64)
    return np.nan_to_num(X), y


def load_and_preprocess_dataset(
    dataset_path: str,
    test_fraction: float = 0.2,
    seed: int = 42,
    validation_fraction: float = 0.0,
    evaluation_split: str = "test",
) -> tuple[DatasetBundle, int, int]:
    """Load RT-IoT22 arrays and create a reusable global train/test split."""
    global class_ids_global, num_classes, num_features

    key = (
        dataset_path,
        float(test_fraction),
        float(validation_fraction),
        str(evaluation_split),
        int(seed),
    )
    if key in dataset_cache:
        bundle = dataset_cache[key]
        return bundle, bundle.num_outputs, bundle.num_features

    X, y = _load_arrays(dataset_path)
    class_ids = tuple(int(c) for c in sorted(np.unique(y)))
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=y,
    )

    # Model selection never sees the held-out test partition. When a validation
    # fraction is requested, it is carved out of ``train_idx`` only, leaving the
    # test split untouched. ``validation_fraction`` is expressed as a fraction of
    # the whole dataset, so the default protocol (0.2 test, 0.16 validation) gives
    # the 64/16/20 train/validation/test partitioning described in the manuscript.
    #
    # With ``validation_fraction == 0`` the two lines above are the only split
    # performed, which is exactly the behaviour of the final training runs: the
    # search is run once with the validation split, the winning configuration is
    # frozen, and the model is then retrained on the recomposed full training
    # partition. That final path is bit-for-bit identical to a run of this file
    # with the argument left at its default.
    val_idx = np.array([], dtype=train_idx.dtype)
    if validation_fraction > 0.0:
        remaining = 1.0 - float(test_fraction)
        if remaining <= 0.0:
            raise ValueError("test-fraction must leave room for a training partition")
        relative = float(validation_fraction) / remaining
        if not 0.0 < relative < 1.0:
            raise ValueError(
                f"validation-fraction={validation_fraction} is not a valid share of the "
                f"{remaining:.2f} training partition"
            )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=relative,
            random_state=seed,
            stratify=y[train_idx],
        )

    train_indices_by_class = {
        class_id: train_idx[y[train_idx] == class_id]
        for class_id in class_ids
    }
    test_indices_by_class = {
        class_id: test_idx[y[test_idx] == class_id]
        for class_id in class_ids
    }
    val_indices_by_class = {
        class_id: val_idx[y[val_idx] == class_id]
        for class_id in class_ids
    }

    split = str(evaluation_split).strip().lower()
    if split not in {"test", "validation"}:
        raise ValueError(f"evaluation-split must be 'test' or 'validation', got '{split}'")
    if split == "validation":
        if validation_fraction <= 0.0:
            raise ValueError("evaluation-split='validation' requires validation-fraction > 0")
        eval_indices_by_class = val_indices_by_class
    else:
        eval_indices_by_class = test_indices_by_class

    bundle = DatasetBundle(
        X=X,
        y=y,
        class_ids=class_ids,
        num_features=X.shape[1],
        num_outputs=max(class_ids) + 1,
        train_indices_by_class=train_indices_by_class,
        test_indices_by_class=test_indices_by_class,
        val_indices_by_class=val_indices_by_class,
        eval_indices_by_class=eval_indices_by_class,
    )
    dataset_cache[key] = bundle

    num_features = bundle.num_features
    num_classes = bundle.num_outputs
    class_ids_global = class_ids

    print(
        f"[Data] RT-IoT22 loaded: samples={len(y)}, "
        f"features={bundle.num_features}, classes={len(class_ids)}"
    )
    return bundle, bundle.num_outputs, bundle.num_features


def parse_class_stages(
    class_stages: str | None,
    class_ids: Sequence[int] | None = None,
) -> Tuple[Tuple[int, ...], ...]:
    """Parse class-stage groups from a semicolon/comma string."""
    available = set(class_ids or LABEL_NAMES.keys())
    groups: list[tuple[int, ...]] = []

    if class_stages:
        for block in class_stages.split(";"):
            parsed = tuple(
                int(token.strip())
                for token in block.split(",")
                if token.strip()
            )
            valid = tuple(class_id for class_id in parsed if class_id in available)
            if valid:
                groups.append(valid)
    elif set(DEFAULT_CLASS_STAGES[0]).issubset(available):
        groups = [tuple(c for c in group if c in available) for group in DEFAULT_CLASS_STAGES]
    else:
        sorted_ids = sorted(available)
        groups = [tuple(chunk) for chunk in np.array_split(sorted_ids, min(5, len(sorted_ids)))]

    seen = {class_id for group in groups for class_id in group}
    missing = tuple(class_id for class_id in sorted(available) if class_id not in seen)
    if missing:
        groups.append(missing)

    return tuple(group for group in groups if group)


def known_classes_until_stage(
    stages: Sequence[Sequence[int]],
    stage: int,
) -> Tuple[int, ...]:
    stage = max(0, min(stage, len(stages) - 1))
    known: list[int] = []
    for group in stages[: stage + 1]:
        known.extend(int(class_id) for class_id in group)
    return tuple(dict.fromkeys(known))


def new_classes_for_stage(stages: Sequence[Sequence[int]], stage: int) -> Tuple[int, ...]:
    stage = max(0, min(stage, len(stages) - 1))
    return tuple(int(class_id) for class_id in stages[stage])


def _client_visible_classes(
    stage: int,
    partition_id: int,
    stages: Sequence[Sequence[int]],
    exposure_mode: str,
    recurrent_class_prob: float,
    seed: int,
) -> Tuple[int, ...]:
    new_classes = set(new_classes_for_stage(stages, stage))

    if exposure_mode == "cumulative":
        return known_classes_until_stage(stages, stage)

    if exposure_mode == "disjoint" or stage == 0:
        return tuple(class_id for class_id in known_classes_until_stage(stages, stage) if class_id in new_classes)

    rng = np.random.default_rng(seed + 1009 * stage + 9176 * partition_id)
    old_classes = [
        class_id
        for class_id in known_classes_until_stage(stages, stage - 1)
        if rng.random() < recurrent_class_prob
    ]
    visible = list(new_classes) + old_classes
    ordered_seen = known_classes_until_stage(stages, stage)
    return tuple(class_id for class_id in ordered_seen if class_id in set(visible))


def _dirichlet_client_indices(
    indices: np.ndarray,
    num_partitions: int,
    partition_id: int,
    alpha: float,
    seed: int,
) -> np.ndarray:
    """Class-wise Dirichlet partitioning, Eq. (12).

    The samples of one class are split across the K clients with proportions drawn
    from Dirichlet(alpha, ..., alpha); lower alpha concentrates a class on fewer
    clients. The seed is derived from (run seed, stage, class), so the same run
    seed gives every method the same partition.
    """
    if len(indices) == 0:
        return indices

    rng = np.random.default_rng(seed)
    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)

    if num_partitions <= 1:
        return shuffled

    proportions = rng.dirichlet(np.repeat(max(alpha, 1e-3), num_partitions))
    cuts = (np.cumsum(proportions) * len(shuffled)).astype(int)[:-1]
    return np.split(shuffled, cuts)[partition_id]


def _limit_indices(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    selected = rng.choice(indices, size=limit, replace=False)
    return np.sort(selected)


def load_stage_partition(
    partition_id: int,
    num_partitions: int,
    dataset_path: str,
    stage: int,
    run_config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Return current-stage train data and all-seen test data for one client."""
    seed = _as_int(run_config, "seed", 42)
    test_fraction = _as_float(run_config, "test-fraction", 0.2)
    alpha = _as_float(run_config, "partition-alpha", 0.2)
    recurrent_prob = _as_float(run_config, "recurrent-class-prob", 0.35)
    exposure_mode = str(run_config.get("exposure-mode", "dynamic-recurrent"))
    max_train = _as_int(run_config, "max-train-samples-per-client", 0)
    max_test = _as_int(run_config, "max-test-samples-per-client", 0)

    bundle, _, _ = load_and_preprocess_dataset(
        dataset_path,
        test_fraction,
        seed,
        validation_fraction=_as_float(run_config, "validation-fraction", 0.0),
        evaluation_split=str(run_config.get("evaluation-split", "test")),
    )
    stages = parse_class_stages(str(run_config.get("class-stages", "")), bundle.class_ids)
    stage = max(0, min(int(stage), len(stages) - 1))

    cache_key = (
        dataset_path,
        partition_id,
        num_partitions,
        stage,
        str(run_config.get("class-stages", "")),
        exposure_mode,
        alpha,
        recurrent_prob,
        test_fraction,
        _as_float(run_config, "validation-fraction", 0.0),
        str(run_config.get("evaluation-split", "test")),
        seed,
        max_train,
        max_test,
    )
    if cache_key in partition_cache:
        return partition_cache[cache_key]

    visible_classes = _client_visible_classes(
        stage,
        partition_id,
        stages,
        exposure_mode,
        recurrent_prob,
        seed,
    )
    known_classes = known_classes_until_stage(stages, stage)
    new_classes = new_classes_for_stage(stages, stage)

    train_parts: list[np.ndarray] = []
    for class_id in visible_classes:
        class_indices = bundle.train_indices_by_class.get(class_id, np.array([], dtype=np.int64))
        client_indices = _dirichlet_client_indices(
            class_indices,
            num_partitions,
            partition_id,
            alpha,
            seed + 7919 * stage + 131 * class_id,
        )
        train_parts.append(client_indices)

    test_parts: list[np.ndarray] = []
    for class_id in known_classes:
        class_indices = bundle.eval_indices_by_class.get(class_id, np.array([], dtype=np.int64))
        client_indices = _dirichlet_client_indices(
            class_indices,
            num_partitions,
            partition_id,
            alpha,
            seed + 123457 + 131 * class_id,
        )
        test_parts.append(client_indices)

    train_indices = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    test_indices = np.concatenate(test_parts) if test_parts else np.array([], dtype=np.int64)

    if len(train_indices) == 0 and visible_classes:
        fallback_class = visible_classes[0]
        fallback = bundle.train_indices_by_class[fallback_class]
        train_indices = fallback[: min(len(fallback), 1)]

    if len(test_indices) == 0 and known_classes:
        fallback_class = known_classes[0]
        fallback = bundle.eval_indices_by_class[fallback_class]
        test_indices = fallback[: min(len(fallback), 1)]

    train_indices = _limit_indices(
        train_indices,
        max_train,
        seed + 3331 * stage + partition_id,
    )
    test_indices = _limit_indices(
        test_indices,
        max_test,
        seed + 7331 * stage + partition_id,
    )

    rng = np.random.default_rng(seed + 17 * stage + partition_id)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    result = (
        bundle.X[train_indices].astype(np.float32, copy=False),
        bundle.X[test_indices].astype(np.float32, copy=False),
        bundle.y[train_indices].astype(np.int64, copy=False),
        bundle.y[test_indices].astype(np.int64, copy=False),
        visible_classes,
        known_classes,
        new_classes,
    )
    partition_cache[cache_key] = result
    return result


class TabularFedACM(nn.Module):
    """Shared backbone: MLP encoder 128 -> 64 -> ``embedding_dim`` with dropout
    after the first hidden layer, plus a linear auxiliary head.

    ``encode`` gives the latent representation in which prototypes live. The head
    is preallocated over the whole taxonomy Y, which is what bounds the entropy of
    Eq. (4c) in [0, 1]; classes not yet introduced are never predicted, because
    evaluation restricts the decision to the classes seen so far.
    """

    def __init__(self, input_dim: int, output_dim: int, embedding_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, output_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))


def get_model(embedding_dim: int = 32) -> TabularFedACM:
    if num_features is None or num_classes is None:
        raise ValueError("Dataset metadata must be loaded before creating the model.")
    return TabularFedACM(num_features, num_classes, embedding_dim)


def get_model_params(model: nn.Module) -> List[np.ndarray]:
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def set_model_params(model: nn.Module, params: Sequence[np.ndarray]) -> nn.Module:
    state_dict = model.state_dict()
    if len(params) != len(state_dict):
        raise ValueError(f"Expected {len(state_dict)} arrays, received {len(params)}")
    new_state = {
        key: torch.tensor(value, dtype=state_dict[key].dtype)
        for key, value in zip(state_dict.keys(), params)
    }
    model.load_state_dict(new_state, strict=True)
    return model


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _memory_tensors(
    memory_class_ids: np.ndarray,
    memory_prototypes: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    class_ids = torch.tensor(memory_class_ids.astype(np.int64), dtype=torch.long, device=device)
    prototypes = torch.tensor(memory_prototypes, dtype=torch.float32, device=device)
    id_to_pos = {int(class_id): pos for pos, class_id in enumerate(memory_class_ids.tolist())}
    return class_ids, prototypes, id_to_pos


def _prototype_alignment_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    id_to_pos: dict[int, int],
) -> torch.Tensor:
    """Prototype alignment, Eq. (3a).

    ``F.mse_loss`` averages over batch and latent dimensions, which is the
    1/(d |B_M|) normalisation of the equation. Samples whose class has no memory
    entry are masked out; an empty B_M gives zero.
    """
    mask = torch.tensor(
        [int(label.item()) in id_to_pos for label in labels],
        dtype=torch.bool,
        device=labels.device,
    )
    if not bool(mask.any()):
        return embeddings.new_tensor(0.0)

    target_positions = torch.tensor(
        [id_to_pos[int(label.item())] for label in labels[mask]],
        dtype=torch.long,
        device=labels.device,
    )
    target_prototypes = prototypes[target_positions]
    return F.mse_loss(embeddings[mask], target_prototypes)


def _prototype_margin_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    id_to_pos: dict[int, int],
    margin: float,
) -> torch.Tensor:
    """Prototype margin, Eq. (3b): hinge between the distance to the own concept
    and the distance to the closest competing one. Disabled with fewer than two
    stored concepts."""
    if len(id_to_pos) < 2:
        return embeddings.new_tensor(0.0)

    mask = torch.tensor(
        [int(label.item()) in id_to_pos for label in labels],
        dtype=torch.bool,
        device=labels.device,
    )
    if not bool(mask.any()):
        return embeddings.new_tensor(0.0)

    target_positions = torch.tensor(
        [id_to_pos[int(label.item())] for label in labels[mask]],
        dtype=torch.long,
        device=labels.device,
    )
    distances = torch.cdist(embeddings[mask], prototypes)
    positive = distances[torch.arange(len(target_positions), device=labels.device), target_positions]
    positive_mask = F.one_hot(target_positions, num_classes=distances.shape[1]).bool()
    negative_distances = distances.masked_fill(positive_mask, float("inf"))
    negative = torch.min(negative_distances, dim=1).values
    return F.relu(margin + positive - negative).mean()


def train_fedacm(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,
    learning_rate: float,
    fedprox_mu: float,
    memory_class_ids: np.ndarray,
    memory_prototypes: np.ndarray,
    lambda_proto: float,
    lambda_margin: float,
    lambda_kd: float,
    kd_temperature: float,
    prototype_margin: float,
    kd_class_ids: Optional[np.ndarray] = None,
    ewc_lambda: float = 0.0,
    ewc_anchor: Optional[Sequence[torch.Tensor]] = None,
    ewc_fisher: Optional[Sequence[torch.Tensor]] = None,
) -> nn.Module:
    """Local optimisation with the composite objective of Eq. (2):

        L = L_CE + lambda_P*L_proto + lambda_M*L_margin + lambda_KD*L_KD
            + (mu_prox/2)*||theta_k - theta_G||^2

    The memory terms are inactive while the memory is empty, so the first round
    reduces to cross-entropy plus the proximal term; the server zeroes the weight
    of any term the active method or ablation does not use. The distillation
    teacher is the received global model, frozen, and the KL is restricted to
    ``kd_class_ids``.
    """
    model.to(device)
    model.train()

    teacher = copy.deepcopy(model).to(device).eval() if lambda_kd > 0 else None
    global_params = [param.detach().clone() for param in model.parameters()]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    has_memory = len(memory_class_ids) > 0 and len(memory_prototypes) > 0
    if has_memory:
        _, memory_proto_t, id_to_pos = _memory_tensors(
            memory_class_ids,
            memory_prototypes,
            device,
        )
    else:
        memory_proto_t = torch.empty(0, 0, dtype=torch.float32, device=device)
        id_to_pos = {}

    if kd_class_ids is None:
        kd_ids_np = memory_class_ids
    else:
        kd_ids_np = kd_class_ids
    kd_ids_t = torch.tensor(kd_ids_np.astype(np.int64), dtype=torch.long, device=device)

    model_params = list(model.parameters())
    has_ewc = (
        ewc_lambda > 0
        and ewc_anchor is not None
        and ewc_fisher is not None
        and len(ewc_anchor) == len(model_params)
        and len(ewc_fisher) == len(model_params)
    )

    for _ in range(max(1, int(epochs))):
        for features, labels in trainloader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            embeddings = model.encode(features)
            logits = model.classifier(embeddings)
            loss = criterion(logits, labels)

            if has_memory and lambda_proto > 0:
                proto_loss = _prototype_alignment_loss(
                    embeddings,
                    labels,
                    memory_proto_t,
                    id_to_pos,
                )
                loss = loss + lambda_proto * proto_loss

            if has_memory and lambda_margin > 0:
                margin_loss = _prototype_margin_loss(
                    embeddings,
                    labels,
                    memory_proto_t,
                    id_to_pos,
                    prototype_margin,
                )
                loss = loss + lambda_margin * margin_loss

            if teacher is not None and len(kd_ids_t) > 0:
                with torch.no_grad():
                    teacher_logits = teacher(features)[:, kd_ids_t]
                student_logits = logits[:, kd_ids_t]
                temperature = max(kd_temperature, 1e-6)
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature**2)
                loss = loss + lambda_kd * kd_loss

            if has_ewc:
                ewc_loss = embeddings.new_tensor(0.0)
                for param, anchor, fisher in zip(model.parameters(), ewc_anchor, ewc_fisher):
                    ewc_loss = ewc_loss + torch.sum(
                        fisher.to(device) * (param - anchor.to(device)) ** 2
                    )
                loss = loss + 0.5 * ewc_lambda * ewc_loss

            if fedprox_mu > 0:
                prox = embeddings.new_tensor(0.0)
                for param, global_param in zip(model.parameters(), global_params):
                    prox = prox + torch.sum((param - global_param) ** 2)
                loss = loss + 0.5 * fedprox_mu * prox

            loss.backward()
            optimizer.step()

    return model


def estimate_ewc_state(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 8,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Estimate a diagonal Fisher approximation for EWC-style local CL."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    fisher = [torch.zeros_like(param, device=device) for param in model.parameters()]
    batches = 0

    for features, labels in dataloader:
        if batches >= max_batches:
            break
        features = features.to(device)
        labels = labels.to(device)
        model.zero_grad(set_to_none=True)
        loss = criterion(model(features), labels)
        loss.backward()
        for idx, param in enumerate(model.parameters()):
            if param.grad is not None:
                fisher[idx] += param.grad.detach() ** 2
        batches += 1

    if batches > 0:
        fisher = [value / batches for value in fisher]

    anchors = [param.detach().cpu().clone() for param in model.parameters()]
    fisher_cpu = [value.detach().cpu().clone() for value in fisher]
    model.zero_grad(set_to_none=True)
    return anchors, fisher_cpu


def compute_local_concepts(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compress each observed class into its evidence tuple, Eqs. (4a)-(4c).

    Returns ``(class_ids, prototypes, supports, dispersions, uncertainties)``: the
    mean embedding of the class, its sample count, the mean distance to the
    prototype and the mean normalised Shannon entropy. One 32-dimensional vector
    and three scalars per class, independent of the number of underlying flows.
    """
    model.to(device)
    model.eval()
    embeddings_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    entropy_batches: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            logits = model(features)
            # Eq. (4c): softmax and log|Y| normaliser both range over the full
            # taxonomy, which is what bounds u in [0, 1].
            probabilities = F.softmax(logits, dim=1)
            entropy = -torch.sum(
                probabilities * torch.log(probabilities + 1e-12),
                dim=1,
            ) / math.log(max(output_dim, 2))
            embeddings_batches.append(model.encode(features).cpu().numpy())
            label_batches.append(labels.numpy())
            entropy_batches.append(entropy.cpu().numpy())

    if not embeddings_batches:
        embedding_dim = getattr(model.classifier, "in_features", 0)
        return (
            np.array([], dtype=np.int64),
            np.empty((0, embedding_dim), dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    embeddings = np.vstack(embeddings_batches).astype(np.float32, copy=False)
    labels = np.concatenate(label_batches).astype(np.int64, copy=False)
    entropies = np.concatenate(entropy_batches).astype(np.float32, copy=False)

    class_ids = sorted(int(class_id) for class_id in np.unique(labels))
    prototypes: list[np.ndarray] = []
    supports: list[float] = []
    dispersions: list[float] = []
    uncertainties: list[float] = []

    for class_id in class_ids:
        mask = labels == class_id
        class_embeddings = embeddings[mask]
        prototype = np.mean(class_embeddings, axis=0)
        dispersion = float(np.mean(np.linalg.norm(class_embeddings - prototype, axis=1)))
        prototypes.append(prototype.astype(np.float32, copy=False))
        supports.append(float(mask.sum()))
        dispersions.append(dispersion)
        uncertainties.append(float(np.mean(entropies[mask])))

    return (
        np.array(class_ids, dtype=np.int64),
        np.vstack(prototypes).astype(np.float32, copy=False),
        np.array(supports, dtype=np.float32),
        np.array(dispersions, dtype=np.float32),
        np.array(uncertainties, dtype=np.float32),
    )


def evaluate_fedacm(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    memory_class_ids: np.ndarray,
    memory_prototypes: np.ndarray,
    known_classes: Sequence[int],
    new_classes: Sequence[int],
    inference_mode: str = "prototype",
) -> tuple[float, dict[str, float]]:
    """Evaluate one partition and return the metrics of Section 4.5.

    ``prototype`` inference is the nearest stored concept, Eq. (11): no task
    identifier, and the decision space is the set of classes in memory.
    ``softmax`` is the argmax of the auxiliary head restricted to the classes seen
    so far, the standard class-incremental convention used by the other methods.

    Metrics cover the classes seen so far: macro-F1 Eq. (13), weighted-F1
    Eq. (14), new-class F1 Eq. (15), old-class F1 Eq. (16), plus per-class recall,
    F1 and support, from which the server derives the forgetting score.
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    losses: list[float] = []
    y_true_batches: list[np.ndarray] = []
    y_pred_batches: list[np.ndarray] = []

    has_memory = (
        inference_mode == "prototype"
        and len(memory_class_ids) > 0
        and len(memory_prototypes) > 0
    )
    if has_memory:
        proto_tensor = torch.tensor(memory_prototypes, dtype=torch.float32, device=device)
        proto_class_ids = np.array(memory_class_ids, dtype=np.int64)

    # For softmax inference, restrict the decision to classes seen so far so
    # that the auxiliary head cannot predict not-yet-introduced classes. This is
    # the standard class-incremental evaluation convention.
    seen_cols = np.array(
        sorted(dict.fromkeys(int(c) for c in known_classes)), dtype=np.int64
    )
    use_seen_mask = (not has_memory) and len(seen_cols) > 0
    if use_seen_mask:
        seen_cols_t = torch.tensor(seen_cols, dtype=torch.long, device=device)

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            losses.append(float(criterion(logits, labels).item()))

            if has_memory:
                embeddings = model.encode(features)
                distances = torch.cdist(embeddings, proto_tensor)
                pred_positions = torch.argmin(distances, dim=1).cpu().numpy()
                predictions = proto_class_ids[pred_positions]
            elif use_seen_mask:
                masked = logits[:, seen_cols_t]
                pred_positions = torch.argmax(masked, dim=1).cpu().numpy()
                predictions = seen_cols[pred_positions]
            else:
                predictions = torch.argmax(logits, dim=1).cpu().numpy()

            y_true_batches.append(labels.cpu().numpy())
            y_pred_batches.append(predictions)

    if not y_true_batches:
        return 0.0, {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
        }

    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)
    labels_seen = list(dict.fromkeys(int(class_id) for class_id in known_classes))

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_seen,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_seen,
        average="weighted",
        zero_division=0,
    )

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }

    if new_classes:
        _, _, new_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(new_classes),
            average="macro",
            zero_division=0,
        )
        metrics["new_class_f1"] = float(new_f1)

    old_classes = [class_id for class_id in labels_seen if class_id not in set(new_classes)]
    if old_classes:
        _, _, old_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=old_classes,
            average="macro",
            zero_division=0,
        )
        metrics["old_class_f1"] = float(old_f1)

    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_seen,
        average=None,
        zero_division=0,
    )
    for class_id, recall, f1_value, support in zip(labels_seen, per_recall, per_f1, per_support):
        metrics[f"recall_class_{class_id}"] = float(recall)
        metrics[f"f1_class_{class_id}"] = float(f1_value)
        metrics[f"support_class_{class_id}"] = float(support)

    return float(np.mean(losses)), metrics


def build_global_test_arrays(
    bundle: DatasetBundle,
    stages: Sequence[Sequence[int]],
    stage: int,
) -> tuple[np.ndarray, np.ndarray, Tuple[int, ...], Tuple[int, ...]]:
    """Pool the global held-out test set over all classes seen up to ``stage``."""
    known = known_classes_until_stage(stages, stage)
    new = new_classes_for_stage(stages, stage)
    parts = [
        bundle.eval_indices_by_class[c]
        for c in known
        if len(bundle.eval_indices_by_class.get(c, [])) > 0
    ]
    if not parts:
        empty = np.array([], dtype=np.int64)
        return (
            bundle.X[empty].astype(np.float32, copy=False),
            empty,
            known,
            new,
        )
    test_idx = np.concatenate(parts)
    return (
        bundle.X[test_idx].astype(np.float32, copy=False),
        bundle.y[test_idx].astype(np.int64, copy=False),
        known,
        new,
    )


def evaluate_global(
    model_arrays: Sequence[np.ndarray],
    embedding_dim: int,
    bundle: DatasetBundle,
    stages: Sequence[Sequence[int]],
    stage: int,
    memory_class_ids: np.ndarray,
    memory_prototypes: np.ndarray,
    inference_mode: str,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, float]:
    """Server-side centralized evaluation of the global model on the pooled test set.

    This complements the per-client federated evaluation. It measures the global
    model and the Attack Concept Memory on a single held-out test set covering
    all classes seen so far, which is the standard class-incremental evaluation
    used in the federated continual-learning literature.
    """
    X, y, known, new = build_global_test_arrays(bundle, stages, stage)
    if len(y) == 0:
        return {}
    model = get_model(embedding_dim=embedding_dim)
    set_model_params(model, model_arrays)
    loader = make_loader(X, y, batch_size=batch_size, shuffle=False)
    _, metrics = evaluate_fedacm(
        model,
        loader,
        device,
        np.array(memory_class_ids, dtype=np.int64),
        np.array(memory_prototypes, dtype=np.float32),
        known,
        new,
        inference_mode=inference_mode,
    )
    return metrics
