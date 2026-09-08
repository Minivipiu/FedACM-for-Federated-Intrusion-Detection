"""Flower ClientApp: local training and concept extraction.

One round on a client: receive the global encoder and the Attack Concept Memory
(the memory travels as the tail of the parameter list, after the first
``num_model_params`` arrays), train with the objective of Eq. (2), summarise each
locally observed class into the evidence tuple of Eqs. (4a)-(4c), and upload both.
No raw traffic leaves the client.
"""

from __future__ import annotations

import numpy as np
import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from fedacm_ids.config import as_bool
from fedacm_ids.task import (
    compute_local_concepts,
    estimate_ewc_state,
    evaluate_fedacm,
    get_model,
    get_model_params,
    load_and_preprocess_dataset,
    load_stage_partition,
    make_loader,
    set_model_params,
    set_random_seed,
    train_fedacm,
)


def _as_int(config: dict, key: str, default: int) -> int:
    return int(config.get(key, default))


def _as_float(config: dict, key: str, default: float) -> float:
    return float(config.get(key, default))


def _empty_memory_tail(embedding_dim: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([], dtype=np.int64),
        np.empty((0, embedding_dim), dtype=np.float32),
    )


def _select_kd_class_ids(
    scope: str,
    known_classes: tuple[int, ...],
    new_classes: tuple[int, ...],
    memory_class_ids: np.ndarray,
) -> np.ndarray:
    if scope == "memory":
        return memory_class_ids.astype(np.int64, copy=False)
    if scope == "known":
        return np.array(known_classes, dtype=np.int64)
    if scope == "old":
        new_set = set(new_classes)
        return np.array([c for c in known_classes if c not in new_set], dtype=np.int64)
    return np.array([], dtype=np.int64)


class FedACMClient(NumPyClient):
    def __init__(
        self,
        cid: str,
        partition_id: int,
        num_partitions: int,
        dataset_path: str,
        run_config: dict,
        model: torch.nn.Module,
    ) -> None:
        self.cid = cid
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.dataset_path = dataset_path
        self.run_config = run_config
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.stage_cache: dict[int, tuple] = {}
        self.local_initialized = False
        self.ewc_anchor: list[torch.Tensor] | None = None
        self.ewc_fisher: list[torch.Tensor] | None = None

    def _stage_data(self, stage: int):
        if stage not in self.stage_cache:
            self.stage_cache[stage] = load_stage_partition(
                self.partition_id,
                self.num_partitions,
                self.dataset_path,
                stage,
                self.run_config,
            )
        return self.stage_cache[stage]

    def _set_payload(self, parameters, config) -> tuple[np.ndarray, np.ndarray]:
        num_model_params = int(config["num_model_params"])
        method = str(config.get("method", "fedacm"))
        should_keep_local = method == "local-finetuning" and self.local_initialized

        if not should_keep_local:
            model_params = parameters[:num_model_params]
            set_model_params(self.model, model_params)
            self.local_initialized = True

        memory_tail = parameters[num_model_params:]
        embedding_dim = int(config["embedding_dim"])
        use_memory = as_bool(config, "use_memory", False)
        if use_memory and len(memory_tail) >= 2:
            return (
                np.array(memory_tail[0], dtype=np.int64),
                np.array(memory_tail[1], dtype=np.float32),
            )
        return _empty_memory_tail(embedding_dim)

    def fit(self, parameters, config):
        stage = int(config["stage"])
        base_seed = _as_int(self.run_config, "seed", 42)
        set_random_seed(base_seed + 1000003 * stage + self.partition_id)
        memory_class_ids, memory_prototypes = self._set_payload(parameters, config)

        X_train, _, y_train, _, visible_classes, known_classes, new_classes = self._stage_data(stage)
        batch_size = int(config["batch_size"])
        trainloader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)

        kd_class_ids = _select_kd_class_ids(
            str(config.get("kd_scope", "none")),
            known_classes,
            new_classes,
            memory_class_ids,
        )

        self.model = train_fedacm(
            self.model,
            trainloader,
            epochs=int(config["local_epochs"]),
            device=self.device,
            learning_rate=float(config["learning_rate"]),
            fedprox_mu=float(config["fedprox_mu"]),
            memory_class_ids=memory_class_ids,
            memory_prototypes=memory_prototypes,
            lambda_proto=float(config["lambda_proto"]),
            lambda_margin=float(config["lambda_margin"]),
            lambda_kd=float(config["lambda_kd"]),
            kd_temperature=float(config["kd_temperature"]),
            prototype_margin=float(config["prototype_margin"]),
            kd_class_ids=kd_class_ids,
            ewc_lambda=float(config.get("ewc_lambda", 0.0)),
            ewc_anchor=self.ewc_anchor,
            ewc_fisher=self.ewc_fisher,
        )

        if as_bool(config, "use_ewc", False):
            self.ewc_anchor, self.ewc_fisher = estimate_ewc_state(
                self.model,
                trainloader,
                self.device,
                max_batches=int(config.get("ewc_max_batches", 8)),
            )

        # Evidence tuple of Eqs. (4a)-(4c), one entry per observed class.
        # Only the prototype-based methods collect it.
        concept_tail: tuple[np.ndarray, ...]
        if as_bool(config, "collect_concepts", False):
            concept_tail = compute_local_concepts(
                self.model,
                trainloader,
                self.device,
                output_dim=int(config["num_classes"]),
            )
        else:
            concept_tail = tuple()

        num_examples = len(trainloader.dataset)
        metrics = {
            "visible_classes": float(len(visible_classes)),
            "local_concepts": float(len(concept_tail[0])) if concept_tail else 0.0,
            "train_examples": float(num_examples),
        }
        return get_model_params(self.model) + list(concept_tail), num_examples, metrics

    def evaluate(self, parameters, config):
        stage = int(config["stage"])
        memory_class_ids, memory_prototypes = self._set_payload(parameters, config)

        _, X_test, _, y_test, _, known_classes, new_classes = self._stage_data(stage)
        batch_size = int(config["batch_size"])
        testloader = make_loader(X_test, y_test, batch_size=batch_size, shuffle=False)

        loss, metrics = evaluate_fedacm(
            self.model,
            testloader,
            self.device,
            memory_class_ids,
            memory_prototypes,
            known_classes,
            new_classes,
            inference_mode=str(config.get("inference_mode", "prototype")),
        )
        metrics["stage"] = float(stage)
        metrics["known_classes"] = float(len(known_classes))
        metrics["memory_classes_available"] = float(len(memory_class_ids))
        return float(loss), len(testloader.dataset), metrics


def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    run_config = dict(context.run_config)
    dataset_path = str(run_config["dataset-path"])
    seed = _as_int(run_config, "seed", 42)
    set_random_seed(seed + partition_id)

    load_and_preprocess_dataset(
        dataset_path,
        test_fraction=_as_float(run_config, "test-fraction", 0.2),
        seed=seed,
        validation_fraction=_as_float(run_config, "validation-fraction", 0.0),
        evaluation_split=str(run_config.get("evaluation-split", "test")),
    )
    model = get_model(embedding_dim=_as_int(run_config, "embedding-dim", 32))
    client = FedACMClient(
        cid=str(partition_id),
        partition_id=partition_id,
        num_partitions=num_partitions,
        dataset_path=dataset_path,
        run_config=run_config,
        model=model,
    )
    return client.to_client()


app = ClientApp(client_fn=client_fn)
