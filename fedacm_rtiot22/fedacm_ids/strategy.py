"""Server-side aggregation: model stream and Attack Concept Memory stream.

Keeps the two coupled global states of the method: the shared encoder, aggregated
by sample-weighted averaging, and the Attack Concept Memory, updated class by class
with Reliability-Aware Concept Aggregation, Eqs. (5)-(7), and then realigned with
the evolving latent space by drift compensation, Eqs. (8)-(10).

Both are broadcast together: the first ``num_model_params`` arrays are the model,
the remaining seven the memory (class ids, prototypes, supports, dispersions,
uncertainties, reliabilities, ages). The strategy also accumulates the runtime and
communication metrics of Section 4.5 and the forgetting score of Eqs. (17)-(18).
"""

from __future__ import annotations

import time

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fedacm_ids.config import MethodSettings
from fedacm_ids.task import known_classes_until_stage, new_classes_for_stage


@dataclass
class ConceptEvidence:
    prototype: np.ndarray
    support: float
    dispersion: float
    uncertainty: float


class FedACMStrategy(FedAvg):
    """Generic strategy for baselines and FedACM variants.

    The encoder is aggregated with a FedAvg-like weighted average. FedProx is
    activated on the client by adding the proximal term to the local objective.
    Prototype baselines and FedACM additionally maintain a server-side memory.
    """

    def __init__(
        self,
        *,
        initial_parameters: Optional[Parameters],
        class_stages: Sequence[Sequence[int]],
        rounds_per_stage: int,
        local_epochs: int,
        learning_rate: float,
        batch_size: int,
        num_model_params: int,
        embedding_dim: int,
        num_classes: int,
        settings: MethodSettings,
        fedprox_mu: float,
        lambda_proto: float,
        lambda_margin: float,
        lambda_kd: float,
        kd_temperature: float,
        prototype_margin: float,
        ewc_lambda: float,
        ewc_max_batches: int,
        raca_alpha: float,
        raca_beta: float,
        raca_gamma: float,
        raca_delta: float,
        raca_epsilon: float,
        memory_eta: float,
        drift_tau: float = 1.0,
        centralized_eval_fn=None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            **kwargs,
        )
        self.class_stages = tuple(tuple(int(c) for c in group) for group in class_stages)
        self.rounds_per_stage = max(1, int(rounds_per_stage))
        self.local_epochs = int(local_epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.num_model_params = int(num_model_params)
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.settings = settings
        self.fedprox_mu = float(fedprox_mu)
        self.lambda_proto = float(lambda_proto)
        self.lambda_margin = float(lambda_margin)
        self.lambda_kd = float(lambda_kd)
        self.kd_temperature = float(kd_temperature)
        self.prototype_margin = float(prototype_margin)
        self.ewc_lambda = float(ewc_lambda)
        self.ewc_max_batches = int(ewc_max_batches)
        self.raca_alpha = float(raca_alpha)
        self.raca_beta = float(raca_beta)
        self.raca_gamma = float(raca_gamma)
        self.raca_delta = float(raca_delta)
        self.raca_epsilon = float(raca_epsilon)
        self.memory_eta = float(memory_eta)
        self.drift_tau = float(drift_tau)
        self.use_drift_compensation = bool(settings.use_drift_compensation)
        self.centralized_eval_fn = centralized_eval_fn
        self.memory: Dict[int, dict[str, float | np.ndarray]] = {}
        self.best_f1_by_class: Dict[int, float] = {}
        self.best_cen_f1_by_class: Dict[int, float] = {}
        self.current_model_arrays = (
            parameters_to_ndarrays(initial_parameters) if initial_parameters is not None else []
        )

        self.fit_round_started_at: Dict[int, float] = {}
        self.evaluate_round_started_at: Dict[int, float] = {}
        self.fit_downlink_bytes_by_round: Dict[int, int] = {}
        self.evaluate_downlink_bytes_by_round: Dict[int, int] = {}
        self.cumulative_fit_seconds = 0.0
        self.cumulative_evaluate_seconds = 0.0
        self.cumulative_downlink_bytes = 0
        self.cumulative_uplink_bytes = 0
        print(
            "[FedACM] Strategy ready: "
            f"method={self.settings.method}, memory={self.settings.memory_aggregation}, "
            f"inference={self.settings.inference_mode}, stages={len(self.class_stages)}, "
            f"rounds/stage={self.rounds_per_stage}"
        )

    def _stage_for_round(self, server_round: int) -> int:
        return min(
            (max(server_round, 1) - 1) // self.rounds_per_stage,
            len(self.class_stages) - 1,
        )

    def _round_config(self, server_round: int) -> dict[str, Scalar]:
        stage = self._stage_for_round(server_round)
        return {
            "method": self.settings.method,
            "server_round": int(server_round),
            "stage": int(stage),
            "local_epochs": int(self.local_epochs),
            "learning_rate": float(self.learning_rate),
            "batch_size": int(self.batch_size),
            "num_model_params": int(self.num_model_params),
            "embedding_dim": int(self.embedding_dim),
            "num_classes": int(self.num_classes),
            "collect_concepts": bool(self.settings.collect_concepts),
            "use_memory": bool(self.settings.use_memory),
            "use_ewc": bool(self.settings.use_ewc),
            "inference_mode": self.settings.inference_mode,
            "kd_scope": self.settings.kd_scope,
            "fedprox_mu": float(self.fedprox_mu if self.settings.use_fedprox else 0.0),
            "lambda_proto": float(self.lambda_proto if self.settings.use_proto_loss else 0.0),
            "lambda_margin": float(self.lambda_margin if self.settings.use_margin_loss else 0.0),
            "lambda_kd": float(self.lambda_kd if self.settings.use_kd_loss else 0.0),
            "kd_temperature": float(self.kd_temperature),
            "prototype_margin": float(self.prototype_margin),
            "ewc_lambda": float(self.ewc_lambda if self.settings.use_ewc else 0.0),
            "ewc_max_batches": int(self.ewc_max_batches),
        }

    def _memory_ndarrays(self) -> NDArrays:
        if not self.settings.use_memory or not self.memory:
            return [
                np.array([], dtype=np.int64),
                np.empty((0, self.embedding_dim), dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            ]

        class_ids = np.array(sorted(self.memory), dtype=np.int64)
        prototypes = np.vstack([self.memory[int(c)]["prototype"] for c in class_ids]).astype(np.float32)
        supports = np.array([self.memory[int(c)]["support"] for c in class_ids], dtype=np.float32)
        dispersions = np.array([self.memory[int(c)]["dispersion"] for c in class_ids], dtype=np.float32)
        uncertainties = np.array([self.memory[int(c)]["uncertainty"] for c in class_ids], dtype=np.float32)
        reliabilities = np.array([self.memory[int(c)]["reliability"] for c in class_ids], dtype=np.float32)
        ages = np.array([self.memory[int(c)]["age"] for c in class_ids], dtype=np.float32)
        return [class_ids, prototypes, supports, dispersions, uncertainties, reliabilities, ages]

    def _parameters_with_memory(self, parameters: Parameters) -> Parameters:
        model_arrays = parameters_to_ndarrays(parameters)
        return ndarrays_to_parameters(model_arrays + self._memory_ndarrays())

    @staticmethod
    def _arrays_nbytes(arrays: Sequence[np.ndarray]) -> int:
        return int(sum(array.nbytes for array in arrays))

    def _parameters_nbytes(self, parameters: Parameters) -> int:
        return self._arrays_nbytes(parameters_to_ndarrays(parameters))

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        self.current_model_arrays = parameters_to_ndarrays(parameters)
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )
        payload = self._parameters_with_memory(parameters)
        downlink_bytes = self._parameters_nbytes(payload) * len(clients)
        self.fit_downlink_bytes_by_round[server_round] = int(downlink_bytes)
        self.cumulative_downlink_bytes += int(downlink_bytes)
        self.fit_round_started_at[server_round] = time.perf_counter()
        fit_ins = FitIns(payload, self._round_config(server_round))
        stage = self._stage_for_round(server_round)
        print(
            f"[FedACM] Round {server_round} stage {stage}: method={self.settings.method}, "
            f"fit_clients={len(clients)}, memory={len(self.memory)}"
        )
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if self.fraction_evaluate == 0.0:
            return []

        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )
        payload = self._parameters_with_memory(parameters)
        downlink_bytes = self._parameters_nbytes(payload) * len(clients)
        self.evaluate_downlink_bytes_by_round[server_round] = int(downlink_bytes)
        self.cumulative_downlink_bytes += int(downlink_bytes)
        self.evaluate_round_started_at[server_round] = time.perf_counter()
        evaluate_ins = EvaluateIns(
            payload,
            self._round_config(server_round),
        )
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        weights_results: list[tuple[NDArrays, float]] = []
        evidence_by_class: dict[int, list[ConceptEvidence]] = {}

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            model_arrays = arrays[: self.num_model_params]
            concept_arrays = arrays[self.num_model_params :]
            model_weight = self._model_weight(fit_res.num_examples, concept_arrays)
            weights_results.append((model_arrays, model_weight))
            self._collect_evidence(concept_arrays, evidence_by_class)

        fit_started_at = self.fit_round_started_at.pop(server_round, None)
        fit_seconds = 0.0 if fit_started_at is None else time.perf_counter() - fit_started_at
        fit_downlink_bytes = self.fit_downlink_bytes_by_round.pop(server_round, 0)
        fit_uplink_bytes = int(sum(self._parameters_nbytes(fit_res.parameters) for _, fit_res in results))
        self.cumulative_fit_seconds += float(fit_seconds)
        self.cumulative_uplink_bytes += int(fit_uplink_bytes)
        if self.settings.aggregate_model:
            aggregated_model = self._aggregate_weighted(weights_results)
            self.current_model_arrays = aggregated_model
        else:
            aggregated_model = self.current_model_arrays

        if self.settings.use_memory and self.settings.memory_aggregation != "none":
            self._update_memory(evidence_by_class)

        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        stage = self._stage_for_round(server_round)
        metrics_aggregated["stage"] = int(stage)
        metrics_aggregated["memory_size"] = int(len(self.memory))
        metrics_aggregated["new_classes"] = int(len(new_classes_for_stage(self.class_stages, stage)))
        metrics_aggregated["aggregation_failures"] = int(len(failures))
        metrics_aggregated["fit_seconds"] = float(fit_seconds)
        metrics_aggregated["fit_downlink_bytes"] = int(fit_downlink_bytes)
        metrics_aggregated["fit_uplink_bytes"] = int(fit_uplink_bytes)
        metrics_aggregated["fit_comm_bytes"] = int(fit_downlink_bytes + fit_uplink_bytes)
        metrics_aggregated["model_payload_bytes"] = int(self._arrays_nbytes(aggregated_model))
        metrics_aggregated["memory_payload_bytes"] = int(self._arrays_nbytes(self._memory_ndarrays()))
        metrics_aggregated["cumulative_fit_seconds"] = float(self.cumulative_fit_seconds)
        metrics_aggregated["cumulative_downlink_bytes"] = int(self.cumulative_downlink_bytes)
        metrics_aggregated["cumulative_uplink_bytes"] = int(self.cumulative_uplink_bytes)
        metrics_aggregated["cumulative_comm_bytes"] = int(self.cumulative_downlink_bytes + self.cumulative_uplink_bytes)

        print(
            f"[FedACM] Round {server_round}: method={self.settings.method}, "
            f"memory={len(self.memory)}, failures={len(failures)}"
        )
        return ndarrays_to_parameters(aggregated_model), metrics_aggregated

    def _collect_evidence(
        self,
        concept_arrays: Sequence[np.ndarray],
        evidence_by_class: dict[int, list[ConceptEvidence]],
    ) -> None:
        if len(concept_arrays) < 5:
            return

        class_ids = np.array(concept_arrays[0], dtype=np.int64)
        prototypes = np.array(concept_arrays[1], dtype=np.float32)
        supports = np.array(concept_arrays[2], dtype=np.float32)
        dispersions = np.array(concept_arrays[3], dtype=np.float32)
        uncertainties = np.array(concept_arrays[4], dtype=np.float32)

        for idx, class_id in enumerate(class_ids.tolist()):
            if idx >= len(prototypes) or idx >= len(supports):
                continue
            support = float(supports[idx])
            if support <= 0 or not np.isfinite(prototypes[idx]).all():
                continue
            evidence_by_class.setdefault(int(class_id), []).append(
                ConceptEvidence(
                    prototype=prototypes[idx].astype(np.float32, copy=False),
                    support=support,
                    dispersion=float(dispersions[idx]),
                    uncertainty=float(uncertainties[idx]),
                )
            )

    def _model_weight(self, num_examples: int, concept_arrays: Sequence[np.ndarray]) -> float:
        if self.settings.model_aggregation != "client-reliability" or len(concept_arrays) < 5:
            return float(num_examples)

        supports = np.array(concept_arrays[2], dtype=np.float32)
        dispersions = np.array(concept_arrays[3], dtype=np.float32)
        uncertainties = np.array(concept_arrays[4], dtype=np.float32)
        if len(supports) == 0:
            return float(num_examples)

        support = float(np.sum(np.maximum(supports, 1.0)))
        uncertainty = float(np.mean(np.clip(uncertainties, 0.0, 1.0)))
        dispersion = float(np.mean(np.maximum(dispersions, 0.0)))
        score = support * max(1e-6, 1.0 - uncertainty) / (self.raca_epsilon + dispersion)
        return float(score if np.isfinite(score) and score > 0 else num_examples)

    @staticmethod
    def _aggregate_weighted(weights_results: list[tuple[NDArrays, float]]) -> NDArrays:
        total_weight = float(sum(max(weight, 0.0) for _, weight in weights_results))
        if total_weight <= 0:
            total_weight = float(len(weights_results))
            weights_results = [(arrays, 1.0) for arrays, _ in weights_results]

        aggregated: NDArrays = []
        for param_idx in range(len(weights_results[0][0])):
            weighted_param = sum(
                arrays[param_idx] * (max(weight, 0.0) / total_weight)
                for arrays, weight in weights_results
            )
            aggregated.append(weighted_param)
        return aggregated

    def _update_memory(self, evidence_by_class: dict[int, list[ConceptEvidence]]) -> None:
        for state in self.memory.values():
            state["age"] = float(state["age"]) + 1.0

        # Snapshot prototypes before the update to estimate embedding drift.
        old_prototypes: Dict[int, np.ndarray] = {
            int(c): np.array(state["prototype"], dtype=np.float32, copy=True)
            for c, state in self.memory.items()
        }

        observed = [c for c, ev in evidence_by_class.items() if ev]
        for class_id in observed:
            evidences = evidence_by_class[class_id]
            if self.settings.memory_aggregation == "simple":
                self._update_memory_simple(class_id, evidences)
            elif self.settings.memory_aggregation == "raca":
                self._update_memory_raca(class_id, evidences)

        if self.use_drift_compensation:
            self._compensate_prototype_drift(old_prototypes, observed)

    def _compensate_prototype_drift(
        self,
        old_prototypes: Dict[int, np.ndarray],
        observed: Sequence[int],
    ) -> None:
        """Semantic drift compensation for stale prototypes.

        When the shared encoder evolves, prototypes of classes that were not
        observed in the current round become misaligned with the current latent
        space. We estimate the local drift field from the classes that were
        refreshed this round and apply a distance-weighted correction to the
        prototypes of the absent classes, following the semantic-drift-
        compensation principle adapted to the federated concept memory.
        """
        anchors = [
            c for c in observed
            if c in old_prototypes and c in self.memory
        ]
        if not anchors:
            return

        drift_vectors: list[np.ndarray] = []
        anchor_old: list[np.ndarray] = []
        for c in anchors:
            new_proto = np.array(self.memory[int(c)]["prototype"], dtype=np.float32)
            delta = new_proto - old_prototypes[int(c)]
            if np.isfinite(delta).all():
                drift_vectors.append(delta)
                anchor_old.append(old_prototypes[int(c)])
        if not drift_vectors:
            return

        drift_arr = np.stack(drift_vectors)
        anchor_arr = np.stack(anchor_old)
        tau = max(self.drift_tau, 1e-6)
        observed_set = set(int(c) for c in observed)

        for class_id, old_proto in old_prototypes.items():
            if int(class_id) in observed_set:
                continue
            distances = np.linalg.norm(anchor_arr - old_proto[None, :], axis=1)
            weights = np.exp(-(distances ** 2) / (2.0 * tau ** 2))
            weight_sum = float(np.sum(weights))
            if weight_sum <= 1e-12:
                continue
            estimated = np.sum(drift_arr * weights[:, None], axis=0) / weight_sum
            corrected = (old_proto + estimated).astype(np.float32)
            if np.isfinite(corrected).all():
                self.memory[int(class_id)]["prototype"] = corrected

    def _update_memory_simple(self, class_id: int, evidences: list[ConceptEvidence]) -> None:
        """Support-weighted prototype averaging: the FedProto rule, and FedACM
        without RACA."""
        weights = np.array([max(e.support, 1.0) for e in evidences], dtype=np.float64)
        weights = weights / np.sum(weights)
        self._commit_memory_update(class_id, evidences, weights, reliability=float(np.sum(weights)))

    def _update_memory_raca(self, class_id: int, evidences: list[ConceptEvidence]) -> None:
        """Reliability-Aware Concept Aggregation, Eqs. (5) and (6a)-(6b).

        rho = n^alpha_r * (1-u)^beta_r / (eps+sigma)^gamma_r * exp(-delta_r*||p-mu||)

        The temporal factor is 1 for a class with no previous entry. Each factor
        can be switched off independently; if every score degenerates to zero the
        update falls back to support weighting.
        """
        previous = self.memory.get(class_id)
        rhos: list[float] = []

        for evidence in evidences:
            if previous is None or not self.settings.use_temporal_weight:
                temporal = 1.0
            else:
                distance = float(np.linalg.norm(evidence.prototype - previous["prototype"]))
                temporal = float(np.exp(-self.raca_delta * distance))

            uncertainty_term = 1.0
            if self.settings.use_uncertainty_weight:
                uncertainty_term = max(1e-6, 1.0 - np.clip(evidence.uncertainty, 0.0, 1.0))

            compactness = 1.0
            if self.settings.use_dispersion_weight:
                compactness = self.raca_epsilon + max(evidence.dispersion, 0.0)

            rho = (
                (max(evidence.support, 1.0) ** self.raca_alpha)
                * (uncertainty_term ** self.raca_beta)
                / (compactness ** self.raca_gamma)
                * temporal
            )
            rhos.append(float(rho) if np.isfinite(rho) else 0.0)

        rho_sum = float(np.sum(rhos))
        if rho_sum <= 0:
            raw_weights = np.array([max(e.support, 1.0) for e in evidences], dtype=np.float64)
            weights = raw_weights / np.sum(raw_weights)
        else:
            weights = np.array(rhos, dtype=np.float64) / rho_sum

        self._commit_memory_update(class_id, evidences, weights, reliability=rho_sum)

    def _commit_memory_update(
        self,
        class_id: int,
        evidences: list[ConceptEvidence],
        weights: np.ndarray,
        reliability: float,
    ) -> None:
        previous = self.memory.get(class_id)
        candidate = np.sum(
            np.stack([e.prototype for e in evidences]) * weights[:, None],
            axis=0,
        ).astype(np.float32)
        dispersion = float(np.sum([e.dispersion * w for e, w in zip(evidences, weights)]))
        uncertainty = float(np.sum([e.uncertainty * w for e, w in zip(evidences, weights)]))
        support = float(np.sum([e.support for e in evidences]))

        # New class: the candidate of Eq. (6b). Existing class: smoothing, Eq. (7).
        if previous is None:
            updated = candidate
            total_support = support
        else:
            eta = min(max(self.memory_eta, 0.0), 1.0)
            updated = ((1.0 - eta) * previous["prototype"] + eta * candidate).astype(np.float32)
            total_support = float(previous["support"]) + support

        self.memory[class_id] = {
            "prototype": updated,
            "support": total_support,
            "dispersion": dispersion,
            "uncertainty": uncertainty,
            "reliability": reliability,
            "age": 0.0,
        }

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        metrics = dict(metrics)
        evaluate_started_at = self.evaluate_round_started_at.pop(server_round, None)
        evaluate_seconds = 0.0 if evaluate_started_at is None else time.perf_counter() - evaluate_started_at
        evaluate_downlink_bytes = self.evaluate_downlink_bytes_by_round.pop(server_round, 0)
        self.cumulative_evaluate_seconds += float(evaluate_seconds)

        stage = self._stage_for_round(server_round)
        forgetting_values: list[float] = []
        for class_id in known_classes_until_stage(self.class_stages, stage):
            key = f"f1_class_{class_id}"
            if key not in metrics:
                continue
            current = float(metrics[key])
            best = self.best_f1_by_class.get(class_id, current)
            best = max(best, current)
            self.best_f1_by_class[class_id] = best
            forgetting_values.append(max(0.0, best - current))

        metrics["stage"] = int(stage)
        metrics["memory_size"] = int(len(self.memory))
        metrics["mean_forgetting"] = float(np.mean(forgetting_values)) if forgetting_values else 0.0
        metrics["known_classes"] = int(len(known_classes_until_stage(self.class_stages, stage)))
        metrics["evaluation_failures"] = int(len(failures))
        metrics["evaluate_seconds"] = float(evaluate_seconds)
        metrics["evaluate_downlink_bytes"] = int(evaluate_downlink_bytes)
        metrics["cumulative_evaluate_seconds"] = float(self.cumulative_evaluate_seconds)
        metrics["cumulative_downlink_bytes"] = int(self.cumulative_downlink_bytes)
        metrics["cumulative_uplink_bytes"] = int(self.cumulative_uplink_bytes)
        metrics["cumulative_comm_bytes"] = int(self.cumulative_downlink_bytes + self.cumulative_uplink_bytes)

        self._inject_centralized_metrics(metrics, stage)
        return loss, metrics

    def _inject_centralized_metrics(self, metrics: Dict[str, Scalar], stage: int) -> None:
        """Run server-side centralized evaluation and add cen_* metrics."""
        if self.centralized_eval_fn is None:
            return
        mem = self._memory_ndarrays()
        memory_class_ids, memory_prototypes = mem[0], mem[1]
        try:
            cen = self.centralized_eval_fn(
                stage,
                self.current_model_arrays,
                memory_class_ids,
                memory_prototypes,
                self.settings.inference_mode,
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[FedACM] Centralized evaluation failed: {exc}")
            return
        if not cen:
            return

        for key, value in cen.items():
            if isinstance(value, (int, float, bool)):
                metrics[f"cen_{key}"] = float(value)

        cen_forgetting: list[float] = []
        for class_id in known_classes_until_stage(self.class_stages, stage):
            key = f"f1_class_{class_id}"
            if key not in cen:
                continue
            current = float(cen[key])
            best = max(self.best_cen_f1_by_class.get(class_id, current), current)
            self.best_cen_f1_by_class[class_id] = best
            cen_forgetting.append(max(0.0, best - current))
        metrics["cen_mean_forgetting"] = (
            float(np.mean(cen_forgetting)) if cen_forgetting else 0.0
        )
