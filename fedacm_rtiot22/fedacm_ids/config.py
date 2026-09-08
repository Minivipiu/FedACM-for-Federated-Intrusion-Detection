"""Method presets and ablation flags.

One implementation covers the seven compared methods: each is a preset over the
same mechanisms, so all of them share the backbone, the data pipeline, the client
partitions and the incremental exposure sequence.

    method            aggregate  memory  inference  KD scope  local extra
    fedavg            yes        --      softmax    --        --
    fedprox           yes        --      softmax    --        mu_prox
    fedproto          yes        simple  prototype  --        lambda_P
    lwf               yes        --      softmax    old       lambda_KD
    ewc               yes        --      softmax    --        Fisher penalty
    local-finetuning  no         --      softmax    --        --
    fedacm            yes        raca    prototype  memory    all of Eq. (2)

The ``ablate-*`` run-config flags switch off one mechanism of the FedACM preset at
a time.
"""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_METHODS = {
    "fedavg",
    "fedprox",
    "fedproto",
    "fedacm",
    "local-finetuning",
    "lwf",
    "ewc",
}


@dataclass(frozen=True)
class MethodSettings:
    method: str
    aggregate_model: bool
    collect_concepts: bool
    use_memory: bool
    memory_aggregation: str
    inference_mode: str
    model_aggregation: str
    use_fedprox: bool
    use_proto_loss: bool
    use_margin_loss: bool
    use_kd_loss: bool
    use_ewc: bool
    use_uncertainty_weight: bool
    use_dispersion_weight: bool
    use_temporal_weight: bool
    use_drift_compensation: bool
    kd_scope: str


def as_bool(config: dict, key: str, default: bool = False) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_method_settings(config: dict) -> MethodSettings:
    """Resolve method and ablation flags from Flower run_config."""
    method = str(config.get("method", "fedacm")).strip().lower()
    if method not in SUPPORTED_METHODS:
        supported = ", ".join(sorted(SUPPORTED_METHODS))
        raise ValueError(f"Unsupported method '{method}'. Supported methods: {supported}")

    settings_by_method = {
        "fedavg": MethodSettings(
            method=method,
            aggregate_model=True,
            collect_concepts=False,
            use_memory=False,
            memory_aggregation="none",
            inference_mode="softmax",
            model_aggregation="num-examples",
            use_fedprox=False,
            use_proto_loss=False,
            use_margin_loss=False,
            use_kd_loss=False,
            use_ewc=False,
            use_uncertainty_weight=False,
            use_dispersion_weight=False,
            use_temporal_weight=False,
            use_drift_compensation=False,
            kd_scope="none",
        ),
        "fedprox": MethodSettings(
            method=method,
            aggregate_model=True,
            collect_concepts=False,
            use_memory=False,
            memory_aggregation="none",
            inference_mode="softmax",
            model_aggregation="num-examples",
            use_fedprox=True,
            use_proto_loss=False,
            use_margin_loss=False,
            use_kd_loss=False,
            use_ewc=False,
            use_uncertainty_weight=False,
            use_dispersion_weight=False,
            use_temporal_weight=False,
            use_drift_compensation=False,
            kd_scope="none",
        ),
        "fedproto": MethodSettings(
            method=method,
            aggregate_model=True,
            collect_concepts=True,
            use_memory=True,
            memory_aggregation="simple",
            inference_mode="prototype",
            model_aggregation="num-examples",
            use_fedprox=False,
            use_proto_loss=True,
            use_margin_loss=False,
            use_kd_loss=False,
            use_ewc=False,
            use_uncertainty_weight=False,
            use_dispersion_weight=False,
            use_temporal_weight=False,
            use_drift_compensation=False,
            kd_scope="none",
        ),
        "fedacm": MethodSettings(
            method=method,
            aggregate_model=True,
            collect_concepts=True,
            use_memory=True,
            memory_aggregation="raca",
            inference_mode="prototype",
            model_aggregation="num-examples",
            use_fedprox=True,
            use_proto_loss=True,
            use_margin_loss=True,
            use_kd_loss=True,
            use_ewc=False,
            use_uncertainty_weight=True,
            use_dispersion_weight=True,
            use_temporal_weight=True,
            use_drift_compensation=True,
            kd_scope="memory",
        ),
        "local-finetuning": MethodSettings(
            method=method,
            aggregate_model=False,
            collect_concepts=False,
            use_memory=False,
            memory_aggregation="none",
            inference_mode="softmax",
            model_aggregation="none",
            use_fedprox=False,
            use_proto_loss=False,
            use_margin_loss=False,
            use_kd_loss=False,
            use_ewc=False,
            use_uncertainty_weight=False,
            use_dispersion_weight=False,
            use_temporal_weight=False,
            use_drift_compensation=False,
            kd_scope="none",
        ),
        "lwf": MethodSettings(
            method=method,
            aggregate_model=True,
            collect_concepts=False,
            use_memory=False,
            memory_aggregation="none",
            inference_mode="softmax",
            model_aggregation="num-examples",
            use_fedprox=False,
            use_proto_loss=False,
            use_margin_loss=False,
            use_kd_loss=True,
            use_ewc=False,
            use_uncertainty_weight=False,
            use_dispersion_weight=False,
            use_temporal_weight=False,
            use_drift_compensation=False,
            kd_scope="old",
        ),
        "ewc": MethodSettings(
            method=method,
            aggregate_model=True,
            collect_concepts=False,
            use_memory=False,
            memory_aggregation="none",
            inference_mode="softmax",
            model_aggregation="num-examples",
            use_fedprox=False,
            use_proto_loss=False,
            use_margin_loss=False,
            use_kd_loss=False,
            use_ewc=True,
            use_uncertainty_weight=False,
            use_dispersion_weight=False,
            use_temporal_weight=False,
            use_drift_compensation=False,
            kd_scope="none",
        ),
    }

    settings = settings_by_method[method]

    memory_aggregation = str(
        config.get("memory-aggregation", "auto")
    ).strip().lower()
    inference_mode = str(config.get("inference-mode", "auto")).strip().lower()
    model_aggregation = str(config.get("model-aggregation", "auto")).strip().lower()

    if memory_aggregation == "auto":
        memory_aggregation = settings.memory_aggregation
    if inference_mode == "auto":
        inference_mode = settings.inference_mode
    if model_aggregation == "auto":
        model_aggregation = settings.model_aggregation

    if as_bool(config, "ablate-raca", False) and memory_aggregation == "raca":
        memory_aggregation = "simple"
    if as_bool(config, "ablate-memory", False):
        memory_aggregation = "none"
        inference_mode = "softmax"

    return MethodSettings(
        method=settings.method,
        aggregate_model=settings.aggregate_model and not as_bool(config, "ablate-model-aggregation", False),
        collect_concepts=settings.collect_concepts and not as_bool(config, "ablate-memory", False),
        use_memory=settings.use_memory and memory_aggregation != "none",
        memory_aggregation=memory_aggregation,
        inference_mode="softmax" if as_bool(config, "ablate-prototype-inference", False) else inference_mode,
        model_aggregation=model_aggregation,
        use_fedprox=settings.use_fedprox and not as_bool(config, "ablate-fedprox", False),
        use_proto_loss=settings.use_proto_loss and not as_bool(config, "ablate-proto", False),
        use_margin_loss=settings.use_margin_loss and not as_bool(config, "ablate-margin", False),
        use_kd_loss=settings.use_kd_loss and not as_bool(config, "ablate-kd", False),
        use_ewc=settings.use_ewc and not as_bool(config, "ablate-ewc", False),
        use_uncertainty_weight=settings.use_uncertainty_weight
        and not as_bool(config, "ablate-uncertainty", False),
        use_dispersion_weight=settings.use_dispersion_weight
        and not as_bool(config, "ablate-dispersion", False),
        use_temporal_weight=settings.use_temporal_weight
        and not as_bool(config, "ablate-temporal", False),
        use_drift_compensation=settings.use_drift_compensation
        and not as_bool(config, "ablate-drift", False),
        kd_scope=(
            settings.kd_scope
            if str(config.get("kd-scope", "auto")).strip().lower() == "auto"
            else str(config.get("kd-scope", settings.kd_scope)).strip().lower()
        ),
    )
