"""Flower ServerApp: wiring of the run configuration into the strategy.

Loads the dataset metadata once, derives the number of communication rounds from
the incremental protocol (``rounds-per-stage`` x ``num-stages``) and builds the
strategy. It also installs the server-side centralized evaluation callback, which
scores the global model and the memory on the pooled held-out test partition
covering all classes seen so far (the ``cen_*`` metrics of Section 4.5).
"""

from __future__ import annotations

import torch

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from fedacm_ids.config import resolve_method_settings
from fedacm_ids.metrics import weighted_average
from fedacm_ids.strategy import FedACMStrategy
from fedacm_ids.task import (
    evaluate_global,
    get_model,
    get_model_params,
    load_and_preprocess_dataset,
    parse_class_stages,
    set_random_seed,
)


def _as_int(config: dict, key: str, default: int) -> int:
    return int(config.get(key, default))


def _as_float(config: dict, key: str, default: float) -> float:
    return float(config.get(key, default))


def server_fn(context: Context):
    run_config = dict(context.run_config)
    settings = resolve_method_settings(run_config)
    dataset_path = str(run_config["dataset-path"])
    seed = _as_int(run_config, "seed", 42)
    set_random_seed(seed)
    test_fraction = _as_float(run_config, "test-fraction", 0.2)
    validation_fraction = _as_float(run_config, "validation-fraction", 0.0)
    evaluation_split = str(run_config.get("evaluation-split", "test"))
    embedding_dim = _as_int(run_config, "embedding-dim", 32)

    print(f"[Server] Loading dataset metadata for method={settings.method}")
    bundle, num_outputs, input_dim = load_and_preprocess_dataset(
        dataset_path,
        test_fraction=test_fraction,
        seed=seed,
        validation_fraction=validation_fraction,
        evaluation_split=evaluation_split,
    )
    if evaluation_split == "validation":
        print(
            "[Server] MODEL SELECTION RUN: evaluating on the validation subset "
            f"({100 * validation_fraction:.0f}% of the data). The held-out test "
            "partition is untouched; results are not comparable with reported ones."
        )
    class_stages = parse_class_stages(str(run_config.get("class-stages", "")), bundle.class_ids)
    configured_stages = _as_int(run_config, "num-stages", len(class_stages))
    class_stages = class_stages[:configured_stages]
    rounds_per_stage = _as_int(run_config, "rounds-per-stage", 1)
    num_rounds = rounds_per_stage * len(class_stages)

    print(
        f"[Server] Features={input_dim}, outputs={num_outputs}, "
        f"stages={len(class_stages)}, rounds={num_rounds}, "
        f"memory={settings.memory_aggregation}, inference={settings.inference_mode}"
    )

    model = get_model(embedding_dim=embedding_dim)
    initial_parameters = ndarrays_to_parameters(get_model_params(model))
    num_model_params = len(get_model_params(model))

    eval_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def centralized_eval_fn(stage, model_arrays, memory_class_ids, memory_prototypes, inference_mode):
        return evaluate_global(
            model_arrays,
            embedding_dim,
            bundle,
            class_stages,
            stage,
            memory_class_ids,
            memory_prototypes,
            inference_mode,
            eval_device,
        )

    min_available_clients = _as_int(run_config, "min-available-clients", 2)
    strategy = FedACMStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_available_clients,
        min_evaluate_clients=min_available_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters,
        class_stages=class_stages,
        rounds_per_stage=rounds_per_stage,
        local_epochs=_as_int(run_config, "local-epochs", 1),
        learning_rate=_as_float(run_config, "learning-rate", 0.001),
        batch_size=_as_int(run_config, "batch-size", 64),
        num_model_params=num_model_params,
        embedding_dim=embedding_dim,
        num_classes=num_outputs,
        settings=settings,
        fedprox_mu=_as_float(run_config, "fedprox-mu", 0.01),
        lambda_proto=_as_float(run_config, "lambda-proto", 0.25),
        lambda_margin=_as_float(run_config, "lambda-margin", 0.05),
        lambda_kd=_as_float(run_config, "lambda-kd", 0.05),
        kd_temperature=_as_float(run_config, "kd-temperature", 2.0),
        prototype_margin=_as_float(run_config, "prototype-margin", 1.0),
        ewc_lambda=_as_float(run_config, "ewc-lambda", 0.01),
        ewc_max_batches=_as_int(run_config, "ewc-max-batches", 8),
        raca_alpha=_as_float(run_config, "raca-alpha", 1.0),
        raca_beta=_as_float(run_config, "raca-beta", 1.0),
        raca_gamma=_as_float(run_config, "raca-gamma", 1.0),
        raca_delta=_as_float(run_config, "raca-delta", 0.1),
        raca_epsilon=_as_float(run_config, "raca-epsilon", 1e-6),
        memory_eta=_as_float(run_config, "memory-eta", 0.65),
        drift_tau=_as_float(run_config, "drift-tau", 1.0),
        centralized_eval_fn=centralized_eval_fn,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)
