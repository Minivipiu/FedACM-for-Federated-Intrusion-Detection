"""FedACM-IDS: federated class-incremental intrusion detection with an Attack
Concept Memory.

- ``task.py``       data pipeline, backbone, local objective, concept extraction,
                    nearest-prototype inference and metrics.
- ``strategy.py``   server side: model aggregation, RACA, drift compensation.
- ``config.py``     method presets and ablation flags.
- ``client_app.py`` / ``server_app.py``  Flower entry points.
- ``metrics.py``    support-weighted aggregation of client-level metrics.

Equation and section numbers in this package refer to the manuscript.
"""

__all__ = ["client_app", "server_app", "strategy", "task", "metrics"]
