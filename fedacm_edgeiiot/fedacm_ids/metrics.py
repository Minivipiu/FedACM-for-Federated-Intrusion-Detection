"""Support-weighted aggregation of client-level metrics.

The client-level evaluation view of Section 4.5: the global state is scored on each
client's local non-IID test partition and averaged with the client evaluation
support as weight. The complementary global view is computed server-side in
``task.evaluate_global`` and reported under the ``cen_*`` prefix.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate scalar metrics, tolerating keys missing on some clients."""
    totals: dict[str, float] = defaultdict(float)
    weights: dict[str, int] = defaultdict(int)

    for num_examples, client_metrics in metrics:
        for key, value in client_metrics.items():
            if isinstance(value, bool):
                numeric_value = float(value)
            elif isinstance(value, (int, float)):
                numeric_value = float(value)
            else:
                continue

            totals[key] += num_examples * numeric_value
            weights[key] += num_examples

    return {
        key: totals[key] / weights[key]
        for key in totals
        if weights[key] > 0
    }
