"""Policy performance and dispersion across valid episode data.

Rewards reduce to undiscounted episode returns; states/actions retain their time
axis. Constraint metrics use the greatest positive residual across constraints.
"""

from abc import ABC

import numpy as np

COMPONENTS = ("x", "u", "r", "g")


def masked_reduce(data, operation, axis=-1):
    """Ignore NaN padding, returning NaN (without warnings) for unobserved cells."""
    values = np.ma.masked_invalid(np.asarray(data, dtype=float))
    return np.ma.filled(getattr(np.ma, operation)(values, axis=axis), np.nan)


def component_values(entry, component):
    """Apply explicit validity masks, also accepting legacy NaN-padded dictionaries."""
    if component not in COMPONENTS:
        raise ValueError(f"Unknown evaluation component: {component}")
    values = np.asarray(entry[component], dtype=float)
    key = "transition_mask" if component == "u" else "state_mask"
    if key in entry:
        mask = np.asarray(entry[key], dtype=bool)
        if component == "g":
            mask = mask[:, None, :]
        values = np.where(mask, values, np.nan)
    return values


def episode_returns(entry):
    """One undiscounted return per repetition, excluding reset and unused columns."""
    return np.nansum(component_values(entry, "r")[:, 1:, :], axis=1)


class metric_base(ABC):
    def __init__(self, scalarised_weight):
        self.scalarised_weight = scalarised_weight

    def evaluate(self, policy_evaluator):
        raise NotImplementedError

    def policy_dispersion_metric(self, data):
        raise NotImplementedError

    def policy_performance_metric(self, data):
        raise NotImplementedError

    def scalarised_performance(self, data):
        raise NotImplementedError


class standard_deviation:
    def __init__(self, data):
        self.data = data

    def get_value(self):
        return masked_reduce(self.data, "std")


class median_absolute_deviation:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)

    def get_value(self):
        median = np.expand_dims(masked_reduce(self.data, "median"), -1)
        return masked_reduce(np.abs(self.data - median), "median")


class mean_performance:
    def __init__(self, data):
        self.data = data

    def get_value(self):
        return masked_reduce(self.data, "mean")


class median_performance:
    def __init__(self, data):
        self.data = data

    def get_value(self):
        return masked_reduce(self.data, "median")


class reproducibility_metric(metric_base):
    """Reduce valid repetitions; scalarisation is performance + weight * dispersion."""

    def __init__(self, dispersion: str, performance: str, scalarised_weight: float):
        super().__init__(scalarised_weight)
        dispersions = {"std": standard_deviation, "mad": median_absolute_deviation}
        performances = {"mean": mean_performance, "median": median_performance}
        if dispersion not in dispersions:
            raise ValueError("Invalid dispersion metric")
        if performance not in performances:
            raise ValueError("Invalid performance metric")
        self.dispersion = dispersions[dispersion]
        self.performance = performances[performance]

    def evaluate(self, policy_evaluator, component=None):
        self.data = getattr(policy_evaluator, "data", None)
        if self.data is None:
            self.data = policy_evaluator.get_rollouts()
        return self.scalarised_performance(self.data, component)

    def _measure(self, data, component, measure):
        if component is not None:
            self.determine_op(component)  # Validate even for empty data.
        result = {}
        for policy, entry in data.items():
            components = [component] if component else [key for key in COMPONENTS if key in entry]
            result[policy] = {
                key: measure(self.determine_op(key)(component_values(entry, key))).get_value() for key in components
            }
        return result

    def policy_dispersion_metric(self, data, component=None):
        return self._measure(data, component, self.dispersion)

    def policy_performance_metric(self, data, component=None):
        return self._measure(data, component, self.performance)

    def scalarised_performance(self, data, component=None):
        performance = self.policy_performance_metric(data, component)
        dispersion = self.policy_dispersion_metric(data, component)
        return {
            policy: {key: value + self.scalarised_weight * dispersion[policy][key] for key, value in entry.items()}
            for policy, entry in performance.items()
        }

    def determine_op(self, component):
        if component in ("x", "u"):
            return lambda values: values
        if component == "r":
            return lambda values: np.nansum(values[:, 1:, :], axis=1)
        if component == "g":
            return lambda values: masked_reduce(np.maximum(values, 0), "max", axis=0)
        raise ValueError(f"Unknown evaluation component: {component}")
