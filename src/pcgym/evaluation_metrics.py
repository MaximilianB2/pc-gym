"""
Policy evaluation metrics.

This module implements various metrics for evaluating policies, following the general approach
described in the paper by Manon Flageat et al (2024) - https://arxiv.org/pdf/2312.07178.pdf
"""

import numpy as np
from abc import ABC


class metric_base(ABC):
    """
    Abstract base class for policy evaluation metrics.
    """

    def __init__(self, scalarised_weight):
        """
        Initialize the metric base.

        Args:
            scalarised_weight: The weight for scalarised performance.
        """
        pass

    def evaluate(self, policy_evaluator):
        """
        Evaluate the given policy using the specified environment.

        Args:
            policy_evaluator: The policy evaluator to generate data for a number of policy rollouts.

        Returns:
            The evaluation metric value.
        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def policy_dispersion_metric(self, data):
        """
        Evaluate the dispersion of the policy.

        Args:
            data: Nested dictionary containing policy data.

        Returns:
            The policy dispersion metric value.
        """
        raise NotImplementedError(
            "Subclasses must implement the policy_dispersion_metric method."
        )

    def policy_performance_metric(self, data):
        """
        Evaluate the performance of the policy.

        Args:
            data: Nested dictionary containing policy data.

        Returns:
            The policy performance metric value.
        """
        raise NotImplementedError(
            "Subclasses must implement the policy_performance_metric method."
        )

    def scalarised_performance(self, data):
        """
        Evaluate the scalarised performance of the policy.

        Args:
            data: Nested dictionary containing policy data.

        Returns:
            The scalarised policy performance metric value.
        """
        raise NotImplementedError(
            "Subclasses must implement the scalarised_performance method."
        )


class standard_deviation:
    """
    Class for calculating standard deviation.
    """

    def __init__(self, data):
        """
        Initialize the standard deviation calculator.

        Args:
            data: Input data for standard deviation calculation.
        """
        self.data = data

    def get_value(self):
        """
        Calculate the standard deviation of the data.

        Returns:
            The standard deviation value.
        """
        return np.std(self.data, axis=-1)


class median_absolute_deviation:
    """
    Class for calculating median absolute deviation.
    """

    def __init__(self, data):
        """
        Initialize the median absolute deviation calculator.

        Args:
            data: Input data for median absolute deviation calculation.
        """
        if data.ndim < 2:
            data = data.reshape((data.shape[0], 1))
        self.data = data

    def get_value(self):
        """
        Calculate the median absolute deviation of the data.

        Returns:
            The median absolute deviation value.
        """
        return np.median(
            np.abs(self.data - np.median(self.data, axis=-1)),
            axis=-1,  # Currently only works for the reward component
        )


class mean_performance:
    """
    Class for calculating mean performance.
    """

    def __init__(self, data):
        """
        Initialize the mean performance calculator.

        Args:
            data: Input data for mean performance calculation.
        """
        self.data = data

    def get_value(self):
        """
        Calculate the mean performance of the data.

        Returns:
            The mean performance value.
        """
        return np.mean(self.data, axis=-1)


class median_performance:
    """
    Class for calculating median performance.
    """

    def __init__(self, data):
        """
        Initialize the median performance calculator.

        Args:
            data: Input data for median performance calculation.
        """
        self.data = data

    def get_value(self):
        """
        Calculate the median performance of the data.

        Returns:
            The median performance value.
        """
        return np.median(self.data, axis=-1)


class reproducibility_metric(metric_base):
    """
    Class for calculating reproducibility metrics.
    """

    def __init__(self, dispersion: str, performance: str, scalarised_weight: float):
        """
        Initialize the reproducibility metric.

        Args:
            dispersion: The dispersion metric to use ('std' or 'mad').
            performance: The performance metric to use ('mean' or 'median').
            scalarised_weight: The weight for scalarised performance.
        """
        # scalarised weight is defined in terms of the upper confidence bound
        # it should be negative for the lower confidence bound
        self.scalarised_weight = scalarised_weight
        if dispersion == "std":
            self.dispersion = standard_deviation
        elif dispersion == "mad":
            self.dispersion = median_absolute_deviation
        else:
            raise ValueError("Invalid dispersion metric")

        if performance == "mean":
            self.performance = mean_performance
        elif performance == "median":
            self.performance = median_performance
        else:
            raise ValueError("Invalid performance metric")

    def evaluate(self, policy_evaluator, component: str = None):
        """
        Evaluate the given policy using the specified environment.

        Args:
            policy_evaluator: The policy evaluator to generate data for a number of policy rollouts.
            component: The specific component to evaluate (optional).

        Returns:
            The evaluation metric value.
        """
        try:
            self.data = policy_evaluator.data  # Try to get data from policy evaluator if this fails then call the get_rollouts method to generate the data
        except Exception:
            self.data = policy_evaluator.get_rollouts()

        return self.scalarised_performance(self.data, component)

    def policy_dispersion_metric(self, data: dict, component: str):
        """
        Evaluate the dispersion of the policy.

        Args:
            data: Nested dictionary containing policy data.
            component: The specific component to evaluate.

        Returns:
            The policy dispersion metric value.
        """
        values = {k: {} for k in data.keys()}

        for policy in data.keys():  # has structure n_x x T x reps  - operation always applied along the reps row
            if component is None:
                for comp in data[policy].keys():
                    operation = self.determine_op(comp)
                    values[policy][comp] = self.dispersion(
                        operation(data[policy][comp])
                    ).get_value()
            else:
                operation = self.determine_op(component)
                values[policy][component] = self.dispersion(
                    operation(data[policy][component])
                ).get_value()

        return values

    def policy_performance_metric(self, data: dict, component: str):
        """
        Evaluate the performance of the policy.

        Args:
            data: Nested dictionary containing policy data.
            component: The specific component to evaluate.

        Returns:
            The policy performance metric value.
        """
        values = {k: {} for k in data.keys()}

        for policy in data.keys():
            if component is None:
                for comp in data[policy].keys():
                    operation = self.determine_op(comp)
                    values[policy][comp] = self.performance(
                        operation(data[policy][comp])
                    ).get_value()
            else:
                operation = self.determine_op(component)
                values[policy][component] = self.performance(
                    operation(data[policy][component])
                ).get_value()

        return values

    def scalarised_performance(self, data: dict, component: str):
        """
        Evaluate the scalarised performance of the policy.

        Args:
            data: Nested dictionary containing policy data.
            component: The specific component to evaluate (set to None to scalarise over all components).

        Returns:
            The scalarised policy performance metric value.
        """
        performance = self.policy_performance_metric(data, component)
        dispersion = self.policy_dispersion_metric(data, component)
        return {
            k: {
                comp: performance[k][comp]
                + self.scalarised_weight * dispersion[k][comp]
                for comp in performance[k].keys()
            }
            for k in performance.keys()
        }

    def determine_op(self, component):
        """
        Determine the operation to be applied based on the component.

        Args:
            component: The component to determine the operation for.

        Returns:
            A lambda function representing the operation to be applied.
        """
        if component == "x":
            return lambda x: x
        elif component == "u":
            return lambda x: x
        elif component == "r":
            return lambda x: x  # sum over discrete time indices (undiscounted)
        elif component == "g":
            return lambda x: np.max(
                x, axis=0
            )  # return the greatest constraint violation of n_g defined