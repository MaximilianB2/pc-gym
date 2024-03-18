""" policy evaluation metrics
- general flavour follows the paper by Manon Flageat et al (2024) - https://arxiv.org/pdf/2312.07178.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class metric_base(ABC):
    def __init__(self, scalarised_weight):
        pass

    def evaluate(self, policy_evaluator):
        """
        Evaluate the given policy using the specified environment.

        Parameters:
        - policy_evaluator: The policy evaluator to generate data for a number of policy rollouts.

        Returns:
        - The evaluation metric value.
        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def policy_dispersion_metric(self, data):
        """
        Evaluate the dispersion of the policy.

        Returns:
        - The policy dispersion metric value.
        - data: nested dictionary
        """
        raise NotImplementedError("Subclasses must implement the policy_dispersion_metric method.")

    def policy_performance_metric(self, data):
        """
        Evaluate the performance of the policy.

        Returns:
        - The policy performance metric value.
        """
        raise NotImplementedError("Subclasses must implement the policy_performance_metric method.")

    def scalarised_performance(self, data):
        """
        Evaluate the scalarised performance of the policy.

        Returns:
        - The scalarised policy performance metric value.
        """
        raise NotImplementedError("Subclasses must implement the scalarised_performance method.")



class standard_deviation:
    def __init__(self, data):
        self.data = data

    def get_value(self):
        """
        Evaluate the dispersion of the policy.

        Returns:
        - The policy dispersion metric value.
        - data: numpy array
        """
        return np.std(data, axis=-1)



class median_absolute_deviation:
    def __init__(self, data):
        self.data = data

    def get_value(self):
        """
        Evaluate the dispersion of the policy.

        Returns:
        - The policy dispersion metric value.
        - data: numpy array
        """
        return np.median(np.abs(data - np.median(data,axis=-1)), axis=-1)


class mean_performance:
    def __init__(self, data):
        self.data = data

    def get_value(self):
        """
        Evaluate the performance of the policy.

        Returns:
        - The policy performance metric value.
        """
        return np.mean(data, axis=-1)


class median_performance:
    def __init__(self, data):
        self.data = data

    def get_value(self):
        """
        Evaluate the performance of the policy.

        Returns:
        - The policy performance metric value.
        """
        return np.median(data, axis=-1)


class reproducibility_metric(metric_base):
    def __init__(self, dispersion:str, performance:str, scalarised_weight:float):

        # scalarised weight is defined in terms of the upper confidence bound
        # it should be negative for the lower confidence bound
        self.scalarised_weight = scalarised_weight
        if dispersion == 'std':
            self.dispersion = standard_deviation
        elif dispersion == 'mad':
            self.dispersion = median_absolute_deviation
        else:
            raise ValueError("Invalid dispersion metric")
        
        if performance == 'mean':
            self.performance = mean_performance
        elif performance == 'median':
            self.performance = median_performance
        else:
            raise ValueError("Invalid performance metric")

    def evaluate(self, policy_evaluator):
        """
        Evaluate the given policy using the specified environment.

        Parameters
        ----------
        policy_evaluator : The policy evaluator to generate data for a number of policy rollouts.
                            must be constructed prior to passing to evaluate
        """
        self.data = policy_evaluator.get_rollouts()

        return self.scalarised_performance(self.data)


    def policy_dispersion_metric(self, data:dict, component:str):
        """
        Evaluate the dispersion of the policy.

        Returns
        -------
        The policy dispersion metric value.
        """
        values = {k: {} for k in data.keys()}


        for policy in data.keys(): # has structure n_x x T x reps  - operation always applied along the reps row
            if component is None:
                for component in data[policy].keys():
                    operation = self.determine_op(component)
                    values[policy][component] = self.dispersion(operation(data[policy][component])).get_value()
            else:
                operation = self.determine_op(component)
                values[policy][component] = self.dispersion(operation(data[policy][component])).get_value()
            

        return values

    def policy_performance_metric(self, data:dict, component:str):
        """
        Evaluate the performance of the policy.

        Returns
        -------
        The policy performance metric value.
        """
        values = {k: {} for k in data.keys()}

        for policy in data.keys():
            if component is None:
                for component in data[policy].keys():
                    operation = self.determine_op(component)
                    values[policy][component] = self.performance(operation(data[policy][component])).get_value()
            else:
                operation = self.determine_op(component)
                values[policy][component] = self.performance(operation(data[policy][component])).get_value()

        return values

    def scalarised_performance(self, data:dict, component:str):
        """
        Evaluate the scalarised performance of the policy.

        Returns
        -------
        The scalarised policy performance metric value.
        set component to None to scalarise over all components
        """
        performance = self.policy_performance_metric(data, component)
        dispersion = self.policy_dispersion_metric(data, component)

        return {k: {component: performance[k][component] + self.scalarised_weight * dispersion[k][component] for component in performance[k].keys()} for k in performance.keys()}

    def determine_op(self, component):

        if component == 'x': 
            return lambda x: x
        elif component == 'u':
            return lambda x: x
        elif component == 'r':
            return lambda x: np.sum(x, axis=1) # sum over discrete time indices (undiscounted)
        elif component == 'g':
            return lambda x: np.max(x, axis=0) # return the greatest constraint violation of n_g defined