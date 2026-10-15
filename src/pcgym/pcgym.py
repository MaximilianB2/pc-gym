import copy
from numbers import Integral, Real
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pcgym.integrator import integration_engine
from pcgym.model_defaults import get_default, has_defaults
from pcgym.models import Batch, CustomReward, Regulation, get_model_spec
from pcgym.policy_evaluation import policy_eval


def _seq_at(seq, i: int):
    """Return seq[i], holding the last value for indices past the end (e.g. the final state x_N)."""
    return seq[min(i, len(seq) - 1)]


class make_env(gym.Env):
    def __init__(self, env_params: dict) -> None:
        """Initialize the environment with given parameters.

        Args:
            env_params (dict): Environment configuration parameters including model selection,
                                spaces, simulation parameters, constraints, and custom functions.

        """
        if not isinstance(env_params, dict):
            raise ValueError("env_params must be a dictionary")
        self.env_params = copy.deepcopy(env_params)
        self._validate_simulation_config()
        self._initialize_action_config()
        self._apply_space_defaults()
        for key, normalise in (("a_space", self.normalise_a), ("o_space", self.normalise_o)):
            self.env_params[key] = self._validate_bounds(
                key, self.env_params[key], finite=key == "a_space" or normalise, strict=normalise
            )
        self._apply_reward_defaults()
        self._setup_spaces()
        self._configure_reward()
        self._setup_simulation_params()
        self._initialize_model()
        self._setup_state_dimensions()
        self._validate_task_config()
        self._validate_x0()
        self._validate_model_episode(self.x0)
        self._setup_constraints()
        self._setup_disturbances()
        self._setup_custom_reward()
        self._setup_uncertainty()
        self._setup_episode_type()
        self._noise_percentage_setup()
        self._setup_partial_observations()
        self._set_public_observation_support()
        self.Nx = len(self.observation_info())

    @staticmethod
    def _finite_scalar(value, name, nonnegative=False):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if nonnegative and value < 0:
            raise ValueError(f"{name} must be nonnegative")
        return float(value)

    def _validate_simulation_config(self):
        for name in ("normalise_a", "normalise_o", "a_delta", "noise"):
            if name in self.env_params and not isinstance(self.env_params[name], bool):
                raise ValueError(f"{name} must be a boolean")
        n = self.env_params.get("N")
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n <= 0:
            raise ValueError("N must be a positive integer")
        tsim = self._finite_scalar(self.env_params.get("tsim"), "tsim")
        if tsim <= 0 or not np.isfinite(tsim / n) or tsim / n <= 0:
            raise ValueError("tsim and tsim / N must be finite and positive")
        if self.env_params.get("integration_method", "casadi") not in ("casadi", "jax"):
            raise ValueError("integration_method must be 'casadi' or 'jax'")
        if self.env_params.get("SP") is not None and not isinstance(self.env_params["SP"], dict):
            raise ValueError("SP must be a dictionary of named schedules")

    @staticmethod
    def _validate_bounds(name, bounds, finite=False, strict=False, size=None):
        if not isinstance(bounds, dict) or not {"low", "high"} <= bounds.keys():
            raise ValueError(f"{name} must contain low and high vectors")
        try:
            low, high = (np.asarray(bounds[k], dtype=np.float64) for k in ("low", "high"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric bounds") from exc
        if low.ndim != 1 or high.shape != low.shape or (size is not None and low.size != size):
            raise ValueError(
                f"{name} must contain matching one-dimensional bounds"
                + (f" with {size} entries" if size is not None else "")
            )
        if np.isnan(low).any() or np.isnan(high).any() or np.any(low > high):
            raise ValueError(f"{name} requires ordered bounds without NaN")
        if finite and (not np.isfinite(low).all() or not np.isfinite(high).all()):
            raise ValueError(f"{name} requires finite bounds")
        if strict:
            with np.errstate(over="ignore", invalid="ignore"):
                width = high - low
            if np.any(width <= 0) or not np.isfinite(width).all():
                raise ValueError(f"{name} requires finite, positive normalization widths")
        return {"low": low.copy(), "high": high.copy()}

    @staticmethod
    def _validate_names(name, names, allowed):
        if isinstance(names, str) or not isinstance(names, (list, tuple, dict)):
            raise ValueError(f"{name} must use names from {allowed}")
        unknown = [key for key in names if key not in allowed]
        if unknown:
            raise ValueError(f"{name} contains unknown names {unknown}; expected names from {allowed}")

    @classmethod
    def _validate_schedules(cls, name, schedules, allowed):
        if not isinstance(schedules, dict):
            raise ValueError(f"{name} must be a dictionary of named schedules")
        cls._validate_names(name, schedules, allowed)
        for key, values in schedules.items():
            cls._finite_vector(values, f"{name}[{key!r}]")

    @staticmethod
    def _finite_vector(values, name):
        try:
            values = np.asarray(values, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name} must be a finite numeric vector") from exc
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a nonempty finite one-dimensional vector")
        return values

    def _validate_task_config(self):
        states = self.model.info()["states"]
        if self.action_space.shape != (self.Nu,):
            raise ValueError(f"a_space must have {self.Nu} entries for model inputs {self.model.info()['inputs']}")
        if self.SP is not None:
            self._validate_schedules("SP", self.SP, states)
        if hasattr(self, "reward_states"):
            self._validate_names("reward_states", self.reward_states, states)
        if self.env_params.get("r_scale") is not None:
            scales = self.env_params["r_scale"]
            if not isinstance(scales, dict):
                raise ValueError("r_scale must be a dictionary")
            self._validate_names("r_scale", scales, states)
            for name, value in scales.items():
                self._finite_scalar(value, f"r_scale[{name!r}]")
        if self.a_delta:
            bounds = self._validate_bounds("a_space_act", self.env_params.get("a_space_act"), finite=True, size=self.Nu)
            self.env_params["a_space_act"] = bounds
            self.a_0 = self._finite_vector(self.a_0, "a_0").copy()
            if self.a_0.shape != (self.Nu,) or np.any(self.a_0 < bounds["low"]) or np.any(self.a_0 > bounds["high"]):
                raise ValueError("a_0 must match the inputs and lie within a_space_act")
        self._validate_model_input_bounds()

    def _validate_model_input_bounds(self):
        validator = getattr(self.model, "validate_input_bounds", None)
        if validator is not None:
            bounds = self.env_params["a_space_act" if self.a_delta else "a_space"]
            validator(bounds["low"], bounds["high"])

    def _validate_model_episode(self, state):
        validator = getattr(self.model, "validate_episode", None)
        if validator is not None:
            bounds = self.env_params["a_space_act" if self.a_delta else "a_space"]
            validator(state[: self.Nx_oracle], bounds["low"], bounds["high"], self.tsim)

    def _initialize_action_config(self):
        self.a_delta = self.env_params.get("a_delta", False)
        if self.a_delta:
            self.a_0 = self.env_params.get("a_0")
        self.normalise_a = self.env_params.get("normalise_a", True)
        self.normalise_o = self.env_params.get("normalise_o", True)

    def _apply_space_defaults(self):
        """Fill in a_space / o_space / x0 from the model's canonical defaults.

        For any of these keys the user omits, fall back to the model's native
        bounds / initial state (see pcgym.model_defaults) so large models don't
        need to be fully hand-specified. Custom models have no registered
        defaults and must supply all three explicitly.
        """
        model_name = self.env_params.get("model")
        is_custom = self.env_params.get("custom_model") is not None
        if not is_custom:
            get_model_spec(model_name)  # fail early with the list of models if the name is unknown

        for key in ("a_space", "o_space", "x0"):
            if self.env_params.get(key) is not None:
                continue
            if is_custom:
                raise ValueError(f"'{key}' must be provided in env_params when using a custom_model.")
            if not has_defaults(model_name):
                raise ValueError(
                    f"'{key}' is missing from env_params and model '{model_name}' has no registered "
                    f"defaults. Please specify '{key}' explicitly."
                )
            self.env_params[key] = get_default(model_name, key)

    def _apply_reward_defaults(self):
        """Use the model's default task when no reward is configured (no SP, custom_reward or reward_states).

        Regulation models track constant setpoints: ``SP`` is filled in, x0's setpoint entries are set to the
        setpoint values (added if x0 only has the model states), and unless ``r_scale`` is given, each
        squared tracking error is scaled by ``1 / (o_space range)**2``. Batch models get their
        ``reward_states`` / ``maximise_reward``. Explicitly configured
        rewards are never modified.
        """
        p = self.env_params
        if p.get("custom_model") is not None or any(
            p.get(key) is not None for key in ("SP", "custom_reward", "reward_states")
        ):
            return
        spec = get_model_spec(p.get("model"))
        task = spec.task
        if task is None:
            return
        if isinstance(task, Batch):
            p["reward_states"] = list(task.reward_states)
            p.setdefault("maximise_reward", task.maximise)
            return
        if task.finite_horizon:
            p.setdefault("finite_horizon", True)
        if isinstance(task, CustomReward):
            p["custom_reward"] = task.reward
            return
        assert isinstance(task, Regulation)
        try:
            states = spec.cls(int_method="casadi", **(p.get("model_params") or {})).info()["states"]
        except TypeError:
            return  # invalid model_params; reported with a clear message by _initialize_model
        n_x, names = len(states), list(task.setpoint)
        values = [float(task.setpoint[k]) for k in names]
        p["SP"] = {k: [v] * p["N"] for k, v in zip(names, values)}

        x0 = np.asarray(p["x0"], dtype=float).reshape(-1)
        if x0.shape[0] in (n_x, n_x + len(names)):
            p["x0"] = np.concatenate([x0[:n_x], values])

        low = np.asarray(p["o_space"]["low"], dtype=float)
        high = np.asarray(p["o_space"]["high"], dtype=float)
        if low.size not in (n_x, n_x + len(names)):
            raise ValueError(f"o_space requires {n_x} model-state entries or {n_x + len(names)} state/setpoint entries")
        idx = [states.index(k) for k in names]
        if low.shape[0] == n_x:
            p["o_space"] = {"low": np.concatenate([low, low[idx]]), "high": np.concatenate([high, high[idx]])}
        if p.get("r_scale") is None:
            widths = high[idx] - low[idx]
            if not np.isfinite(widths).all() or np.any(widths <= 0):
                raise ValueError("o_space requires finite positive tracked-state widths for default r_scale")
            p["r_scale"] = {k: 1.0 / (high[i] - low[i]) ** 2 for k, i in zip(names, idx)}

    def _noise_percentage_setup(self):
        self.noise_percentage = self.env_params.get("noise_percentage")
        self._noise_scale = np.zeros(self.Nx_oracle)
        if not self.env_params.get("noise", False):
            return
        states = self.model.info()["states"]
        if isinstance(self.noise_percentage, dict):
            self._validate_names("noise_percentage", self.noise_percentage, states)
            for name, value in self.noise_percentage.items():
                self._noise_scale[states.index(name)] = self._finite_scalar(value, "noise_percentage", True)
        else:
            self._noise_scale[:] = self._finite_scalar(self.noise_percentage, "noise_percentage", True)

    def _setup_spaces(self):
        if self.normalise_a:
            dim = self.env_params["a_space"]["low"].shape[0]
            self.action_space = spaces.Box(low=np.array([-1] * dim), high=np.array([1] * dim))
        else:
            self.action_space = spaces.Box(
                low=self.env_params["a_space"]["low"].astype(np.float32),
                high=self.env_params["a_space"]["high"].astype(np.float32),
                dtype=np.float32,
            )

        self._set_observation_spaces(self.env_params["o_space"]["low"], self.env_params["o_space"]["high"])

    def _set_observation_spaces(self, low, high):
        """Keep physical bounds at simulation precision and public observations in float32."""
        self.observation_space_base = spaces.Box(
            low=np.asarray(low, dtype=np.float64), high=np.asarray(high, dtype=np.float64), dtype=np.float64
        )
        if self.normalise_o:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=self.observation_space_base.shape, dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=self.observation_space_base.low.astype(np.float32),
                high=self.observation_space_base.high.astype(np.float32),
                dtype=np.float32,
            )

    def _configure_reward(self):
        self.maximise_reward = True
        self.SP = self.env_params.get("SP")
        custom_reward = self.env_params.get("custom_reward")
        if custom_reward is not None and not callable(custom_reward):
            raise ValueError("custom_reward must be callable")

        if self.SP is not None and self.env_params.get("custom_reward") is None:
            self.reward = "SP_reward_fn"
        elif self.SP is None and self.env_params.get("custom_reward") is None:
            self.reward = "batch_reward_fn"
            if "reward_states" not in self.env_params:
                raise ValueError(
                    "No reward is configured: give a setpoint ('SP'), a 'custom_reward', or 'reward_states' "
                    "(with 'maximise_reward') for an end-of-episode batch reward. This model has no default task."
                )
            self.reward_states = self.env_params["reward_states"]
            self.maximise_reward = self.env_params.get("maximise_reward", True)

    def _setup_simulation_params(self):
        self.N = int(self.env_params["N"])
        self.tsim = float(self.env_params["tsim"])
        self.x0 = np.array(self.env_params["x0"], dtype=np.float64, copy=True).reshape(-1)
        self.integration_method = self.env_params.get("integration_method", "casadi")
        self.dt = self.tsim / self.N
        self.done = False
        self._has_reset = False

    def _setup_constraints(self):
        self.constraint_active = False
        self.r_penalty = False
        self.custom_constraint_active = False
        self.info = {}

        constraints = self.env_params.get("constraints")
        if constraints is None:
            return

        if isinstance(constraints, dict):
            constraints = self._dict_constraints_to_callable(constraints)
        elif not callable(constraints):
            raise ValueError("constraints must be a dictionary or callable")

        self.constraints = constraints
        self.done_on_constraint = self.env_params["done_on_cons_vio"]
        self.r_penalty = self.env_params["r_penalty"]
        self.constraint_active = True
        self.n_con = self.constraints(self.x0, self.action_space.sample()).shape[0]
        # One column per visited state x_0..x_N.
        self.info["cons_info"] = np.zeros((self.n_con, self.N + 1, 1))

    def _dict_constraints_to_callable(self, constraints_dict):
        cons_type = self.env_params.get("cons_type", {})
        state_names = self.model.info()["states"]
        self._validate_names("constraints", constraints_dict, state_names)
        if not isinstance(cons_type, dict):
            raise ValueError("cons_type must be a dictionary")
        self._validate_names("cons_type", cons_type, list(constraints_dict))
        for state_name, bounds in constraints_dict.items():
            bounds = self._finite_vector(bounds, f"constraints[{state_name!r}]")
            ops = cons_type.get(state_name, [">=", "<="])
            if (
                not isinstance(ops, (list, tuple))
                or len(ops) != len(bounds)
                or any(op not in (">=", "<=") for op in ops)
            ):
                raise ValueError("cons_type must provide one '>=' or '<=' per constraint bound")

        def _constraints(x, u):
            g = []
            for state_name, bounds in constraints_dict.items():
                idx = state_names.index(state_name)
                types = cons_type.get(state_name, [">=", "<="])
                for bound, op in zip(bounds, types):
                    if op == ">=":
                        g.append(bound - x[idx])
                    else:
                        g.append(x[idx] - bound)
            return np.array(g).reshape(-1)

        return _constraints

    def _initialize_model(self):

        model_params = self.env_params.get("model_params") or {}
        if not isinstance(model_params, dict):
            raise ValueError("model_params must be a dictionary of {parameter_name: value}")

        if self.env_params.get("custom_model") is not None:
            m = self.env_params["custom_model"]
            m.int_method = self.integration_method
            self._apply_model_params(m, model_params)
            self.model = m
        else:
            model_cls = get_model_spec(self.env_params.get("model")).cls
            self._validate_model_params(model_cls, model_params)
            # Pass overrides to the constructor so any __post_init__ derived
            # quantities (e.g. state lists sized by a parameter) are rebuilt.
            self.model = model_cls(int_method=self.integration_method, **model_params)

    @staticmethod
    def _validate_model_params(model_cls, model_params):
        """Raise a helpful error if an override is not a parameter of the model."""
        valid = getattr(model_cls, "__dataclass_fields__", None)
        if valid is None:
            return
        unknown = [k for k in model_params if k not in valid]
        if unknown:
            settable = sorted(
                k for k in valid if k not in ("int_method", "states", "inputs", "disturbances", "uncertainties")
            )
            raise ValueError(
                f"Unknown model_params {unknown} for model '{model_cls.__name__}'. Settable parameters: {settable}"
            )

    @staticmethod
    def _apply_model_params(model, model_params):
        """Override parameters on an already-constructed (custom) model instance."""
        for name, value in model_params.items():
            if not hasattr(model, name):
                raise ValueError(f"Unknown model_params '{name}' for custom model '{type(model).__name__}'")
            setattr(model, name, value)

    def _setup_state_dimensions(self):
        self.Nx = len(self.model.info()["states"])
        if self.SP is not None:
            self.Nx += len(self.SP)
        self.Nx_oracle = len(self.model.info()["states"])
        self.Nu = len(self.model.info()["inputs"])

    def _validate_x0(self):
        """Check x0 and o_space have one entry per model state followed by one entry per setpoint."""
        layout = list(self.model.info()["states"])
        if self.SP is not None:
            layout += [f"{k}_SP" for k in self.SP]
        hint = (
            f"the model states followed by one entry per setpoint, in this order: {layout}. "
            "Disturbance, uncertain-parameter and time entries are appended automatically and must not be included."
        )
        n_x0 = np.asarray(self.x0).reshape(-1).shape[0]
        if n_x0 != len(layout):
            raise ValueError(f"x0 has {n_x0} entries but {len(layout)} are expected: {hint}")
        n_obs = self.observation_space_base.shape[0]
        if n_obs != len(layout):
            raise ValueError(f"o_space has {n_obs} entries but {len(layout)} are expected: {hint}")
        if not np.isfinite(self.x0).all():
            raise ValueError("x0 must contain only finite values")
        validator = getattr(self.model, "validate_initial_state", None)
        if validator is not None:
            validator(self.x0[: self.Nx_oracle])

    def _setup_disturbances(self):
        self.disturbance_active = False
        self.Nd = self.Nd_model = 0

        if self.env_params.get("disturbances") is not None:
            self._validate_schedules(
                "disturbances", self.env_params["disturbances"], self.model.info().get("disturbances") or []
            )
            self.disturbance_active = True
            self.disturbances = self.env_params["disturbances"]
            self.Nd = len(self.disturbances)
            self.Nd_model = len(self.model.info()["disturbances"])
            self.Nu += self.Nd_model
            self.Nx += self.Nd

            bounds = self._validate_bounds(
                "disturbance_bounds",
                self.env_params.get("disturbance_bounds"),
                finite=self.normalise_o,
                strict=self.normalise_o,
                size=self.Nd,
            )
            dist_low, dist_high = bounds["low"], bounds["high"]
            extended_obs_low = np.concatenate((self.observation_space_base.low, dist_low))
            extended_obs_high = np.concatenate((self.observation_space_base.high, dist_high))
            self._set_observation_spaces(extended_obs_low, extended_obs_high)

    def _setup_custom_reward(self):
        self.custom_reward = False
        if self.env_params.get("custom_reward") is not None:
            self.custom_reward = True
            self.custom_reward_f = self.env_params["custom_reward"]

    def _setup_partial_observations(self):
        self.partial_observation = False
        if self.env_params.get("partial_observation") is not None:
            self.partial_observation = self.env_params["partial_observation"]
            if self.partial_observation is not False:
                self._validate_names("partial_observation", self.partial_observation, self.model.info()["states"])

    def _setup_uncertainty(self):
        self.uncertainty = False
        self.NUn = 0
        self.uncertainty_percentages = self.env_params.get("uncertainty_percentages")
        self.empirical_distribution = self.env_params.get("empirical_distribution")
        self.original_param_values = {}
        self._x0_uncertainty = {}
        self._unbounded_observation_indices = []
        self.distribution = self.env_params.get("distribution", "uniform")
        if self.distribution is None:
            self.distribution = "uniform"
        if self.uncertainty_percentages is not None and self.empirical_distribution is not None:
            raise ValueError("Choose only one of uncertainty_percentages and empirical_distribution")
        percentage_mode = self.uncertainty_percentages is not None
        config = self.uncertainty_percentages if percentage_mode else self.empirical_distribution
        if config is None:
            return
        self.uncertainty = True
        label = "uncertainty_percentages" if percentage_mode else "empirical_distribution"
        if not isinstance(config, dict):
            raise ValueError(f"{label} must be a dictionary")
        if self.distribution not in ("uniform", "normal"):
            raise ValueError("distribution must be 'uniform' or 'normal'")

        x0_config = config.get("x0", {})
        if percentage_mode and not isinstance(x0_config, dict):
            x0_config = dict(enumerate(self._finite_vector(x0_config, f"{label} x0")))
        if not isinstance(x0_config, dict):
            raise ValueError(f"{label} x0 must map state indices to sample vectors")
        for idx, values in x0_config.items():
            if isinstance(idx, (bool, np.bool_)) or not isinstance(idx, Integral) or not 0 <= idx < self.Nx_oracle:
                raise ValueError(f"{label} x0 index {idx!r} must refer to a model state (0..{self.Nx_oracle - 1})")
            if percentage_mode:
                self._x0_uncertainty[idx] = self._finite_scalar(values, f"{label} x0", True)
                radius = self._finite_scalar(values * abs(self.x0[idx]), f"{label} x0 scale", True)
                if self.distribution == "normal" and radius > 0:
                    self._unbounded_observation_indices.append(idx)
                    continue
                interval = (self.x0[idx] - radius, self.x0[idx] + radius)
            else:
                self._x0_uncertainty[idx] = self._finite_vector(values, f"{label} x0")
                interval = (self._x0_uncertainty[idx].min(), self._x0_uncertainty[idx].max())
            if (
                interval[0] < self.observation_space_base.low[idx]
                or interval[1] > self.observation_space_base.high[idx]
            ):
                raise ValueError("o_space must contain the full uniform or empirical x0 uncertainty support")

        support = []
        parameters = self.model.info().get("parameters", {})
        for name, values in config.items():
            if name == "x0":
                continue
            if name not in parameters or not hasattr(self.model, name):
                raise ValueError(f"{label} contains unknown model parameter {name!r}")
            nominal = self._finite_scalar(getattr(self.model, name), f"{label} parameter {name!r}")
            self.original_param_values[name] = nominal
            if percentage_mode:
                pct = self._finite_scalar(values, f"{label}[{name!r}]", True)
                radius = self._finite_scalar(abs(nominal) * pct, f"{label} scale", True)
                if self.distribution == "normal" and radius > 0:
                    self._unbounded_observation_indices.append(self.observation_space_base.shape[0] + len(support))
                    support.append(None)
                else:
                    support.append((nominal - radius, nominal + radius))
            else:
                samples = self._finite_vector(values, f"{label}[{name!r}]")
                support.append((samples.min(), samples.max()))

        self.NUn = len(self.original_param_values)
        if not self.NUn and self.env_params.get("uncertainty_bounds") is None:
            return  # Initial-state uncertainty adds no observation entries.
        bounds = self._validate_bounds(
            "uncertainty_bounds",
            self.env_params.get("uncertainty_bounds"),
            finite=self.normalise_o,
            strict=self.normalise_o,
            size=self.NUn,
        )
        for i, interval in enumerate(support):
            if interval is not None and (interval[0] < bounds["low"][i] or interval[1] > bounds["high"][i]):
                raise ValueError("uncertainty_bounds must contain the full uniform or empirical support")
        self._set_observation_spaces(
            np.concatenate((self.observation_space_base.low, bounds["low"])),
            np.concatenate((self.observation_space_base.high, bounds["high"])),
        )

    def _setup_episode_type(self):
        batch = not self.custom_reward and self.reward == "batch_reward_fn" and bool(self.reward_states)
        self.finite_horizon = self.env_params.get("finite_horizon", batch)
        if not isinstance(self.finite_horizon, bool):
            raise ValueError("finite_horizon must be a boolean")
        if batch and not self.finite_horizon:
            raise ValueError("A terminal batch reward requires finite_horizon=True")
        if self.finite_horizon:
            self._set_observation_spaces(
                np.append(self.observation_space_base.low, 0.0),
                np.append(self.observation_space_base.high, 1.0),
            )

    def _set_public_observation_support(self):
        """Keep finite scaling bounds separate from Gaussian support and masked values."""
        low, high = self.observation_space.low.copy(), self.observation_space.high.copy()
        indices = self._unbounded_observation_indices + list(np.flatnonzero(self._noise_scale))
        low[indices], high[indices] = -np.inf, np.inf
        if self.partial_observation is not False:
            for i, name in enumerate(self.model.info()["states"]):
                if name not in self.partial_observation:
                    low[i] = high[i] = 0.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _uncertain_param_names(self) -> list:
        if not self.uncertainty:
            return []
        if self.uncertainty_percentages is not None:
            return [p for p in self.uncertainty_percentages if p != "x0"]
        return [p for p in self.empirical_distribution if p != "x0"]

    def observation_info(self) -> list[tuple[str, str]]:
        """Describe the observation (and state) vector, one ``(name, kind)`` entry per index.

        The layout is: model states (``"state"``), one entry per setpoint (``"setpoint"``, named
        ``"<state>_SP"``), active disturbances (``"disturbance"``), uncertain model parameters
        (``"uncertainty"``) and, for finite-horizon tasks, ``("time_remaining", "time")``.
        ``x0`` and ``o_space`` cover the states and setpoints only; the
        disturbance and uncertainty entries are appended automatically (their bounds come from
        ``disturbance_bounds`` / ``uncertainty_bounds``). Time is a fraction with bounds [0, 1].

        Returns:
            list[tuple[str, str]]: e.g. ``[("Ca", "state"), ("T", "state"), ("Ca_SP", "setpoint")]``.
        """
        info = [(name, "state") for name in self.model.info()["states"]]
        if self.SP is not None:
            info += [(f"{k}_SP", "setpoint") for k in self.SP]
        if self.disturbance_active:
            info += [(k, "disturbance") for k in self.model.info()["disturbances"] if k in self.disturbances]
        info += [(p, "uncertainty") for p in self._uncertain_param_names()]
        if self.finite_horizon:
            info += [("time_remaining", "time")]
        return info

    def build_obs(
        self,
        states,
        setpoints=None,
        disturbances=None,
        uncertainties=None,
        normalise: bool = False,
        time_remaining: float = 1.0,
    ) -> np.ndarray:
        """Assemble a correctly ordered observation vector from named parts.

        Each part is either a dict keyed by name (state name, setpoint state name, disturbance or
        parameter name) or a sequence in ``observation_info()`` order. Omitted optional parts
        default to the first setpoint value, the first disturbance value and the nominal parameter
        value respectively.

        Args:
            states: Values of the model states.
            setpoints: Setpoint values, keyed by the tracked state name (e.g. ``{"Ca": 0.85}``).
            disturbances: Disturbance values, keyed by disturbance name.
            uncertainties: Uncertain parameter values, keyed by parameter name.
            normalise: If True, scale the result like the env does when ``normalise_o`` is set.
            time_remaining: Fraction of the horizon remaining, in [0, 1]. Used only by finite-horizon tasks.

        Returns:
            np.ndarray: A float64 vector, before partial-observation masking. When not normalised,
            its first ``len(states) + len(SP)`` entries are a valid ``x0``. Reset/step cast policy
            observations to ``observation_space.dtype`` separately.
        """
        defaults = {
            "setpoint": {k: _seq_at(v, 0) for k, v in (self.SP or {}).items()},
            "disturbance": {k: _seq_at(v, 0) for k, v in self.disturbances.items()} if self.disturbance_active else {},
            "uncertainty": getattr(self, "original_param_values", {}),
        }
        parts = {"state": states, "setpoint": setpoints, "disturbance": disturbances, "uncertainty": uncertainties}
        layout = self.observation_info()

        values = []
        for kind in ("state", "setpoint", "disturbance", "uncertainty"):
            names = [name[:-3] if kind == "setpoint" else name for name, k in layout if k == kind]
            part = parts[kind]
            if part is None:
                if kind == "state":
                    raise ValueError("states are required")
                part = defaults[kind]
            if isinstance(part, dict):
                missing = [n for n in names if n not in part]
                unknown = [n for n in part if n not in names]
                if missing or unknown:
                    raise ValueError(
                        f"{kind} values must be given for exactly {names} (missing {missing}, unknown {unknown})"
                    )
                values += [part[n] for n in names]
            else:
                part = np.asarray(part, dtype=float).reshape(-1)
                if part.shape[0] != len(names):
                    raise ValueError(f"expected {len(names)} {kind} values ({names}), got {part.shape[0]}")
                values += list(part)

        if self.finite_horizon:
            time_remaining = self._finite_scalar(time_remaining, "time_remaining")
            if not 0 <= time_remaining <= 1:
                raise ValueError("time_remaining must lie in [0, 1]")
            values.append(time_remaining)

        obs = np.asarray(values, dtype=float)
        if normalise:
            obs = self._normalise_observation(obs)
        return obs

    def _normalise_observation(self, obs):
        low, high = self.observation_space_base.low, self.observation_space_base.high
        return 2 * (obs - low) / (high - low) - 1

    def _update_scheduled_state(self):
        """Expose setpoints and disturbances at the current state index, including x_N."""
        start = self.Nx_oracle
        if self.SP is not None:
            self.state[start : start + len(self.SP)] = [_seq_at(values, self.t) for values in self.SP.values()]
            start += len(self.SP)
        if self.disturbance_active:
            self.state[start : start + self.Nd] = [
                _seq_at(self.disturbances[name], self.t)
                for name in self.model.info()["disturbances"]
                if name in self.disturbances
            ]
        if self.finite_horizon:
            self.state[-1] = 1 - self.t / self.N

    def _get_observation(self):
        """Format a policy observation without rounding or masking the physical measurement."""
        if self.normalise_o:
            self.normobs = self._normalise_observation(self.obs)
            full_obs = self.normobs
        else:
            full_obs = self.obs

        # Evaluation retains the full measurement at simulation precision. No clipping:
        # physical excursions must remain visible rather than being hidden at the bounds.
        self.info["obs"] = full_obs.copy()
        obs = full_obs.astype(self.observation_space.dtype, copy=True)
        if self.partial_observation is not False:
            for i, name in enumerate(self.model.info()["states"]):
                if name not in self.partial_observation:
                    obs[i] = 0
        return obs

    def apply_uncertainties(self, value, percentage, distribution):
        if distribution == "normal":
            return self.np_random.normal(value, percentage * abs(value))
        # default: uniform around the nominal value
        noise = self.np_random.uniform(-percentage, percentage)
        return value * (1 + noise)

    def _measure_state(self):
        self.obs = self.state.copy()
        if np.any(self._noise_scale):
            self.obs[: self.Nx_oracle] += (
                self.np_random.normal(size=self.Nx_oracle) * self.state[: self.Nx_oracle] * self._noise_scale
            )

    def reset(self, seed: int | None = None, options: dict | None = None, **kwargs) -> tuple[np.array, dict]:
        """
        Reset the environment to its initial state.

        This method resets the environment's state, time, and other relevant variables.
        It's called at the beginning of each episode.

        Args:
            seed (int, optional): Seeds the environment's random number generator (``self.np_random``),
                which drives observation noise and parametric/initial-state uncertainty. If None, the
                generator is left as is (seeded from OS entropy on first use).
            options (dict, optional): Unused; accepted for Gymnasium API compatibility.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing:
                - numpy.array: An independent float32 observation.
                - dict: An independent information snapshot (e.g., initial reward).
        """
        super().reset(seed=seed)
        self._has_reset = False
        self.t = 0

        # Initialize state with potential random uncertainties in x0
        state = np.array(self.env_params["x0"], dtype=np.float64, copy=True).reshape(-1)
        for idx, values in self._x0_uncertainty.items():
            state[idx] = (
                self.apply_uncertainties(state[idx], values, self.distribution)
                if self.uncertainty_percentages is not None
                else self.np_random.choice(values)
            )

        # If disturbances are active, expand the initial state with disturbances
        if self.disturbance_active:
            initial_disturbances = []
            for k in self.model.info()["disturbances"]:
                if k in self.disturbances:
                    initial_disturbances.append(self.disturbances[k][0])

            # Append initial disturbances to the state
            state = np.concatenate((state, initial_disturbances))

        # Handle initial uncertainties
        if self.uncertainty:
            uncertain_params = []
            if self.uncertainty_percentages is not None:
                for param, percentage in self.uncertainty_percentages.items():
                    if param != "x0":  # x0 handled separately
                        original_value = self.original_param_values[param]
                        new_value = self.apply_uncertainties(original_value, percentage, self.distribution)
                        setattr(self.model, param, new_value)
                        uncertain_params.append(new_value)
                state = np.concatenate((state, uncertain_params))
            elif self.empirical_distribution is not None:
                for param in self._uncertain_param_names():
                    sample = self.np_random.choice(self.empirical_distribution[param])
                    setattr(self.model, param, sample)
                    uncertain_params.append(sample)
                state = np.concatenate((state, uncertain_params))

        if self.finite_horizon:
            state = np.append(state, 1.0)

        validator = getattr(self.model, "validate_initial_state", None)
        if validator is not None:
            validator(state[: self.Nx_oracle])
        self._validate_model_input_bounds()

        self._validate_model_episode(state)

        # Built once from this env's own model; the engine rebuilds itself if model parameters change
        # (e.g. resampled uncertain parameters above, or user edits to env.model).
        if getattr(self, "int_eng", None) is None:
            self.int_eng = integration_engine(self)

        if self.a_delta:
            self.a_save = self.a_0.copy()

        self.info = {}
        if self.constraint_active:
            self.info["cons_info"] = np.zeros((self.n_con, self.N + 1, 1))

        self.state = state
        self._update_scheduled_state()
        self._measure_state()
        self.done = False
        self._has_reset = True

        obs_to_return = self._get_observation()
        self.info["r_init"] = 0.0
        return obs_to_return, copy.deepcopy(self.info)

    def step(self, action: np.array) -> tuple[np.array, float, bool, bool, dict]:
        """
        Perform one time step in the environment.

        This method takes an action, applies it to the environment, and returns
        the next state, reward, and other information.

        Args:
            action (numpy.array): The action to be taken in the environment.

        Returns:
            tuple: A tuple containing:
                - numpy.array: An independent float32 observation of the next state.
                - float: The reward for the current step.
                - bool: Whether the episode has ended.
                - bool: Whether the episode was truncated.
                - dict: An independent information snapshot for the step.
        """

        if not self._has_reset or self.done:
            raise gym.error.ResetNeeded("Call reset() before stepping a new or completed episode")
        try:
            action = np.asarray(action, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("action must be a finite numeric vector") from exc
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError(f"action must be a finite vector with shape {self.action_space.shape}")
        low, high = self.action_space.low, self.action_space.high
        if not self.normalise_a:
            # Accept exact physical endpoints and their float32 public representations.
            low = np.minimum(low, self.env_params["a_space"]["low"])
            high = np.maximum(high, self.env_params["a_space"]["high"])
        if np.any(action < low) or np.any(action > high):
            raise ValueError("action must lie within action_space bounds")

        # Create control vector
        uk = np.zeros(self.Nu)
        if self.normalise_a is True:
            action = (action + 1) * (
                self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
            ) / 2 + self.env_params["a_space"]["low"]
        else:
            action = np.clip(action, self.env_params["a_space"]["low"], self.env_params["a_space"]["high"])
        if self.a_delta:
            # `action` is a change in input: accumulate it and apply the clipped absolute input.
            action = np.clip(
                self.a_save + action,
                self.env_params["a_space_act"]["low"],
                self.env_params["a_space_act"]["high"],
            )

        # Add disturbance to control vector
        if self.disturbance_active:
            uk[: self.Nu - self.Nd_model] = action
            for i, k in enumerate(self.model.info()["disturbances"]):
                if k in self.disturbances:
                    # d_t acts over the interval taking x_t to x_(t+1).
                    uk[self.Nu - self.Nd_model + i] = _seq_at(self.disturbances[k], self.t)
                else:
                    uk[self.Nu - self.Nd_model + i] = self.model.info()["parameters"][str(k)]
        else:
            uk = action  # Add action to control vector

        # An unsuccessful solve must leave state, time, applied input and info untouched.
        try:
            if self.integration_method == "casadi":
                result = self.int_eng.casadi_step(self.state.copy(), uk.copy())["xf"].full()
                next_state = np.asarray(result, dtype=np.float64)
                if next_state.shape != (self.Nx_oracle, 1):
                    raise ValueError(f"unexpected solver state shape {next_state.shape}")
                next_state = next_state[:, 0]
            else:
                next_state = np.asarray(self.int_eng.jax_step(self.state.copy(), uk.copy()), dtype=np.float64)
                if next_state.shape != (self.Nx_oracle,):
                    raise ValueError(f"unexpected solver state shape {next_state.shape}")
            projector = getattr(self.model, "project_state", None)
            if projector is not None:
                next_state = np.asarray(projector(next_state), dtype=np.float64)
                if next_state.shape != (self.Nx_oracle,):
                    raise ValueError(f"unexpected projected state shape {next_state.shape}")
            if not np.isfinite(next_state).all():
                raise ValueError("solver returned nonfinite state values")
        except Exception as exc:
            raise RuntimeError(
                f"{self.integration_method} integration failed at step {self.t} for "
                f"{type(self.model).__name__} with inputs {uk.tolist()}: {exc}"
            ) from exc

        initial_violation = False
        if self.t == 0 and self.constraint_active:
            initial_violation = self.constraint_check(self.state, uk)
        self.state[: self.Nx_oracle] = next_state
        if self.a_delta:
            self.a_save = action.copy()
        self.info["u"] = action.copy()

        # Observations and rewards describe x_(t+1), with SP[t+1] and the next disturbance d[t+1].
        self.t += 1
        self._update_scheduled_state()

        # Check if constraints are violated
        constraint_violated = False
        if self.constraint_active:
            constraint_violated = self.constraint_check(self.state, uk) or initial_violation

        # Batch objectives end the MDP; continuing regulation uses a time-limit truncation.
        terminated = bool(self.done or (self.finite_horizon and self.t >= self.N))
        truncated = self.t >= self.N and not terminated
        self.done = terminated or truncated

        # Copy the obs from the state and add noise if the user requests this
        self._measure_state()

        if self.custom_reward:
            rew = self.custom_reward_f(self, self.obs, uk, constraint_violated)

        elif not self.custom_reward and self.reward == "SP_reward_fn":
            rew = self.SP_reward_fn(self.state, constraint_violated)

        elif not self.custom_reward and self.reward != "SP_reward_fn":
            rew = self.batch_reward_fn(self.state, constraint_violated)

        else:
            raise ValueError("Reward not valid function")
        obs_to_return = self._get_observation()
        return obs_to_return, rew, terminated, truncated, copy.deepcopy(self.info)

    def simulate(self, x0, u_seq, params: dict | None = None) -> np.ndarray:
        """Integrate the model from x0 under an input sequence without touching the episode.

        Intended for model-based control (MPC, shooting, MBRL): candidate input sequences can be
        evaluated with the env's model, dt and integration backend while the env's t, state,
        history and info stay unchanged.

        Args:
            x0: Initial model states (length ``Nx`` of the model). A longer vector, such as a full
                state/observation vector in physical units, is truncated to the model states.
            u_seq: Input sequence of shape ``(T, n_u)`` in physical units. ``n_u`` is either the
                number of control inputs or, when the model has disturbance inputs, the full model input
                vector (controls followed by disturbances). If only controls are given, disturbance
                inputs are held at their nominal model values.
            params: Optional model parameter overrides for this simulation only (e.g. for robust
                MPC), such as ``{"k0": 8e10}``.

        Returns:
            np.ndarray: State trajectory of shape ``(T + 1, Nx)``, starting with x0.
        """
        model = self.model
        if params:
            unknown = [k for k in params if not hasattr(model, k)]
            if unknown:
                raise ValueError(f"Unknown model parameter(s) {unknown}")
            model = copy.copy(model)
            for k, v in params.items():
                setattr(model, k, v)
            engine = integration_engine(
                SimpleNamespace(
                    model=model,
                    Nx_oracle=self.Nx_oracle,
                    Nu=self.Nu,
                    dt=self.dt,
                    integration_method=self.integration_method,
                )
            )
        else:
            if getattr(self, "int_eng", None) is None:
                self.int_eng = integration_engine(self)
            engine = self.int_eng

        u_seq = np.asarray(u_seq, dtype=float)
        if u_seq.ndim == 1:
            u_seq = u_seq.reshape(-1, 1)
        n_controls = len(model.info()["inputs"])
        disturbance_names = model.info().get("disturbances") or []
        if self.disturbance_active and u_seq.shape[1] == n_controls:
            nominal = [model.info()["parameters"][str(k)] for k in disturbance_names]
            u_seq = np.hstack([u_seq, np.tile(nominal, (u_seq.shape[0], 1))])
        if u_seq.shape[1] != self.Nu:
            raise ValueError(f"u_seq must have {n_controls} (controls) or {self.Nu} (model inputs) columns")

        x = np.asarray(x0, dtype=float).reshape(-1)[: self.Nx_oracle].copy()
        validator = getattr(model, "validate_initial_state", None)
        if validator is not None:
            validator(x)
        if len(u_seq):
            low, high = u_seq[:, :n_controls].min(axis=0), u_seq[:, :n_controls].max(axis=0)
            validator = getattr(model, "validate_input_bounds", None)
            if validator is not None:
                validator(low, high)
            validator = getattr(model, "validate_episode", None)
            if validator is not None:
                validator(x, low, high, len(u_seq) * self.dt)
        traj = np.zeros((u_seq.shape[0] + 1, self.Nx_oracle))
        traj[0] = x
        for k, uk in enumerate(u_seq):
            if self.integration_method == "casadi":
                x = np.array(engine.casadi_step(x, uk)["xf"].full()).reshape(self.Nx_oracle)
            else:
                x = np.asarray(engine.jax_step(x, uk))
            projector = getattr(model, "project_state", None)
            if projector is not None:
                x = np.asarray(projector(x), dtype=float)
            traj[k + 1] = x
        return traj

    def batch_reward_fn(self, state: np.array, c_violated: bool) -> float:
        """
        Compute the reward function for a batch

        Args:
            states (np.array): Current State of the system
            c_violated (bool): Whether any constraints were violated

        Returns:
            float: the computed reward
        """

        r = 0.0
        if self.t == self.N:
            # Get the full list of states from the model
            all_states = self.model.info()["states"]
            # Find indices of reward states that actually exist in the model
            reward_state_indices = [
                all_states.index(state_name) for state_name in self.reward_states if str(state_name) in all_states
            ]
            # Calculate reward based on those indices
            r_scale = self.env_params.get("r_scale", {})
            for state_index in reward_state_indices:
                state_name = all_states[state_index]
                if self.maximise_reward:
                    r += state[state_index] * r_scale.get(state_name, 1)
                else:
                    r -= state[state_index] * r_scale.get(state_name, 1)

            if self.r_penalty and c_violated:
                r -= 1000

        return r

    def SP_reward_fn(self, state: np.array, c_violated: bool) -> float:
        """
        Compute the reward for the current state and action.

        This method calculates the reward based on the current state and whether
        any constraints were violated.

        Args:
            state (numpy.array): The current state of the system.
            c_violated (bool): Whether any constraints were violated.

        Returns:
            float: The computed reward.
        """

        r = 0.0

        for k in self.SP:
            i = self.model.info()["states"].index(k)
            r_scale = self.env_params.get("r_scale", {})
            r += (-((state[i] - np.array(_seq_at(self.SP[k], self.t))) ** 2)) * r_scale.get(k, 1)
            if self.r_penalty and c_violated:
                r -= 1000
        return r

    def con_checker(self, curr_state: np.array, inputs: np.array) -> bool:
        """
        Check if any constraints are violated for the given states.

        Args:
            model_states (list): List of state or input names to check.
            curr_state (list): List of corresponding state or input values.

        Returns:
            bool: True if any constraint is violated, False otherwise.
        """
        constraint = self.constraints
        g = constraint(curr_state, inputs)
        self.info["cons_info"][:, self.t, :] = g.reshape(g.shape[0], 1)
        if np.any(self.info["cons_info"][:, self.t, :] > 0):
            return True
        else:
            return False

    def constraint_check(self, state: np.array, input: np.array) -> bool:
        """
        Check if any constraints are violated in the current step.

        This method checks both state and input constraints, as well as any
        custom constraints defined by the user.

        Args:
            state (numpy.array): The current state of the system, in physical units.
            input (numpy.array): The current control vector applied to the system, in physical units.

        Returns:
            bool: True if any constraint is violated, False otherwise.
        """

        self.con_i = 0

        # `state` and `input` are already in physical units (step() passes the raw state and the
        # de-normalised control vector), so no de-normalisation is applied here.
        constraint_violated = self.con_checker(state, input)  # Check both inputs and states

        if constraint_violated and self.done_on_constraint:
            self.done = True
        return constraint_violated

    def get_rollouts(
        self,
        policies: dict,
        reps: int,
        oracle: bool = False,
        dist_reward: bool = False,
        MPC_params: bool = False,
        cons_viol: bool = False,
        *,
        seed: int = 0,
        seeds=None,
    ) -> tuple[policy_eval, dict]:
        """
        Generate rollouts for the given policies.

        This method simulates the environment for multiple episodes using the provided policies.

        Args:
            policies (dict): Dictionary of policies to evaluate.
            reps (int): Number of rollouts to perform.
            oracle (bool, optional): Whether to use an oracle model for evaluation. Defaults to False.
            dist_reward (bool, optional): Whether to use reward distribution. Defaults to False.
            MPC_params (bool, optional): Whether to use MPC parameters. Defaults to False.
            cons_viol (bool, optional): Whether to track constraint violations. Defaults to False.
            seed (int): Base seed used to generate the matched episode seeds. Defaults to 0.
            seeds: Explicit nonnegative seeds, one per repetition; overrides seed.

        Returns:
            tuple: A tuple containing:
                - policy_eval: The policy evaluator object.
                - dict: Data from the rollouts.
        """
        # construct evaluator
        evaluator = policy_eval(
            make_env, policies, reps, self.env_params, oracle, MPC_params, cons_viol, seed=seed, seeds=seeds
        )
        # generate rollouts
        data = evaluator.get_rollouts()
        # return evaluator and data
        return evaluator, data

    def plot_rollout(
        self,
        policies: dict,
        reps: int,
        oracle: bool = False,
        dist_reward: bool = False,
        MPC_params: bool = False,
        cons_viol: bool = False,
        save_fig: bool = False,
        *,
        seed: int = 0,
        seeds=None,
    ) -> tuple[policy_eval, dict]:
        """
        Generate and plot rollouts for the given policies.

        This method simulates the environment for multiple episodes using the provided policies
        and plots the results.

        Args:
            policies (dict): Dictionary of policies to evaluate.
            reps (int): Number of rollouts to perform.
            oracle (bool, optional): Whether to use an oracle model for evaluation. Defaults to False.
            dist_reward (bool, optional): Whether to use reward distribution for plotting. Defaults to False.
            MPC_params (bool, optional): Whether to use MPC parameters. Defaults to False.
            cons_viol (bool, optional): Whether to track constraint violations. Defaults to False.
            save_fig (bool, optional): Whether to save the generated figures. Defaults to False.
            seed (int): Base seed used to generate the matched episode seeds. Defaults to 0.
            seeds: Explicit nonnegative seeds, one per repetition; overrides seed.

        Returns:
            tuple: A tuple containing:
                - policy_eval: The policy evaluator object.
                - dict: Data from the rollouts.
        """
        # construct evaluator
        evaluator = policy_eval(
            make_env, policies, reps, self.env_params, oracle, MPC_params, cons_viol, save_fig, seed=seed, seeds=seeds
        )
        # generate rollouts
        data = evaluator.get_rollouts()
        # plot data from rollouts via the evaluator method
        evaluator.plot_data(data, dist_reward)
        # return constructed evaluator and data
        return evaluator, data
