"""Matched policy evaluation, with physical measurements and explicit episode lengths."""

import copy
import inspect

import matplotlib.pyplot as plt
import numpy as np

from pcgym.evaluation_metrics import component_values, episode_returns, masked_reduce
from pcgym.oracle import oracle


class policy_eval:
    """Evaluate policies on the same seeded scenarios.

    ``x`` contains full unmasked measurements in physical units; ``u`` contains applied
    absolute controls. State/reward columns run from reset to the terminal observation;
    action columns describe the intervening transitions. Unused columns are NaN.
    """

    def __init__(
        self,
        make_env: callable,
        policies: dict,
        reps: int,
        env_params: dict,
        oracle: bool = False,
        MPC_params: dict = False,
        cons_viol: bool = False,
        save_fig: bool = False,
        *,
        seed: int = 0,
        seeds=None,
    ):
        if isinstance(reps, bool) or not isinstance(reps, (int, np.integer)) or reps < 1:
            raise ValueError("reps must be a positive integer")
        if seeds is None:
            seeds = np.random.SeedSequence(seed).generate_state(reps).tolist()
        if len(seeds) != reps or any(
            isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0 for s in seeds
        ):
            raise ValueError("seeds must contain one nonnegative integer per repetition")
        self.seeds = [int(s) for s in seeds]
        self.make_env = make_env
        self.env_params = copy.deepcopy(env_params)
        self.env = make_env(copy.deepcopy(env_params))
        self.policies = policies
        self.n_pi = len(policies)
        self.reps = int(reps)
        self.oracle = oracle
        self.cons_viol = cons_viol
        self.save_fig = save_fig
        self.MPC_params = MPC_params
        if oracle:
            self._validate_oracle()

    def _validate_oracle(self):
        if (
            self.env.custom_reward
            or self.env.reward != "SP_reward_fn"
            or self.env.env_params.get("noise", False)
            or self.env.uncertainty
            or self.env.a_delta
            or (self.env.constraint_active and self.env.done_on_constraint)
        ):
            raise ValueError(
                "oracle comparisons support only nominal setpoint tracking with absolute actions "
                "and without early constraint termination; batch/custom rewards, noise, uncertainty "
                "and delta actions need a scenario-matched oracle implementation"
            )

    def rollout(self, policy_i, *, seed=None, return_metadata=False):
        """Run one episode, stopping at either done flag.

        The default return remains ``(rewards, x, u, g)``. Arrays have their full horizon
        shapes, with NaN padding after termination. ``return_metadata=True`` appends a
        dictionary containing length, masks, final flags and the scenario seed.
        Policies with SB3 recurrent arguments receive fresh state/episode-start flags.
        Stochastic custom policies can implement ``reset(seed=...)`` for matched actions.
        """
        rewards = np.full(self.env.N + 1, np.nan)
        states = np.full((self.env.Nx, self.env.N + 1), np.nan)
        actions = np.full((self.env.action_space.shape[0], self.env.N), np.nan)
        obs, info = self.env.reset(seed=seed)
        reset_policy = getattr(policy_i, "reset", None)
        if callable(reset_policy):
            reset_policy(seed=seed)
        rewards[0] = float(np.asarray(info["r_init"]).item())
        states[:, 0] = self._physical_obs(info)
        if not np.isfinite(rewards[0]) or not np.isfinite(states[:, 0]).all():
            raise ValueError("Nonfinite evaluation data at reset")
        recurrent = {"state", "episode_start"}.issubset(inspect.signature(policy_i.predict).parameters)
        policy_state = None
        for i in range(self.env.N):
            kwargs = {"deterministic": True}
            if recurrent:
                kwargs.update(state=policy_state, episode_start=np.array([i == 0]))
            action, policy_state = policy_i.predict(obs, **kwargs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            actions[:, i] = self._applied_action(action, info)
            states[:, i + 1] = self._physical_obs(info)
            rewards[i + 1] = float(np.asarray(reward).item())
            if (
                not np.isfinite(rewards[i + 1])
                or not np.isfinite(states[:, i + 1]).all()
                or not np.isfinite(actions[:, i]).all()
            ):
                raise ValueError(f"Nonfinite evaluation data at transition {i + 1}")
            if terminated or truncated:
                break
        length = i + 1
        constraints = np.full((self.env.n_con if self.env.constraint_active else 1, self.env.N + 1, 1), np.nan)
        if self.env.constraint_active:
            constraints[:, : length + 1] = info["cons_info"][:, : length + 1]
        else:
            constraints[:, : length + 1] = 0
        result = (rewards, states, actions, constraints)
        if return_metadata:
            return (*result, self._metadata(length, terminated, truncated, seed))
        return result

    def _metadata(self, length, terminated, truncated, seed):
        return {
            "lengths": length,
            "state_mask": np.arange(self.env.N + 1) <= length,
            "transition_mask": np.arange(self.env.N) < length,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "seeds": seed,
        }

    def _physical_obs(self, info):
        """Full measurement, including entries hidden from the policy, in physical units."""
        obs = np.asarray(info["obs"], dtype=float)
        if self.env.normalise_o:
            low, high = self.env.observation_space_base.low, self.env.observation_space_base.high
            obs = (obs + 1) * (high - low) / 2 + low
        return obs

    def _applied_action(self, action, info):
        if "u" in info:
            return info["u"]
        if self.env.normalise_a:
            low, high = self.env.env_params["a_space"]["low"], self.env.env_params["a_space"]["high"]
            return (action + 1) * (high - low) / 2 + low
        return action

    def _oracle_rewards_and_constraints(self, x, u):
        self._validate_oracle()
        # A separate environment keeps rewards/constraint histories out of the live episode.
        reward_env = self.make_env(copy.deepcopy(self.env_params))
        try:
            reward_env.reset(seed=self.seeds[0])
            rewards = [0.0]
            for t in range(1, x.shape[1]):
                reward_env.t = t - 1
                violated = False
                if reward_env.constraint_active:
                    violated = reward_env.constraint_check(x[:, t - 1], u[:, t - 1])
                reward_env.t = t
                if reward_env.constraint_active:
                    violated = reward_env.constraint_check(x[:, t], u[:, t - 1]) or violated
                rewards.append(float(reward_env.SP_reward_fn(x[:, t], violated)))
            constraints = copy.deepcopy(reward_env.info.get("cons_info"))
            return rewards, constraints
        finally:
            reward_env.close()

    def oracle_reward_fn(self, x: np.ndarray, u: np.ndarray) -> list:
        """Score a nominal tracking trajectory without changing the evaluator's episode."""
        return self._oracle_rewards_and_constraints(x, u)[0]

    def get_rollouts(self) -> dict:
        """Collect fixed-shape arrays and per-repetition validity metadata for every policy."""
        data = {}
        if self.oracle:
            self._validate_oracle()
            oracle_instance = oracle(self.make_env, self.env_params, self.MPC_params)
            x, u = oracle_instance.mpc()
            rewards, constraints = self._oracle_rewards_and_constraints(x, u)
            entry = {"r": np.asarray(rewards)[None, :], "x": x, "u": u}
            if constraints is not None:
                entry["g"] = constraints
            entry.update(self._metadata(self.env.N, self.env.finite_horizon, not self.env.finite_horizon, 0))
            data["oracle"] = {k: np.stack([v] * self.reps, axis=-1) for k, v in entry.items()}
            data["oracle"]["seeds"] = np.asarray(self.seeds)
        for name, policy in self.policies.items():
            episodes = []
            for seed in self.seeds:
                rewards, states, actions, constraints, metadata = self.rollout(policy, seed=seed, return_metadata=True)
                entry = {"r": rewards[None, :], "x": states, "u": actions, **metadata}
                if self.env.constraint_active:
                    entry["g"] = constraints
                episodes.append(entry)
            data[name] = {k: np.stack([episode[k] for episode in episodes], axis=-1) for k in episodes[0]}
        self.data = data
        return data

    def plot_data(self, data, reward_dist=False):
        """Plot physical trajectories and episode returns, excluding invalid columns."""
        model_info = self.env.model.info()
        state_names = model_info["states"]
        input_names = model_info["inputs"]
        disturbances = self.env.disturbances if self.env.disturbance_active else {}
        constraints = self.env.env_params.get("constraints", {})
        constraints = constraints if isinstance(constraints, dict) else {}
        time = np.linspace(0, self.env.tsim, self.env.N + 1)
        colors = {name: plt.get_cmap("tab20")(i % 20) for i, name in enumerate(data)}

        def schedule(values):
            values = np.asarray(values).reshape(-1)
            return values[np.minimum(np.arange(self.env.N + 1), len(values) - 1)]

        def band(ax, values, name, step=False):
            center = masked_reduce(values, "median", axis=-1)
            lower = masked_reduce(values, "min", axis=-1)
            upper = masked_reduce(values, "max", axis=-1)
            if step:
                center, lower, upper = (np.append(v, v[-1]) for v in (center, lower, upper))
            ax.plot(time, center, label=name, color=colors[name], drawstyle="steps-post" if step else "default")
            ax.fill_between(time, lower, upper, color=colors[name], alpha=0.2, step="post" if step else None)

        nrows = len(state_names) + len(input_names) + len(disturbances)
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 2.5 * nrows), squeeze=False)
        for i, label in enumerate(state_names + input_names):
            ax = axes[i, 0]
            state = i < len(state_names)
            component, index = ("x", i) if state else ("u", i - len(state_names))
            for name, entry in data.items():
                band(ax, component_values(entry, component)[index], name, step=not state)
            if label in (self.env.SP or {}):
                ax.step(
                    time, schedule(self.env.SP[label]), where="post", color="black", linestyle="--", label="Setpoint"
                )
            if label in constraints:
                ax.hlines(constraints[label], time[0], time[-1], color="black", label="Constraint")
            ax.set_ylabel(label)
        for offset, (name, values) in enumerate(disturbances.items(), len(state_names) + len(input_names)):
            ax = axes[offset, 0]
            ax.step(time, schedule(values), where="post", label=name, color="tab:orange")
            ax.set_ylabel(name)
        for ax in axes[:, 0]:
            ax.set(xlabel="Time (model units)", xlim=(time[0], time[-1]))
            ax.grid(True)
            ax.legend(loc="best")
        fig.tight_layout()
        if self.save_fig:
            fig.savefig("rollout.pdf")
        plt.show()

        if self.cons_viol and self.env.constraint_active:
            fig, axes = plt.subplots(self.env.n_con, 1, figsize=(10, 3 * self.env.n_con), squeeze=False)
            names = [name for name, bounds in constraints.items() for _ in bounds]
            for i, ax in enumerate(axes[:, 0]):
                for name, entry in data.items():
                    if "g" in entry:
                        # Positive residuals are violations; feasible slack cannot cancel them.
                        band(ax, np.maximum(component_values(entry, "g")[i, :, 0, :], 0), name)
                ax.set(xlabel="Time (model units)", ylabel=names[i] if names else f"g{i}", title="Positive violation")
                ax.grid(True)
                ax.legend(loc="best")
            fig.tight_layout()
            plt.show()

        if reward_dist:
            fig, ax = plt.subplots(figsize=(10, 6))
            returns = {name: episode_returns(entry).ravel() for name, entry in data.items()}
            bins = np.histogram_bin_edges(np.concatenate(list(returns.values())), bins="auto")
            for name, values in returns.items():
                ax.hist(values, bins=bins, alpha=0.5, label=name, color=colors[name])
            ax.set(xlabel="Episode return", ylabel="Episodes", title="Distribution of episode returns")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.show()
